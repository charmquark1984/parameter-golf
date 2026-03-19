"""
CPU training script v3: Training schedule experiments.
Adds configurable LR schedule (linear vs cosine warmdown),
label smoothing, and other training tricks.
"""

from __future__ import annotations

import glob
import io
import math
import os
import random
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train_cpu import (
    Muon, zeropower_via_newtonschulz5,
    build_sentencepiece_luts, load_data_shard, load_validation_tokens,
    TokenStream, RMSNorm, Rotary, apply_rotary_emb,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    CONTROL_TENSOR_NAME_PATTERNS,
)
from train_cpu_v2 import (
    CausalSelfAttention, make_mlp, Block, GPT,
)


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/sp1024")
    train_files = os.environ.get("TRAIN_FILES", os.path.join(data_path, "train_*.bin"))
    val_files = os.environ.get("VAL_FILES", os.path.join(data_path, "val_*.bin"))
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/sp1024.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 8192))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 25))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 5))

    iterations = int(os.environ.get("ITERATIONS", 100))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 20))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 0))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 4096))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 256))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 4))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))
    model_dim = int(os.environ.get("MODEL_DIM", 128))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 50))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Architecture extensions
    mlp_type = os.environ.get("MLP_TYPE", "relu2")
    share_kv = bool(int(os.environ.get("SHARE_KV", "0")))
    parallel_block = bool(int(os.environ.get("PARALLEL_BLOCK", "0")))

    # Training schedule extensions
    lr_schedule = os.environ.get("LR_SCHEDULE", "linear")  # linear, cosine, wsd
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", 0.0))
    warmup_frac = float(os.environ.get("WARMUP_FRAC", 0.0))  # fraction of training for LR warmup


def eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_seqs = max(args.val_batch_size // args.train_seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    val_loss_sum = 0.0
    val_token_count = 0.0
    val_byte_count = 0.0

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, total_seqs)
            raw_start = batch_start * args.train_seq_len
            raw_end = batch_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(dtype=torch.int64)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            # Always eval with no label smoothing
            logits = model.forward_logits(x)
            targets = y.reshape(-1)
            batch_loss = F.cross_entropy(logits, targets, reduction="mean").detach().item()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum().item()

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / val_byte_count
    model.train()
    return val_loss, bits_per_token * tokens_per_byte


class GPTv3(GPT):
    """Extended GPT with forward_logits for eval and label smoothing support."""

    def forward_logits(self, input_ids):
        """Return logits without computing loss (for eval)."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.float()

    def forward_with_smoothing(self, input_ids, target_ids, label_smoothing=0.0):
        logits = self.forward_logits(input_ids)
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits, targets, reduction="mean", label_smoothing=label_smoothing)


def main():
    args = Hyperparameters()
    device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    print(f"Val tokens: {val_tokens.numel() - 1}")

    model = GPTv3(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, mlp_type=args.mlp_type,
        share_kv=args.share_kv, parallel_block=args.parallel_block,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    print(f"MLP: {args.mlp_type}, LR schedule: {args.lr_schedule}, Label smoothing: {args.label_smoothing}")

    block_named_params = list(model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps
        )
        optimizers.insert(1, optimizer_head)

    train_stream = TokenStream(args.train_files)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step):
        # Warmup phase
        warmup_steps = int(args.warmup_frac * args.iterations)
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps

        if args.warmdown_iters <= 0:
            return 1.0

        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if step < warmdown_start:
            return 1.0

        # Warmdown phase
        progress = (step - warmdown_start) / max(args.warmdown_iters, 1)
        progress = min(progress, 1.0)

        if args.lr_schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif args.lr_schedule == "wsd":
            # Warmup-Stable-Decay: linear warmup, constant, then sqrt decay
            return max(1.0 - progress, 0.0) ** 0.5
        else:
            # Linear decay (default)
            return max(1.0 - progress, 0.0)

    val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"step:0/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    t0 = time.perf_counter()
    best_val_bpb = val_bpb

    model.train()
    for step in range(1, args.iterations + 1):
        scale = lr_mul(step)
        zero_grad_all()

        per_rank_span = args.train_batch_tokens + 1
        chunk = train_stream.take(per_rank_span)
        local = chunk.to(dtype=torch.int64)
        x = local[:-1].reshape(-1, args.train_seq_len)
        y = local[1:].reshape(-1, args.train_seq_len)

        loss = model.forward_with_smoothing(x, y, label_smoothing=args.label_smoothing)
        loss.backward()

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        if args.train_log_every > 0 and (step <= 5 or step % args.train_log_every == 0):
            elapsed = time.perf_counter() - t0
            print(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} time:{elapsed:.1f}s lr_mul:{scale:.4f}")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            elapsed = time.perf_counter() - t0
            improved = " *BEST*" if val_bpb < best_val_bpb else ""
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
            print(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} time:{elapsed:.1f}s{improved}")

    val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"\nFinal: val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    quant_obj = quantize_state_dict_int8(model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    os.makedirs("output", exist_ok=True)
    with open("output/model.int8.ptz", "wb") as f:
        f.write(quant_blob)
    model_bytes = len(quant_blob)
    print(f"Model size (int8+zlib): {model_bytes:,} bytes ({model_bytes/1024/1024:.2f} MB)")

    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu", weights_only=False)
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    q_val_loss, q_val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"Roundtrip: val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    print(f"Quantization degradation: {q_val_bpb - val_bpb:.6f} BPB")

    return q_val_bpb


if __name__ == "__main__":
    main()
