"""
CPU training script v4: Novel architectural ideas.
- Weight sharing between encoder/decoder layers
- Learned embedding scaling (logit temperature)
- Different vocab sizes
- SwiGLU with adjusted hidden size
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
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 4096))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 256))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 4))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))
    model_dim = int(os.environ.get("MODEL_DIM", 128))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.002))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.1))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.1))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.08))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.8))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 50))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.99))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    mlp_type = os.environ.get("MLP_TYPE", "gelu")
    lr_schedule = os.environ.get("LR_SCHEDULE", "cosine")

    # Novel features
    share_layers = bool(int(os.environ.get("SHARE_LAYERS", "0")))  # Share weights between encoder/decoder pairs
    learned_logit_temp = bool(int(os.environ.get("LEARNED_LOGIT_TEMP", "0")))
    embed_scale = float(os.environ.get("EMBED_SCALE", 0.0))  # If > 0, multiply embeddings by this constant


# ----- Attention -----
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, mlp_type="gelu"):
        super().__init__()
        self.mlp_type = mlp_type
        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        if self.mlp_type == "relu2":
            x = torch.relu(self.fc(x))
            return self.proj(x.square())
        else:
            return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, mlp_type="gelu"):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, mlp_type)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPTv4(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, mlp_type="gelu",
                 share_layers=False, learned_logit_temp=False, embed_scale=0.0):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.learned_logit_temp = learned_logit_temp
        self.embed_scale_val = embed_scale
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        if share_layers:
            # Share weights between encoder and decoder pairs
            # Create N/2 unique blocks, each used twice
            unique_blocks = num_layers // 2
            blocks = []
            for i in range(unique_blocks):
                block = Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, mlp_type)
                blocks.append(block)
                blocks.append(block)  # Same block used twice
            if num_layers % 2 == 1:
                blocks.append(Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, mlp_type))
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = nn.ModuleList([
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, mlp_type)
                for _ in range(num_layers)
            ])

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else nn.Linear(model_dim, vocab_size, bias=False)
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

        if learned_logit_temp:
            self.logit_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.embed_scale_val > 0:
            x = x * self.embed_scale_val
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
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if self.learned_logit_temp:
            logits = logits * self.logit_temperature.abs()
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


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
            batch_loss = model(x, y).detach().item()
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


def main():
    args = Hyperparameters()
    device = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    model = GPTv4(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, mlp_type=args.mlp_type,
        share_layers=args.share_layers, learned_logit_temp=args.learned_logit_temp,
        embed_scale=args.embed_scale,
    ).to(device)

    n_params = model.count_parameters()
    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Model parameters: {n_params:,} (unique: {unique_params:,})")
    print(f"Share layers: {args.share_layers}, Learned temp: {args.learned_logit_temp}")

    block_named_params = list(model.blocks.named_parameters())
    seen = set()
    matrix_params = []
    scalar_params = []
    for name, p in block_named_params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)
    if args.learned_logit_temp and hasattr(model, 'logit_temperature'):
        scalar_params.append(model.logit_temperature)

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

    train_stream = TokenStream(args.train_files)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step):
        if args.warmdown_iters <= 0:
            return 1.0
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if step < warmdown_start:
            return 1.0
        progress = (step - warmdown_start) / max(args.warmdown_iters, 1)
        if args.lr_schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(1.0 - progress, 0.0)

    val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"step:0/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    t0 = time.perf_counter()
    best_val_bpb = val_bpb
    model.train()

    for step in range(1, args.iterations + 1):
        scale = lr_mul(step)
        zero_grad_all()
        chunk = train_stream.take(args.train_batch_tokens + 1)
        local = chunk.to(dtype=torch.int64)
        x = local[:-1].reshape(-1, args.train_seq_len)
        y = local[1:].reshape(-1, args.train_seq_len)
        loss = model(x, y)
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
            print(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} time:{elapsed:.1f}s")

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


if __name__ == "__main__":
    main()
