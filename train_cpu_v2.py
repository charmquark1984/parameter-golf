"""
CPU training script v2: architectural innovations for parameter golf.
Based on train_cpu.py but with configurable MLP types, activation functions,
and other architecture experiments.
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


# Import shared components from train_cpu
from train_cpu import (
    Hyperparameters as BaseHyperparameters,
    Muon, zeropower_via_newtonschulz5,
    build_sentencepiece_luts, load_data_shard, load_validation_tokens,
    TokenStream, RMSNorm, Rotary, apply_rotary_emb,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    eval_val, CONTROL_TENSOR_NAME_PATTERNS,
)


class Hyperparameters(BaseHyperparameters):
    # Architecture extensions
    mlp_type = os.environ.get("MLP_TYPE", "relu2")  # relu2, swiglu, geglu, gelu
    share_kv = bool(int(os.environ.get("SHARE_KV", "0")))  # Share K and V weights
    parallel_block = bool(int(os.environ.get("PARALLEL_BLOCK", "0")))  # Parallel attn+MLP


# ----- MLP Variants -----

class ReluSquaredMLP(nn.Module):
    """Original ReLU^2 MLP from the baseline."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP - uses gated linear unit with SiLU activation.
    More parameter-efficient per quality than standard MLP.
    Note: For same hidden size, this uses 3x input projections instead of 1x.
    We adjust hidden to keep param count similar.
    """
    def __init__(self, dim, mlp_mult):
        super().__init__()
        # Adjusted hidden size: standard MLP uses 2*dim*hidden params
        # SwiGLU uses 3*dim*hidden params, so use hidden = 2/3 * mlp_mult * dim
        hidden = int(2 * mlp_mult * dim / 3)
        # Round to nearest multiple of 8 for efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.up(x))


class GeGLUMLP(nn.Module):
    """GeGLU MLP - uses gated linear unit with GELU activation."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(2 * mlp_mult * dim / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(F.gelu(self.gate(x)) * self.up(x))


class GELUMLP(nn.Module):
    """Standard GELU MLP (no gating)."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


def make_mlp(mlp_type, dim, mlp_mult):
    if mlp_type == "relu2":
        return ReluSquaredMLP(dim, mlp_mult)
    elif mlp_type == "swiglu":
        return SwiGLUMLP(dim, mlp_mult)
    elif mlp_type == "geglu":
        return GeGLUMLP(dim, mlp_mult)
    elif mlp_type == "gelu":
        return GELUMLP(dim, mlp_mult)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")


# ----- Attention with optional KV sharing -----

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, share_kv=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.share_kv = share_kv
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        if share_kv:
            # Share K and V weights - saves kv_dim * dim params
            self.c_v = self.c_k  # Same projection for K and V
        else:
            self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.share_kv:
            v = k.clone()  # Same linear transform but no RoPE applied
        else:
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


# ----- Block variants -----

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 mlp_type="relu2", share_kv=False, parallel=False):
        super().__init__()
        self.parallel = parallel
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, share_kv)
        self.mlp = make_mlp(mlp_type, dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.parallel:
            # Parallel attention + MLP (like PaLM)
            normed = self.attn_norm(x)
            attn_out = self.attn(normed)
            mlp_out = self.mlp(self.mlp_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out + \
                    self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        else:
            attn_out = self.attn(self.attn_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, mlp_type="relu2", share_kv=False, parallel_block=False):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  mlp_type=mlp_type, share_kv=share_kv, parallel=parallel_block)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else nn.Linear(model_dim, vocab_size, bias=False)
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

    def forward(self, input_ids, target_ids):
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
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def main():
    args = Hyperparameters()
    device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    print(f"Tokenizer: vocab_size={sp.vocab_size()}")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    print(f"Val tokens: {val_tokens.numel() - 1}")

    model = GPT(
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
    est_size_mb = n_params / (1024 * 1024)
    print(f"Estimated model size (int8): ~{est_size_mb:.1f} MB")
    print(f"MLP type: {args.mlp_type}, share_kv: {args.share_kv}, parallel: {args.parallel_block}")

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
        if args.warmdown_iters <= 0:
            return 1.0
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if warmdown_start <= step < args.iterations:
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
        return 1.0

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

    return q_val_bpb


if __name__ == "__main__":
    main()
