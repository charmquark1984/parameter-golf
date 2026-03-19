"""
CPU-friendly training script for parameter golf experimentation.
Adapted from train_gpt.py to run without CUDA for rapid iteration on architecture ideas.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------
# HYPERPARAMETERS (CPU-tuned defaults)
# -----------------------------

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

    # Model shape
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

    # Optimizer
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


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :].clone()
            self._sin_cached = freqs.sin()[None, None, :, :].clone()
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype).clone(), self._sin_cached.to(dtype=dtype).clone()


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


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
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
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


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
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


# -----------------------------
# QUANTIZATION (from original)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes = {}, {}, {}
    passthrough, passthrough_orig_dtypes = {}, {}
    qmeta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point():
            passthrough[name] = t
            continue
        if t.numel() <= 65536:
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                passthrough[name] = t.float().contiguous()
            else:
                if t.dtype in {torch.float32, torch.bfloat16}:
                    passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                    passthrough[name] = t.to(dtype=torch.float16).contiguous()
                else:
                    passthrough[name] = t
            continue
        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), 0.9999984, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
            clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), 0.9999984).item()) if t32.numel() else 0.0
            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
        if t32.ndim == 2:
            scales[name] = scale.to(dtype=torch.float16).contiguous()
        else:
            scales[name] = scale
        quantized[name] = q
        dtypes[name] = str(t.dtype).removeprefix("torch.")
    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# EVALUATION
# -----------------------------

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


# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------

def main():
    args = Hyperparameters()
    device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={sp.vocab_size()}")

    # Load validation data
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    print(f"Validation tokens: {val_tokens.numel() - 1}")

    # Build model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    # Estimate model size after int8+zlib
    # Rough estimate: ~1 byte per param + overhead
    est_size_mb = n_params / (1024 * 1024)
    print(f"Estimated model size (int8): ~{est_size_mb:.1f} MB")

    # Set up optimizers (Muon for matrices, Adam for rest)
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

    # Training data
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

    # Initial validation
    val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"step:0/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    t0 = time.perf_counter()
    best_val_bpb = val_bpb

    # Main loop
    model.train()
    for step in range(1, args.iterations + 1):
        scale = lr_mul(step)
        zero_grad_all()

        # Get batch
        per_rank_span = args.train_batch_tokens + 1
        chunk = train_stream.take(per_rank_span)
        local = chunk.to(dtype=torch.int64)
        x = local[:-1].reshape(-1, args.train_seq_len)
        y = local[1:].reshape(-1, args.train_seq_len)

        loss = model(x, y)
        loss.backward()

        # Update Muon momentum warmup
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
            print(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} time:{elapsed:.1f}s step_avg:{elapsed/step:.2f}s")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            elapsed = time.perf_counter() - t0
            improved = " *BEST*" if val_bpb < best_val_bpb else ""
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
            print(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} time:{elapsed:.1f}s{improved}")

    # Final validation
    val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"\nFinal: val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # Quantize and save
    quant_obj = quantize_state_dict_int8(model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    os.makedirs("output", exist_ok=True)
    with open("output/model.int8.ptz", "wb") as f:
        f.write(quant_blob)
    model_bytes = len(quant_blob)
    print(f"Model size (int8+zlib): {model_bytes:,} bytes ({model_bytes/1024/1024:.2f} MB)")

    # Roundtrip validation
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu", weights_only=False)
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    q_val_loss, q_val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"Roundtrip: val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    print(f"Quantization degradation: {q_val_bpb - val_bpb:.6f} BPB")

    return q_val_bpb


if __name__ == "__main__":
    main()
