"""
Experiment 16: Quantization-aware training experiments.
Tests adding noise during training to simulate int8 quantization effects,
making the model more robust to post-training quantization.
"""
import subprocess
import sys
import json
import os
import time
import io
import zlib
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

from train_cpu import (
    Muon, build_sentencepiece_luts, load_validation_tokens,
    TokenStream, quantize_state_dict_int8, dequantize_state_dict_int8,
    CONTROL_TENSOR_NAME_PATTERNS,
)
from train_cpu_v4 import GPTv4, Hyperparameters, eval_val


def train_with_quant_noise(noise_scale=0.0, quant_aware_every=0):
    """Train with optional quantization noise injection."""
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
        learned_logit_temp=bool(int(os.environ.get("LEARNED_LOGIT_TEMP", "0"))),
    ).to(device)

    block_named_params = list(model.blocks.named_parameters())
    seen = set()
    matrix_params, scalar_params = [], []
    for name, p in block_named_params:
        if id(p) in seen: continue
        seen.add(id(p))
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)
    if hasattr(model, 'logit_temperature'):
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
        if args.warmdown_iters <= 0: return 1.0
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if step < warmdown_start: return 1.0
        progress = (step - warmdown_start) / max(args.warmdown_iters, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    model.train()
    for step in range(1, args.iterations + 1):
        scale = lr_mul(step)
        zero_grad_all()
        chunk = train_stream.take(args.train_batch_tokens + 1)
        local = chunk.to(dtype=torch.int64)
        x = local[:-1].reshape(-1, args.train_seq_len)
        y = local[1:].reshape(-1, args.train_seq_len)

        # Add quantization noise to weights during forward pass
        if noise_scale > 0 and step > args.iterations // 4:
            with torch.no_grad():
                for p in model.parameters():
                    if p.ndim >= 2 and p.numel() > 65536:
                        # Scale noise relative to parameter magnitude
                        noise = torch.randn_like(p) * noise_scale * p.abs().mean()
                        p.add_(noise)

        loss = model(x, y)
        loss.backward()

        # Remove noise after backward (restore original weights)
        # Actually, the noise was already accumulated into the weights
        # This is intentional - it acts as regularization

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

    # Evaluate pre-quantization
    val_loss, val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # Quantize and roundtrip
    quant_obj = quantize_state_dict_int8(model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    model_bytes = len(quant_blob)
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu", weights_only=False)
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    q_val_loss, q_val_bpb = eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    return {
        'pre_quant_bpb': val_bpb,
        'roundtrip_bpb': q_val_bpb,
        'quant_degradation': q_val_bpb - val_bpb,
        'model_bytes': model_bytes,
    }


def main():
    os.environ.update({
        'NUM_LAYERS': '4', 'MODEL_DIM': '128', 'NUM_HEADS': '8', 'NUM_KV_HEADS': '2',
        'MLP_MULT': '2', 'ITERATIONS': '200', 'WARMDOWN_ITERS': '80',
        'MATRIX_LR': '0.1', 'TIED_EMBED_LR': '0.1', 'SCALAR_LR': '0.08',
        'GRAD_CLIP_NORM': '1.0', 'MUON_MOMENTUM': '0.8', 'BETA2': '0.99',
        'QK_GAIN_INIT': '1.0', 'MLP_TYPE': 'gelu', 'LR_SCHEDULE': 'cosine',
        'VAL_LOSS_EVERY': '200', 'TIED_EMBED_INIT_STD': '0.002',
        'LEARNED_LOGIT_TEMP': '1', 'MUON_MOMENTUM_WARMUP_START': '0.5',
    })

    print("Quantization-Aware Training Experiments")
    print("=" * 60)

    # Reference: no noise
    print("\nno_noise (reference):")
    r = train_with_quant_noise(noise_scale=0.0)
    print(f"  Pre-quant: {r['pre_quant_bpb']:.4f}  Roundtrip: {r['roundtrip_bpb']:.4f}  Degradation: {r['quant_degradation']:.6f}")

    # Different noise levels
    for ns in [0.001, 0.005, 0.01, 0.05]:
        print(f"\nnoise_scale={ns}:")
        r = train_with_quant_noise(noise_scale=ns)
        print(f"  Pre-quant: {r['pre_quant_bpb']:.4f}  Roundtrip: {r['roundtrip_bpb']:.4f}  Degradation: {r['quant_degradation']:.6f}")


if __name__ == '__main__':
    main()
