# Proposed Changes to train_gpt.py for Competition

**Updated after experiments 1-13 on synthetic data. All findings need validation on real FineWeb data.**

## Final Recommended Architecture

```
NUM_LAYERS=14        (was 9)    — 5 extra transformer layers
NUM_KV_HEADS=2       (was 4)    — extreme GQA saves params for more layers
MLP_MULT=2           (unchanged) — 2x MLP is sufficient with other improvements
MODEL_DIM=512        (unchanged)
NUM_HEADS=8          (unchanged)
```

**Estimated size**: ~12.5 MB int8+zlib (78% of 16MB budget)
**vs baseline**: 9x512/4KV/2x = 7.3 MB (46% of budget) — we use 1.7x more of the budget

Alternative: 16x512 at 14.2 MB (89% of budget) for maximum depth.

## High Confidence Changes

### 1. Architecture: 14 layers, 2 KV heads (Exp 8, 12, 13)
- More layers is the single biggest improvement
- 2KV heads allow more layers within budget
- At 15 iters on synthetic data: 0.9319 BPB vs 0.9477 baseline (-0.0158)

### 2. Cosine LR Warmdown (Exp 6: -0.0087 BPB vs linear)
```python
# In lr_mul(), replace linear warmdown with cosine
progress = (step - warmdown_start) / max(args.warmdown_iters, 1)
return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
```
Zero-cost change. Standard in modern LLM training.

### 3. Warmdown 40% of training (Exp 6, 7)
```python
warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 8000))  # was 1200
```

### 4. Lower Muon Momentum: 0.8 (Exp 3: -0.0106 BPB)
```python
muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.8))  # was 0.95
```

### 5. Adam beta2: 0.99 (Exp 3: -0.0069 BPB)
```python
beta2 = float(os.environ.get("BETA2", 0.99))  # was 0.95
```

### 6. GELU activation (Exp 4: -0.0078 BPB; confirmed Exp 5)
```python
def forward(self, x):
    return self.proj(F.gelu(self.fc(x)))  # was relu(x).square()
```

### 7. Lower QK gain init: 1.0 (Exp 3, 9: -0.0018 BPB)
```python
qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.0))  # was 1.5
```

### 8. Gradient clipping at 1.0 (Exp 2: needed with higher LR)
```python
grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))  # was 0.0
```

### 9. Lower tied_embed_init_std: 0.002 (Exp 9: -0.0017 BPB)
```python
tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.002))  # was 0.005
```

## Medium Confidence Changes

### 10. Learned logit temperature (Exp 10: -0.0028 BPB)
Add a single scalar parameter that scales output logits. Essentially free.
```python
self.logit_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
# In forward: logits = logits * self.logit_temperature.abs()
```

### 11. Muon momentum warmup from 0.5 (Exp 11: marginal improvement)
```python
muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.5))  # was 0.85
```

## Changes in train_gpt_modified.py

The file `train_gpt_modified.py` contains all high-confidence changes applied to the original `train_gpt.py`. It is ready for GPU testing on real FineWeb data.

## Summary of All Changes vs Baseline

| Change | BPB Impact | Confidence | Cost |
|--------|-----------|------------|------|
| 14 layers (was 9) | ~0.016 | High | More params, more compute |
| 2 KV heads (was 4) | Enables above | High | None (fewer params) |
| Cosine warmdown | ~0.005-0.009 | Very High | None |
| Muon momentum 0.8 | ~0.005-0.010 | High | None |
| Beta2 0.99 | ~0.003-0.007 | High | None |
| GELU activation | ~0.004-0.008 | Medium-High | None |
| QK gain 1.0 | ~0.002 | High | None |
| Grad clip 1.0 | ~0.002 | High | None |
| Init std 0.002 | ~0.002 | Medium | None |
| Learned logit temp | ~0.003 | Medium | 1 parameter |

**Estimated total improvement: 0.030-0.060 BPB**

Baseline: 1.2244 BPB → Target: ~1.16-1.19 BPB

## Experiment Summary

| # | Focus | Key Finding |
|---|-------|-------------|
| 1 | Architecture sweep | 3x MLP, 8Q/2KV most efficient |
| 2 | Combinations | Grad clip helps, combos don't always stack |
| 3 | Hyperparameter tuning | Muon momentum 0.8, beta2 0.99 |
| 4 | MLP activations | GELU > GeGLU > SwiGLU > ReLU² |
| 5 | GELU scaling | GELU advantage holds at 500 iters |
| 6 | LR schedules | Cosine warmdown 40% best |
| 7 | Combined best | GELU 3x at competition scale: 11.7MB |
| 8 | Competition architecture | 12x512/2KV/3x fits at 15.25MB |
| 9 | Misc tuning | init_std=0.002, matrix_lr=0.1 |
| 10 | Novel ideas | Learned logit temp -0.0028; weight sharing dead end |
| 11 | Depth + sharing | Sharing increases file size; muon warmup 0.5 helps |
| 12 | Parameter allocation | MLP 2x ≈ 3x; confirms 8Q/2KV optimal |
| 13 | Final architecture | **14x512/2KV/2x = 12.5MB, best BPB** |
