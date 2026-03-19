# Proposed Changes to train_gpt.py for Competition

Based on experiments 1-7 on synthetic data. All findings need validation on real FineWeb data.

## High Confidence Changes (likely to help)

### 1. Cosine LR Warmdown (Exp 6: -0.0087 BPB vs linear)
Replace linear warmdown with cosine warmdown in the `lr_mul` function.
```python
# In lr_mul(), change warmdown calculation from:
return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
# To:
progress = (step - warmdown_start) / max(args.warmdown_iters, 1)
return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
```
**Rationale**: Cosine decay is standard in modern LLM training. Zero-cost change.

### 2. Warmdown fraction: increase to ~40% of training (Exp 6, 7)
Change `WARMDOWN_ITERS` default from 1200 to something proportional to ~40% of expected total steps.
For 20K iterations: `WARMDOWN_ITERS = 8000`.

### 3. Lower Muon Momentum: 0.8 instead of 0.95 (Exp 3: -0.0106 BPB)
```python
muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.8))  # was 0.95
```
**Rationale**: Lower momentum allows faster adaptation. Especially helpful for short training runs (10 min). Need to verify on real data - optimal value may differ.

### 4. Adam beta2: 0.99 instead of 0.95 (Exp 3: -0.0069 BPB)
```python
beta2 = float(os.environ.get("BETA2", 0.99))  # was 0.95
```
**Rationale**: Higher beta2 stabilizes second-moment estimates with noisy gradients.

## Medium Confidence Changes (promising but less certain)

### 5. GELU activation instead of ReLU² (Exp 4: -0.0078 BPB)
Replace the MLP class:
```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))  # was relu(x).square()
```
**Caution**: ReLU² was deliberately chosen by the baseline authors. GELU advantage might not transfer to full-scale training on real data.

### 6. 3x MLP expansion (Exp 1: -0.0073 BPB)
```python
mlp_mult = int(os.environ.get("MLP_MULT", 3))  # was 2
```
This increases model size but stays within 16MB budget (~11.7MB at 9x512).

### 7. Lower QK gain init: 1.0 instead of 1.5 (Exp 3: -0.0018 BPB)
```python
qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.0))  # was 1.5
```

## Lower Confidence Changes (need real data validation)

### 8. Gradient clipping at 1.0 (Exp 2: helped at high LR)
May not be needed if LR is properly tuned for the full run.

### 9. 8Q/2KV instead of 8Q/4KV (Exp 1: saved params with similar quality)
Only helps if we want to reallocate those params to MLP.

## Summary of Expected Impact
If all high-confidence changes transfer to real data:
- Cosine warmdown: ~0.005-0.009 BPB improvement
- Lower Muon momentum: ~0.005-0.010 BPB improvement
- Higher beta2: ~0.003-0.007 BPB improvement
- Total estimated: ~0.013-0.026 BPB improvement

Combined with medium-confidence changes (GELU, 3x MLP):
- Potential total: ~0.020-0.040 BPB improvement

The baseline scores 1.2244 BPB. If our changes save 0.020-0.040, we'd target ~1.18-1.20 BPB.
The 4-hour run scores 1.2074, so beating it in 10 minutes would be very impressive.
