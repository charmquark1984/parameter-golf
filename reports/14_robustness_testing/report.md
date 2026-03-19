# Experiment 14: Robustness Testing Across Random Seeds

## Date: 2026-03-19

## What was tested
Compared the full best config vs original baseline across 5 random seeds (1337, 42, 123, 777, 2024) to verify improvements are robust and not artifacts of a single seed.

## Data
Synthetic data, 200 iterations each.

## Configs compared
- **Best**: GELU, cosine warmdown 40%, muon=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0, init_std=0.002, matrix_lr=0.1, learned logit temp, muon warmup from 0.5
- **Baseline**: ReLU², linear warmdown, muon=0.95, beta2=0.95, qk_gain=1.5, no grad clip

## Results

| Seed | Best BPB | Baseline BPB | Delta |
|------|----------|-------------|-------|
| 1337 | 0.7441 | 0.7555 | 0.0114 |
| 42 | 0.7426 | 0.7571 | 0.0145 |
| 123 | 0.7445 | 0.7534 | 0.0089 |
| 777 | 0.7482 | 0.7553 | 0.0071 |
| 2024 | 0.7509 | 0.7583 | 0.0074 |

### Statistics
| Metric | Best | Baseline |
|--------|------|----------|
| Mean | 0.7461 | 0.7559 |
| Std | 0.0034 | 0.0019 |
| Range | 0.7426-0.7509 | 0.7534-0.7583 |
| **Mean delta** | | **0.0099** |

## Key findings

1. **Improvement is robust**: All 5 seeds show positive improvement (range: 0.0071-0.0145)
2. **Mean improvement of 0.0099 BPB** at the toy model scale
3. **Best config has higher variance** (std 0.0034 vs 0.0019) - the more aggressive optimizer settings make it more seed-sensitive
4. **Worst-case improvement is still 0.0071** - even in the worst seed, we still clearly beat baseline

## Importance
**Very High** - This confirms our improvements are not noise. The consistent improvement across seeds validates the combined changes.

## Notes
- Both configs use the same model architecture (4x128, 8Q/2KV, 2x MLP) - only training/optimizer changes differ
- The higher variance of the best config suggests the optimizer is exploring more aggressively (lower momentum, higher LR). This is a feature for short training but might need tuning for longer runs.
- On real data with much more training, the relative improvement may be different (likely larger for architecture changes, possibly smaller for optimizer changes that are overfitting to the short-run regime).
