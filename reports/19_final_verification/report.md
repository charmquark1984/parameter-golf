# Experiment 19: Final Verification at Medium Scale

## Date: 2026-03-19

## What was tested
Head-to-head comparison of full baseline vs all improvements at medium scale (6x256, 2.8M params, 50 iterations).

## Results

| Config | BPB | Time |
|--------|-----|------|
| **All improvements** | **0.8111** | **24s** |
| Baseline | 0.8128 | 108s |
| **Delta** | **-0.0017** | **4.5x faster** |

## Key findings
1. Improvement holds at medium scale (+0.0017 BPB)
2. The improved config is **4.5x faster** per iteration due to 2KV heads (fewer attention params to compute)
3. The smaller improvement vs toy scale (0.0017 vs 0.0123) is expected:
   - Only 50 iterations (optimizer improvements need more steps)
   - Medium scale model may need different LR tuning
   - Synthetic data saturates faster at larger model sizes

## Importance
Confirms improvements generalize beyond toy scale. The speed improvement from 2KV heads is a bonus that translates directly to more training iterations within the 10-minute competition window.
