# Experiment 02: Combining Winning Hyperparameters

## Date: 2026-03-19

## What was tested
Combined the winning configurations from Experiment 01 and tested additional variations:
- Combined: 3x MLP + 8Q/2KV + 2x LR
- 4x MLP expansion
- Extreme GQA (8Q/1KV)
- 200 iterations (vs 100)
- Scaled up configs (5x256, wide/deep)
- Very high LR (3x)
- Higher logit softcap (50 vs 30)
- Untied embeddings
- Gradient clipping at 1.0

## Data
Same synthetic data as Experiment 01 (~223K tokens/shard, 47K val tokens, vocab_size=1024).

## Reasoning
Each of the individual winners from Exp 01 improved BPB by 0.004-0.007. Combining them should yield cumulative gains, unless they interact negatively.

## Results

| Name | Params | BPB | Time |
|------|--------|-----|------|
| best_200iters | 690,464 | **0.7510** | 39.8s |
| grad_clip_1 | 690,464 | **0.7517** | 22.7s |
| 4x_MLP | 821,536 | 0.7570 | 24.6s |
| extreme_gqa_1kv | 674,080 | 0.7584 | 23.3s |
| softcap_50 | 690,464 | 0.7590 | 22.8s |
| combo_3xMLP_8heads_highLR | 690,464 | 0.7594 | 22.7s |
| untied_embeddings | 821,536 | 0.7615 | 23.0s |
| wide_gqa_5x256 | 3,053,096 | 0.7616 | 55.1s |
| wider_3xMLP_highLR | 1,453,472 | 0.7619 | 33.9s |
| very_high_lr | 690,464 | 0.7712 | 23.4s |

## Key findings
1. **More iterations helps the most** (0.7510 with 200 iters vs ~0.75 at 100)
2. **Gradient clipping at 1.0 helps** (0.7517 at 100 iters - comparable to 200 iters without clip!)
3. **4x MLP is decent** but not clearly better than 3x for the extra params
4. **Very high LR (3x) hurts** - there's a sweet spot around 2x baseline
5. **Untied embeddings don't help** at this model/data scale
6. **Larger models don't help with limited iterations** - 3M params at 100 iters = 0.7616

## Did results match expectations?
- More iterations helping: Expected
- Gradient clipping helping: Somewhat surprising - suggests training instability at 2x LR
- Combined config underperforming individual winners: **Unexpected** - the combo (0.7594) was worse than individual 3x MLP (0.7464 from Exp 01). This may be because 2x LR combined with other changes creates instability
- Very high LR hurting: Expected, but good to confirm the boundary

## State of other changes
Testing combinations of Exp 01 winners. Each run varied one or more settings from the combo baseline.

## Model size
Toy models (674K - 3M params).

## Suggested followups
- Use gradient clipping + moderate LR as default going forward
- Test Muon momentum values (the combo changed learning dynamics)
- More iterations with best configs
- Focus on training efficiency (more learning per step) rather than raw model size

## Importance
**Medium-High** - Gradient clipping as default is important. The finding that simple combinations don't always stack is a useful caution.

## Notes
The combo being worse than individuals suggests hyperparameters interact non-linearly. Each "improvement" was tuned in the context of the baseline; changing multiple things at once can destabilize the optimization landscape.
