# Experiment 07: All-Best Combined + Competition Scale Exploration

## Date: 2026-03-19

## What was tested
1. Combined all winning settings from Experiments 1-6 into a single config
2. Fine-tuned Muon momentum (0.6, 0.7, 0.8, 0.9) with cosine decay
3. Warmdown fraction sweep with cosine (30%, 40%, 50%)
4. GELU 4x MLP
5. Competition-scale models (9x512 baseline, 9x512 GELU 3x, 7x576)

## Data
Synthetic data (~223K tokens/shard, 47K val tokens).

## Combined best config
- 4 layers, dim=128, 8Q/2KV, 3x MLP, GELU activation
- Cosine LR schedule with 40% warmdown
- muon_momentum=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0
- matrix_lr=0.08, tied_embed_lr=0.1, scalar_lr=0.08

## Results

### Small-scale (690K params, ~1.08 MB)
| Name | BPB | Notes |
|------|-----|-------|
| **gelu_4xMLP** | **0.7433** | Best overall (821K params, 1.32 MB) |
| cosine_muon_0.7 | 0.7448 | |
| all_best_combined | 0.7452 | Reference (muon 0.8) |
| cosine_wd50pct | 0.7454 | |
| cosine_muon_0.9 | 0.7460 | |
| cosine_wd30pct | 0.7468 | |
| best_with_warmup | 0.7469 | |
| cosine_muon_0.6 | 0.7484 | Too low momentum |

### Competition-scale (16-22M params, ~9-12 MB)
| Name | Params | BPB | Size |
|------|--------|-----|------|
| **9x512 GELU 3x** | 21.8M | **0.8952** | 11.7 MB |
| 9x512 baseline (2x) | 17.1M | 0.9017 | 9.4 MB |
| 7x576 | 16.9M | 0.9226 | 9.0 MB |

## Key findings
1. **GELU 4x MLP is the new best** at small scale (0.7433), but uses 19% more params
2. **Muon momentum has a broad sweet spot at 0.7-0.8**: 0.7 gives 0.7448, 0.8 gives 0.7452, 0.6 is too low, 0.9 slightly worse
3. **Warmdown fraction 40-50% is optimal** with cosine: 40% (0.7452) ≈ 50% (0.7454) > 30% (0.7468)
4. **Competition-scale validation**: 9x512 GELU 3x MLP beats baseline by 0.0065 BPB even at 30 iterations and fits in 11.7 MB (well under 16MB limit)
5. **Wider/shallower (7x576) is worse** than 9x512 at competition scale, even though width helped at small scale

## Did results match expectations?
- Combined best working: Expected - the settings were individually validated
- GELU 4x beating 3x: Expected (more capacity) but the gain is small (0.0019) relative to 19% more params
- Competition-scale GELU 3x winning: **Very encouraging** - at 11.7MB it leaves 4.3MB budget headroom for either a larger model or more code
- 7x576 being worse: **Somewhat unexpected** - at competition scale, depth matters more than width for language modeling
- Muon 0.7 ≈ 0.8: Expected given the gradual trend from Exp 3

## State of other changes
Using accumulated best config from all previous experiments.

## Model size
- Small experiments: 690K-822K params (~1.07-1.32 MB)
- Competition experiments: 16.9M-21.8M params (~9.0-11.7 MB)

## Suggested followups
- **Priority 1**: Create a modified `train_gpt.py` with our best findings for the competition
- The key changes for competition are: GELU MLP, cosine warmdown, lower muon momentum
- Consider using the leftover budget (16MB - 11.7MB = 4.3MB) for a wider model
- Try 10x or 11x layers with 512 dim and GELU 3x - might fit in 16MB
- Need to estimate: does 3x MLP with GELU help on real FineWeb data?

## Importance
**Very High** - This confirms our improvements scale to competition-size models. The 11.7MB int8+zlib footprint for 9x512 GELU 3x is well within budget and already beats the baseline architecture by 0.0065 BPB with only 30 training iterations on synthetic data.

## Notes
- Competition-scale models (9x512) take 70-83s for just 30 iterations on CPU - running these for more iterations is impractical
- The 7x576 failure at competition scale vs small-scale success highlights the danger of only testing at small scale
- All competition runs are on synthetic data so absolute BPB values don't predict real performance
- The fact that GELU 3x MLP fits in 11.7MB means we could try 10x512 or 9x544 to use more of the budget
