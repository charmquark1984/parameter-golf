# Experiment 17: Muon Optimizer Settings at Medium Scale

## Date: 2026-03-19

## What was tested
Full grid search of Muon momentum (0.6, 0.7, 0.8, 0.9, 0.95) × Matrix LR (0.04, 0.08, 0.1) at medium model scale (6x256, 2.8M params).

## Data
Synthetic data, 50 iterations, batch=2048 tokens.

## Model
6 layers, dim=256, 8Q/2KV, 2x MLP GELU, ~2.83M params (~2.7-2.9 MB compressed).
This is ~4x the toy scale (690K) and ~0.2x competition scale (24M).

## Results (best per momentum highlighted)

| Muon Mom. | LR 0.04 | LR 0.08 | **LR 0.1** |
|-----------|---------|---------|------------|
| 0.6 | 0.8412 | 0.8097 | 0.8078 |
| **0.7** | 0.8427 | 0.8073 | **0.8045** |
| 0.8 | 0.8408 | **0.8050** | 0.8087 |
| 0.9 | 0.8378 | 0.8080 | 0.8108 |
| 0.95 | **0.8369** | 0.8087 | 0.8119 |

**Best overall: muon=0.7, LR=0.1 → BPB 0.8045**
**Second best: muon=0.8, LR=0.08 → BPB 0.8050**

## Key findings

1. **Lower momentum still wins at medium scale**: muon=0.7 (0.8045) beats muon=0.95 (0.8119) at LR=0.1
2. **Optimal momentum shifts slightly with LR**:
   - At LR=0.04: muon=0.95 best (0.8369)
   - At LR=0.08: muon=0.8 best (0.8050)
   - At LR=0.1: muon=0.7 best (0.8045)
3. **Higher LR is almost always better**: LR=0.08 or 0.1 beats 0.04 at every momentum
4. **The muon_momentum × LR interaction is important**: You can't tune them independently

## Recommendations for competition
- With matrix_lr=0.1: use muon_momentum=0.7
- With matrix_lr=0.08: use muon_momentum=0.8
- Either combo gives ~0.8050 BPB (essentially equivalent)
- The default muon_momentum=0.95 is only good at low LR (0.04)

## Importance
**High** - Confirms that lower Muon momentum + higher LR is the right direction at medium scale. The interaction effect (momentum×LR) means the competition script should adjust momentum when LR changes.

## Notes
- At toy scale (690K params), muon=0.8 was best. At medium scale (2.8M), muon=0.7 is slightly better. At competition scale (24M), the optimal value may shift again.
- The competition baseline uses muon=0.95 with matrix_lr=0.04. Switching to muon=0.7-0.8 with matrix_lr=0.08-0.1 is a major improvement at every scale tested.
- Only 50 iterations due to model size - rankings might change with more training.
