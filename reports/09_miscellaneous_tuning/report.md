# Experiment 09: Miscellaneous Hyperparameter Tuning

## Date: 2026-03-19

## What was tested
Fine-grained tuning of remaining hyperparameters:
1. **Muon Newton-Schulz steps**: 3, 5 (default), 7, 10
2. **Tied embedding init std**: 0.002, 0.005 (default), 0.01
3. **Tied embedding LR**: 0.05, 0.1 (current best), 0.15, 0.2
4. **Matrix LR**: 0.06, 0.08 (current best), 0.1, 0.12
5. **Scalar LR**: 0.04, 0.08 (current best), 0.12

## Data
Synthetic data, 200 iterations, GELU MLP, cosine warmdown 40%.

## Results (sorted by BPB)

| Name | BPB | vs reference | Notes |
|------|-----|-------------|-------|
| **init_std_0.002** | **0.7435** | **-0.0017** | Best |
| **matrix_lr_0.1** | **0.7438** | **-0.0014** | Second best |
| ns_7steps | 0.7447 | -0.0005 | Marginal |
| reference | 0.7452 | baseline | |
| scalar_lr_0.12 | 0.7452 | 0.0000 | |
| matrix_lr_0.06 | 0.7454 | +0.0002 | |
| init_std_0.01 | 0.7468 | +0.0016 | |
| scalar_lr_0.04 | 0.7470 | +0.0018 | |
| embed_lr_0.05 | 0.7473 | +0.0021 | |
| embed_lr_0.2 | 0.7478 | +0.0026 | |
| ns_10steps | 0.7485 | +0.0033 | Slower, no benefit |
| embed_lr_0.15 | 0.7489 | +0.0037 | |
| ns_3steps | 0.7499 | +0.0047 | Too few NS steps |
| matrix_lr_0.12 | 0.7515 | +0.0063 | Unstable |

## Key findings
1. **Lower tied_embed_init_std (0.002 vs 0.005)** helps: 0.7435 (-0.0017). Smaller initialization allows the optimizer to shape embeddings more freely.
2. **Matrix LR 0.1 (vs 0.08)** helps: 0.7438 (-0.0014). Slightly higher Muon LR improves convergence. But 0.12 is too high (0.7515).
3. **NS 7 steps** is marginally better than 5 (0.7447 vs 0.7452). The extra orthogonalization quality helps slightly but costs computation time.
4. **NS 3 steps is too few** (0.7499) - insufficient orthogonalization degrades Muon's update quality.
5. **NS 10 steps doesn't help** (0.7485) - diminishing returns from over-orthogonalization, plus slower per-step.
6. **Tied embed LR 0.1 is the right value** - both lower (0.05) and higher (0.15, 0.2) are worse.
7. **Scalar LR is not very sensitive** - 0.08 and 0.12 are equivalent, 0.04 slightly worse.

## Did results match expectations?
- Lower init_std helping: Plausible but magnitude was noteworthy. With tied embeddings, the init_std controls both input and output projections, so a smaller init gives more freedom to learn.
- Matrix LR 0.1: Expected - with lower Muon momentum (0.8), slightly higher LR compensates.
- NS steps: 7 being marginal was expected. The competition baseline uses 5 for speed, and 5 is good enough.
- Embed LR sensitivity: Expected - embedding LR is known to be a sensitive parameter.

## State of other changes
Using full best config: GELU, cosine warmdown 40%, muon=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0.

## Model size
Toy model: 690,464 params.

## Suggested followups
- Apply init_std=0.002 and matrix_lr=0.1 to the competition script
- These are small gains - priority is lower than the major architectural/schedule changes
- On real data, the optimal LR values may differ (larger batch, different data distribution)

## Importance
**Medium** - init_std_0.002 and matrix_lr_0.1 are small but free improvements. Combined they save ~0.003 BPB.

## Notes
- These improvements are at the diminishing returns stage for this synthetic dataset
- Real FineWeb data may have different optimal LR values due to different gradient scales
- The NS steps tradeoff (quality vs speed) matters more on GPU where per-step time is critical
