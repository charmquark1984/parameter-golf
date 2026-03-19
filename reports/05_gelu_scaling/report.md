# Experiment 05: GELU Scaling and Duration Study

## Date: 2026-03-19

## What was tested
Whether GELU's advantage over ReLU² holds at:
1. Different training lengths (200, 500, 1000 iterations)
2. Different model scales (dim=128 vs dim=192)
3. Different depths (4 vs 6 layers - 6 timed out)

## Data
Synthetic data (~223K tokens/shard, 47K val tokens, vocab_size=1024).

## Best-so-far config
4 layers, dim=128, 8Q/2KV, 3x MLP, muon_momentum=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0, matrix_lr=0.08

## Reasoning
Need to verify that GELU's advantage isn't specific to short training runs (200 iterations). If the advantage disappears or reverses at longer training, it may not transfer to the competition (which trains for much longer on real data).

## Results

| Name | Params | BPB | Pre-quant BPB | Time |
|------|--------|-----|---------------|------|
| **gelu_500** | 690,464 | **0.7368** | 0.7369 | 82.6s |
| gelu_200 | 690,464 | 0.7382 | 0.7382 | 35.7s |
| relu2_500 | 690,464 | 0.7414 | 0.7416 | 84.3s |
| gelu_dim192 | 1,453,472 | 0.7429 | 0.7432 | 124.3s |
| **gelu_1000** | 690,464 | **0.7576** | 0.7581 | 158.5s |
| relu2_1000 | 690,464 | 0.7602 | 0.7598 | 163.7s |
| gelu_6layer | - | TIMEOUT | - | >600s |

## Key findings
1. **GELU maintains advantage at 500 iters**: 0.7368 vs 0.7414 (ReLU²), delta = 0.0046
2. **1000 iterations OVERFITS** on this synthetic data: both GELU and ReLU² get worse (0.7576 and 0.7602 respectively). The synthetic data is too repetitive/small for extended training.
3. **GELU still better at 1000 iters**: 0.7576 vs 0.7602, delta = 0.0026 (smaller advantage, suggesting convergence)
4. **Wider model (dim=192) is WORSE** than dim=128 at 500 iters on this data: 0.7429 vs 0.7368. More params need more unique data.
5. **6 layers timed out** at 600s - too slow for CPU experiments at 500 iterations
6. **500 iterations is the sweet spot** for this synthetic dataset

## Did results match expectations?
- GELU holding advantage: Expected and confirmed
- 1000 iters overfitting: **Somewhat surprising in magnitude** - the gap between 500 and 1000 is large (~0.02 BPB). This is a clear sign the synthetic data is too small/simple.
- Wider model being worse: **Unexpected** - usually wider is better. But with limited data diversity, the extra capacity just memorizes noise.
- This strongly suggests that on real FineWeb data (much more diverse), wider models and longer training will behave very differently.

## State of other changes
Using accumulated best config from Experiments 1-4: 8Q/2KV, 3x MLP, muon_momentum=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0.

## Model size
Toy models: 690K params (dim=128) and 1.45M params (dim=192).

## Suggested followups
- Don't train beyond 500 iterations on this synthetic data
- The relative GELU vs ReLU² comparison is the most transferable result
- Need to test on real FineWeb data to validate scaling behaviors
- The overfitting at 1000 iters means all future experiments should use 200-500 iters max

## Importance
**High** - Confirms GELU advantage is robust across training durations. The overfitting finding is critical for experimental design - all future experiments on this synthetic data should cap at ~500 iterations.

## Notes
- The 6-layer experiment hitting 600s timeout highlights CPU limitations - deeper models may need batch size reduction
- Quantization degradation is minimal (<0.001 BPB) across all runs, suggesting the model is quantization-friendly at this scale
