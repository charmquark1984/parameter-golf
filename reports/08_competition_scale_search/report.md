# Experiment 08: Competition-Scale Architecture Search

## Date: 2026-03-19

## What was tested
Systematic search for architectures that fit within the 16MB int8+zlib budget:
- Depth: 7, 9, 10, 11, 12 layers
- Width: 512, 544, 576, 640 dim
- MLP expansion: 2x, 3x, 4x with GELU
- KV heads: 2 vs 4

All at 20 iterations (minimal training, just to see model sizes and initial convergence).

## Data
Synthetic data. Only 20 training iterations per config - results are about **model size** and **initial convergence rate**, not final quality.

## Key constraint
The competition requires total submission size (model int8+zlib + code) < 16MB. Code is ~48KB, so model budget is ~15.95MB.

## Results (sorted by BPB, all fit in 16MB)

| Config | Params | Size (MB) | BPB | Budget Used |
|--------|--------|-----------|-----|-------------|
| **12x512, 2KV, GELU 3x** | **27.3M** | **15.25 MB** | **0.9417** | **95%** |
| 10x512, 4KV, GELU 3x | 24.1M | 11.83 MB | 0.9485 | 74% |
| 12x512, 4KV, GELU 3x | 28.9M | 14.22 MB | 0.9516 | 89% |
| 11x512, 2KV, GELU 3x | 25.1M | 13.96 MB | 0.9525 | 87% |
| 11x512, 4KV, GELU 3x | 26.5M | 12.98 MB | 0.9607 | 81% |
| 9x576, 4KV, GELU 3x | 27.5M | 13.02 MB | 0.9613 | 81% |
| 12x512, 4KV, GELU 2x | 22.6M | 11.34 MB | 0.9616 | 71% |
| 9x640, 4KV, GELU 2x | 26.5M | 12.53 MB | 0.9634 | 78% |
| 9x544, 4KV, GELU 3x | 24.5M | 11.81 MB | 0.9702 | 74% |
| 7x512, 4KV, GELU 4x | 20.7M | 9.86 MB | 0.9741 | 62% |

## Key findings

1. **Best architecture: 12 layers, dim=512, 2 KV heads, GELU 3x MLP**
   - 27.3M params, 15.25 MB int8+zlib (95% of budget)
   - Best BPB even at 20 iterations: 0.9417
   - Uses extreme GQA (8Q/2KV) to save params, allowing 3 more layers than baseline

2. **Depth wins over width**: 12x512 consistently beats 9x576 or 9x640 at similar model sizes
   - 12x512 GELU 3x (14.22 MB): 0.9516
   - 9x576 GELU 3x (13.02 MB): 0.9613
   - 9x640 GELU 2x (12.53 MB): 0.9634

3. **3x MLP > 2x MLP at competition scale**:
   - 12x512 GELU 3x (14.22 MB): 0.9516
   - 12x512 GELU 2x (11.34 MB): 0.9616
   - The extra MLP capacity justifies the size increase

4. **2 KV heads > 4 KV heads when combined with more layers**:
   - 12x512 2KV GELU 3x (15.25 MB): **0.9417**
   - 12x512 4KV GELU 3x (14.22 MB): 0.9516
   - Savings from 2KV allow the same depth with better results

5. **4x MLP is not worth it**: 7x512 GELU 4x (9.86 MB) = 0.9741, much worse than 12x512 3x

## Did results match expectations?
- Depth winning: **Expected** - deeper models are generally better for language modeling at similar param counts
- 2KV helping: **Somewhat surprising in magnitude** - the param savings from 2KV allow 3 more layers, and those layers matter more than the attention quality loss
- 4x MLP being worst: **Expected** - fewer layers with bigger MLPs means less contextual processing
- All fitting in 16MB: **Expected** - the int8+zlib compression ratio is roughly 1.7-1.9x

## State of other changes
Using best training config: GELU MLP, cosine warmdown, muon_momentum=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0.

## Model size
Competition-scale: 20.7M-28.9M params, 9.86-15.25 MB compressed.

## Recommended competition architecture
```
NUM_LAYERS=12
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=2  (changed from 4)
MLP_MULT=3      (changed from 2)
```
With GELU activation, cosine warmdown, muon_momentum=0.8, beta2=0.99.

**Size estimate**: 15.25 MB int8+zlib + 48KB code = ~15.3 MB (under 16MB)

## Suggested followups
- Validate this architecture on real FineWeb data with GPU training
- The 12-layer model has more encoder layers (6) and decoder layers (6) than baseline (4+5), which means more skip connections
- May need to tune warmdown and LR specifically for the larger model
- Check if int8 quantization degrades more for the 12-layer model (more layers = more accumulated quantization error)

## Importance
**Critical** - This defines the recommended architecture for the competition submission. The 12x512 2KV GELU 3x config maximizes budget utilization at 95% while achieving the best initial convergence.

## Notes
- Only 20 training iterations per config! These BPB values are NOT predictive of final quality
- The ranking at 20 iterations may change at 20,000 iterations - but size estimates are exact
- Competition-scale models take 70-90s per 20 iterations on CPU, so longer training is impractical here
- The baseline (9x512, 4KV, ReLU² 2x) scores 0.9017 at 30 iters in Exp 7; our 12x512 2KV GELU 3x scores 0.9417 at 20 iters. This isn't a fair comparison (different iteration counts), but the size utilization is much better.
