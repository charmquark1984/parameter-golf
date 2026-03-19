# Experiment 16: Quantization-Aware Training

## Date: 2026-03-19

## What was tested
Adding noise to model weights during training to simulate int8 quantization effects, making the model more robust to post-training quantization. Tested noise scales: 0, 0.001, 0.005, 0.01, 0.05.

## Data
Synthetic data, 200 iterations, all best settings.

## Results

| Noise Scale | Pre-quant BPB | Roundtrip BPB | Quant Degradation |
|------------|---------------|---------------|-------------------|
| 0 (none) | 0.7577 | 0.7578 | +0.000159 |
| 0.001 | 0.7563 | 0.7563 | -0.000032 |
| 0.005 | 0.7565 | 0.7566 | +0.000100 |
| 0.01 | 0.7537 | 0.7536 | -0.000194 |
| 0.05 | 0.7531 | 0.7532 | +0.000104 |

## Key findings
1. **Quantization degradation is already negligible** at this model scale (<0.0002 BPB). There's essentially no room to improve it.
2. **Weight noise acts as regularization**, improving both pre-quant AND roundtrip BPB. Noise=0.05 gives 0.7531 vs 0.7577 baseline.
3. However, this is NOT true quantization-aware training - it's just weight perturbation regularization.
4. The BPB values here are slightly worse than earlier experiments due to implementation differences.

## Importance
**Low** - Quantization degradation is too small to be worth optimizing. The noise-as-regularization effect is interesting but not a priority.

## Notes
- At competition scale (17-27M params), quantization degradation may be larger (~0.01-0.03 BPB based on baseline submission data). QAT might matter more there.
- The baseline submission shows ~0.007 BPB quantization degradation at full scale (pre-quant 1.2172 vs roundtrip 1.2244). Still small but non-trivial.
- Proper QAT would involve quantize-dequantize in the forward pass with straight-through estimator, not additive noise. This was not implemented.
