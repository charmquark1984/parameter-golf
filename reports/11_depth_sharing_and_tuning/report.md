# Experiment 11: Weight Sharing at Depth + Further Tuning

## Date: 2026-03-19

## What was tested
1. Weight sharing with learned logit temp at different depths (8L, 12L)
2. 6 layers without sharing (comparison)
3. Higher beta1 (0.95 vs 0.9)
4. No softcap with learned temperature
5. Lower Muon momentum warmup start (0.5 vs 0.85)

## Data
Synthetic data, 200 iterations, all best settings + learned logit temp.

## Results

| Name | BPB | vs ref | Params | Size |
|------|-----|--------|--------|------|
| 6L_no_share | **0.7437** | -0.0006 | 970K | 1.57 MB |
| muon_warmup_0.5 | **0.7437** | -0.0006 | 690K | 1.07 MB |
| ref_with_temp | 0.7443 | baseline | 690K | 1.08 MB |
| beta1_0.95 | 0.7455 | +0.0012 | 690K | 1.08 MB |
| shared_8L | 0.7458 | +0.0015 | 691K | 2.07 MB |
| shared_12L | 0.7466 | +0.0023 | 971K | 3.06 MB |
| softcap_999 | 0.7489 | +0.0046 | 690K | 1.07 MB |

## Key findings
1. **Weight sharing increases model size** without proportional quality gain. Shared layers still store all parameters in the int8 state dict (each block appears separately even if weights are shared). This makes weight sharing COUNTERPRODUCTIVE for the competition.
2. **Muon warmup from 0.5 matches best result** (0.7437). Starting momentum warmup from a lower value may help the early training phase.
3. **6 layers > 4 layers** with more params (0.7437 vs 0.7443) but the margin is small at this scale.
4. **Softcap is still needed** even with learned temperature. Removing it (softcap=999) hurts by 0.0046 BPB.
5. **Higher beta1 (0.95) slightly hurts** - default 0.9 is better.

## Did results match expectations?
- Weight sharing size issue: **Unexpected and important** - the int8 serialization doesn't deduplicate shared weights, so sharing layers INCREASES model file size while providing minimal quality benefit. This rules out weight sharing for the competition.
- Muon warmup from 0.5: Small surprise - starting the optimizer with less momentum initially helps it explore more.

## State of other changes
Using full best config with learned logit temperature.

## Importance
**Medium** - The main takeaway is that weight sharing is a dead end for competition (increases file size). Muon warmup from 0.5 is a minor free improvement.

## Notes
- Weight sharing would only work if the quantization/serialization was modified to deduplicate shared weights. This is possible but adds code complexity.
- For the competition, the lesson is clear: use unique layers, as many as fit in the 16MB budget.
