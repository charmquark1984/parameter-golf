# Experiment 18: Encoder/Decoder Asymmetry and Final Tuning

## Date: 2026-03-19

## What was tested
1. Layer counts 3-7 (checking encoder/decoder split effects)
2. Softcap 25, 30, 35, 40 (narrower range)
3. Adam beta1: 0.85, 0.9, 0.95
4. Adam eps: 1e-6, 1e-8, 1e-10

## Data
Synthetic data, 200 iterations, all best settings with MLP 2x.

## Results

### Layer Count / Encoder-Decoder Split
| Layers | Enc/Dec | BPB |
|--------|---------|-----|
| 4 | 2/2 | **0.7439** |
| 5 | 2/3 | 0.7441 |
| 6 | 3/3 | 0.7444 |
| 3 | 1/2 | 0.7484 |
| 7 | 3/4 | 0.7471 |

### Softcap
| Softcap | BPB |
|---------|-----|
| **30** | **0.7439** |
| 40 | 0.7449 |
| 35 | 0.7470 |
| 25 | 0.7473 |

### Beta1 (IMPORTANT)
| Beta1 | BPB |
|-------|-----|
| **0.95** | **0.7414** |
| 0.9 | 0.7439 |
| 0.85 | 0.7505 |

### Eps
| Eps | BPB |
|-----|-----|
| 1e-8 | **0.7439** |
| 1e-10 | 0.7456 |
| 1e-6 | 0.7466 |

## Key findings

1. **beta1=0.95 is BETTER than 0.9** (0.7414 vs 0.7439, -0.0025). This REVERSES the finding from Experiment 11 where beta1=0.95 hurt. The difference is we're now using MLP 2x instead of 3x.
   - **Interaction effect**: With a simpler MLP (2x), higher first-moment smoothing (beta1=0.95) helps the optimizer converge to better minima
   - This is the NEW best result at toy scale: **0.7414 BPB**

2. **Softcap 30 confirmed optimal** (again). The 25-40 range shows 30 is the sweet spot.

3. **Adam eps=1e-8 is correct** - both 1e-6 and 1e-10 are slightly worse.

4. **Layer count shows diminishing returns** beyond 4 on this data - overfitting dominates.

## Updated Competition Recommendation
Change beta1 from 0.9 to 0.95:
```python
beta1 = float(os.environ.get("BETA1", 0.95))  # was 0.9
```

## Importance
**High** - beta1=0.95 gives the new best result (0.7414). This should be added to the competition script.

## Notes
- The beta1 × MLP_MULT interaction is a good example of why hyperparameters can't be tuned independently
- Previous finding (Exp 11) that beta1=0.95 hurts was with MLP 3x. With MLP 2x, beta1=0.95 helps.
- Since our final recommendation uses MLP 2x, beta1=0.95 is the right choice.
