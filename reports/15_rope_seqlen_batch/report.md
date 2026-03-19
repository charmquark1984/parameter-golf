# Experiment 15: RoPE Base, Sequence Length, Batch Size

## Date: 2026-03-19

## What was tested
1. RoPE base: 500, 1000, 5000, 10000 (default), 50000
2. Sequence length: 64, 128, 256 (default), 512
3. Batch tokens: 2048, 4096 (default), 8192, 16384
4. Softcap with learned temperature: 15, 20, 30 (default), 50

## Data
Synthetic data, 200 iterations, all best settings.

## Results

### RoPE Base
| Base | BPB | vs default (10000) |
|------|-----|-------------------|
| **50000** | **0.7432** | **-0.0007** |
| 500 | 0.7436 | -0.0003 |
| 10000 | 0.7439 | baseline |
| 5000 | 0.7456 | +0.0017 |
| 1000 | 0.7475 | +0.0036 |

### Sequence Length
| Seq Len | BPB | vs default (256) |
|---------|-----|-----------------|
| **512** | **0.7424** | **-0.0015** |
| 256 | 0.7439 | baseline |
| 64 | 0.7490 | +0.0051 |
| 128 | 0.7496 | +0.0057 |

### Batch Tokens
| Batch | BPB | vs default (4096) |
|-------|-----|------------------|
| **4096** | **0.7439** | **baseline** |
| 2048 | 0.7466 | +0.0027 |
| 8192 | 0.7474 | +0.0035 |
| 16384 | 0.7585 | +0.0146 |

### Softcap + Learned Temp
| Softcap | BPB | vs 30 |
|---------|-----|-------|
| **30** | **0.7439** | **baseline** |
| 50 | 0.7470 | +0.0031 |
| 15 | 0.7484 | +0.0045 |
| 20 | 0.7520 | +0.0081 |

## Key findings
1. **Seq len 512 helps** (0.7424 vs 0.7439). The competition uses seq_len=1024 by default, which should be even better. This is NOT actionable since the competition already uses 1024.
2. **RoPE base 50000 marginal** (-0.0007). Not significant enough to change. The competition uses 10000.
3. **Batch 4096 is optimal for this data** - larger batches see the same data too many times.
4. **Softcap 30 confirmed optimal** even with learned temperature.

## Importance
**Low** - No actionable changes for the competition. The sequence length finding confirms the competition default (1024) is reasonable. The batch size finding is data-specific.

## Notes
- The RoPE base 500 being good (0.7436) suggests that for short sequences on simple data, stronger position information helps. For real data with seq_len=1024, the default 10000 is likely correct.
- Larger batch sizes hurting is a clear overfitting signal on the tiny synthetic dataset.
