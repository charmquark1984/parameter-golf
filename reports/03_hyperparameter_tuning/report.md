# Experiment 03: Hyperparameter Tuning

## Date: 2026-03-19

## What was tested
Fine-tuning of optimizer and training hyperparameters with the best architecture from Exp 02 as reference:
- Warmdown length: 40, 60, 100 (out of 200 iterations)
- Muon momentum: 0.9, 0.95 (default), 0.98
- Adam beta2: 0.95 (default), 0.99
- Sequence length: 128, 256 (default), 512
- RoPE base: 1000, 10000 (default)
- QK gain init: 1.0, 1.5 (default), 2.0
- Extended training: 300 iterations

## Data
Same synthetic data (~223K tokens/shard, 47K val tokens).

## Reference config
4 layers, dim=128, 8Q/2KV, 3x MLP, 200 iterations, warmdown=40, matrix_lr=0.08, tied_embed_lr=0.1, scalar_lr=0.08, grad_clip=1.0

## Reasoning
After finding good architecture and LR ranges, optimizer-level tuning can extract additional performance. Muon momentum, beta2, and QK gain are all training dynamics parameters that affect convergence quality.

## Results

| Name | BPB | Change vs ref |
|------|-----|---------------|
| **muon_momentum_0.9** | **0.7468** | **-0.0106** |
| beta2_0.99 | 0.7505 | -0.0069 |
| qk_gain_1.0 | 0.7556 | -0.0018 |
| 300_iters | 0.7562 | -0.0012 |
| seq_len_128 | 0.7564 | -0.0010 |
| warmdown_60 | 0.7568 | -0.0006 |
| reference_best | 0.7574 | baseline |
| muon_momentum_0.98 | 0.7577 | +0.0003 |
| rope_base_1000 | 0.7584 | +0.0010 |
| seq_len_512 | 0.7586 | +0.0012 |
| qk_gain_2.0 | 0.7634 | +0.0060 |
| warmdown_100 | 0.7638 | +0.0064 |

## Key findings
1. **Muon momentum 0.9 is significantly better** than default 0.95 (0.7468, -0.0106 improvement)
2. **Beta2=0.99 helps** (0.7505, -0.0069 improvement)
3. **Lower QK gain init (1.0 vs 1.5) helps slightly** - suggests default over-scales queries
4. **Warmdown 50% of training is too aggressive** - 20% (40/200) is about right
5. **Higher Muon momentum (0.98) slightly hurts** - confirms lower is better
6. **Higher QK gain (2.0) hurts** - the model benefits from less aggressive attention

## Did results match expectations?
- Muon momentum 0.9: **Surprisingly large effect**. Lower momentum = less smoothing = faster adaptation. With short training runs, the model benefits from reacting quickly to gradients
- Beta2: Expected - higher beta2 means slower second-moment decay, which can stabilize with noisy gradients
- QK gain: The direction (lower=better) was slightly unexpected; it suggests the baseline over-emphasizes attention patterns relative to what the model can learn with limited capacity

## State of other changes
All experiments used the best config from Exp 02 (8Q/2KV, 3x MLP, 2x LR, grad_clip=1.0) as reference. One hyperparameter varied at a time.

## Model size
Toy model: 690,464 params (~1.08 MB compressed).

## Suggested followups
- Try even lower Muon momentum (0.7, 0.8, 0.85)
- Combine muon_momentum=0.9 + beta2=0.99 + qk_gain=1.0
- These findings are specific to short runs on synthetic data; verify on longer runs

## Importance
**High** - Muon momentum is a major lever. The 0.0106 BPB improvement from momentum alone is larger than most architectural changes. This likely transfers to real training since it's about optimizer dynamics, not data distribution.

## Notes
- Lower Muon momentum works better for short training because the model needs to learn quickly from limited gradient signals
- For the competition (10 min on 8xH100), the optimal Muon momentum might be different than for 4+ hour runs
- The interaction between Muon momentum and LR is important - may need to re-tune LR when momentum changes significantly
