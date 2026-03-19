# Experiment 10: Novel Architectural Ideas

## Date: 2026-03-19

## What was tested
1. **Weight sharing between layers**: Encoder/decoder layer pairs share weights
2. **Learned logit temperature**: A single scalar parameter that scales the output logits
3. **Embedding scaling**: Multiply embeddings by sqrt(model_dim) before RMSNorm
4. **Logit softcap values**: 15, 20, 30 (default), 50, 100

## Data
Synthetic data, 200 iterations, best accumulated config.

## Results

| Name | BPB | vs reference | Notes |
|------|-----|-------------|-------|
| **learned_temp** | **0.7443** | **-0.0028** | 1 extra parameter! |
| reference | 0.7471 | baseline | |
| embed_scale_sqrt | 0.7479 | +0.0008 | |
| softcap_15 | 0.7479 | +0.0008 | |
| share_layers | 0.7483 | +0.0012 | Half the unique params |
| softcap_50 | 0.7498 | +0.0027 | |
| share_8layers | 0.7502 | +0.0031 | |
| softcap_20 | 0.7527 | +0.0056 | |
| softcap_100 | 0.7540 | +0.0069 | |

## Key findings

1. **Learned logit temperature helps** (0.7443, -0.0028). A single scalar parameter that scales logits gives the model freedom to adjust its confidence level during training. This is essentially free (1 parameter, trivial compute).

2. **Weight sharing barely hurts** (0.7483, +0.0012) while halving the unique parameters. At competition scale, this means you could use weight sharing to either:
   - Fit a much deeper model in the same 16MB budget
   - Have the same architecture with much smaller compressed size

3. **Embedding scaling doesn't help** - RMSNorm after embedding already normalizes scale.

4. **Logit softcap 30 (default) is about right**. Lower values (15, 20) restrict the model too much. Higher values (50, 100) effectively remove the cap and are slightly worse.

## Did results match expectations?
- Learned temperature: **Surprising in effectiveness** - a single parameter providing -0.0028 BPB is remarkable. The model benefits from learning its own "confidence calibration."
- Weight sharing: Expected to hurt somewhat. The +0.0012 is smaller than expected, making it an interesting option for extreme parameter efficiency.
- Softcap: Expected that 30 is reasonable. The non-monotonic behavior (15 better than 20, but 30 better than both) suggests softcap acts as a regularizer, not just a stability mechanism.

## State of other changes
Using full best config: GELU, cosine warmdown 40%, muon=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0, init_std=0.002, matrix_lr=0.1.

## Model size
Toy model: 690,464 params (410,896 unique params for shared version).

## Suggested followups
- Add learned_logit_temp to the competition script - it's a free improvement
- Explore weight sharing at competition scale: a 24-layer model with sharing would have the same unique params as a 12-layer but potentially better results due to implicit regularization
- Test whether softcap is needed at all with learned temperature

## Importance
**Medium-High** - Learned logit temperature is a simple, free improvement (-0.0028 BPB). Weight sharing is interesting for future exploration but not immediately useful.

## Notes
- The learned temperature could be viewed as undoing or modifying the logit_softcap. If the model learns temp > 1, it's effectively increasing the softcap's effective range.
- Weight sharing creates an implicit regularization effect (Universal Transformer-like). The fact it barely hurts quality while halving params is promising for extreme compression scenarios.
- train_cpu_v4.py was created for this experiment.
