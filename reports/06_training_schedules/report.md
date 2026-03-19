# Experiment 06: Training Schedule and Regularization

## Date: 2026-03-19

## What was tested
1. **LR decay schedules**: Linear (default), Cosine, WSD (warmup-stable-decay with sqrt)
2. **Label smoothing**: 0.0 (default), 0.05, 0.1
3. **LR warmup**: 0% (default), 5% of training
4. **Combinations**: warmup+cosine, cosine+smoothing
5. **Warmdown length with cosine**: 20, 40 (default), 80 out of 200 iters

## Data
Synthetic data (~223K tokens/shard, 47K val tokens, vocab_size=1024).

## Config
4 layers, dim=128, 8Q/2KV, 3x MLP, GELU activation, 200 iterations, all best hyperparams from previous experiments.

## Reasoning
- **Cosine decay** is widely used in LLM training (GPT-3, LLaMA, etc.) and typically outperforms linear decay because it decays more gradually at the start and faster at the end
- **WSD** is a newer schedule that maintains high LR longer then drops with sqrt
- **Label smoothing** can improve generalization by preventing overconfident predictions
- **LR warmup** prevents early-training instability from large gradient steps

## Results

| Name | BPB | vs linear ref | Notes |
|------|-----|--------------|-------|
| **cosine_long_wd** (80/200) | **0.7452** | **-0.0087** | Best result |
| warmup_cosine | 0.7469 | -0.0070 | |
| cosine_decay (40/200) | 0.7489 | -0.0050 | |
| wsd_decay | 0.7503 | -0.0036 | |
| warmup_5pct | 0.7523 | -0.0016 | |
| **linear_decay** (reference) | **0.7539** | baseline | |
| cosine_short_wd (20/200) | 0.7578 | +0.0039 | |
| cosine_smooth (0.05) | 0.7641 | +0.0102 | |
| smooth_0.05 | 0.7666 | +0.0127 | |
| smooth_0.1 | 0.7853 | +0.0314 | |

## Key findings
1. **Cosine decay with long warmdown (40% of training) is best**: 0.7452, -0.0087 vs linear
2. **Standard cosine warmdown is second best**: 0.7489, -0.0050 vs linear
3. **LR warmup helps slightly** (0.7523 vs 0.7539) and combines well with cosine (0.7469)
4. **Label smoothing HURTS significantly**: +0.0127 at 0.05, +0.0314 at 0.1. On this repetitive synthetic data, the model benefits from being confident.
5. **Longer warmdown is better** with cosine schedule: 80/200 > 40/200 > 20/200
6. **WSD is decent** but not as good as cosine

## Did results match expectations?
- Cosine beating linear: **Expected and confirmed**. This is well-established in the literature.
- Longer warmdown helping with cosine: **Expected** - cosine's gradual decay means you want it to start decaying earlier to benefit from the full curve shape.
- Label smoothing hurting: **Somewhat unexpected in magnitude**. On synthetic repetitive data, the model needs to be very confident about patterns. Label smoothing fights this. On real diverse data, label smoothing might behave differently (possibly neutral or slightly positive).
- LR warmup helping: Expected but the effect is small (0.0016). With grad_clip=1.0 already controlling early instability, warmup adds less value.

## State of other changes
Used best accumulated config: GELU MLP, 8Q/2KV, 3x MLP, muon_momentum=0.8, beta2=0.99, qk_gain=1.0, grad_clip=1.0.

## Model size
Toy model: 690,464 params (~1.08 MB compressed).

## Suggested followups
- Use cosine decay with 40% warmdown as default going forward
- Try cosine with even longer warmdown (50-60% of training)
- Skip label smoothing for now (reassess on real data)
- Consider combining LR warmup with cosine long warmdown
- The original train_gpt.py uses LINEAR warmdown - switching to COSINE should be an easy improvement

## Importance
**Very High** - Cosine LR schedule is a simple, zero-cost improvement that gives ~0.005-0.009 BPB. This is one of the most important findings because:
1. It's trivially easy to implement in the competition script
2. It doesn't affect model architecture or size
3. It's well-established in the literature
4. The baseline uses linear warmdown, so this is a clear improvement

## Notes
- The original `train_gpt.py` computes warmdown based on wallclock time, not step count. Need to adapt cosine decay to the wallclock-based schedule for the competition script.
- Label smoothing hurting is likely specific to synthetic data. Don't completely rule it out for real data, but it's low priority.
- train_cpu_v3.py was created for this experiment, adding `forward_logits()` method and configurable LR schedules
