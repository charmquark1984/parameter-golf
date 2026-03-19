# Experiment 04: Architecture Innovations (MLP Types, KV Sharing, Parallel Blocks)

## Date: 2026-03-19

## What was tested
Structural modifications to the transformer architecture:
1. **MLP activation functions**: ReLU² (baseline), SwiGLU, GeGLU, standard GELU
2. **KV weight sharing**: using same linear projection for both K and V
3. **Parallel blocks**: computing attention and MLP in parallel (like PaLM) instead of sequentially
4. **SwiGLU + parallel**: combination of both innovations

## Data
Same synthetic data (~223K tokens/shard, 47K val tokens, vocab_size=1024).

## Best-so-far config used as baseline
4 layers, dim=128, 8Q/2KV, 3x MLP, 200 iters, warmdown=40, matrix_lr=0.08, tied_embed_lr=0.1, scalar_lr=0.08, grad_clip=1.0, muon_momentum=0.8, beta2=0.99, qk_gain=1.0

## Reasoning
- **SwiGLU/GeGLU**: Used in LLaMA, Gemma, and other modern LLMs. The gating mechanism adds expressiveness. For same param count, we use hidden = 2/3 * mlp_mult * dim to compensate for the extra gate projection.
- **GELU**: Simpler than ReLU² (no squaring step) but smoother than ReLU. May be better suited for small models.
- **KV sharing**: Reduces attention params by sharing K/V projections. If K and V carry similar information, this is free compression.
- **Parallel blocks**: Reduces sequential dependency, potentially allowing better gradient flow.

## Results

| Name | Params | BPB | vs reference |
|------|--------|-----|-------------|
| **gelu** | 690,464 | **0.7382** | **-0.0078** |
| swiglu_4x | 825,632 | 0.7387 | -0.0073 |
| geglu | 690,464 | 0.7397 | -0.0063 |
| swiglu | 690,464 | 0.7420 | -0.0040 |
| ref_relu2 | 690,464 | 0.7460 | baseline |
| shared_kv | 674,080 | 0.7489 | +0.0029 |
| swiglu_parallel | 690,464 | 0.7499 | +0.0039 |
| parallel | 690,464 | 0.7525 | +0.0065 |

## Key findings
1. **Standard GELU is the best MLP activation** (0.7382), beating ReLU² by 0.0078. This is a substantial architectural improvement.
2. **GeGLU is second best** (0.7397), slightly better than SwiGLU (0.7420)
3. **SwiGLU with 4x expansion** (0.7387) uses more params to match GELU - GELU is more parameter-efficient
4. **KV sharing hurts** (-0.0029) - K and V need distinct representations even with GQA
5. **Parallel blocks hurt** (-0.0065) - sequential attention→MLP is better, the MLP benefits from seeing attention-transformed representations

## Did results match expectations?
- **GELU winning over ReLU² was unexpected.** ReLU² is a deliberate design choice in modded-nanogpt. However:
  - ReLU² has an aggressive nonlinearity (squaring after ReLU amplifies large values, zeroes small ones)
  - With small models and short training, the smoother GELU gradient landscape may allow faster learning
  - GELU is also simpler (fewer ops), which matters less here but is notable
  - **IMPORTANT CAVEAT**: This may not hold on real data. ReLU² might be better at larger scales or with more training. The baseline authors chose ReLU² for a reason.
- SwiGLU not winning: Somewhat expected at this scale - the 2/3 hidden reduction may hurt more than gating helps with tiny dimensions
- KV sharing hurting: Expected - at 2 KV heads, each head is already doing a lot of work; sharing K/V reduces representational capacity
- Parallel blocks hurting: Expected for small models - they lose the benefit of MLP processing attention-refined features

## State of other changes
All experiments used the accumulated best config: 8Q/2KV, 3x MLP, 2x LR, grad_clip=1.0, muon_momentum=0.8, beta2=0.99, qk_gain=1.0.

## Model size
Toy model: 690,464 params (~1.08 MB compressed). Note: SwiGLU/GeGLU have same param count because hidden size is adjusted to compensate for the gate projection.

## Suggested followups
- Run GELU at longer training (500, 1000 iterations) to confirm it maintains advantage
- Test GELU at larger model scales (closer to 16MB budget)
- Compare GELU vs ReLU² on real FineWeb data - this is the critical test
- Try GELU with different MLP multipliers (2x, 4x)
- Consider that ReLU² might be better for quantization (sharper weight distributions)

## Importance
**Very High** - If GELU vs ReLU² holds on real data, this is one of the simplest changes that yields a meaningful BPB improvement. However, the caveat about synthetic data is critical - must verify on real data before making it the default.

## Notes
- train_cpu_v2.py was created for this experiment, importing shared code from train_cpu.py and adding configurable MLP types
- The GELU MLP is the simplest implementation: just `proj(gelu(fc(x)))` with no squaring or gating
- SwiGLU with adjusted hidden size (2/3 * mlp_mult * dim, rounded to multiple of 8) maintains approximately the same param count as the standard MLP
- Parallel blocks save no params but change the computation graph - they're used in PaLM for throughput, not quality
