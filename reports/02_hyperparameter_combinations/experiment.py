"""
Second round of experiments: combining winning strategies from round 1.
Key findings from round 1:
- 3x MLP expansion helps (0.7464)
- 8Q/2KV heads (more GQA groups) helps (0.7490) with fewer params
- Higher LR helps (0.7514)
- Wider > deeper at small scale
"""
import subprocess
import sys
import json
import os
import time


def run_experiment(name, env_overrides, description=""):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    if description:
        print(f"  {description}")
    print(f"{'='*70}")
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "train_cpu.py"], env=env, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - start
    lines = (result.stdout + result.stderr).strip().split('\n')
    val_bpb = roundtrip_bpb = model_size = params = quant_degradation = None
    for line in lines:
        if 'Model parameters:' in line: params = line.split('Model parameters:')[1].strip()
        if 'Model size (int8+zlib):' in line: model_size = line.split('Model size (int8+zlib):')[1].strip()
        if line.startswith('Final:'): val_bpb = float(line.split('val_bpb:')[1].strip())
        if line.startswith('Roundtrip:'): roundtrip_bpb = float(line.split('val_bpb:')[1].strip())
        if 'Quantization degradation:' in line: quant_degradation = line.split('Quantization degradation:')[1].strip()
    bpb = roundtrip_bpb if roundtrip_bpb is not None else val_bpb
    if result.returncode == 0:
        print(f"  Params: {params} | Size: {model_size} | BPB: {bpb} | Quant deg: {quant_degradation} | Time: {elapsed:.1f}s")
    else:
        print(f"  FAILED!")
        for line in lines[-5:]: print(f"  > {line}")
    return {"name": name, "success": result.returncode == 0, "val_bpb": bpb, "pre_quant_bpb": val_bpb,
            "params": params, "model_size": model_size, "elapsed": f"{elapsed:.1f}s", "config": env_overrides}


def main():
    results = []

    # Combine: 3x MLP + more heads + higher LR
    results.append(run_experiment(
        "combo_3xMLP_8heads_highLR",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08},
        "Best combo: 3x MLP, 8Q/2KV, 2x LR"
    ))

    # Combine: wider + 3x MLP + higher LR
    results.append(run_experiment(
        "wider_3xMLP_highLR",
        {"NUM_LAYERS": 4, "MODEL_DIM": 192, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08},
        "Wider + 3x MLP + high LR"
    ))

    # Try 4x MLP
    results.append(run_experiment(
        "4x_MLP",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 4, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08},
        "4x MLP expansion"
    ))

    # Try 1 KV head (extreme GQA)
    results.append(run_experiment(
        "extreme_gqa_1kv",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 1,
         "MLP_MULT": 3, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08},
        "Extreme GQA: 8Q/1KV"
    ))

    # Try more iterations (200) with the best config
    results.append(run_experiment(
        "best_200iters",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 200,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08,
         "WARMDOWN_ITERS": 40, "VAL_LOSS_EVERY": 50},
        "Best config with 200 iterations"
    ))

    # Larger model at ~16MB budget target
    # At int8+zlib ratio ~2.5x, need ~42M params for 16MB
    # Let's try different configurations near that budget:
    # Option A: 9x512 (baseline) = 17M params = 9.6MB - we can go bigger
    # Option B: 12x512 = ~22M params
    # Option C: 9x640 = ~26M params
    # But for CPU testing, scale down proportionally

    # Try wider with fewer KV heads to save params
    results.append(run_experiment(
        "wide_gqa_5x256",
        {"NUM_LAYERS": 5, "MODEL_DIM": 256, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08},
        "5 layers, dim=256, 8Q/2KV, 3x MLP"
    ))

    # Even higher LR
    results.append(run_experiment(
        "very_high_lr",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100,
         "MATRIX_LR": 0.12, "TIED_EMBED_LR": 0.15, "SCALAR_LR": 0.12},
        "Very high LR: 3x base"
    ))

    # Try different logit_softcap
    results.append(run_experiment(
        "softcap_50",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08,
         "LOGIT_SOFTCAP": 50.0},
        "Higher logit softcap (50 vs 30)"
    ))

    # Try no tied embeddings (separate lm_head)
    results.append(run_experiment(
        "untied_embeddings",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100, "TIE_EMBEDDINGS": 0,
         "MATRIX_LR": 0.08, "EMBED_LR": 0.6, "HEAD_LR": 0.008, "SCALAR_LR": 0.08},
        "Untied embeddings"
    ))

    # Gradient clipping
    results.append(run_experiment(
        "grad_clip_1",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08,
         "GRAD_CLIP_NORM": 1.0},
        "With gradient clipping at 1.0"
    ))

    # Print summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY - ROUND 2")
    print(f"{'='*80}")
    print(f"{'Name':<30} {'Params':<15} {'BPB':<10} {'Time':<8}")
    print(f"{'-'*63}")
    successful = sorted([r for r in results if r["success"]], key=lambda x: x["val_bpb"] or 999)
    for r in successful:
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] else "N/A"
        print(f"{r['name']:<30} {r['params'] or 'N/A':<15} {bpb:<10} {r['elapsed']:<8}")

    with open("output/experiment_results2.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
