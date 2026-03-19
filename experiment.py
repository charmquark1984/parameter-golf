"""
Experiment runner for parameter golf.
Tests different architecture configurations and reports results.
"""
import subprocess
import sys
import json
import os
import time


def run_experiment(name, env_overrides, iterations=100, description=""):
    """Run a single experiment with given environment overrides."""
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    if description:
        print(f"  {description}")
    print(f"  Config: {env_overrides}")
    print(f"{'='*70}")

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "train_cpu.py"],
        env=env, capture_output=True, text=True, timeout=600
    )
    elapsed = time.perf_counter() - start

    # Parse output for key metrics
    output = result.stdout + result.stderr
    lines = output.strip().split('\n')

    val_bpb = None
    roundtrip_bpb = None
    model_size = None
    params = None
    quant_degradation = None

    for line in lines:
        if 'Model parameters:' in line:
            params = line.split('Model parameters:')[1].strip()
        if 'Model size (int8+zlib):' in line:
            model_size = line.split('Model size (int8+zlib):')[1].strip()
        if line.startswith('Final:'):
            try:
                val_bpb = float(line.split('val_bpb:')[1].strip())
            except (IndexError, ValueError):
                pass
        if line.startswith('Roundtrip:'):
            try:
                roundtrip_bpb = float(line.split('val_bpb:')[1].strip())
            except (IndexError, ValueError):
                pass
        if 'Quantization degradation:' in line:
            try:
                quant_degradation = line.split('Quantization degradation:')[1].strip()
            except (IndexError, ValueError):
                pass

    success = result.returncode == 0

    result_dict = {
        "name": name,
        "success": success,
        "val_bpb": roundtrip_bpb if roundtrip_bpb is not None else val_bpb,
        "pre_quant_bpb": val_bpb,
        "params": params,
        "model_size": model_size,
        "quant_degradation": quant_degradation,
        "elapsed": f"{elapsed:.1f}s",
        "config": env_overrides,
    }

    if not success:
        print(f"  FAILED! Return code: {result.returncode}")
        # Print last 10 lines of output for debugging
        for line in lines[-10:]:
            print(f"  > {line}")
    else:
        print(f"  Params: {params}")
        print(f"  Model size: {model_size}")
        print(f"  Val BPB (pre-quant): {val_bpb}")
        print(f"  Val BPB (roundtrip): {roundtrip_bpb}")
        print(f"  Quant degradation: {quant_degradation}")
        print(f"  Time: {elapsed:.1f}s")

    return result_dict


def main():
    results = []

    # ===== BASELINE =====
    results.append(run_experiment(
        "baseline_tiny",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
         "MLP_MULT": 2, "ITERATIONS": 100},
        description="Baseline: 4 layers, dim=128"
    ))

    # ===== EXPERIMENT 1: Deeper model =====
    results.append(run_experiment(
        "deeper_6layers",
        {"NUM_LAYERS": 6, "MODEL_DIM": 128, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
         "MLP_MULT": 2, "ITERATIONS": 100},
        description="Deeper: 6 layers, dim=128"
    ))

    # ===== EXPERIMENT 2: Wider model =====
    results.append(run_experiment(
        "wider_dim192",
        {"NUM_LAYERS": 4, "MODEL_DIM": 192, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
         "MLP_MULT": 2, "ITERATIONS": 100},
        description="Wider: 4 layers, dim=192"
    ))

    # ===== EXPERIMENT 3: More KV heads =====
    results.append(run_experiment(
        "more_kv_heads",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 4, "NUM_KV_HEADS": 4,
         "MLP_MULT": 2, "ITERATIONS": 100},
        description="More KV heads: 4 KV heads (MHA instead of GQA)"
    ))

    # ===== EXPERIMENT 4: Larger MLP =====
    results.append(run_experiment(
        "larger_mlp_3x",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
         "MLP_MULT": 3, "ITERATIONS": 100},
        description="Larger MLP: 3x expansion instead of 2x"
    ))

    # ===== EXPERIMENT 5: Scale to ~16MB budget =====
    # Target: ~42M params for 16MB int8+zlib (like the baseline submission)
    # 9 layers * 512 dim = baseline. Let's try alternatives:
    results.append(run_experiment(
        "baseline_full_size",
        {"NUM_LAYERS": 9, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4,
         "MLP_MULT": 2, "ITERATIONS": 50, "TRAIN_BATCH_TOKENS": 2048,
         "TRAIN_SEQ_LEN": 256, "VAL_LOSS_EVERY": 25},
        description="Full baseline config (9x512) - few iterations to verify"
    ))

    # ===== EXPERIMENT 6: Different width/depth ratio =====
    results.append(run_experiment(
        "wider_fewer_layers",
        {"NUM_LAYERS": 6, "MODEL_DIM": 256, "NUM_HEADS": 8, "NUM_KV_HEADS": 4,
         "MLP_MULT": 2, "ITERATIONS": 100},
        description="Wider/shallower: 6 layers, dim=256"
    ))

    # ===== EXPERIMENT 7: More heads with same dim =====
    results.append(run_experiment(
        "many_heads",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
         "MLP_MULT": 2, "ITERATIONS": 100},
        description="More Q heads: 8 heads with 2 KV (more GQA groups)"
    ))

    # ===== EXPERIMENT 8: Deeper narrow =====
    results.append(run_experiment(
        "deep_narrow_8x96",
        {"NUM_LAYERS": 8, "MODEL_DIM": 96, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
         "MLP_MULT": 2, "ITERATIONS": 100,
         "TRAIN_SEQ_LEN": 256},
        description="Deep narrow: 8 layers, dim=96"
    ))

    # ===== EXPERIMENT 9: Higher learning rates =====
    results.append(run_experiment(
        "higher_lr",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
         "MLP_MULT": 2, "ITERATIONS": 100,
         "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08},
        description="Higher learning rates: 2x matrix/scalar/embed LR"
    ))

    # ===== EXPERIMENT 10: Lower LR =====
    results.append(run_experiment(
        "lower_lr",
        {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
         "MLP_MULT": 2, "ITERATIONS": 100,
         "MATRIX_LR": 0.02, "TIED_EMBED_LR": 0.025, "SCALAR_LR": 0.02},
        description="Lower learning rates: 0.5x matrix/scalar/embed LR"
    ))

    # Print summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<25} {'Params':<15} {'Size':<15} {'BPB':<10} {'Time':<8}")
    print(f"{'-'*73}")

    successful = [r for r in results if r["success"]]
    successful.sort(key=lambda x: x["val_bpb"] if x["val_bpb"] is not None else 999)

    for r in successful:
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] is not None else "N/A"
        print(f"{r['name']:<25} {r['params'] or 'N/A':<15} {r['model_size'] or 'N/A':<15} {bpb:<10} {r['elapsed']:<8}")

    failed = [r for r in results if not r["success"]]
    if failed:
        print(f"\nFailed experiments: {', '.join(r['name'] for r in failed)}")

    # Save results
    with open("output/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to output/experiment_results.json")


if __name__ == "__main__":
    main()
