"""
Round 4: Combine best hyperparams + try architectural innovations.
Best so far: muon_momentum=0.9, beta2=0.99, qk_gain=1.0, grad_clip=1.0
"""
import subprocess
import sys
import json
import os
import time


def run_experiment(name, env_overrides, script="train_cpu.py", description=""):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    if description:
        print(f"  {description}")
    start = time.perf_counter()
    result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - start
    lines = (result.stdout + result.stderr).strip().split('\n')
    val_bpb = roundtrip_bpb = params = model_size = None
    for line in lines:
        if 'Model parameters:' in line: params = line.split('Model parameters:')[1].strip()
        if 'Model size (int8+zlib):' in line: model_size = line.split('Model size (int8+zlib):')[1].strip()
        if line.startswith('Final:'): val_bpb = float(line.split('val_bpb:')[1].strip())
        if line.startswith('Roundtrip:'): roundtrip_bpb = float(line.split('val_bpb:')[1].strip())
    bpb = roundtrip_bpb if roundtrip_bpb is not None else val_bpb
    if result.returncode == 0:
        print(f"  Params: {params} | Size: {model_size} | BPB: {bpb} | Time: {elapsed:.1f}s")
    else:
        print(f"  FAILED!")
        for line in lines[-5:]: print(f"  > {line}")
    return {"name": name, "success": result.returncode == 0, "val_bpb": bpb, "params": params,
            "model_size": model_size, "elapsed": f"{elapsed:.1f}s", "config": env_overrides}


def main():
    results = []

    # Best combined hyperparams
    best = {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
            "MLP_MULT": 3, "ITERATIONS": 200, "WARMDOWN_ITERS": 40,
            "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08,
            "GRAD_CLIP_NORM": 1.0, "VAL_LOSS_EVERY": 50,
            "MUON_MOMENTUM": 0.9, "BETA2": 0.99, "QK_GAIN_INIT": 1.0}

    results.append(run_experiment("combined_best", best, description="All best hyperparams combined"))

    # Try with more iterations
    results.append(run_experiment("combined_300", {**best, "ITERATIONS": 300, "WARMDOWN_ITERS": 60, "VAL_LOSS_EVERY": 75},
        description="Combined best + 300 iterations"))

    # Try with 500 iterations
    results.append(run_experiment("combined_500", {**best, "ITERATIONS": 500, "WARMDOWN_ITERS": 100, "VAL_LOSS_EVERY": 100},
        description="Combined best + 500 iterations"))

    # Now test architectural modifications via train_cpu_v2.py
    # We need to create that script first - let's test with standard script but different configs

    # Scale up to fill 16MB budget
    # For 16MB int8+zlib at ~2x compression on raw int8:
    # ~16MB / 1 byte per param = ~16M params in int8
    # But zlib helps, typical ratio is ~0.6 so we need ~27M params
    # Baseline uses 42M params -> 15.8MB. Let's see param-to-size ratios.

    # Wider model (closer to full budget)
    results.append(run_experiment("scaled_7x384",
        {**best, "NUM_LAYERS": 7, "MODEL_DIM": 384, "NUM_HEADS": 8, "NUM_KV_HEADS": 4,
         "ITERATIONS": 50, "WARMDOWN_ITERS": 10, "TRAIN_BATCH_TOKENS": 2048,
         "VAL_LOSS_EVERY": 25},
        "Scaled up: 7x384 (closer to 16MB budget)"))

    # Even wider
    results.append(run_experiment("scaled_9x448",
        {**best, "NUM_LAYERS": 9, "MODEL_DIM": 448, "NUM_HEADS": 8, "NUM_KV_HEADS": 4,
         "ITERATIONS": 50, "WARMDOWN_ITERS": 10, "TRAIN_BATCH_TOKENS": 2048,
         "VAL_LOSS_EVERY": 25},
        "Scaled up: 9x448"))

    # Try different MLP ratios at larger scale
    results.append(run_experiment("scaled_7x384_4xMLP",
        {**best, "NUM_LAYERS": 7, "MODEL_DIM": 384, "NUM_HEADS": 8, "NUM_KV_HEADS": 4,
         "MLP_MULT": 4, "ITERATIONS": 50, "WARMDOWN_ITERS": 10,
         "TRAIN_BATCH_TOKENS": 2048, "VAL_LOSS_EVERY": 25},
        "Scaled: 7x384, 4x MLP"))

    # 1 KV head at larger scale
    results.append(run_experiment("scaled_8x384_1kv",
        {**best, "NUM_LAYERS": 8, "MODEL_DIM": 384, "NUM_HEADS": 8, "NUM_KV_HEADS": 1,
         "ITERATIONS": 50, "WARMDOWN_ITERS": 10, "TRAIN_BATCH_TOKENS": 2048,
         "VAL_LOSS_EVERY": 25},
        "Scaled: 8x384, extreme GQA (1 KV head)"))

    # Muon momentum sweep at combined best
    results.append(run_experiment("muon_0.85",
        {**best, "MUON_MOMENTUM": 0.85}, "Muon momentum 0.85"))

    results.append(run_experiment("muon_0.8",
        {**best, "MUON_MOMENTUM": 0.8}, "Muon momentum 0.8"))

    # Print summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY - ROUND 4")
    print(f"{'='*80}")
    print(f"{'Name':<30} {'Params':<15} {'BPB':<10} {'Time':<8}")
    print(f"{'-'*63}")
    successful = sorted([r for r in results if r["success"]], key=lambda x: x["val_bpb"] or 999)
    for r in successful:
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] else "N/A"
        print(f"{r['name']:<30} {r['params'] or 'N/A':<15} {bpb:<10} {r['elapsed']:<8}")
    with open("output/experiment_results4.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
