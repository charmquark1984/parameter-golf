"""
Round 3: Architectural innovations and training tricks.
Focus on ideas that transfer to full-scale training.
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
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "train_cpu.py"], env=env, capture_output=True, text=True, timeout=600)
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
    base = {"NUM_LAYERS": 4, "MODEL_DIM": 128, "NUM_HEADS": 8, "NUM_KV_HEADS": 2,
            "MLP_MULT": 3, "ITERATIONS": 200, "WARMDOWN_ITERS": 40,
            "MATRIX_LR": 0.08, "TIED_EMBED_LR": 0.1, "SCALAR_LR": 0.08,
            "GRAD_CLIP_NORM": 1.0, "VAL_LOSS_EVERY": 50}

    # Reference: best config from round 2
    results.append(run_experiment("reference_best", base, "Best config from round 2 with grad clip"))

    # Try different warmdown ratios
    results.append(run_experiment("warmdown_60",
        {**base, "WARMDOWN_ITERS": 60}, "Longer warmdown (60 vs 40)"))

    results.append(run_experiment("warmdown_100",
        {**base, "WARMDOWN_ITERS": 100}, "Very long warmdown (100/200)"))

    # Try different Muon momentum
    results.append(run_experiment("muon_momentum_0.9",
        {**base, "MUON_MOMENTUM": 0.9}, "Lower Muon momentum (0.9 vs 0.95)"))

    results.append(run_experiment("muon_momentum_0.98",
        {**base, "MUON_MOMENTUM": 0.98}, "Higher Muon momentum (0.98)"))

    # Try different beta2
    results.append(run_experiment("beta2_0.99",
        {**base, "BETA2": 0.99}, "Higher beta2 (0.99 vs 0.95)"))

    # Try shorter sequence length
    results.append(run_experiment("seq_len_128",
        {**base, "TRAIN_SEQ_LEN": 128, "TRAIN_BATCH_TOKENS": 4096},
        "Shorter seq len (128 vs 256), same tokens per batch"))

    # Try longer sequence
    results.append(run_experiment("seq_len_512",
        {**base, "TRAIN_SEQ_LEN": 512, "TRAIN_BATCH_TOKENS": 4096},
        "Longer seq len (512 vs 256), same tokens per batch"))

    # Try lower rope base (helps with shorter contexts)
    results.append(run_experiment("rope_base_1000",
        {**base, "ROPE_BASE": 1000.0}, "Lower RoPE base (1000 vs 10000)"))

    # Try different QK gain init
    results.append(run_experiment("qk_gain_1.0",
        {**base, "QK_GAIN_INIT": 1.0}, "Lower QK gain init (1.0 vs 1.5)"))

    results.append(run_experiment("qk_gain_2.0",
        {**base, "QK_GAIN_INIT": 2.0}, "Higher QK gain init (2.0 vs 1.5)"))

    # 300 iterations to see if more training helps
    results.append(run_experiment("300_iters",
        {**base, "ITERATIONS": 300, "WARMDOWN_ITERS": 60, "VAL_LOSS_EVERY": 75},
        "300 iterations"))

    # Print summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY - ROUND 3")
    print(f"{'='*80}")
    print(f"{'Name':<25} {'Params':<12} {'BPB':<10} {'Time':<8}")
    print(f"{'-'*55}")
    successful = sorted([r for r in results if r["success"]], key=lambda x: x["val_bpb"] or 999)
    for r in successful:
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] else "N/A"
        print(f"{r['name']:<25} {r['params'] or 'N/A':<12} {bpb:<10} {r['elapsed']:<8}")
    with open("output/experiment_results3.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
