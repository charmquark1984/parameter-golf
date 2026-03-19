"""
Experiment 7: Combine all best settings + quantization-aware experiments.
Tests the full accumulated improvements and explores quantization-friendliness.
"""
import subprocess
import sys
import json
import os
import time


def run(name, env_overrides, script='train_cpu_v3.py', description=''):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    print(f'\n{"="*70}')
    print(f'EXP: {name} - {description}')
    start = time.perf_counter()
    result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - start
    lines = (result.stdout + result.stderr).strip().split('\n')
    bpb = params = size = pre_quant_bpb = quant_deg = None
    for line in lines:
        if 'Model parameters:' in line: params = line.split('Model parameters:')[1].strip()
        if 'Model size (int8+zlib):' in line: size = line.split('Model size (int8+zlib):')[1].strip()
        if line.startswith('Final:'):
            pre_quant_bpb = float(line.split('val_bpb:')[1].strip())
        if line.startswith('Roundtrip:'):
            bpb = float(line.split('val_bpb:')[1].strip())
        if 'Quantization degradation:' in line:
            quant_deg = line.split('Quantization degradation:')[1].strip()
    if bpb is None:
        bpb = pre_quant_bpb
    r = {'name': name, 'bpb': bpb, 'pre_quant_bpb': pre_quant_bpb, 'params': params,
         'size': size, 'quant_deg': quant_deg, 'ok': result.returncode == 0,
         'elapsed': f'{elapsed:.1f}s', 'description': description, 'config': env_overrides}
    if result.returncode == 0:
        print(f'  Params:{params} Size:{size} BPB:{bpb} (pre-quant:{pre_quant_bpb}) Quant:{quant_deg} Time:{elapsed:.1f}s')
    else:
        print(f'  FAILED!')
        for line in lines[-5:]: print(f'  > {line}')
    return r


def main():
    results = []

    # All-best config combining findings from Experiments 1-6
    best = {
        'NUM_LAYERS': 4, 'MODEL_DIM': 128, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 2,
        'MLP_MULT': 3,
        'ITERATIONS': 200, 'WARMDOWN_ITERS': 80,  # 40% warmdown
        'MATRIX_LR': 0.08, 'TIED_EMBED_LR': 0.1, 'SCALAR_LR': 0.08,
        'GRAD_CLIP_NORM': 1.0,
        'MUON_MOMENTUM': 0.8, 'BETA2': 0.99, 'QK_GAIN_INIT': 1.0,
        'MLP_TYPE': 'gelu',
        'LR_SCHEDULE': 'cosine',
        'VAL_LOSS_EVERY': 50,
    }

    # === Reference: all-best combined ===
    results.append(run('all_best_combined', best,
        description='All best settings combined: GELU + cosine + muon0.8 + beta2=0.99'))

    # === Try with warmup ===
    results.append(run('best_with_warmup', {**best, 'WARMUP_FRAC': 0.05},
        description='Best + 5% LR warmup'))

    # === Muon momentum sweep with cosine ===
    results.append(run('cosine_muon_0.7', {**best, 'MUON_MOMENTUM': 0.7},
        description='Cosine + muon 0.7'))

    results.append(run('cosine_muon_0.6', {**best, 'MUON_MOMENTUM': 0.6},
        description='Cosine + muon 0.6'))

    results.append(run('cosine_muon_0.9', {**best, 'MUON_MOMENTUM': 0.9},
        description='Cosine + muon 0.9'))

    # === Different warmdown fractions with cosine ===
    results.append(run('cosine_wd50pct', {**best, 'WARMDOWN_ITERS': 100},
        description='Cosine 50% warmdown'))

    results.append(run('cosine_wd30pct', {**best, 'WARMDOWN_ITERS': 60},
        description='Cosine 30% warmdown'))

    # === Try 4x MLP with GELU ===
    results.append(run('gelu_4xMLP', {**best, 'MLP_MULT': 4},
        description='GELU with 4x MLP'))

    # === Competition-relevant: estimate 16MB budget model ===
    # The baseline 9x512 with 2x MLP = ~17M params = ~9.6MB int8+zlib
    # With 3x MLP + GELU: need to find a config that fits 16MB
    # Try the baseline-equivalent with our improvements
    results.append(run('competition_9x512', {**best,
        'NUM_LAYERS': 9, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4,
        'MLP_MULT': 2,  # Stay at 2x to match size
        'ITERATIONS': 30, 'WARMDOWN_ITERS': 10, 'TRAIN_BATCH_TOKENS': 2048,
        'VAL_LOSS_EVERY': 15},
        description='Competition baseline equivalent (9x512, 2xMLP) - few iters'))

    # Same but with GELU 3x MLP (will be larger)
    results.append(run('competition_9x512_gelu3x', {**best,
        'NUM_LAYERS': 9, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4,
        'MLP_MULT': 3,
        'ITERATIONS': 30, 'WARMDOWN_ITERS': 10, 'TRAIN_BATCH_TOKENS': 2048,
        'VAL_LOSS_EVERY': 15},
        description='Competition config with GELU 3xMLP (may exceed 16MB)'))

    # === Try with fewer layers but wider (param-equivalent) ===
    results.append(run('competition_7x576', {**best,
        'NUM_LAYERS': 7, 'MODEL_DIM': 576, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4,
        'MLP_MULT': 2,
        'ITERATIONS': 30, 'WARMDOWN_ITERS': 10, 'TRAIN_BATCH_TOKENS': 2048,
        'VAL_LOSS_EVERY': 15},
        description='Wider/shallower: 7x576'))

    # Print summary
    print(f'\n\n{"="*80}')
    print('EXPERIMENT 7 SUMMARY: All-Best Combined + Competition Scale')
    print(f'{"="*80}')
    print(f'{"Name":<30} {"Params":<15} {"BPB":<10} {"Pre-Q":<10} {"Size":<15} {"Time":<8}')
    print(f'{"-"*88}')
    ok = sorted([r for r in results if r['ok']], key=lambda x: x['bpb'] or 999)
    for r in ok:
        bpb = f"{r['bpb']:.4f}" if r['bpb'] else "N/A"
        pq = f"{r['pre_quant_bpb']:.4f}" if r['pre_quant_bpb'] else "N/A"
        print(f"{r['name']:<30} {r['params'] or 'N/A':<15} {bpb:<10} {pq:<10} {r['size'] or 'N/A':<15} {r['elapsed']:<8}")

    failed = [r for r in results if not r['ok']]
    if failed:
        print(f"\nFailed: {', '.join(r['name'] for r in failed)}")

    os.makedirs('output', exist_ok=True)
    with open('output/experiment_results7.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
