"""
Experiment 6: Training schedule and regularization experiments.
Tests cosine LR decay, label smoothing, LR warmup, and WSD schedule.
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
    result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=600)
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
        print(f'  Params:{params} Size:{size} BPB:{bpb} Quant:{quant_deg} Time:{elapsed:.1f}s')
    else:
        print(f'  FAILED!')
        for line in lines[-5:]: print(f'  > {line}')
    return r


def main():
    results = []

    base = {'NUM_LAYERS': 4, 'MODEL_DIM': 128, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 2,
            'MLP_MULT': 3, 'ITERATIONS': 200, 'WARMDOWN_ITERS': 40,
            'MATRIX_LR': 0.08, 'TIED_EMBED_LR': 0.1, 'SCALAR_LR': 0.08,
            'GRAD_CLIP_NORM': 1.0, 'MUON_MOMENTUM': 0.8, 'BETA2': 0.99,
            'QK_GAIN_INIT': 1.0, 'MLP_TYPE': 'gelu', 'VAL_LOSS_EVERY': 50}

    # Reference: linear warmdown (default)
    results.append(run('linear_decay', {**base, 'LR_SCHEDULE': 'linear'},
        description='Linear warmdown (default)'))

    # Cosine warmdown
    results.append(run('cosine_decay', {**base, 'LR_SCHEDULE': 'cosine'},
        description='Cosine warmdown'))

    # WSD schedule (sqrt decay)
    results.append(run('wsd_decay', {**base, 'LR_SCHEDULE': 'wsd'},
        description='WSD (sqrt decay) warmdown'))

    # Label smoothing variants
    results.append(run('smooth_0.05', {**base, 'LABEL_SMOOTHING': 0.05},
        description='Label smoothing 0.05'))

    results.append(run('smooth_0.1', {**base, 'LABEL_SMOOTHING': 0.1},
        description='Label smoothing 0.1'))

    # LR warmup (5% of training)
    results.append(run('warmup_5pct', {**base, 'WARMUP_FRAC': 0.05},
        description='5% LR warmup'))

    # LR warmup + cosine
    results.append(run('warmup_cosine', {**base, 'WARMUP_FRAC': 0.05, 'LR_SCHEDULE': 'cosine'},
        description='5% warmup + cosine decay'))

    # Cosine + label smoothing
    results.append(run('cosine_smooth', {**base, 'LR_SCHEDULE': 'cosine', 'LABEL_SMOOTHING': 0.05},
        description='Cosine + label smoothing 0.05'))

    # Longer warmdown
    results.append(run('cosine_long_wd', {**base, 'LR_SCHEDULE': 'cosine', 'WARMDOWN_ITERS': 80},
        description='Cosine with longer warmdown (80/200)'))

    # Shorter warmdown
    results.append(run('cosine_short_wd', {**base, 'LR_SCHEDULE': 'cosine', 'WARMDOWN_ITERS': 20},
        description='Cosine with shorter warmdown (20/200)'))

    # Print summary
    print(f'\n\n{"="*80}')
    print('EXPERIMENT 6 SUMMARY: Training Schedule Experiments')
    print(f'{"="*80}')
    print(f'{"Name":<25} {"BPB":<10} {"Quant Deg":<15} {"Time":<8}')
    print(f'{"-"*58}')
    ok = sorted([r for r in results if r['ok']], key=lambda x: x['bpb'] or 999)
    for r in ok:
        bpb = f"{r['bpb']:.4f}" if r['bpb'] else "N/A"
        print(f"{r['name']:<25} {bpb:<10} {r['quant_deg'] or 'N/A':<15} {r['elapsed']:<8}")

    os.makedirs('output', exist_ok=True)
    with open('output/experiment_results6.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to output/experiment_results6.json')


if __name__ == '__main__':
    main()
