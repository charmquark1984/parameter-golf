"""
Experiment 9: Miscellaneous improvements.
Tests: Muon Newton-Schulz steps, different tied_embed_init_std, different tied_embed_lr.
"""
import subprocess
import sys
import json
import os
import time


def run(name, env_overrides, script='train_cpu_v3.py', description=''):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    print(f'\n{"="*60}')
    print(f'EXP: {name} - {description}')
    start = time.perf_counter()
    result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=180)
    elapsed = time.perf_counter() - start
    lines = (result.stdout + result.stderr).strip().split('\n')
    bpb = params = size = None
    for line in lines:
        if 'Model parameters:' in line: params = line.split('Model parameters:')[1].strip()
        if 'Model size (int8+zlib):' in line: size = line.split('Model size (int8+zlib):')[1].strip()
        if line.startswith('Roundtrip:'): bpb = float(line.split('val_bpb:')[1].strip())
        elif line.startswith('Final:') and bpb is None: bpb = float(line.split('val_bpb:')[1].strip())
    if result.returncode == 0:
        print(f'  BPB:{bpb} Params:{params} Time:{elapsed:.1f}s')
    else:
        print(f'  FAILED!')
        for line in lines[-3:]: print(f'  > {line}')
    return {'name': name, 'bpb': bpb, 'params': params, 'size': size, 'ok': result.returncode == 0,
            'elapsed': f'{elapsed:.1f}s', 'description': description, 'config': env_overrides}


def main():
    results = []
    best = {
        'NUM_LAYERS': 4, 'MODEL_DIM': 128, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 2,
        'MLP_MULT': 3, 'ITERATIONS': 200, 'WARMDOWN_ITERS': 80,
        'MATRIX_LR': 0.08, 'TIED_EMBED_LR': 0.1, 'SCALAR_LR': 0.08,
        'GRAD_CLIP_NORM': 1.0, 'MUON_MOMENTUM': 0.8, 'BETA2': 0.99,
        'QK_GAIN_INIT': 1.0, 'MLP_TYPE': 'gelu', 'LR_SCHEDULE': 'cosine',
        'VAL_LOSS_EVERY': 100,
    }

    results.append(run('reference', best, description='Reference (all best)'))

    # Muon Newton-Schulz steps
    results.append(run('ns_3steps', {**best, 'MUON_BACKEND_STEPS': 3}, description='Muon 3 NS steps'))
    results.append(run('ns_7steps', {**best, 'MUON_BACKEND_STEPS': 7}, description='Muon 7 NS steps'))
    results.append(run('ns_10steps', {**best, 'MUON_BACKEND_STEPS': 10}, description='Muon 10 NS steps'))

    # Tied embed init std
    results.append(run('init_std_0.01', {**best, 'TIED_EMBED_INIT_STD': 0.01}, description='Tied embed init std 0.01'))
    results.append(run('init_std_0.002', {**best, 'TIED_EMBED_INIT_STD': 0.002}, description='Tied embed init std 0.002'))

    # Tied embed LR
    results.append(run('embed_lr_0.05', {**best, 'TIED_EMBED_LR': 0.05}, description='Tied embed LR 0.05 (default)'))
    results.append(run('embed_lr_0.15', {**best, 'TIED_EMBED_LR': 0.15}, description='Tied embed LR 0.15'))
    results.append(run('embed_lr_0.2', {**best, 'TIED_EMBED_LR': 0.2}, description='Tied embed LR 0.2'))

    # Matrix LR
    results.append(run('matrix_lr_0.06', {**best, 'MATRIX_LR': 0.06}, description='Matrix LR 0.06'))
    results.append(run('matrix_lr_0.1', {**best, 'MATRIX_LR': 0.1}, description='Matrix LR 0.1'))
    results.append(run('matrix_lr_0.12', {**best, 'MATRIX_LR': 0.12}, description='Matrix LR 0.12'))

    # Scalar LR relative to matrix
    results.append(run('scalar_lr_0.04', {**best, 'SCALAR_LR': 0.04}, description='Scalar LR 0.04 (lower)'))
    results.append(run('scalar_lr_0.12', {**best, 'SCALAR_LR': 0.12}, description='Scalar LR 0.12 (higher)'))

    print(f'\n\n{"="*60}')
    print('EXPERIMENT 9 SUMMARY')
    print(f'{"="*60}')
    ok = sorted([r for r in results if r['ok']], key=lambda x: x['bpb'] or 999)
    for r in ok:
        bpb = f"{r['bpb']:.4f}" if r['bpb'] else "N/A"
        print(f"{r['name']:<20} BPB:{bpb} Time:{r['elapsed']}")
    with open('output/experiment_results9.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
