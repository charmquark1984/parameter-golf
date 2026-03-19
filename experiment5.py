"""
Experiment 5: GELU activation at longer training runs and different model scales.
Tests whether GELU advantage holds with more training, and at scales closer to competition budget.
"""
import subprocess
import sys
import json
import os
import time


def run(name, env_overrides, script='train_cpu_v2.py', description=''):
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
        print(f'  Params:{params} Size:{size} BPB:{bpb} (pre-quant:{pre_quant_bpb}) Quant:{quant_deg} Time:{elapsed:.1f}s')
    else:
        print(f'  FAILED!')
        for line in lines[-5:]: print(f'  > {line}')
    return r


def main():
    results = []

    # Best config
    best = {'NUM_LAYERS': 4, 'MODEL_DIM': 128, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 2,
            'MLP_MULT': 3, 'WARMDOWN_ITERS': 40,
            'MATRIX_LR': 0.08, 'TIED_EMBED_LR': 0.1, 'SCALAR_LR': 0.08,
            'GRAD_CLIP_NORM': 1.0, 'MUON_MOMENTUM': 0.8, 'BETA2': 0.99,
            'QK_GAIN_INIT': 1.0, 'MLP_TYPE': 'gelu', 'VAL_LOSS_EVERY': 100}

    # === GELU at different training lengths ===
    results.append(run('gelu_200', {**best, 'ITERATIONS': 200, 'WARMDOWN_ITERS': 40},
        description='GELU baseline 200 iters'))

    results.append(run('gelu_500', {**best, 'ITERATIONS': 500, 'WARMDOWN_ITERS': 100},
        description='GELU 500 iters'))

    results.append(run('gelu_1000', {**best, 'ITERATIONS': 1000, 'WARMDOWN_ITERS': 200},
        description='GELU 1000 iters'))

    # === ReLU² at same lengths for comparison ===
    results.append(run('relu2_500', {**best, 'MLP_TYPE': 'relu2', 'ITERATIONS': 500, 'WARMDOWN_ITERS': 100},
        description='ReLU² 500 iters (comparison)'))

    results.append(run('relu2_1000', {**best, 'MLP_TYPE': 'relu2', 'ITERATIONS': 1000, 'WARMDOWN_ITERS': 200},
        description='ReLU² 1000 iters (comparison)'))

    # === GELU at different model scales ===
    results.append(run('gelu_dim192', {**best, 'MODEL_DIM': 192, 'ITERATIONS': 500, 'WARMDOWN_ITERS': 100},
        description='GELU dim=192, 500 iters'))

    results.append(run('gelu_6layer', {**best, 'NUM_LAYERS': 6, 'ITERATIONS': 500, 'WARMDOWN_ITERS': 100},
        description='GELU 6 layers, 500 iters'))

    results.append(run('gelu_dim192_6layer', {**best, 'MODEL_DIM': 192, 'NUM_LAYERS': 6,
        'ITERATIONS': 300, 'WARMDOWN_ITERS': 60},
        description='GELU dim=192, 6 layers'))

    # === GeGLU at same lengths (second best activation) ===
    results.append(run('geglu_500', {**best, 'MLP_TYPE': 'geglu', 'ITERATIONS': 500, 'WARMDOWN_ITERS': 100},
        description='GeGLU 500 iters'))

    results.append(run('geglu_1000', {**best, 'MLP_TYPE': 'geglu', 'ITERATIONS': 1000, 'WARMDOWN_ITERS': 200},
        description='GeGLU 1000 iters'))

    # Print summary
    print(f'\n\n{"="*80}')
    print('EXPERIMENT 5 SUMMARY: GELU vs alternatives at different scales')
    print(f'{"="*80}')
    print(f'{"Name":<25} {"Params":<12} {"BPB":<10} {"Pre-Q BPB":<12} {"Size":<15} {"Time":<8}')
    print(f'{"-"*82}')
    ok = sorted([r for r in results if r['ok']], key=lambda x: x['bpb'] or 999)
    for r in ok:
        bpb = f"{r['bpb']:.4f}" if r['bpb'] else "N/A"
        pq = f"{r['pre_quant_bpb']:.4f}" if r['pre_quant_bpb'] else "N/A"
        print(f"{r['name']:<25} {r['params'] or 'N/A':<12} {bpb:<10} {pq:<12} {r['size'] or 'N/A':<15} {r['elapsed']:<8}")

    failed = [r for r in results if not r['ok']]
    if failed:
        print(f"\nFailed: {', '.join(r['name'] for r in failed)}")

    os.makedirs('output', exist_ok=True)
    with open('output/experiment_results5.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to output/experiment_results5.json')


if __name__ == '__main__':
    main()
