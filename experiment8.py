"""
Experiment 8: Competition-scale architecture search.
Find the best architecture that fits in 16MB int8+zlib budget.
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
        print(f'  Params:{params} Size:{size} BPB:{bpb} (pre-quant:{pre_quant_bpb}) Time:{elapsed:.1f}s')
    else:
        print(f'  FAILED!')
        for line in lines[-3:]: print(f'  > {line}')
    return r


def main():
    results = []

    # Base training config for competition-scale tests
    train = {
        'MATRIX_LR': 0.08, 'TIED_EMBED_LR': 0.1, 'SCALAR_LR': 0.08,
        'GRAD_CLIP_NORM': 1.0, 'MUON_MOMENTUM': 0.8, 'BETA2': 0.99,
        'QK_GAIN_INIT': 1.0, 'MLP_TYPE': 'gelu', 'LR_SCHEDULE': 'cosine',
        'ITERATIONS': 20, 'WARMDOWN_ITERS': 8, 'TRAIN_BATCH_TOKENS': 2048,
        'VAL_LOSS_EVERY': 10, 'TRAIN_SEQ_LEN': 256,
    }

    # The goal is to find configs that produce int8+zlib artifacts ~15-16MB
    # From Exp 7:
    #   9x512, 2xMLP = 17.1M params = 9.4 MB  (too small)
    #   9x512, 3xMLP = 21.8M params = 11.7 MB (headroom)
    # We need to find configs in 12-16 MB range

    # === GELU 3x MLP, vary depth and width ===

    # Try more layers with 3x MLP
    results.append(run('10x512_gelu3x', {**train,
        'NUM_LAYERS': 10, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 3},
        description='10 layers, dim=512, GELU 3x'))

    results.append(run('11x512_gelu3x', {**train,
        'NUM_LAYERS': 11, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 3},
        description='11 layers, dim=512, GELU 3x'))

    results.append(run('12x512_gelu3x', {**train,
        'NUM_LAYERS': 12, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 3},
        description='12 layers, dim=512, GELU 3x'))

    # Try wider dim with 3x MLP
    results.append(run('9x544_gelu3x', {**train,
        'NUM_LAYERS': 9, 'MODEL_DIM': 544, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 3},
        description='9 layers, dim=544, GELU 3x'))

    results.append(run('9x576_gelu3x', {**train,
        'NUM_LAYERS': 9, 'MODEL_DIM': 576, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 3},
        description='9 layers, dim=576, GELU 3x'))

    # Try 2x MLP but much wider/deeper
    results.append(run('12x512_gelu2x', {**train,
        'NUM_LAYERS': 12, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 2},
        description='12 layers, dim=512, GELU 2x'))

    results.append(run('9x640_gelu2x', {**train,
        'NUM_LAYERS': 9, 'MODEL_DIM': 640, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 2},
        description='9 layers, dim=640, GELU 2x'))

    # Try extreme GQA to save params (2 KV heads)
    results.append(run('11x512_2kv_gelu3x', {**train,
        'NUM_LAYERS': 11, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 2, 'MLP_MULT': 3},
        description='11 layers, dim=512, 2 KV heads, GELU 3x'))

    results.append(run('12x512_2kv_gelu3x', {**train,
        'NUM_LAYERS': 12, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 2, 'MLP_MULT': 3},
        description='12 layers, dim=512, 2 KV heads, GELU 3x'))

    # Try 4x MLP with fewer layers
    results.append(run('7x512_gelu4x', {**train,
        'NUM_LAYERS': 7, 'MODEL_DIM': 512, 'NUM_HEADS': 8, 'NUM_KV_HEADS': 4, 'MLP_MULT': 4},
        description='7 layers, dim=512, GELU 4x'))

    # Print summary
    print(f'\n\n{"="*80}')
    print('EXPERIMENT 8: Competition-Scale Architecture Search')
    print(f'{"="*80}')
    print(f'{"Name":<30} {"Params":<15} {"Size (MB)":<15} {"BPB":<10} {"Time":<8}')
    print(f'{"-"*78}')
    ok = sorted([r for r in results if r['ok']], key=lambda x: x['bpb'] or 999)
    for r in ok:
        bpb = f"{r['bpb']:.4f}" if r['bpb'] else "N/A"
        # Extract MB from size string
        size_str = r['size'] or 'N/A'
        print(f"{r['name']:<30} {r['params'] or 'N/A':<15} {size_str:<15} {bpb:<10} {r['elapsed']:<8}")

    # Also print size-sorted view
    print(f'\n--- Size-sorted (for 16MB budget) ---')
    for r in sorted([r for r in results if r['ok']], key=lambda x: float(x['size'].split('(')[1].split(' ')[0]) if x['size'] else 999):
        size_str = r['size'] or 'N/A'
        bpb = f"{r['bpb']:.4f}" if r['bpb'] else "N/A"
        fits = "<<< FITS" if r['size'] and float(r['size'].split('(')[1].split(' ')[0]) < 16.0 else "OVER"
        print(f"{r['name']:<30} {size_str:<20} {bpb:<10} {fits}")

    os.makedirs('output', exist_ok=True)
    with open('output/experiment_results8.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
