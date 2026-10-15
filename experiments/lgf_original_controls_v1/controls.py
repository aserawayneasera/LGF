#!/usr/bin/env python3
"""Evaluate Table S6 gate controls on saved lgf_original EMA checkpoints only."""
import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

from gate_controls import apply_control, blocks, definition, names, parameter_cost

ROOT = Path(__file__).resolve().parent
VERSION = '1.0.0'
DATASETS = ('coco_nw', 'coco_weather', 'acdc')
METRICS = ('AP_small', 'mAP', 'AP50', 'AP75', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    temp = Path(str(path) + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def bootstrap_package(package):
    package = Path(package).resolve()
    required = ('core_bridge.py', 'data.py', 'evaluation.py', 'ugs.py', 'utils.py',
                'vendor/original_core.py', 'vendor/source_identity.json')
    for name in required:
        if not (package / name).is_file():
            raise FileNotFoundError(f'Missing {package/name}; supply the UGS comparison package directory')
    recorded = ROOT / 'source_runner/SOURCE_HASHES.json'
    bundled = json.loads(recorded.read_text())
    actual = {name: sha256(package/name) for name in required}
    if actual['vendor/original_core.py'] != bundled['vendor/original_core.py']:
        raise ValueError('LGF model source differs from the inspected original. Do not guess compatibility.')
    for module in ('core_bridge', 'data', 'evaluation', 'ugs', 'utils'):
        if module in sys.modules:
            loaded = Path(sys.modules[module].__file__).resolve().parent
            if loaded != package:
                raise RuntimeError(f'{module} already imported from another directory; run in a fresh process')
    sys.path.insert(0, str(package))
    return actual


def setup(args):
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    os.environ.setdefault('TORCH_HOME', '/tmp/asera/paper1_original/torch_cache')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    source_hashes = bootstrap_package(args.lgf_package)
    import torch
    from utils import require_supported_torch
    require_supported_torch()
    if args.device.isdigit():
        args.device = 'cuda:' + args.device
    if args.device == 'cuda':
        args.device = 'cuda:0'
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use --device cpu, cuda, cuda:0, or 0')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    if os.getenv('MAX_TRAIN_IMAGES'):
        raise ValueError('Unset MAX_TRAIN_IMAGES to avoid hidden dataset overrides')
    return device, source_hashes


def checkpoint_path(args, dataset, seed):
    if args.checkpoint:
        return args.checkpoint.resolve()
    folder = args.run_root / dataset / 'lgf_original' / f'seed_{seed}'
    candidates = [folder/(args.selection+suffix) for suffix in ('.pth', '.pt')]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f'Expected one selected checkpoint in {folder}: '
                                f'{args.selection}.pth or .pt; found {len(existing)}. '
                                'Finish training, narrow --datasets/--seeds, or set --run-root.')
    return existing[0].resolve()


def inspect_checkpoint(path, dataset, seed, level, selection):
    import torch
    if not path.is_file():
        raise FileNotFoundError(path)
    before = sha256(path)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get('ema_state_dict'), dict):
        raise ValueError(f'{path}: a checkpoint dictionary with ema_state_dict is required')
    cfg = ck.get('config', {})
    if (cfg.get('block_type'), cfg.get('gating_type'), cfg.get('BRANCH_PRESET')) != (
            'lgf', 'gated_spatial', 'local_global'):
        raise ValueError('Only lgf_original with both branches and spatial sigmoid gates is accepted')
    if ck.get('package_version'):
        identity = (ck.get('method'), ck.get('dataset'), ck.get('seed'), ck.get('level'))
        if identity != ('lgf_original', dataset, seed, level):
            raise ValueError(f'Checkpoint identity {identity} differs from requested lgf_original/{dataset}/{seed}/{level}')
        if path.stem != selection:
            raise ValueError('Use the selected checkpoint filename matching --selection; last.pth is not accepted')
        epoch = ck['completed_epochs']
        origin = 'ugs_comparison_runner'
        # The sidecar is optional, but if present its provenance must match.
        sidecar = path.with_name(selection + '.metrics.json')
        if sidecar.is_file():
            meta = json.loads(sidecar.read_text())
            if meta.get('checkpoint_sha256') != before:
                raise ValueError('Selected checkpoint and sidecar hash differ; training may be updating this run')
        if selection not in ck.get('improved_selection_keys', []):
            raise ValueError('Checkpoint metadata does not identify this epoch as the requested selection')
    else:
        identity = re.fullmatch(r'(coco_nw|coco_weather|acdc)_(C[345])_.+_s(-?\d+)',
                                str(ck.get('experiment_name', '')))
        if not identity or (identity[1], identity[2], int(identity[3])) != (dataset, level, seed):
            raise ValueError('Original notebook checkpoint requires matching experiment_name metadata')
        if selection != 'best_map':
            raise ValueError('The inspected original notebook selected by mAP. Use --selection best_map.')
        if not path.name.startswith('BEST_'):
            raise ValueError('Original checkpoint must be an original BEST_... file')
        epoch = int(ck.get('epoch', -1)) + 1
        origin = 'original_notebook'
    if sha256(path) != before:
        raise RuntimeError('Checkpoint changed while being read; rerun after this training save completes')
    state = ck['ema_state_dict']
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if not state:
        raise ValueError('Empty EMA state')
    meta = {'checkpoint': str(path), 'checkpoint_sha256': before, 'dataset': dataset,
            'seed': seed, 'level': level, 'method': 'lgf_original', 'epoch': epoch,
            'selection': selection, 'checkpoint_origin': origin,
            'original_config': cfg, 'training_protocol_sha256': ck.get('protocol_sha256')}
    return ck, state, meta


def inventory(args):
    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            path = checkpoint_path(args, dataset, seed)
            ck, state, meta = inspect_checkpoint(path, dataset, seed, args.level, args.selection)
            jobs.append(meta)
            print(f'CHECKPOINT PASS {dataset} seed={seed} epoch={meta["epoch"]}: {path}', flush=True)
            del ck, state
    return jobs


def dataset_record(args, dataset):
    from data import audit
    rec = audit(dataset, args.data_root, val_only=True)
    print(f'DATA PASS {dataset}: {rec["val"]["image_count"]} validation images', flush=True)
    return rec


def verify_record(ck, record):
    expected = ck.get('protocol', {}).get('validation_content_sha256')
    if expected and expected != record['val']['content_sha256']:
        raise ValueError('Validation files differ from the training record. Restore the matching dataset copy.')
    return bool(expected)


def state_digest(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(ck, state, level, device):
    from core_bridge import build_model
    model = build_model('lgf_original', 6, level, pretrained=False, original_config=ck['config'])
    model.load_state_dict(state, strict=True)
    model.to(device).float().eval()
    blocks(model)
    from evaluation import model_protocol
    protocol = model_protocol(model)
    recorded = ck.get('protocol', {}).get('model_protocol')
    if recorded and signature(protocol) != signature(recorded):
        raise ValueError('Reconstructed model inference settings differ from the training record')
    return model


def check_reference(ck, result, batch_size, tolerance):
    ref = ck.get('epoch_result') or {}
    expected = ref.get('metrics') or ck.get('eval_results')
    same_batch = ck.get('protocol', {}).get('batch_size') == batch_size
    if not expected or not same_batch:
        return {'status': 'not_compared', 'reason': 'Stored metrics or matching recorded batch size unavailable'}
    differences = {}
    for name in ('AP_small', 'mAP'):
        a, b = result['metrics'].get(name), expected.get(name)
        if a is not None and b is not None and a >= 0 and b >= 0:
            differences[name] = float(a - b)
    if any(abs(v) > tolerance for v in differences.values()):
        raise RuntimeError(f'Native AP differs from saved evaluation: {differences}. '
                           'Check environment and data before interpreting controls. '
                           'Differences are in 0-1 AP units.')
    return {'status': 'pass' if differences else 'not_compared',
            'differences_0_to_1': differences, 'tolerance_0_to_1': tolerance}


def timed_inference(model, images, device, warmup, repeats):
    import torch
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(images)
        sync()
        times = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            model(images)
            sync()
            times.append((time.perf_counter()-start)*1000)
    median = statistics.median(times)
    return {'batch_size': len(images), 'median_batch_ms': median,
            'median_amortized_ms_per_image': median/len(images),
            'median_throughput_images_per_second': len(images)*1000/median,
            'batch_ms_samples': times, 'warmup': warmup, 'repeats': repeats,
            'input_shapes': [list(x.shape) for x in images],
            'scope': 'Repeated first validation batch, preloaded on device; includes detector postprocessing. '
                     'Amortized timing is not batch-one latency or a whole-dataset timing estimate.'}


def run_one(args, job, record, device, code_hashes, environment):
    import torch
    from core_bridge import load_core
    from data import make_loaders
    from evaluation import evaluate, model_protocol
    checkpoint = Path(job['checkpoint'])
    ck, state, meta = inspect_checkpoint(checkpoint, job['dataset'], job['seed'], args.level, args.selection)
    if meta['checkpoint_sha256'] != job['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint changed after inventory. Use a new output directory for new weights.')
    data_verified = verify_record(ck, record)
    core = load_core()
    core.set_seed(job['seed'])
    model = load_model(ck, state, args.level, device)
    del state
    original_state = state_digest(model)
    _, loader, _ = make_loaders(record, args.batch_size, args.num_workers, args.prefetch_factor,
                               job['seed'], val_only=True)
    first_images, _ = next(iter(loader))
    first_images = [im.to(device) for im in first_images]
    with torch.inference_mode(), apply_control(model, 'native', verify_native=True) as evidence:
        model(first_images)
    if len(evidence['verified_blocks']) != len(blocks(model)):
        raise RuntimeError('Not every LGF block executed during native parity verification')
    print(f'NATIVE PARITY PASS {job["dataset"]} seed={job["seed"]}: exact block outputs', flush=True)
    cost = parameter_cost(model)
    gate_macs = 0
    for name, m in blocks(model):
        n, c, h, w = evidence['block_shapes'][name]
        gate_macs += n*h*w*(m.gate_reduce.weight.numel()+m.gate_expand.weight.numel())
    cost.update({'gate_conv_MACs_first_batch': gate_macs,
                 'gate_conv_FLOPs_first_batch_2_per_MAC': 2*gate_macs,
                 'cost_scope': 'Gate convolutions only; excludes normalization, sigmoid, elementwise ops. '
                               'Not total detector FLOPs.', 'block_input_shapes': evidence['block_shapes']})
    protocol = {'version': VERSION, 'method': 'lgf_original', 'suite': args.suite,
                'controls': list(names(args.suite)), 'batch_size': args.batch_size,
                'shuffle_seed': args.shuffle_seed, 'model_protocol': model_protocol(model),
                'dataset': job['dataset'], 'level': args.level, 'selection': args.selection,
                'validation_content_sha256': record['val']['content_sha256'],
                'code_hashes': code_hashes, 'environment': environment,
                'benchmark': args.benchmark, 'benchmark_warmup': args.benchmark_warmup,
                'benchmark_repeats': args.benchmark_repeats,
                'reference_ap_tolerance': args.reference_ap_tolerance}
    group_signature = signature(protocol)
    run_signature = signature({'protocol': protocol, 'checkpoint_sha256': meta['checkpoint_sha256'],
                               'seed': job['seed']})
    folder = args.output_root/job['dataset']/f'seed_{job["seed"]}'
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'checkpoint.json', {**meta, 'cost': cost, 'native_parity': evidence,
               'training_validation_bytes_verified': data_verified, 'protocol': protocol})
    native = None
    for control in names(args.suite):
        target = folder/control
        result_path = target/'control_result.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get('run_signature') != run_signature:
                raise ValueError(f'{target}: results belong to different inputs/settings; choose a new --output-root')
            pred = target/'predictions.json'
            if not pred.is_file() or sha256(pred) != result['predictions_sha256']:
                raise ValueError(f'{target}: saved prediction file changed or is missing')
            print(f'RESUME {job["dataset"]} seed={job["seed"]} {control}', flush=True)
        else:
            core.set_seed(job['seed'])
            print(f'EVALUATE {job["dataset"]} seed={job["seed"]} {control}', flush=True)
            with apply_control(model, control, args.suite, args.shuffle_seed):
                result = evaluate(model, loader, record['paths']['val_annotations'], device, target,
                                  record, meta)
            result.update({'suite': args.suite, 'control': control,
                           'control_definition': definition(args.suite, control),
                           'run_signature': run_signature, 'group_signature': group_signature,
                           'shuffle_seed': args.shuffle_seed, 'batch_size': args.batch_size,
                           'training_validation_bytes_verified': data_verified,
                           'scope': 'Post-training validation controls; not separately trained ungated models',
                           'native_parity_verified': True})
            if control == 'native':
                result['saved_native_reference'] = check_reference(
                    ck, result, args.batch_size, args.reference_ap_tolerance)
            if args.benchmark:
                with apply_control(model, control, args.suite, args.shuffle_seed):
                    result['benchmark'] = timed_inference(model, first_images, device,
                                                          args.benchmark_warmup, args.benchmark_repeats)
            if control != 'native' and native is None:
                raise RuntimeError('Native evaluation must complete first')
            base = result if control == 'native' else native
            result['delta_vs_native_ap_points'] = {
                k: (100*(result['metrics'][k]-base['metrics'][k])
                    if result['metrics'].get(k) is not None and base['metrics'].get(k) is not None
                    and result['metrics'][k] >= 0 and base['metrics'][k] >= 0 else None)
                for k in METRICS}
            write_json(result_path, result)
        if control == 'native':
            native = result
        print(f'RESULT {control}: APsmall={result["metrics"]["AP_small"]} '
              f'mAP={result["metrics"]["mAP"]} '
              f'delta_APsmall_points={result["delta_vs_native_ap_points"]["AP_small"]}', flush=True)
    if state_digest(model) != original_state:
        raise RuntimeError('Model parameters or buffers changed during control evaluation')
    if sha256(checkpoint) != meta['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint changed during evaluation. Results were generated from the original loaded weights.')
    write_json(folder/'completed.json', {'run_signature': run_signature,
               'checkpoint_sha256': meta['checkpoint_sha256'], 'state_unchanged': True,
               'controls': list(names(args.suite))})
    del model, loader, first_images, ck
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def mean_std(values):
    valid = [float(v) for v in values if v is not None and math.isfinite(v)]
    return (statistics.mean(valid) if valid else None,
            statistics.stdev(valid) if len(valid) > 1 else None, len(valid))


def summarize(root):
    root = Path(root)
    experiment = json.loads((root/'experiment.json').read_text())
    suite = experiment['settings']['suite']
    expected_jobs = experiment['jobs']
    rows, missing = [], []
    groups = {}
    for job in expected_jobs:
        folder = root/job['dataset']/f'seed_{job["seed"]}'
        if not (folder/'completed.json').is_file():
            missing.append(f'{job["dataset"]}/seed_{job["seed"]}: not complete')
            continue
        complete = json.loads((folder/'completed.json').read_text())
        for control in names(suite):
            path = folder/control/'control_result.json'
            if not path.is_file():
                raise ValueError(f'Completed run has missing result: {path}')
            r = json.loads(path.read_text())
            if (r['checkpoint_sha256'] != job['checkpoint_sha256']
                    or r['run_signature'] != complete['run_signature']
                    or r['control'] != control or r['suite'] != suite):
                raise ValueError('Result provenance differs from the experiment or completion record')
            pred = path.parent/'predictions.json'
            if not pred.is_file() or sha256(pred) != r['predictions_sha256']:
                raise ValueError(f'Prediction integrity check failed: {pred}')
            row = {'dataset': r['dataset'], 'seed': r['seed'], 'suite': suite,
                   'control': control, 'selection': r['selection'], 'epoch': r['epoch'],
                   'checkpoint_sha256': r['checkpoint_sha256']}
            for name in METRICS:
                value = r['metrics'].get(name)
                row[name] = value if value is not None and value >= 0 else None
                row['delta_'+name+'_points'] = r['delta_vs_native_ap_points'][name]
            rows.append(row)
            key = (r['dataset'], control)
            groups.setdefault(key, []).append((row, r['group_signature']))
    aggregate = []
    for (dataset, control), entries in groups.items():
        if len({sig for _, sig in entries}) != 1:
            raise ValueError('Refusing to aggregate seeds with different datasets or evaluation settings')
        seeds = [row['seed'] for row, _ in entries]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate training seeds')
        out = {'dataset': dataset, 'suite': suite, 'control': control,
               'seeds': ','.join(str(x) for x in sorted(seeds)), 'n_seeds': len(seeds),
               'complete_requested_grid': not bool(missing)}
        for metric in METRICS:
            for col in (metric, 'delta_'+metric+'_points'):
                avg, std, count = mean_std([row[col] for row, _ in entries])
                out[col+'_mean'], out[col+'_std'], out[col+'_n'] = avg, std, count
        aggregate.append(out)
    write_csv(root/'controls_per_seed.csv', rows)
    write_csv(root/'controls_mean_std.csv', aggregate)
    status = {'complete': not bool(missing), 'missing': missing, 'rows': len(rows),
              'std': 'Sample standard deviation across training seeds, ddof=1; null for one seed',
              'delta': 'Control minus native within the same checkpoint; positive means the control performed better',
              'units': 'AP values 0-1; delta columns in AP points (100 times the AP difference)',
              'scope': 'Validation controls. Seed SD is not a significance test or confidence interval.'}
    write_json(root/'summary_status.json', status)
    print(f'SUMMARY {"COMPLETE" if status["complete"] else "PARTIAL"}: {root}', flush=True)
    return status


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'run', 'summarize'))
    p.add_argument('--run-root', type=Path, default=Path('/tmp/asera/paper1_original/ugs_comparison/runs'))
    p.add_argument('--data-root', type=Path, default=Path('/tmp/asera/paper1_original/DATASET'))
    p.add_argument('--output-root', type=Path, default=Path('/tmp/asera/paper1_original/lgf_controls_v1/paper'))
    p.add_argument('--lgf-package', type=Path, default=ROOT/'source_runner')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1337, 2025])
    p.add_argument('--level', choices=('C3', 'C4', 'C5'), default='C3')
    p.add_argument('--selection', choices=('best_ap_small', 'best_map'), default='best_ap_small')
    p.add_argument('--checkpoint', type=Path, help='Explicit checkpoint, requires one dataset and one seed')
    p.add_argument('--suite', choices=('paper', 'notebook'), default='paper')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--prefetch-factor', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--shuffle-seed', type=int, default=123)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--benchmark-warmup', type=int, default=5)
    p.add_argument('--benchmark-repeats', type=int, default=20)
    p.add_argument('--reference-ap-tolerance', type=float, default=1e-4)
    args = p.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate datasets or seeds')
    if args.checkpoint and (len(args.datasets) != 1 or len(args.seeds) != 1):
        p.error('--checkpoint requires one --datasets value and one --seeds value')
    if min(args.batch_size, args.prefetch_factor, args.threads, args.benchmark_repeats) < 1:
        p.error('Batch size, prefetch factor, threads, and benchmark repeats must be positive')
    if min(args.num_workers, args.benchmark_warmup) < 0:
        p.error('Workers and warmup must be nonnegative')
    if not math.isfinite(args.reference_ap_tolerance) or args.reference_ap_tolerance < 0:
        p.error('Reference tolerance must be finite and nonnegative')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.action == 'summarize':
        summarize(args.output_root)
        return
    device, source_hashes = setup(args)
    jobs = inventory(args)
    records = {dataset: dataset_record(args, dataset) for dataset in args.datasets}
    # Check every requested checkpoint and dataset before starting a long grid.
    for job in jobs:
        ck, state, _ = inspect_checkpoint(Path(job['checkpoint']), job['dataset'], job['seed'],
                                          args.level, args.selection)
        verified = verify_record(ck, records[job['dataset']])
        if args.action == 'check':
            model = load_model(ck, state, args.level, device)
            print(f'MODEL LOAD PASS {job["dataset"]} seed={job["seed"]}; '
                  f'validation_hash_verified={verified}; {parameter_cost(model)}', flush=True)
            del model
        del ck, state
    if args.action == 'check':
        print(f'CHECK PASS: {len(jobs)} lgf_original checkpoints; '
              f'{len(jobs)*len(names(args.suite))} planned evaluations. No training.', flush=True)
        return
    import fcntl
    from utils import environment
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root/'evaluation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another control evaluation is writing to this output directory') from exc
        code_hashes = {**source_hashes, 'controls.py': sha256(ROOT/'controls.py'),
                       'gate_controls.py': sha256(ROOT/'gate_controls.py')}
        env = environment()
        env.update({'requested_device': str(device),
                    'cuda_launch_blocking': os.getenv('CUDA_LAUNCH_BLOCKING'),
                    'threads': args.threads})
        settings = {k: (str(v.resolve()) if isinstance(v, Path) else v)
                    for k, v in vars(args).items() if k not in ('action', 'checkpoint')}
        identity = {'version': VERSION, 'settings': settings, 'jobs': jobs,
                    'code_hashes': code_hashes, 'environment': env}
        exp = args.output_root/'experiment.json'
        if exp.is_file() and signature(json.loads(exp.read_text())) != signature(identity):
            raise ValueError('Output directory belongs to another grid/settings/checkpoint version. '
                             'Repeat the exact command to resume, or use a new --output-root.')
        write_json(exp, identity)
        for job in jobs:
            run_one(args, job, records[job['dataset']], device, code_hashes, env)
            summarize(args.output_root)
        print('ALL CONTROLS COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Repeat the identical command to resume completed controls.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
