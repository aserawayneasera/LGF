#!/usr/bin/env python3
"""Evaluate saved C3/C4 EMA checkpoints on paired fog, rain and snow images."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import datetime as dt
import gc
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = Path('/nas.dbms/asera/paper1_original')
CONDITIONS = ('fog', 'rain', 'snow')
METRICS = ('mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')
CLASSES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}
VERSION = '1.0.0'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_versions():
    versions = {'python': sys.version}
    for package in ('torch', 'torchvision', 'numpy', 'pycocotools', 'albumentations',
                    'opencv-python', 'opencv-python-headless', 'Pillow'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows for ' + str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_archived_sources():
    entries = read_json(ROOT / 'archived_sources.json')
    for entry in entries:
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Archived source changed: ' + entry['path'])
    return {entry['path']: entry['sha256'] for entry in entries}


def paths(a):
    weather = a.data_root / 'COCO/weather-mini'
    source = a.source_annotations or weather / 'annotations/mini_val2017_weather-500k6c.json'
    images = a.images or weather / 'images/val2017_weather-500k6c'
    training = a.train_annotations or weather / 'annotations/mini_train2017_weather-2400k6c.json'
    return source.resolve(), images.resolve(), training.resolve()


def prepare(a):
    import prepare_coco_weather_variants as builder
    if not a.confirm_aligned_variants:
        raise ValueError('Use prepare --confirm-aligned-variants for your existing geometry-preserving weather images.')
    validate_archived_sources()
    source, images, training = paths(a)
    audit = {'dataset': 'coco_weather', 'evaluation_split': 'validation',
             'annotations': str(source), 'annotations_sha256': sha256(source),
             'image_root': str(images), 'training_annotation_checked': str(training)}
    audit_path = a.output / 'preparation/source_audit.json'
    if audit_path.exists() and read_json(audit_path) != audit:
        raise ValueError('Preparation inputs changed. Select a new --output directory.')
    write_json(audit_path, audit)
    extra = a.data_root / 'COCO/non_weather-mini/annotations/mini_train2017_non_weather-2400k6c.json'
    args = SimpleNamespace(audit=audit_path, output_dir=a.output / 'manifests',
                           confirm_aligned_variants=True, expect_sources=a.expect_sources,
                           train_annotations=[extra] if extra.is_file() else [])
    builder.prepare(args)


def validate_bundle(directory, images, training, expect_sources=500, check_images=True):
    import prepare_coco_weather_variants as builder
    from PIL import Image
    directory = Path(directory)
    report = read_json(directory / 'preparation_report.json')
    annotation = directory / builder.COMBINED_NAME
    if sha256(annotation) != report['output_sha256'][builder.COMBINED_NAME]:
        raise ValueError('Combined evaluation annotations differ from preparation records.')
    source = Path(report['source_annotations'])
    if sha256(source) != report['source_annotations_sha256']:
        raise ValueError('Original validation annotations changed after preparation.')
    doc = read_json(annotation)
    builder.validate_source(doc)
    train_doc = read_json(training)
    cats = sorted((c['id'], c['name']) for c in doc['categories'])
    train_cats = sorted((c['id'], c['name']) for c in train_doc['categories'])
    if cats != train_cats or {name for _, name in cats} != CLASSES:
        raise ValueError('Training/evaluation category IDs or names differ.')
    groups = defaultdict(dict)
    rows = []
    hashes = {int(row['image_id']): row for row in report['image_records']}
    for im in doc['images']:
        condition = im.get('weather')
        group = im.get('source_group_id')
        if condition not in CONDITIONS or not isinstance(group, str) or not group:
            raise ValueError('Missing weather/source group metadata for image ' + str(im['id']))
        if condition in groups[group]:
            raise ValueError('Duplicate condition in source group ' + group)
        groups[group][condition] = im['id']
        if im['id'] not in hashes or hashes[im['id']]['file_name'] != im['file_name']:
            raise ValueError('Image differs from preparation records.')
        file = builder.checked_path(images, im['file_name'])
        if not file.is_file():
            raise FileNotFoundError(file)
        if check_images:
            if sha256(file) != hashes[im['id']]['sha256']:
                raise ValueError('Weather image changed after preparation: ' + str(file))
            with Image.open(file) as decoded:
                if decoded.size != (im['width'], im['height']):
                    raise ValueError('Unexpected image dimensions: ' + str(file))
                decoded.load()
        rows.append({'image_id': im['id'], 'condition': condition})
    if expect_sources and len(groups) != expect_sources:
        raise ValueError(f'Expected {expect_sources} source scenes, found {len(groups)}.')
    if any(set(value) != set(CONDITIONS) for value in groups.values()):
        raise ValueError('Every source scene must have fog, rain and snow versions.')
    if len(doc['images']) != 3 * len(groups) or len(hashes) != len(doc['images']):
        raise ValueError('Evaluation/preparation image inventories differ.')
    annotations = defaultdict(list)
    for ann in doc['annotations']:
        original = copy.deepcopy(ann)
        original.pop('id')
        original.pop('image_id')
        annotations[ann['image_id']].append(signature(original))
    for group, members in groups.items():
        values = [sorted(annotations[members[c]]) for c in CONDITIONS]
        if values[0] != values[1] or values[0] != values[2]:
            raise ValueError('Annotation geometry differs across weather variants: ' + group)
    fingerprint = {'annotations_sha256': sha256(annotation),
                   'source_annotations_sha256': report['source_annotations_sha256'],
                   'training_annotations_sha256': sha256(training),
                   'images': [{'id': im['id'], 'sha256': hashes[im['id']]['sha256']}
                              for im in sorted(doc['images'], key=lambda x: x['id'])],
                   'category_order': cats, 'source_scenes': len(groups)}
    return doc, {'rows': rows}, fingerprint


def entries(a):
    if len(set(a.placements)) != len(a.placements) or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Duplicate saved run labels or seeds.')
    result = []
    for placement in a.placements:
        folder = 'runs' if placement == 'C3' else 'runs_C4'
        for seed in a.seeds:
            run = a.runs_base / folder / 'coco_weather/lgf_original' / f'seed_{seed}'
            result.append({'placement': placement, 'seed': seed,
                           'checkpoint': run / (a.selection + '.pth'),
                           'metrics': run / (a.selection + '.metrics.json')})
    return result


def selected_metrics(entry, selection, digest):
    if not entry['checkpoint'].is_file() or not entry['metrics'].is_file():
        raise FileNotFoundError(f'Missing checkpoint/selected metrics: {entry["checkpoint"]}')
    value = read_json(entry['metrics'])
    for key, expected in [('seed', entry['seed']), ('dataset', 'coco_weather'), ('selection', selection)]:
        if key in value and value[key] != expected:
            raise ValueError('Selected metrics disagree with requested ' + key)
    if value.get('checkpoint_sha256') and value['checkpoint_sha256'] != digest:
        raise ValueError('Selected metrics reference a different checkpoint.')
    return value


def checkpoint_data_audit(entry, fingerprint):
    path = Path(entry['checkpoint']).parent / 'data_audit.json'
    if not path.is_file():
        return None
    audit = read_json(path)
    val = audit.get('val', {})
    if val.get('annotations_sha256') and val['annotations_sha256'] != fingerprint['source_annotations_sha256']:
        raise ValueError('Checkpoint data audit references different source validation annotations.')
    if val.get('categories'):
        categories = sorted((row['id'], row['name']) for row in val['categories'])
        if categories != fingerprint['category_order']:
            raise ValueError('Checkpoint data audit has different category IDs or names.')
    return sha256(path)


def preflight(a):
    sources = validate_archived_sources()
    _, images, training = paths(a)
    doc, audit, fingerprint = validate_bundle(a.output / 'manifests', images, training,
                                              a.expect_sources)
    checkpoint_rows = []
    for entry in entries(a):
        if not entry['checkpoint'].is_file():
            raise FileNotFoundError(entry['checkpoint'])
        digest = sha256(entry['checkpoint'])
        selected_metrics(entry, a.selection, digest)
        checkpoint_rows.append({**entry, 'checkpoint': str(entry['checkpoint']),
                                'metrics': str(entry['metrics']), 'checkpoint_sha256': digest,
                                'selected_metrics_sha256': sha256(entry['metrics']),
                                'data_audit_sha256': checkpoint_data_audit(entry, fingerprint)})
    counts = Counter(im['weather'] for im in doc['images'])
    print('DATA CHECK:', dict(counts), 'source scenes:', fingerprint['source_scenes'], flush=True)
    for entry in checkpoint_rows:
        print('CHECKPOINT:', entry['placement'], 'seed', entry['seed'], entry['checkpoint'], flush=True)
    print('CHECK PASSED. Saved run labels and checkpoint selection remain fixed.', flush=True)
    return doc, audit, fingerprint, sources, images, checkpoint_rows


def load_model(entry, a):
    import torch
    import torchvision
    import archived_profile_loader as loader
    if not torch.__version__.startswith('2.10.') or not torchvision.__version__.startswith('0.25.'):
        raise RuntimeError('Use your placement training environment: torch 2.10.x / torchvision 0.25.x.')
    sys.path.insert(0, str(ROOT / 'model_source'))
    import core_bridge
    core_bridge.load_core().set_seed(entry['seed'])
    ck = torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or 'ema_state_dict' not in ck:
        raise ValueError('Expected the saved EMA checkpoint.')
    for key, expected in [('seed', entry['seed']), ('dataset', 'coco_weather')]:
        if key in ck and ck[key] != expected:
            raise ValueError('Checkpoint disagrees with requested ' + key)
    metadata = selected_metrics({**entry, 'checkpoint': Path(entry['checkpoint']),
                                  'metrics': Path(entry['metrics'])}, a.selection,
                                entry['checkpoint_sha256'])
    if sha256(entry['metrics']) != entry['selected_metrics_sha256']:
        raise ValueError('Selected metrics changed after preflight.')
    state = loader.normalize_state(ck['ema_state_dict'])
    if not loader.checkpoint_keys(state):
        raise ValueError('The requested LGF checkpoint has no saved LGF block.')
    model, details = loader.reconstruct(ck, state, core_bridge)
    protocol = loader.validate_protocol(model, ck, metadata)
    del ck, state
    gc.collect()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable.')
    return model.to(device).eval(), device, protocol, metadata, details


def infer(model, device, doc, image_root, batch_size):
    import torch
    import cv2
    label_to_category = {label: category['id'] for label, category in enumerate(
                         sorted(doc['categories'], key=lambda x: x['id']))}
    records = sorted(doc['images'], key=lambda im: im['id'])
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            tensors = []
            for im in batch:
                file = image_root / im['file_name']
                image = cv2.imread(str(file))
                if image is None or image.shape[:2] != (im['height'], im['width']):
                    raise ValueError('Unreadable or mismatched image: ' + str(file))
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().float().div(255).to(device))
            outputs = model(tensors)
            if len(outputs) != len(batch):
                raise ValueError('Detector output count differs from the batch image count.')
            for im, output in zip(batch, outputs):
                boxes = output['boxes'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()
                labels = output['labels'].detach().cpu().numpy()
                if not len(boxes) == len(scores) == len(labels):
                    raise ValueError('Detector box, score and label counts differ.')
                for box, score, label in zip(boxes, scores, labels):
                    if int(label) not in label_to_category:
                        raise ValueError('Unknown prediction class label: ' + str(label))
                    x1, y1, x2, y2 = box
                    row = {'image_id': im['id'], 'category_id': label_to_category[int(label)],
                           'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                           'score': float(score)}
                    if not all(math.isfinite(v) for v in row['bbox'] + [row['score']]):
                        raise ValueError('Non-finite detector prediction.')
                    predictions.append(row)
            if start == 0 or (start + len(batch)) % 100 == 0 or start + len(batch) == len(records):
                print(f'  INFERENCE {start + len(batch)}/{len(records)} images', flush=True)
    return predictions


def fog_reference(results, selected, tolerance):
    fog = next(row for row in results if row['condition'] == 'fog')
    original = selected.get('metrics', {})
    comparisons = []
    for metric in METRICS[:6]:
        if original.get(metric) is not None and fog.get(metric) is not None:
            difference = fog[metric] - original[metric]
            comparisons.append({'metric': metric, 'new_fog': fog[metric],
                                'saved_fog': original[metric], 'difference': difference})
    return {'status': ('match' if all(abs(row['difference']) <= tolerance for row in comparisons)
                       else 'different') if comparisons else 'no_reference',
            'tolerance': tolerance, 'comparisons': comparisons}


def verify_result(path):
    path = Path(path)
    result = read_json(path)
    digest = result.get('result_sha256')
    payload = {key: value for key, value in result.items() if key != 'result_sha256'}
    if digest != signature(payload):
        raise ValueError('Saved metrics changed: ' + str(path))
    if signature(result['protocol']) != result['protocol_sha256']:
        raise ValueError('Result protocol changed: ' + str(path))
    if sha256(path.parent / 'predictions.json') != result['predictions_sha256']:
        raise ValueError('Predictions changed: ' + str(path))
    return result


def summarize_results(results, placements, seeds):
    by_job = {(result['placement'], result['seed']): result for result in results}
    expected = {(placement, seed) for placement in placements for seed in seeds}
    if len(by_job) != len(results) or set(by_job) != expected:
        raise ValueError('Incomplete or duplicate seed/placement results.')
    if len({result['dataset_sha256'] for result in results}) != 1:
        raise ValueError('Results use different evaluation datasets.')
    if len({signature(result['evaluation_protocol']) for result in results}) != 1:
        raise ValueError('Results use different inference protocols.')
    per_seed, summary, deltas = [], [], []
    for placement in placements:
        for condition in (*CONDITIONS, 'ALL'):
            rows = []
            for seed in seeds:
                result = by_job[(placement, seed)]
                found = [row for row in result['results'] if row['condition'] == condition]
                if len(found) != 1:
                    raise ValueError('Missing or duplicate condition: ' + condition)
                row = found[0]
                rows.append(row)
                per_seed.append({'placement': placement, 'seed': seed, 'condition': condition,
                                 'images': row['images'], 'source_scenes': result['source_scenes'],
                                 **{metric: row[metric] for metric in METRICS}})
            if len({row['images'] for row in rows}) != 1:
                raise ValueError('Condition image counts differ between seeds.')
            aggregate = {'placement': placement, 'condition': condition, 'images': rows[0]['images'],
                         'source_scenes': by_job[(placement, seeds[0])]['source_scenes'],
                         'n_seeds': len(seeds), 'seeds': ','.join(map(str, seeds))}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                if any(value is None for value in values):
                    aggregate[metric + '_mean'] = aggregate[metric + '_sd'] = None
                else:
                    aggregate[metric + '_mean'] = statistics.mean(values)
                    aggregate[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(aggregate)
    if set(placements) == {'C3', 'C4'}:
        for condition in (*CONDITIONS, 'ALL'):
            row = {'comparison': 'C4 minus C3', 'condition': condition,
                   'n_seeds': len(seeds), 'seeds': ','.join(map(str, seeds))}
            for metric in METRICS[:6]:
                differences = []
                for seed in seeds:
                    left = next(r for r in by_job[('C3', seed)]['results'] if r['condition'] == condition)
                    right = next(r for r in by_job[('C4', seed)]['results'] if r['condition'] == condition)
                    if left[metric] is None or right[metric] is None:
                        differences = []
                        break
                    differences.append((right[metric] - left[metric]) * 100)
                row[metric + '_difference_pp'] = statistics.mean(differences) if differences else None
                row[metric + '_paired_seed_sd_pp'] = statistics.stdev(differences) if len(differences) > 1 else None
            deltas.append(row)
    return per_seed, summary, deltas


def export_summary(a):
    records = []
    for entry in entries(a):
        path = a.output / 'results' / entry['placement'] / f'seed_{entry["seed"]}/metrics.json'
        result = verify_result(path)
        records.append(result)
    per_seed, mean_sd, deltas = summarize_results(records, a.placements, a.seeds)
    write_csv(a.output / 'summary/placement_weather_per_seed.csv', per_seed)
    write_csv(a.output / 'summary/placement_weather_mean_std.csv', mean_sd)
    if deltas:
        write_csv(a.output / 'summary/c4_minus_c3_percentage_points.csv', deltas)
    matches = [{k: result[k] for k in ('placement', 'seed', 'fog_reference')} for result in records]
    write_json(a.output / 'summary/fog_reproduction_checks.json', matches)
    write_json(a.output / 'completion.json', {'complete': True, 'placements': a.placements,
               'seeds': a.seeds, 'conditions': list(CONDITIONS), 'source_scenes': records[0]['source_scenes'],
               'variant_images': sum(next(row for row in records[0]['results'] if row['condition'] == c)['images']
                                     for c in CONDITIONS),
               'fog_reference_status': {r['placement'] + '_seed' + str(r['seed']): r['fog_reference']['status']
                                        for r in records},
               'evaluation_status': 'fixed selected-checkpoint validation analysis', 'training_performed': False})
    zip_path = a.output.with_suffix('.zip')
    temporary = zip_path.with_name(zip_path.name + '.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(a.output.rglob('*')):
            if file.is_file():
                archive.write(file, Path(a.output.name) / file.relative_to(a.output))
    os.replace(temporary, zip_path)
    print('RESULTS ZIP:', zip_path, flush=True)
    return records


def run(a):
    import weather_metrics
    doc, audit, fingerprint, sources, image_root, jobs = preflight(a)
    dataset_sha = signature(fingerprint)
    versions = runtime_versions()
    for job in jobs:
        destination = a.output / 'results' / job['placement'] / f'seed_{job["seed"]}'
        protocol = {'version': VERSION, 'checkpoint': job, 'dataset': fingerprint,
                    'selection': a.selection, 'source_sha256': sources,
                    'runner_sha256': sha256(__file__), 'batch_size': a.batch_size,
                    'runtime_versions': versions,
                    'device': a.device, 'fog_tolerance': a.fog_tolerance,
                    'COCO_evaluator': {'iou_thresholds': [.5 + i * .05 for i in range(10)],
                                       'max_detections': [1, 10, 100], 'area': 'native COCO bbox area rules'}}
        protocol_sha = signature(protocol)
        result_file = destination / 'metrics.json'
        if result_file.exists():
            old = verify_result(result_file)
            if old['protocol_sha256'] != protocol_sha or signature(old['protocol']) != protocol_sha:
                raise ValueError('Existing result uses different inputs/settings. Select a new --output directory.')
            if sha256(destination / 'predictions.json') != old['predictions_sha256']:
                raise ValueError('Saved predictions changed: ' + str(destination))
            print('REUSING VERIFIED RESULT:', job['placement'], job['seed'], flush=True)
            continue
        print('EVALUATING:', job['placement'], 'seed', job['seed'], flush=True)
        model, device, settings, selected, schema = load_model(job, a)
        started = time.perf_counter()
        predictions = infer(model, device, doc, image_root, a.batch_size)
        results = weather_metrics.all_condition_metrics(doc, predictions, audit)
        reference = fog_reference(results, selected, a.fog_tolerance)
        if sha256(job['checkpoint']) != job['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during evaluation.')
        write_json(destination / 'predictions.json', predictions)
        result = {'placement': job['placement'], 'seed': job['seed'], 'selection': a.selection,
                  'source_scenes': fingerprint['source_scenes'], 'dataset_sha256': dataset_sha,
                  'protocol': protocol, 'protocol_sha256': protocol_sha, 'evaluation_protocol': settings,
                  'results': results, 'fog_reference': reference, 'saved_model_schema': schema,
                  'predictions_sha256': sha256(destination / 'predictions.json'),
                  'evaluation_seconds': time.perf_counter() - started,
                  'evaluation_status': 'fixed selected-checkpoint validation analysis',
                  'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        result['result_sha256'] = signature(result)
        write_json(result_file, result)
        print('FOG REFERENCE:', reference['status'], flush=True)
        del model, predictions
        gc.collect()
        import torch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    export_summary(a)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'check', 'run', 'summarize'))
    p.add_argument('--data-root', type=Path, default=BASE / 'DATASET')
    p.add_argument('--runs-base', type=Path, default=BASE / 'ugs_comparison')
    p.add_argument('--output', type=Path, default=BASE / 'ugs_comparison/c4_weather_evaluation_v1')
    p.add_argument('--source-annotations', type=Path)
    p.add_argument('--images', type=Path)
    p.add_argument('--train-annotations', type=Path)
    p.add_argument('--placements', nargs='+', choices=('C3', 'C4'), default=['C4'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 1337, 2025])
    p.add_argument('--selection', choices=('best_ap_small', 'best_map'), default='best_ap_small')
    p.add_argument('--expect-sources', type=int, default=500)
    p.add_argument('--confirm-aligned-variants', action='store_true')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--fog-tolerance', type=float, default=0.0001)
    a = p.parse_args(argv)
    if a.batch_size < 1 or a.expect_sources < 1 or a.fog_tolerance < 0:
        p.error('Batch size and source count must be positive; fog tolerance must be nonnegative.')
    a.output = a.output.expanduser().resolve()
    a.data_root = a.data_root.expanduser().resolve()
    a.runs_base = a.runs_base.expanduser().resolve()
    source, images, training = paths(a)
    protected = [images, source.parent.parent, a.runs_base / 'runs', a.runs_base / 'runs_C4']
    if any(a.output == path or path in a.output.parents for path in protected):
        p.error('Choose an output folder outside the original dataset and checkpoint run folders.')
    return a


def main(argv=None):
    a = arguments(argv)
    if a.action == 'prepare':
        prepare(a)
    elif a.action == 'check':
        preflight(a)
    elif a.action == 'run':
        run(a)
    else:
        export_summary(a)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print('ERROR:', exc, file=sys.stderr, flush=True)
        sys.exit(2)
