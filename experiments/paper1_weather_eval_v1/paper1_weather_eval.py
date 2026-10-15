#!/usr/bin/env python3
"""Weather-stratified evaluation of frozen Paper 1 models. No source edits or training.

Commands: audit, evaluate, summarize. See README.md for verified package interfaces.
The default evaluation data are the EXISTING labelled validation JSONs, not tests.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import statistics
import subprocess
import sys
from types import SimpleNamespace
import zipfile

VERSION = '1.0.0'
BASE = Path('/tmp/asera/paper1_original')
PACKAGE = Path('/tmp/asera/NEW/paper1_recovery_1_0_1/paper1_all_codes_with_test_evaluation')
ARCHIVE = Path('/tmp/asera/acdc_direct/rgb_anon_trainvaltest.zip')
METHODS = ('baseline', 'se', 'cbam', 'lgf_original', 'ugs_reimpl', 'yolo11m', 'rtdetr-l')
RECENT = ('yolo11m', 'rtdetr-l')
CONDITIONS = ('clear', 'fog', 'night', 'rain', 'snow')
ALIASES = {'foggy': 'fog', 'rainy': 'rain', 'snowy': 'snow', 'nighttime': 'night', 'night-time': 'night'}
CLASSES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}
METRICS = ('mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')
FRAME = re.compile(r'(?P<sequence>[A-Za-z0-9]+)_frame_(?P<frame>\d{6})_rgb_anon(?:\.png|_png\.rf\.[0-9a-fA-F]{32}\.(?:jpg|jpeg|png))', re.I)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: ('NA' if v is None else v) for k, v in row.items()})


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def existing(path):
    path = Path(path).expanduser().absolute()
    if not path.is_file():
        raise ValueError(f'File not found: {path}')
    return path


def safe_image(root, name):
    rel = PurePosixPath(str(name).replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Expected a safe relative image filename: {name}')
    return Path(root).joinpath(*rel.parts)


def normal_condition(value):
    if not isinstance(value, str):
        return None
    value = ALIASES.get(value.strip().lower(), value.strip().lower())
    return value if value in CONDITIONS else None


def frame_key(name):
    match = FRAME.fullmatch(PurePosixPath(str(name).replace('\\', '/')).name)
    return f"{match['sequence'].lower()}_frame_{match['frame']}_rgb_anon.png" if match else None


def path_conditions(name, dataset):
    path = PurePosixPath(str(name).replace('\\', '/'))
    values = {p.lower() for p in path.parts[:-1] if p.lower() in CONDITIONS}
    if dataset == 'coco_weather':
        # This exact suffix convention is used by the original weather notebook.
        tag = normal_condition(path.stem.rsplit('_', 1)[-1])
        if tag:
            values.add(tag)
    return values


def original_acdc_record(path):
    parts = PurePosixPath(str(path).replace('\\', '/')).parts
    key = frame_key(path)
    if not key:
        return None
    for i in range(len(parts) - 1):
        if parts[i] in ('fog', 'night', 'rain', 'snow') and parts[i+1] in ('train', 'val', 'test'):
            return key, {'condition': parts[i], 'official_split': parts[i+1], 'source_path': str(path)}
    # Reference (train_ref/val_ref/test_ref) paths are deliberately not adverse images.
    return None


def build_acdc_index(source):
    """Read ZIP directory entries only, or a supplied original metadata JSON/directory."""
    index, descriptors = {}, []
    if source is None:
        return index, descriptors
    source = Path(source).expanduser()
    if not source.exists():
        raise ValueError(f'ACDC mapping source not found: {source}')

    def add(name):
        parsed = original_acdc_record(name)
        if parsed is None:
            return
        key, value = parsed
        if key in index and any(index[key][k] != value[k] for k in ('condition', 'official_split')):
            raise ValueError(f'Ambiguous original ACDC frame: {key}')
        index[key] = value

    if source.is_dir():
        for path in source.rglob('*_rgb_anon.png'):
            add(path.relative_to(source).as_posix())
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            infos = sorted(z.infolist(), key=lambda i: i.filename)
            for info in infos:
                if '/rgb_anon/' in '/' + info.filename:
                    add(info.filename)
            # Also support the much smaller official gt_detection_trainval.zip.
            if not index:
                for info in infos:
                    if info.filename.endswith('.json') and 'gt_detection' in info.filename:
                        doc = json.loads(z.read(info))
                        for im in doc.get('images', []):
                            add(im.get('file_name', ''))
            descriptors.append({'source': str(source.absolute()), 'type': 'ZIP directory or metadata',
                                'directory_signature': signature([(i.filename, i.CRC, i.file_size) for i in infos])})
    else:
        doc = read_json(source)
        for im in doc.get('images', []):
            add(im.get('file_name', ''))
        descriptors.append({'source': str(source.absolute()), 'sha256': sha256(source)})
    if not index:
        raise ValueError('No original ACDC adverse frame paths in mapping source. Use the full RGB train/val/test ZIP, the original rgb_anon directory, or official combined detection metadata.')
    descriptors.append({'indexed_frames': len(index), 'mapping_signature': signature(index)})
    return index, descriptors


def dataset_paths(root, dataset):
    root = Path(root)
    if dataset == 'acdc':
        folder = root / 'ACDC-1/ACDC-1-NEW'
        return folder / 'images/val', folder / 'annotations/mini_val.json', folder / 'annotations/mini_train.json'
    tag = 'non_weather' if dataset == 'coco_nw' else 'weather'
    folder = root / 'COCO' / ('non_weather-mini' if dataset == 'coco_nw' else 'weather-mini')
    return (folder / f'images/val2017_{tag}-500k6c',
            folder / f'annotations/mini_val2017_{tag}-500k6c.json',
            folder / f'annotations/mini_train2017_{tag}-2400k6c.json')


def validate_gt(doc):
    if not doc.get('images') or 'annotations' not in doc or not doc['annotations']:
        raise ValueError('A labelled COCO-format evaluation JSON is required. Image-info or hidden-label test manifests cannot provide AP.')
    for key in ('images', 'annotations', 'categories'):
        ids = [r['id'] for r in doc[key]]
        if any(type(v) is not int for v in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'{key} needs unique integer IDs.')
    cats = {c['id']: c['name'] for c in doc['categories']}
    if set(cats.values()) != CLASSES or len(cats) != 6:
        raise ValueError('Evaluation JSON must contain exactly the six Paper 1 classes.')
    images = {im['id']: im for im in doc['images']}
    if len({im['file_name'] for im in images.values()}) != len(images):
        raise ValueError('Duplicate evaluation filenames.')
    for im in images.values():
        if min(im.get('width', 0), im.get('height', 0)) <= 0:
            raise ValueError(f'Invalid image dimensions: {im}')
    for ann in doc['annotations']:
        if ann['image_id'] not in images or ann['category_id'] not in cats:
            raise ValueError('Annotation refers to an unknown image/category.')
        if len(ann['bbox']) != 4 or any(not math.isfinite(float(x)) for x in ann['bbox']):
            raise ValueError('Invalid annotation bbox.')
        if min(ann['bbox'][2:]) <= 0 or 'area' not in ann or not math.isfinite(float(ann['area'])) or ann['area'] < 0:
            raise ValueError('Expected positive bbox dimensions and a finite existing annotation area.')


def support_counts(doc, ids):
    ids = set(ids)
    anns = [a for a in doc['annotations'] if a['image_id'] in ids and not a.get('iscrowd', 0)]
    small = [a for a in anns if 0 <= a['area'] <= 32**2]
    return {'images': len(ids), 'objects_non_crowd': len(anns), 'small_objects_non_crowd': len(small),
            'classes_with_objects': len({a['category_id'] for a in anns}),
            'classes_with_small_objects': len({a['category_id'] for a in small})}


def audit_one(args, dataset, index=None, descriptors=None):
    image_root, annotations, training = dataset_paths(args.data_root, dataset)
    if args.annotations:
        image_root, annotations = Path(args.images), Path(args.annotations)
    doc = read_json(existing(annotations))
    validate_gt(doc)
    index, descriptors = index or {}, descriptors or []
    external = {}
    if args.condition_map:
        with open(existing(args.condition_map), newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                name, cond = r['file_name'], normal_condition(r['condition'])
                if cond is None or name in external:
                    raise ValueError('Condition-map CSV needs unique exact file_name rows and valid condition labels.')
                external[name] = cond
    rows, missing, conflicts = [], [], []
    for im in sorted(doc['images'], key=lambda x: x['id']):
        evidence = []
        for container_name, container in [('image', im), ('metadata', im.get('metadata', {}))]:
            if not isinstance(container, dict):
                continue
            for field in ('weather', 'condition', 'acdc_condition', 'weather_condition'):
                value = container.get(field)
                cond = normal_condition(value)
                if cond:
                    evidence.append((cond, f'{container_name}.{field}'))
                elif value not in (None, '', 'unknown', 'unavailable'):
                    conflicts.append({'image_id': im['id'], 'field': field, 'unrecognized_value': value})
        for field in ('file_name', 'original_file_name', 'source_file_name', 'original_path'):
            for cond in path_conditions(im.get(field, ''), dataset):
                evidence.append((cond, field))
        if im['file_name'] in external:
            evidence.append((external[im['file_name']], 'explicit condition-map CSV'))
        key = frame_key(im['file_name']) if dataset == 'acdc' else None
        original = index.get(key) if key else None
        if original:
            evidence.append((original['condition'], 'exact original ACDC frame join'))
        values = {v for v, _ in evidence}
        if len(values) > 1:
            conflicts.append({'image_id': im['id'], 'file_name': im['file_name'], 'evidence': evidence})
        cond = next(iter(values)) if len(values) == 1 else None
        if cond is None:
            missing.append(im['id'])
        path = safe_image(image_root, im['file_name'])
        rows.append({'image_id': im['id'], 'file_name': im['file_name'], 'condition': cond or 'unknown',
                     'evidence': ' | '.join(s for _, s in evidence), 'canonical_frame': key or '',
                     'official_split': original['official_split'] if original else '', 'image_exists': path.is_file()})
    groups = {c: [r['image_id'] for r in rows if r['condition'] == c] for c in CONDITIONS}
    counts = {c: support_counts(doc, ids) for c, ids in groups.items() if ids}
    # Read-only diagnostic, not a complete leakage audit of all historical data.
    overlaps = []
    if training.is_file():
        train_doc = read_json(training)
        if dataset == 'acdc':
            train_keys = {frame_key(im['file_name']) for im in train_doc['images']} - {None}
            overlaps = sorted({r['canonical_frame'] for r in rows} & train_keys)
        else:
            def coco_key(name):
                m = re.fullmatch(r'(\d+)(?:_(?:clear|fog|rain|snow))?\.[^.]+', Path(name).name)
                return str(int(m[1])) if m else None
            train_keys = {coco_key(im['file_name']) for im in train_doc['images']} - {None}
            overlaps = sorted(({coco_key(r['file_name']) for r in rows} - {None}) & train_keys)
    report = {'schema': 'paper1-weather-audit-' + VERSION, 'dataset': dataset,
              'evaluation_split': args.split, 'annotations': str(annotations.absolute()),
              'annotations_sha256': sha256(annotations), 'image_root': str(image_root.absolute()),
              'image_count': len(rows), 'condition_counts': counts, 'unknown_images': len(missing),
              'unknown_image_ids': missing, 'conflicts': conflicts,
              'missing_image_files': [r['file_name'] for r in rows if not r['image_exists']],
              'official_split_counts': dict(collections.Counter(r['official_split'] or 'unverified' for r in rows)),
              'training_annotation_checked': str(training) if training.is_file() else None,
              'source_frame_overlap_with_current_training': overlaps,
              'complete_historical_scene_independence_verified': False,
              'mapping_sources': descriptors,
              'condition_map_sha256': sha256(args.condition_map) if args.condition_map else None,
              'small_support_definition': 'Non-crowd ground-truth annotation area in [0, 1024], following COCOeval small-area range.',
              'rows': rows}
    out = Path(args.output_root) / 'audits' / dataset
    write_json(out / 'condition_audit.json', report)
    write_csv(out / 'image_conditions.csv', rows)
    write_json(out / 'condition_image_ids.json', groups)
    print(f'\n{dataset}: {len(rows)} images; evaluation split={args.split}', flush=True)
    for c, s in counts.items():
        print(f"  {c:6}: images={s['images']}, small objects={s['small_objects_non_crowd']}", flush=True)
    print(f'  unknown={len(missing)}; conflicts={len(conflicts)}; missing files={len(report["missing_image_files"])}', flush=True)
    print('  Official ACDC split labels:', report['official_split_counts'], flush=True) if dataset == 'acdc' else None
    if overlaps:
        print(f'  SOURCE OVERLAP: {len(overlaps)} frames also occur in current training JSON. This is not independent test evidence.', flush=True)
    print('  Audit:', out / 'condition_audit.json', flush=True)
    if missing or conflicts:
        raise ValueError(f'{dataset}: incomplete/contradictory weather labels. Read the audit. Supply --acdc-source or an exact --condition-map CSV. No unknown images were silently omitted.')
    if report['missing_image_files']:
        raise ValueError(f'{dataset}: image files are missing. Check the recorded image_root.')
    return report


def notebook_checkpoint(root, dataset, method, seed, level, selection):
    if selection != 'best_map':
        raise ValueError('The original notebook BEST files select mAP, not APsmall. Use best_map, or explicitly supply a verified checkpoint with --checkpoint.')
    mode = 'lgf_gated_spatial' if method == 'lgf_original' else method
    prefix = f'BEST_{dataset}_{level}_{mode}_s{seed}_epoch_'
    pattern = re.compile(re.escape(prefix) + r'(\d+)_map_(\d+(?:\.\d+)?)\.pth')
    found = []
    for path in Path(root).glob(prefix + '*.pth'):
        m = pattern.fullmatch(path.name)
        if m:
            found.append((int(m[1]), float(m[2]), str(path.absolute())))
    found.sort()
    if not found:
        raise ValueError(f'No original notebook checkpoint: {root}/{prefix}*.pth. Set --original-root or --checkpoint.')
    for prev, curr in zip(found, found[1:]):
        if curr[0] == prev[0] or curr[1] < prev[1]:
            raise ValueError('Ambiguous original notebook run. Use --checkpoint for one completed run.')
    return existing(found[-1][2])


def checkpoint_for(a, dataset, method, seed):
    if a.checkpoint:
        path = existing(a.checkpoint)
        if path.stem in ('best_map', 'best_ap_small') and path.stem != a.selection:
            raise ValueError('Checkpoint filename and --selection disagree.')
        if path.name.startswith('BEST_') and a.selection != 'best_map':
            raise ValueError('Original notebook BEST checkpoints were selected by mAP.')
        return path
    if method in RECENT:
        return existing(Path(a.recent_root) / dataset / method / f'seed_{seed}' / (a.selection + '.pt'))
    ugs = method == 'ugs_reimpl' or (method == 'lgf_original' and a.lgf_source == 'ugs') or (method == 'baseline' and a.baseline_source == 'ugs')
    if ugs:
        return existing(Path(a.ugs_root) / dataset / method / f'seed_{seed}' / (a.selection + '.pth'))
    return notebook_checkpoint(a.original_root, dataset, method, seed, a.level, a.selection)


def coco_metrics(doc, predictions, ids):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not ids:
        return {k: None for k in METRICS}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(doc)
        gt.dataset.setdefault('info', {})
        for a in gt.dataset['annotations']:
            a.setdefault('iscrowd', 0)
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = {'images': copy.deepcopy(doc['images']), 'categories': copy.deepcopy(doc['categories']), 'annotations': []}
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = list(ids)  # Includes evaluated images with ZERO detections.
        ev.params.catIds = sorted(c['id'] for c in doc['categories'])
        ev.params.maxDets = [1, 10, 100]
        ev.params.iouThrs = np.linspace(.5, .95, 10)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) if float(v) >= 0 else None for k, v in zip(METRICS, ev.stats)}


def all_condition_metrics(doc, predictions, audit):
    ids = {'ALL': [im['id'] for im in doc['images']]}
    for c in CONDITIONS:
        ids[c] = [r['image_id'] for r in audit['rows'] if r['condition'] == c]
    out = []
    for cond, group in ids.items():
        if not group:
            continue
        print(f'  COCO evaluation: {cond}, {len(group)} images', flush=True)
        out.append({'condition': cond, **support_counts(doc, group), **coco_metrics(doc, predictions, group)})
    return out


def load_adapter(package):
    path = existing(Path(package) / 'paper1_test_evaluation_v1/models.py')
    name = 'paper1_weather_unmodified_model_adapter'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_job(job):
    """Executed in a separate process so each detector uses its existing environment."""
    import fcntl
    import torch
    from PIL import Image
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'run.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = read_json(job['audit'])
        doc = read_json(audit['annotations'])
        if sha256(audit['annotations']) != audit['annotations_sha256']:
            raise ValueError('Evaluation annotation changed after audit.')
        cp_hash = sha256(job['checkpoint'])
        images = sorted(doc['images'], key=lambda x: x['id'])
        image_hashes = []
        for im in images:
            path = existing(safe_image(audit['image_root'], im['file_name']))
            with Image.open(path) as image:
                if image.size != (im['width'], im['height']):
                    raise ValueError('Image/annotation dimensions differ: ' + str(path))
            image_hashes.append([im['id'], sha256(path)])
        training_ann = dataset_paths(job['data_root'], job['train_dataset'])[2]
        cats = sorted(read_json(existing(training_ann))['categories'], key=lambda c: c['id'])
        names = [c['name'] for c in cats]
        if set(names) != CLASSES or len(names) != 6:
            raise ValueError('Training JSON does not define the six expected classes.')
        cat_by_name = {c['name']: c['id'] for c in doc['categories']}
        package = Path(job['package_root'])
        adapter_files = [package / 'paper1_test_evaluation_v1/models.py']
        adapter_files += ([package / 'paper1_reviewer34_v1/reviewer34_compare.py'] if job['method'] in RECENT else
                          [package / 'paper1_ugs_comparison_v1/core_bridge.py', package / 'paper1_ugs_comparison_v1/ugs.py',
                           package / 'paper1_ugs_comparison_v1/vendor/original_core.py'])
        import importlib.metadata
        import pycocotools.cocoeval as coco_impl
        software = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'pycocotools')}
        software['python'] = sys.version.split()[0]
        evaluation = {'annotations_sha256': audit['annotations_sha256'], 'images_sha256': signature(image_hashes),
                      'condition_assignment_sha256': signature([(r['image_id'], r['condition']) for r in audit['rows']]),
                      'split': audit['evaluation_split'], 'score_floor': .05, 'max_dets': [1, 10, 100],
                      'min_size': 800, 'max_size': 1333, 'precision': 'FP32', 'batch_size': job['batch_size'],
                      'evaluator': VERSION, 'cocoeval_sha256': sha256(coco_impl.__file__), 'script_sha256': sha256(__file__)}
        protocol = {**job, 'checkpoint_sha256': cp_hash, 'evaluation': evaluation,
                    'adapter_hashes': {str(p): sha256(existing(p)) for p in adapter_files},
                    'training_annotation_sha256': sha256(training_ann), 'training_class_order': names,
                    'mapping_provenance': audit['mapping_sources'], 'software_versions': software}
        proto_hash = signature(protocol)
        report_file = out / 'weather_metrics.json'
        if report_file.exists():
            old = read_json(report_file)
            if old['protocol_sha256'] != proto_hash or sha256(out / 'predictions_coco.json') != old['predictions_sha256']:
                raise ValueError('Existing result has different data/checkpoint/settings. Choose a new output root.')
            print('SKIP verified completed run:', out, flush=True)
            return
        write_json(out / 'run_protocol.json', protocol)
        adapter = load_adapter(package)
        a = SimpleNamespace(method=job['method'], train_dataset=job['train_dataset'], seed=job['seed'],
                            level=job['level'], device=job['device'], threads=job['threads'], batch_size=job['batch_size'],
                            checkpoint=Path(job['checkpoint']), lgf_package=package / 'paper1_ugs_comparison_v1',
                            recent_package=package / 'paper1_reviewer34_v1', runtime_root=out / 'runtime')
        model, device, metadata, parameters, infer = adapter.load_model(a, names)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(images), a.batch_size):
                batch = images[start:start + a.batch_size]
                outputs = infer(adapter.rgb_tensors(batch, audit['image_root'], device))
                if len(outputs) != len(batch):
                    raise ValueError('Prediction batch has incorrect length.')
                for im, pred in zip(batch, outputs):
                    boxes = pred['boxes'].detach().cpu().tolist()
                    scores = pred['scores'].detach().cpu().tolist()
                    labels = pred['labels'].detach().cpu().tolist()
                    if not (len(boxes) == len(scores) == len(labels)) or len(boxes) > 100:
                        raise ValueError('Invalid detector output lengths/detection cap.')
                    for box, score, label in zip(boxes, scores, labels):
                        if not all(math.isfinite(v) for v in [*box, score]) or not 0 <= score <= 1:
                            raise ValueError('Nonfinite/invalid prediction.')
                        if int(label) != label or int(label) not in range(6):
                            raise ValueError('Unexpected detector label. Existing adapters must emit zero-based labels.')
                        x1, y1, x2, y2 = box
                        if x2 < x1 or y2 < y1:
                            raise ValueError('Reversed prediction box.')
                        predictions.append({'image_id': im['id'], 'category_id': cat_by_name[names[int(label)]],
                                            'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
                if start == 0 or (start // a.batch_size) % 20 == 0 or start + len(batch) == len(images):
                    print(f"INFERENCE {a.method} seed={a.seed} {start+len(batch)}/{len(images)}", flush=True)
        if sha256(job['checkpoint']) != cp_hash:
            raise ValueError('Checkpoint changed during evaluation. Finish training before evaluating.')
        write_json(out / 'predictions_coco.json', predictions)
        write_json(out / 'evaluated_image_ids.json', [im['id'] for im in images])
        rows = all_condition_metrics(doc, predictions, audit)
        report = {'schema': 'paper1-weather-metrics-' + VERSION, 'method': a.method, 'seed': a.seed,
                  'train_dataset': a.train_dataset, 'dataset': job['dataset'], 'selection': job['selection'],
                  'evaluation_split': audit['evaluation_split'], 'evaluation_signature': signature(evaluation),
                  'protocol_sha256': proto_hash, 'protocol': protocol, 'model_metadata': metadata,
                  'parameters': parameters, 'rows': rows, 'AP_scale': '0-1',
                  'predictions_sha256': sha256(out / 'predictions_coco.json'),
                  'independent_test_evidence': False, 'condition_breakdown_only': True,
                  'source_frame_overlap_with_current_training': audit['source_frame_overlap_with_current_training']}
        write_json(report_file, report)
        write_csv(out / 'weather_metrics.csv', rows)
        print('COMPLETE:', report_file, flush=True)


def summarize(root):
    files = sorted((Path(root) / 'runs').rglob('weather_metrics.json'))
    if not files:
        raise ValueError('No completed weather results under ' + str(root))
    per_seed, groups, identities, eval_sigs = [], collections.defaultdict(list), set(), {}
    for path in files:
        r = read_json(path)
        context = (r['dataset'], r['train_dataset'], r['selection'], r['evaluation_split'])
        if context in eval_sigs and eval_sigs[context] != r['evaluation_signature']:
            raise ValueError('Refusing to combine different images/condition groups/inference settings for ' + str(context))
        eval_sigs[context] = r['evaluation_signature']
        identity = (*context, r['method'], r['seed'])
        if identity in identities:
            raise ValueError('Duplicate method/dataset/seed reports.')
        identities.add(identity)
        for row in r['rows']:
            record = {'dataset': r['dataset'], 'train_dataset': r['train_dataset'], 'selection': r['selection'],
                      'split': r['evaluation_split'], 'method': r['method'], 'seed': r['seed'], **row}
            per_seed.append(record)
            groups[(*context, r['method'], row['condition'])].append(record)
    means = []
    for key, items in sorted(groups.items()):
        dataset, train, selection, split, method, condition = key
        item = {'dataset': dataset, 'train_dataset': train, 'selection': selection, 'split': split,
                'method': method, 'condition': condition, 'n_seeds': len(items),
                'seeds': ','.join(str(x) for x in sorted(i['seed'] for i in items)),
                **{k: items[0][k] for k in ('images', 'objects_non_crowd', 'small_objects_non_crowd', 'classes_with_objects', 'classes_with_small_objects')}}
        for metric in METRICS:
            values = [r[metric] for r in items]
            valid = all(v is not None for v in values)
            item[metric + '_mean'] = statistics.mean(values) if valid else None
            item[metric + '_std'] = statistics.stdev(values) if valid and len(values) > 1 else None
        means.append(item)
    deltas = []
    for key, items in groups.items():
        dataset, train, selection, split, method, condition = key
        if method != 'lgf_original':
            continue
        baseline = groups.get((dataset, train, selection, split, 'baseline', condition))
        if not baseline:
            continue
        b = {r['seed']: r for r in baseline}
        l = {r['seed']: r for r in items}
        if set(b) != set(l):
            print('No paired delta for mismatched baseline/LGF seed sets:', key, flush=True)
            continue
        row = {'dataset': dataset, 'train_dataset': train, 'selection': selection, 'split': split,
               'condition': condition, 'n_seed_pairs': len(b), 'seeds': ','.join(map(str, sorted(b)))}
        for metric in METRICS:
            values = [(l[s][metric]-b[s][metric])*100 for s in sorted(b)] if all(l[s][metric] is not None and b[s][metric] is not None for s in b) else None
            row[metric + '_delta_points_mean'] = statistics.mean(values) if values is not None else None
            row[metric + '_delta_points_std'] = statistics.stdev(values) if values is not None and len(values) > 1 else None
        deltas.append(row)
    output = Path(root) / 'summary'
    write_csv(output / 'weather_per_seed.csv', per_seed)
    write_csv(output / 'weather_mean_std.csv', means)
    if deltas:
        write_csv(output / 'lgf_minus_baseline.csv', deltas)
    elif (output / 'lgf_minus_baseline.csv').exists():
        (output / 'lgf_minus_baseline.csv').unlink()
    print('SUMMARY:', output, flush=True)
    return means


def evaluate_grid(a, audits):
    if a.checkpoint and len(a.datasets) * len(a.methods) * len(a.seeds) != 1:
        raise ValueError('--checkpoint requires one dataset, one method, and one seed.')
    package = Path(a.package_root).expanduser().absolute()
    existing(package / 'paper1_test_evaluation_v1/models.py')
    jobs, errors = [], []
    for dataset in a.datasets:
        train = a.train_dataset or dataset
        for method in a.methods:
            for seed in a.seeds:
                try:
                    cp = checkpoint_for(a, train, method, seed)
                    python = existing(a.recent_python if method in RECENT else a.legacy_python)
                    if not os.access(python, os.X_OK):
                        raise ValueError(f'Python is not executable: {python}')
                    output = Path(a.output_root).absolute() / 'runs' / f'{train}_to_{dataset}' / a.selection / method / f'seed_{seed}'
                    job = {'dataset': dataset, 'train_dataset': train, 'method': method, 'seed': seed,
                           'level': a.level, 'selection': a.selection, 'checkpoint': str(cp),
                           'package_root': str(package), 'data_root': str(Path(a.data_root).absolute()),
                           'audit': str(Path(a.output_root).absolute() / 'audits' / dataset / 'condition_audit.json'),
                           'device': a.device, 'batch_size': a.batch_size, 'threads': a.threads, 'output': str(output)}
                    job_file = Path(a.output_root) / 'jobs' / f'{train}_to_{dataset}_{a.selection}_{method}_{seed}.json'
                    write_json(job_file, job)
                    cmd = [str(python), '-u', str(Path(__file__).absolute()), '_run', '--job', str(job_file.absolute())]
                    jobs.append((job, cmd))
                    print(f'JOB {dataset}/{method}/seed_{seed}: {cp}', flush=True)
                except (ValueError, OSError) as exc:
                    errors.append(f'{dataset}/{method}/seed_{seed}: {exc}')
    if errors:
        raise ValueError('Preflight failed; no inference started:\n' + '\n'.join(errors))
    if a.check:
        print('\nPATH/LABEL CHECK PASSED. No checkpoint was loaded and no inference was run.', flush=True)
        return
    for job, cmd in jobs:
        log = Path(job['output']) / 'evaluation.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        print('\nRUN:', shlex.join(cmd), flush=True)
        with open(log, 'a', encoding='utf-8') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            if proc.wait():
                raise RuntimeError(f'Evaluation stopped. Read {log}')
    summarize(a.output_root)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for command in ('audit', 'evaluate'):
        s = sub.add_parser(command)
        s.add_argument('--datasets', nargs='+', choices=('acdc', 'coco_weather'), default=['coco_weather', 'acdc'])
        s.add_argument('--data-root', type=Path, default=BASE / 'DATASET')
        s.add_argument('--output-root', type=Path, default=BASE / 'weather_evaluation_v1')
        s.add_argument('--acdc-source', type=Path, help='Full original RGB ZIP/directory or official detection metadata. Default: known downloaded RGB ZIP if present.')
        s.add_argument('--condition-map', type=Path, help='Optional CSV with exact file_name,condition columns; evidence must be independently verified.')
        s.add_argument('--annotations', type=Path, help='Explicit labelled evaluation JSON, one dataset only.')
        s.add_argument('--images', type=Path, help='Image root for explicit --annotations.')
        s.add_argument('--split', choices=('validation', 'heldout_labelled'), default='validation', help='Descriptive label, not an independence certification.')
        if command == 'evaluate':
            s.add_argument('--methods', nargs='+', choices=METHODS, default=['baseline', 'lgf_original'])
            s.add_argument('--seeds', nargs='+', type=int, default=[42, 1337, 2025])
            s.add_argument('--package-root', type=Path, default=PACKAGE)
            s.add_argument('--original-root', type=Path, default=BASE / 'NEW')
            s.add_argument('--ugs-root', type=Path, default=BASE / 'ugs_comparison/runs')
            s.add_argument('--recent-root', type=Path, default=BASE / 'reviewer34/runs')
            s.add_argument('--baseline-source', choices=('notebook', 'ugs'), default='notebook')
            s.add_argument('--lgf-source', choices=('notebook', 'ugs'), default='ugs')
            s.add_argument('--checkpoint', type=Path)
            s.add_argument('--selection', choices=('best_map', 'best_ap_small'), default='best_map')
            s.add_argument('--train-dataset', choices=('coco_nw', 'coco_weather', 'acdc'), help='Explicit source training dataset for transfer, otherwise matches evaluation dataset.')
            s.add_argument('--level', choices=('C3', 'C4', 'C5'), default='C3')
            s.add_argument('--device', default='0')
            s.add_argument('--batch-size', type=int, default=4)
            s.add_argument('--threads', type=int, default=4)
            s.add_argument('--legacy-python', default=os.environ.get('P1PY', '/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python'))
            s.add_argument('--recent-python', default=os.environ.get('P34PY', '/tmp/asera/paper1_reviewer34_env/bin/python'))
            s.add_argument('--check', action='store_true')
    s = sub.add_parser('summarize')
    s.add_argument('--output-root', type=Path, default=BASE / 'weather_evaluation_v1')
    s = sub.add_parser('_run', help='Internal single-job worker; use evaluate for normal runs')
    s.add_argument('--job', type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    a = arguments(argv)
    if a.action == '_run':
        return run_job(read_json(a.job))
    if a.action == 'summarize':
        return summarize(a.output_root)
    if bool(a.annotations) != bool(a.images):
        raise ValueError('Supply --annotations and --images together.')
    if (a.annotations or a.condition_map) and len(a.datasets) != 1:
        raise ValueError('--annotations/--condition-map requires exactly one dataset.')
    for name in ('datasets', 'methods', 'seeds'):
        values = getattr(a, name, [])
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate --{name} entries.')
    if a.action == 'evaluate' and (a.batch_size < 1 or a.threads < 1 or (a.device != 'cpu' and not a.device.isdigit())):
        raise ValueError('Positive batch size/threads and --device 0 (or cpu) are required.')
    index, sources = {}, []
    if 'acdc' in a.datasets:
        source = a.acdc_source or (ARCHIVE if ARCHIVE.is_file() else None)
        if source:
            print('Indexing original ACDC paths without extracting images:', source, flush=True)
            index, sources = build_acdc_index(source)
    audits = {}
    errors = []
    for dataset in a.datasets:
        try:
            audits[dataset] = audit_one(a, dataset, index if dataset == 'acdc' else {}, sources if dataset == 'acdc' else [])
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError('\n'.join(errors))
    if a.action == 'evaluate':
        evaluate_grid(a, audits)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(2)
