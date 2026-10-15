#!/usr/bin/env python3
"""Build NEW evaluation manifests for existing, aligned COCO fog/rain/snow files.

No training, augmentation, image copying, or source-file edits. Reads the original
weather evaluator audit to find the selected validation JSON and image directory.
Requires numeric COCO filenames ending in _fog, _rain, or _snow. The source JSON
must contain exactly one variant per original image. Intended for the original
geometry-preserving Paper 1 generator, not independently cropped/warped images.
Python >=3.8 and Pillow are required. Model libraries are not imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional

VERSION = '1.0.0'
CONDITIONS = ('fog', 'rain', 'snow')
CLASSES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}
VARIANT = re.compile(r'^(\d+)_(fog|rain|snow)\.(jpg|jpeg|png)$', re.I)
SOURCE = re.compile(r'^(\d+)(?:_(?:clear|fog|rain|snow))?\.(?:jpg|jpeg|png)$', re.I)
COMBINED_NAME = 'mini_val2017_weather_fog_rain_snow.json'
CONDITION_KEYS = {'weather', 'condition', 'acdc_condition', 'weather_condition'}
OLD_PATH_KEYS = {'original_file_name', 'source_file_name', 'original_path'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> Dict[str, Any]:
    with path.open(encoding='utf-8') as stream:
        doc = json.load(stream)
    if not isinstance(doc, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return doc


def encoded(doc: Any) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def checked_path(root: Path, name: str) -> Path:
    rel = PurePosixPath(str(name).replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts or not rel.name:
        raise ValueError('Unsafe relative image filename: ' + str(name))
    return root.joinpath(*rel.parts)


def source_key(name: str) -> Optional[str]:
    match = SOURCE.fullmatch(PurePosixPath(str(name).replace('\\', '/')).name)
    return 'coco:' + str(int(match[1])) if match else None


def validate_source(doc: Dict[str, Any]) -> None:
    for field in ('images', 'annotations', 'categories'):
        if not isinstance(doc.get(field), list) or not doc[field]:
            raise ValueError('Expected a nonempty source ' + field + ' list.')
        ids = [row.get('id') for row in doc[field]]
        if any(type(x) is not int or x < 0 for x in ids) or len(ids) != len(set(ids)):
            raise ValueError('Source ' + field + ' IDs must be unique nonnegative integers.')
    if len(doc['categories']) != 6 or {c.get('name') for c in doc['categories']} != CLASSES:
        raise ValueError('Expected the six original Paper 1 category names, unchanged.')
    image_ids = {im['id'] for im in doc['images']}
    category_ids = {c['id'] for c in doc['categories']}
    seen = set()
    for im in doc['images']:
        name = im.get('file_name')
        if not isinstance(name, str) or name in seen:
            raise ValueError('Missing or duplicate source filename: ' + str(name))
        seen.add(name)
        if any(type(im.get(k)) is not int or im[k] < 1 for k in ('width', 'height')):
            raise ValueError('Invalid source image dimensions: ' + name)
    for ann in doc['annotations']:
        if ann.get('image_id') not in image_ids or ann.get('category_id') not in category_ids:
            raise ValueError('Source annotation references an unknown image/category.')
        box = ann.get('bbox')
        if (not isinstance(box, list) or len(box) != 4 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
                min(box[2:]) <= 0):
            raise ValueError('Invalid source bounding box: ' + str(ann.get('id')))
        area = ann.get('area')
        if not isinstance(area, (int, float)) or not math.isfinite(area) or area < 0:
            raise ValueError('Invalid/missing source area. Areas are never recalculated.')
    encoded(doc)  # Reject NaN/Infinity anywhere before creating outputs.


def training_diagnostic(path: Path, eval_sources: set) -> Dict[str, Any]:
    doc = load(path)
    if not isinstance(doc.get('images'), list):
        raise ValueError('Training JSON has no images list: ' + str(path))
    keys, suffixes = set(), Counter()
    unparsed = []
    fields = Counter()
    for im in doc['images']:
        name = str(im.get('file_name', ''))
        key = source_key(name)
        if key:
            keys.add(key)
        else:
            unparsed.append(name)
        match = VARIANT.fullmatch(PurePosixPath(name.replace('\\', '/')).name)
        suffixes[match[2].lower() if match else 'no_fog_rain_snow_suffix'] += 1
        fields[str(im.get('weather', '<absent>'))] += 1
    return {
        'annotations': str(path), 'sha256': digest(path.read_bytes()),
        'image_count': len(doc['images']), 'filename_condition_counts': dict(suffixes),
        'raw_weather_field_counts': dict(fields),
        'unparsed_source_count': len(unparsed), 'unparsed_examples': unparsed[:10],
        'overlapping_source_groups': sorted(eval_sources & keys),
        'scope': 'Current supplied training JSON only. Historical training inputs and online augmentations are not verified.'}


def support(doc: Dict[str, Any], condition: str) -> Dict[str, int]:
    ids = {im['id'] for im in doc['images'] if im['weather'] == condition}
    anns = [a for a in doc['annotations'] if a['image_id'] in ids and not a.get('iscrowd', 0)]
    return {'images': len(ids), 'objects_non_crowd': len(anns),
            'small_objects_non_crowd': sum(0 <= a['area'] <= 1024 for a in anns)}


def publish(directory: Path, payloads: Dict[str, bytes]) -> str:
    """Publish a complete new bundle; never replace a different existing file."""
    if directory.exists():
        expected = set(payloads)
        actual = {p.name for p in directory.iterdir()}
        if actual != expected or any(
                not (directory / name).is_file() or (directory / name).read_bytes() != data
                for name, data in payloads.items()):
            raise ValueError('Output directory already exists with different/incomplete contents. '
                             'Choose a NEW --output-dir; nothing was overwritten: ' + str(directory))
        return 'VERIFIED EXISTING MANIFESTS'
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + directory.name + '.', dir=str(directory.parent)))
    try:
        for name, data in payloads.items():
            with (stage / name).open('xb') as stream:
                stream.write(data)
        if directory.exists():
            raise ValueError('Output directory appeared during preparation. Rerun safely.')
        os.rename(str(stage), str(directory))
    finally:
        if stage.exists():
            shutil.rmtree(str(stage))
    return 'CREATED NEW MANIFESTS'


def prepare(a: argparse.Namespace) -> Dict[str, Any]:
    if not a.confirm_aligned_variants:
        raise ValueError('Use --confirm-aligned-variants only for the original geometry-preserving '
                         'fog/rain/snow outputs. Matching dimensions alone do not prove box alignment.')
    from PIL import Image  # Present in the existing evaluation environment.
    audit_path = a.audit.expanduser().resolve()
    audit_bytes = audit_path.read_bytes()
    audit = load(audit_path)
    if audit.get('dataset') != 'coco_weather':
        raise ValueError('Supply the coco_weather condition_audit.json, not an ACDC audit.')
    if audit.get('evaluation_split') != 'validation':
        raise ValueError('This helper is for the current validation scenes, not a new test split.')
    source_ann = Path(audit['annotations']).expanduser().resolve()
    image_root = Path(audit['image_root']).expanduser().resolve()
    original_bytes = source_ann.read_bytes()
    if audit.get('annotations_sha256') and digest(original_bytes) != audit['annotations_sha256']:
        raise ValueError('Source annotation changed after the audit. Do not silently change evaluated data.')
    if not image_root.is_dir():
        raise ValueError('Image directory missing: ' + str(image_root))
    out_dir = a.output_dir.expanduser().resolve()
    if source_ann.parent.name == 'annotations':
        dataset_dir = source_ann.parent.parent
        if out_dir == dataset_dir or dataset_dir in out_dir.parents:
            raise ValueError('Choose an output directory OUTSIDE the original dataset.')
    if out_dir == image_root or image_root in out_dir.parents:
        raise ValueError('Do not place outputs inside the image directory.')
    doc = json.loads(original_bytes)
    validate_source(doc)
    if a.expect_sources and len(doc['images']) != a.expect_sources:
        raise ValueError('Expected %s source images, found %s.' % (a.expect_sources, len(doc['images'])))
    records, source_groups = [], set()
    for im in sorted(doc['images'], key=lambda x: x['id']):
        path = PurePosixPath(im['file_name'].replace('\\', '/'))
        checked_path(image_root, path.as_posix())
        match = VARIANT.fullmatch(path.name)
        if not match:
            raise ValueError('Expected a numeric COCO weather filename: ' + im['file_name'])
        group = source_key(path.name)
        if group in source_groups:
            raise ValueError('More than one source record for ' + str(group) + '. Do not expand an already expanded JSON.')
        source_groups.add(group)
        for container in (im, im.get('metadata', {})):
            if not isinstance(container, dict):
                continue
            for key in CONDITION_KEYS:
                if key in container and container[key] not in (None, '', match[2].lower()):
                    raise ValueError('Source metadata conflicts with filename: ' + im['file_name'])
        variants = {}
        for condition in CONDITIONS:
            relative = path.with_name('%s_%s.%s' % (match[1], condition, match[3])).as_posix()
            file_path = checked_path(image_root, relative)
            if not file_path.is_file():
                raise ValueError('Missing matching variant; no images were dropped: ' + str(file_path))
            variants[condition] = (relative, file_path)
        records.append((im, group, variants))

    train_paths = [Path(p).expanduser().resolve() for p in (a.train_annotations or [])]
    current_training = audit.get('training_annotation_checked')
    if current_training:
        train_paths.insert(0, Path(current_training).expanduser().resolve())
    train_paths = list(dict.fromkeys(train_paths))
    training = [training_diagnostic(path, source_groups) for path in train_paths]
    for result in training:
        print('TRAINING JSON:', result['annotations'], flush=True)
        print('  Filename tags:', result['filename_condition_counts'], flush=True)
        if result['overlapping_source_groups']:
            raise ValueError('Source overlap with current training JSON: ' +
                             ', '.join(result['overlapping_source_groups'][:10]) +
                             '. No evaluation manifests were written. Review the split before proceeding.')
        print('  Source-ID overlap: 0; unparsed training filenames: %s' % result['unparsed_source_count'], flush=True)

    by_image = defaultdict(list)
    for ann in doc['annotations']:
        by_image[ann['image_id']].append(ann)
    original_condition = {im['id']: VARIANT.fullmatch(PurePosixPath(im['file_name'].replace('\\', '/')).name)[2].lower()
                          for im in doc['images']}
    image_ids, next_image = {}, max(im['id'] for im in doc['images']) + 1
    for condition in CONDITIONS:
        for im, group, variants in records:
            if condition == original_condition[im['id']]:
                image_ids[(im['id'], condition)] = im['id']
            else:
                image_ids[(im['id'], condition)] = next_image
                next_image += 1
    all_images, all_anns, mapping = [], [], []
    next_ann = max(ann['id'] for ann in doc['annotations']) + 1
    variant_hashes = defaultdict(dict)
    for condition in CONDITIONS:
        for position, (im, group, variants) in enumerate(records, 1):
            relative, file_path = variants[condition]
            image_bytes = file_path.read_bytes()
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                if image.size != (im['width'], im['height']):
                    raise ValueError('Variant/source dimensions differ: ' + relative)
            image_sha = digest(image_bytes)
            variant_hashes[group][condition] = image_sha
            # Store original provenance separately. Old *_fog paths must not become
            # contradictory weather evidence on the new rain/snow image records.
            new_im = {k: copy.deepcopy(v) for k, v in im.items()
                      if k not in CONDITION_KEYS | OLD_PATH_KEYS | {'metadata'}}
            metadata = copy.deepcopy(im.get('metadata'))
            if isinstance(metadata, dict):
                metadata = {k: v for k, v in metadata.items() if k not in CONDITION_KEYS | OLD_PATH_KEYS}
                if metadata:
                    new_im['metadata'] = metadata
            new_im.update(id=image_ids[(im['id'], condition)], file_name=relative, weather=condition,
                          source_image_id=im['id'], source_group_id=group)
            all_images.append(new_im)
            for ann in by_image[im['id']]:
                new_ann = copy.deepcopy(ann)
                if condition != original_condition[im['id']]:
                    new_ann['id'] = next_ann
                    next_ann += 1
                new_ann['image_id'] = new_im['id']
                new_ann['source_annotation_id'] = ann['id']
                all_anns.append(new_ann)
            mapping.append({'image_id': new_im['id'], 'file_name': relative, 'condition': condition,
                            'source_group_id': group, 'source_image_id': im['id'],
                            'reference_file_name': im['file_name'], 'sha256': image_sha})
            if position % 100 == 0 or position == len(records):
                print('CHECKED %s: %s/%s files' % (condition, position, len(records)), flush=True)
    duplicate_variants = [group for group, hashes in variant_hashes.items() if len(set(hashes.values())) < 3]
    if duplicate_variants:
        raise ValueError('Two weather variants have identical file bytes for %s source(s). '
                         'Check these before evaluation: %s' % (len(duplicate_variants), duplicate_variants[:10]))
    combined = copy.deepcopy(doc)
    combined.update(images=all_images, annotations=all_anns)
    combined['weather_evaluation'] = {
        'schema': 'paper1-aligned-weather-evaluation-' + VERSION,
        'source_annotation_sha256': digest(original_bytes),
        'source_images': len(records), 'conditions': list(CONDITIONS),
        'evaluation_split': 'validation', 'independent_test_evidence': False,
        'box_alignment': 'Original geometry-preserving generator asserted; exact source filename joins and decoded dimensions checked. Pixel alignment not independently established.',
        'source_group_field': 'source_group_id',
        'small_support_definition': 'Existing annotation area in [0,1024], excluding crowd; unchanged from supplied weather evaluator.',
        'note': 'Three variants per existing validation source. Not 1500 independent scenes. Training datasets are unchanged.'}
    validate_source(combined)
    counts = {condition: support(combined, condition) for condition in CONDITIONS}
    payloads = {COMBINED_NAME: encoded(combined)}
    for condition in CONDITIONS:
        subset = copy.deepcopy(combined)
        subset['images'] = [im for im in all_images if im['weather'] == condition]
        ids = {im['id'] for im in subset['images']}
        subset['annotations'] = [ann for ann in all_anns if ann['image_id'] in ids]
        subset['weather_evaluation']['conditions'] = [condition]
        payloads['mini_val2017_weather_%s.json' % condition] = encoded(subset)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(mapping[0]))
    writer.writeheader()
    writer.writerows(mapping)
    payloads['source_groups.csv'] = buffer.getvalue().encode('utf-8')
    provenance = {
        'builder_version': VERSION, 'builder_sha256': digest(Path(__file__).read_bytes()),
        'source_audit': str(audit_path), 'source_audit_sha256': digest(audit_bytes),
        'source_annotations': str(source_ann), 'source_annotations_sha256': digest(original_bytes),
        'image_root': str(image_root), 'source_images': len(records), 'variant_images': len(all_images),
        'source_annotation_records': len(doc['annotations']), 'variant_annotation_records': len(all_anns),
        'condition_counts': counts, 'training_diagnostics': training,
        'independent_test_evidence': False, 'training_file_access': 'read-only',
        'geometry_assumption': combined['weather_evaluation']['box_alignment'],
        'source_image_records': doc['images'], 'image_records': mapping,
        'output_sha256': {name: digest(data) for name, data in payloads.items()}}
    payloads['preparation_report.json'] = encoded(provenance)
    if source_ann.read_bytes() != original_bytes or audit_path.read_bytes() != audit_bytes:
        raise ValueError('Source annotation/audit changed during preparation. No manifests published.')
    for result in training:
        if digest(Path(result['annotations']).read_bytes()) != result['sha256']:
            raise ValueError('Training JSON changed during preparation. No manifests published.')
    status = publish(out_dir, payloads)
    print('\n' + status, out_dir, flush=True)
    print('Existing source images: %s; evaluated variants: %s' % (len(records), len(all_images)), flush=True)
    for condition, count in counts.items():
        print('%s: %s images; %s small annotated objects' %
              (condition, count['images'], count['small_objects_non_crowd']), flush=True)
    print('Combined JSON:', out_dir / COMBINED_NAME, flush=True)
    print('Image root (unchanged):', image_root, flush=True)
    print('Training annotations, image files, checkpoints, and original evaluator were not edited.', flush=True)
    return provenance


def arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True, help='Existing coco_weather/condition_audit.json.')
    p.add_argument('--output-dir', type=Path, required=True, help='NEW directory outside the original dataset.')
    p.add_argument('--confirm-aligned-variants', action='store_true',
                   help='Confirm original photometric-only fog/rain/snow generation, no crop/resize/warp.')
    p.add_argument('--expect-sources', type=int, default=500, help='Expected selected source count, default 500; 0 disables.')
    p.add_argument('--train-annotations', type=Path, action='append',
                   help='Additional current training JSON to check for source overlap. Repeat for multiple inputs.')
    a = p.parse_args(argv)
    if a.expect_sources < 0:
        p.error('--expect-sources must be nonnegative.')
    return a


if __name__ == '__main__':
    try:
        prepare(arguments())
    except (ValueError, OSError, KeyError, ImportError, TypeError) as exc:
        print('ERROR:', exc, file=sys.stderr, flush=True)
        sys.exit(2)
