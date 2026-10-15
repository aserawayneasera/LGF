#!/usr/bin/env python3
"""Read-only annotation checks before exporting, scoring, or bootstrapping."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

CLASSES = {'person', 'car', 'truck', 'bicycle', 'bus', 'motorcycle'}

def audit(path, positive_ids=False):
    data = json.loads(Path(path).read_text())
    errors = []
    images = data.get('images', [])
    anns = data.get('annotations', [])
    cats = data.get('categories', [])
    ids = [x['id'] for x in images]
    ann_ids = [x['id'] for x in anns]
    category_ids = [x['id'] for x in cats]
    for label, values in [('image', ids), ('annotation', ann_ids), ('category', category_ids)]:
        if len(values) != len(set(values)):
            errors.append(f'Duplicate {label} IDs')
    if {x['name'] for x in cats} != CLASSES or len(cats) != 6:
        errors.append('Expected exactly the six manuscript class names')
    if positive_ids and any(not isinstance(x, int) or x <= 0 for x in ann_ids):
        errors.append('Bootstrap requires strictly positive integer ground-truth annotation IDs')
    image_set, cat_set = set(ids), set(category_ids)
    for a in anns:
        if a['image_id'] not in image_set or a['category_id'] not in cat_set:
            errors.append(f"Annotation {a['id']} references an absent image/category")
        bbox = a.get('bbox', [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0 or a.get('area', 0) <= 0:
            errors.append(f"Annotation {a['id']} has invalid bbox/area")
    visible = [a for a in anns if not a.get('iscrowd', 0) and not a.get('ignore', 0)]
    size_counts = Counter('small' if a['area'] < 32**2 else 'medium' if a['area'] < 96**2 else 'large' for a in visible if a.get('area',0)>0)
    return {'path': str(Path(path).resolve()), 'images': len(images), 'annotations': len(anns),
            'nonignored_annotations': len(visible), 'nonignored_sizes': dict(size_counts),
            'category_mapping': {str(c['id']): c['name'] for c in cats}, 'errors': errors}, data

def coco_scenes(data):
    """Only infer lineage for the explicit 12-digit COCO filename convention."""
    scenes = set()
    unknown = []
    for image in data['images']:
        match = re.fullmatch(r'(\d{12})(?:_(?:clear|fog|rain|snow))?\.[A-Za-z0-9]+', Path(image['file_name']).name)
        if match:
            scenes.add(match[1])
        else:
            unknown.append(image['file_name'])
    return scenes, unknown

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', nargs='+', required=True, type=Path)
    p.add_argument('--require-positive-ann-ids', action='store_true')
    p.add_argument('--train', type=Path, help='Optional COCO source-scene overlap check with --val')
    p.add_argument('--val', type=Path)
    args = p.parse_args()
    if bool(args.train) != bool(args.val):
        p.error('Provide --train and --val together')
    results = [audit(path, args.require_positive_ann_ids)[0] for path in args.annotations]
    report = {'annotations': results}
    failed = any(r['errors'] for r in results)
    if args.train:
        train = json.loads(args.train.read_text())
        val = json.loads(args.val.read_text())
        t, tu = coco_scenes(train)
        v, vu = coco_scenes(val)
        overlap = sorted(t & v)
        report['source_scene_audit'] = {'convention': '12-digit COCO basename', 'overlap': overlap,
            'unresolved_train': len(tu), 'unresolved_val': len(vu), 'complete': not tu and not vu}
        failed = failed or bool(overlap) or bool(tu) or bool(vu)
    print(json.dumps(report, indent=2))
    return int(failed)

if __name__ == '__main__':
    raise SystemExit(main())
