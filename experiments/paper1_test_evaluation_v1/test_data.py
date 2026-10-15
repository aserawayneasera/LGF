#!/usr/bin/env python3
"""Download and prepare official hidden-label tests or a supplied labelled holdout."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_ROOT = '/tmp/asera/paper1_original/TEST_DATASET'
DEFAULT_DATA_ROOT = '/tmp/asera/paper1_original/DATASET'
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
COCO_IDS = dict(zip(CLASSES, [1, 2, 3, 4, 6, 8]))
COCO_BASE = 'https://s3.amazonaws.com/images.cocodataset.org/'
COCO_INFO_SHA256 = 'e52f412dd7195ac8f98d782b44c6dd30ea10241e9f42521f67610fbe055a74f8'
COCO_TESTDEV_INFO_SHA256 = '6b5347aad42531c10d144921b5d07e07c76298223672d851edc530717b5b2892'
SCHEMA = 'paper1-test-data-1.0'

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_json(path): return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in p.parts[0]:
        raise ValueError('Unsafe relative image/archive path')
    return Path(*p.parts)

def download(url, destination, expected_sha256=None, expected_md5=None):
    """Resumable download. Private ACDC URLs never enter saved reports or logs."""
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    state_path = destination.with_name(destination.name + '.download.json')
    source = signature(url)
    def check_file():
        if expected_sha256 and digest(destination) != expected_sha256: raise ValueError('Download SHA-256 mismatch')
        if expected_md5 and digest(destination, 'md5') != expected_md5.lower(): raise ValueError('Download MD5 mismatch')
    if destination.exists():
        check_file()
        if not zipfile.is_zipfile(destination): raise ValueError('Expected a ZIP archive, received another file type')
        print('DOWNLOAD EXISTS', destination, flush=True)
        return destination
    state = read_json(state_path) if state_path.exists() else {}
    if partial.exists() and (state.get('source_sha256') != source or not state.get('etag')):
        raise ValueError('Partial download has no matching source/ETag record; move the .part file before restarting')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'User-Agent': 'Paper1-test-evaluation/1.0', 'Accept-Encoding': 'identity'}
    if offset: headers.update({'Range': f'bytes={offset}-', 'If-Range': state['etag']})
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Download HTTP {exc.code}. For ACDC, obtain a current link from your signed-in account.') from None
    except Exception:
        raise RuntimeError('Download connection failed. Rerun the command to resume.') from None
    with response:
        status = response.status
        if offset and status == 206:
            content_range = response.headers.get('Content-Range', '')
            if not content_range.startswith(f'bytes {offset}-'): raise ValueError('Server returned a mismatched byte range')
            if response.headers.get('ETag') != state['etag']: raise ValueError('Remote archive changed during resume')
            mode = 'ab'
        elif status == 200:
            offset = 0; mode = 'wb'
        else: raise RuntimeError(f'Unexpected download status {status}')
        length = response.headers.get('Content-Length')
        expected_size = offset + int(length) if length is not None else None
        state = {'source_sha256': source, 'etag': response.headers.get('ETag'), 'expected_size': expected_size}
        write_json(state_path, state)
        count = offset; last = time.monotonic()
        with open(partial, mode) as f:
            while True:
                b = response.read(4 * 1024 * 1024)
                if not b: break
                f.write(b); count += len(b)
                if time.monotonic() - last >= 10:
                    print(f'DOWNLOAD {destination.name}: {count / 1e9:.2f} GB' + (f' / {expected_size / 1e9:.2f} GB' if expected_size else ''), flush=True)
                    last = time.monotonic()
        if expected_size is not None and count != expected_size: raise IOError('Incomplete download; rerun to resume')
    if not zipfile.is_zipfile(partial): raise ValueError('Expected ZIP bytes, not a login or error page')
    if expected_sha256 and digest(partial) != expected_sha256: raise ValueError('Download SHA-256 mismatch')
    if expected_md5 and digest(partial, 'md5') != expected_md5.lower(): raise ValueError('Download MD5 mismatch')
    os.replace(partial, destination)
    write_json(state_path, {**state, 'complete': True, 'bytes': count, 'sha256': digest(destination),
                            'publisher_checksum_verified': bool(expected_sha256 or expected_md5)})
    print('DOWNLOAD COMPLETE', destination, flush=True)
    return destination

def extract_selected(archive, destination, predicate):
    """Extract selected files, checking CRCs and refusing traversal/symlinks."""
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    record_path = destination / ('extract_' + Path(archive).name + '.json')
    archive_hash = digest(archive)
    if record_path.exists():
        record = read_json(record_path)
        if record['archive_sha256'] != archive_hash: raise ValueError('Archive changed since extraction')
        if all((destination / r['name']).is_file() and digest(destination / r['name']) == r['sha256'] for r in record['files']):
            return record
        raise ValueError('Extracted data changed; use a fresh destination')
    records = []; names = set()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            relative = safe_relative(member.filename)
            if member.is_dir() or not predicate(member.filename): continue
            if member.filename in names: raise ValueError('Duplicate ZIP member')
            names.add(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16): raise ValueError('ZIP symlinks are not accepted')
            path = destination / relative
            if not path.resolve().is_relative_to(destination): raise ValueError('Archive path escapes destination')
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + '.extracting')
            with z.open(member) as src, open(tmp, 'wb') as dst: shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            if tmp.stat().st_size != member.file_size: raise ValueError('Extracted size mismatch')
            if path.exists() and digest(path) != digest(tmp): raise ValueError('Existing extraction file differs')
            os.replace(tmp, path)
            records.append({'name': member.filename, 'sha256': digest(path)})
            if len(records) % 1000 == 0: print('EXTRACT', len(records), flush=True)
    if not records: raise ValueError('Archive contained no matching test files')
    record = {'archive_sha256': archive_hash, 'files': records}
    write_json(record_path, record)
    return record

def acdc_test_path(name):
    parts = PurePosixPath(name).parts
    return ('rgb_anon' in parts and 'test' in parts and any(c in parts for c in ['fog', 'night', 'rain', 'snow'])
            and '_ref_' not in name and name.endswith('_rgb_anon.png'))

def download_action(args):
    root = Path(args.test_root); archives = root / 'archives'
    if args.dataset == 'coco':
        info = download(COCO_BASE + 'annotations/image_info_test2017.zip', archives / 'image_info_test2017.zip', COCO_INFO_SHA256)
        extract_selected(info, root / 'COCO', lambda n: n.endswith('.json'))
        if not args.metadata_only:
            images = Path(args.archive) if args.archive else download(COCO_BASE + 'zips/test2017.zip', archives / 'test2017.zip')
            extract_selected(images, root / 'COCO', lambda n: n.startswith('test2017/') and n.endswith('.jpg'))
    else:
        if args.metadata_only: raise ValueError('--metadata-only applies to COCO')
        if args.archive:
            archive = Path(args.archive)
            if args.md5 and digest(archive, 'md5') != args.md5.lower(): raise ValueError('ACDC archive MD5 mismatch')
        elif args.url_file:
            url = Path(args.url_file).read_text().strip()
            if not url.startswith('https://'): raise ValueError('Use the HTTPS download link issued to your ACDC account')
            archive = download(url, archives / 'rgb_anon_trainvaltest.zip', expected_md5=args.md5)
        else:
            raise ValueError('ACDC requires its account download: use --archive PATH or --url-file PATH. See https://acdc.vision.ee.ethz.ch/download')
        extract_selected(archive, root / 'ACDC', acdc_test_path)
    print('DOWNLOAD/EXTRACTION PASS', root, flush=True)

def data_paths(dataset, root):
    root = Path(root)
    if dataset.startswith('coco'):
        tag = 'non_weather' if dataset == 'coco_nw' else 'weather'
        folder = root / 'COCO' / ('non_weather-mini' if dataset == 'coco_nw' else 'weather-mini')
        return [(folder / f'images/{s}2017_{tag}-{n}k6c', folder / f'annotations/mini_{s}2017_{tag}-{n}k6c.json') for s,n in [('train',2400),('val',500)]]
    folder = root / 'ACDC-1/ACDC-1-NEW'
    return [(folder / 'images' / s, folder / 'annotations' / f'mini_{s}.json') for s in ['train','val']]

def source_key(dataset, filename):
    if dataset.startswith('coco'):
        match = re.match(r'^(\d+)(?:_|\.)', Path(filename).name)
        return 'coco:' + str(int(match[1])) if match else None
    match = re.search(r'(GOPR\d+_frame_\d+)', filename)
    return 'acdc:' + match[1] if match else None

def image_record(path, filename, image_id, dataset, condition=None):
    from PIL import Image
    with Image.open(path) as im: width, height = im.size; im.verify()
    return {'id': image_id, 'file_name': filename, 'width': width, 'height': height,
            'sha256': digest(path), 'source_key': source_key(dataset, filename), 'condition': condition}

def exclusion_audit(dataset, data_root, extra_jsons=()):
    family = ['coco_nw', 'coco_weather'] if dataset.startswith('coco') else ['acdc']
    keys, hashes, sources = set(), set(), []
    for name in family:
        for folder, annotations in data_paths(name, data_root):
            if not annotations.is_file(): raise FileNotFoundError(f'Original split needed for overlap audit: {annotations}')
            doc = read_json(annotations)
            for im in doc['images']:
                p = folder / safe_relative(im['file_name'])
                if not p.is_file(): raise FileNotFoundError(p)
                hashes.add(digest(p)); key = source_key(name, im['file_name'])
                if key: keys.add(key)
            sources.append({'annotations': str(annotations.resolve()), 'sha256': digest(annotations), 'images': len(doc['images'])})
    for p in extra_jsons:
        for im in read_json(p)['images']:
            key = source_key(dataset, im['file_name'])
            if key: keys.add(key)
        sources.append({'extra_history_json': str(Path(p).resolve()), 'sha256': digest(p)})
    return keys, hashes, sources

def check_overlap(records, keys, hashes):
    conflict = [r['file_name'] for r in records if (r.get('source_key') and r['source_key'] in keys) or r['sha256'] in hashes]
    if conflict: raise ValueError(f'Test/train-validation overlap: {len(conflict)} image(s), first: {conflict[0]}')

def save_manifest(path, dataset, image_root, records, categories, audit_sources, **extra):
    if Path(path).exists(): raise FileExistsError(f'Manifest exists: {path}. Reuse the existing manifest or choose a new output.')
    if not records or len({str(r['id']) for r in records}) != len(records): raise ValueError('Empty test set or duplicate image IDs')
    if len({r['file_name'] for r in records}) != len(records): raise ValueError('Duplicate test filenames')
    value = {'schema': SCHEMA, 'dataset': dataset, 'image_root': str(Path(image_root).resolve()),
             'images': records, 'categories': categories, 'ground_truth_available': False,
             'exclusion_sources': audit_sources, 'overlap_check': 'no matching bytes or parsed source-frame IDs',
             'complete_scene_history_verified': False, **extra}
    value['content_sha256'] = signature({k:v for k,v in value.items() if k != 'image_root'})
    write_json(path, value); print('TEST MANIFEST', path, 'images=', len(records), flush=True)
    return value

def prepare_coco(args):
    root = Path(args.test_root) / 'COCO'; doc_path = root / 'annotations/image_info_test-dev2017.json'
    if digest(doc_path) != COCO_TESTDEV_INFO_SHA256: raise ValueError('COCO test-dev image metadata differs from the verified official release')
    doc = read_json(doc_path); images = sorted(doc['images'], key=lambda x:x['id'])
    if len(images) != 20288: raise ValueError('Official COCO test-dev2017 must contain 20,288 images')
    keys, hashes, sources = exclusion_audit('coco_nw', args.data_root, args.exclude_json)
    records = []
    for im in images:
        rec = image_record(root / 'test2017' / safe_relative(im['file_name']), im['file_name'], im['id'], 'coco_nw', 'clear')
        if (rec['width'],rec['height']) != (im['width'],im['height']): raise ValueError('COCO image dimensions differ from official metadata')
        records.append(rec)
        if len(records) % 1000 == 0: print('INDEX', len(records), flush=True)
    check_overlap(records, keys, hashes)
    categories = [{ 'id': COCO_IDS[n], 'name': n} for n in CLASSES]
    return save_manifest(args.output or Path(args.test_root)/'coco_nw_test.json', 'coco_nw', root/'test2017', records, categories, sources,
                         benchmark='COCO test-dev2017', server_mapping_verified=True,
                         official_info_sha256=digest(doc_path), official_class_count=80)

def prepare_acdc(args):
    root = Path(args.images or Path(args.test_root)/'ACDC').resolve()
    files = sorted(p for p in root.rglob('*.png') if acdc_test_path(str(p.relative_to(root))))
    if len(files) != 2000: raise ValueError(f'Expected 2,000 ACDC adverse test images, found {len(files)}. Point --images above rgb_anon/.')
    keys, hashes, sources = exclusion_audit('acdc', args.data_root, args.exclude_json)
    records = []
    for p in files:
        relative = str(p.relative_to(root)); condition = next(c for c in ['fog','night','rain','snow'] if c in p.parts)
        records.append(image_record(p, relative, relative, 'acdc', condition))
    if collections.Counter(r['condition'] for r in records) != dict.fromkeys(['fog','night','rain','snow'], 500):
        raise ValueError('Expected 500 images for each ACDC adverse test condition')
    check_overlap(records, keys, hashes)
    # Filename IDs are internal only. No server image ID is invented.
    categories = [{'id':i+1,'name':n} for i,n in enumerate(CLASSES)]
    return save_manifest(args.output or Path(args.test_root)/'acdc_test.json', 'acdc', root, records, categories, sources,
                         benchmark='ACDC adverse test', server_mapping_verified=False,
                         official_class_count=8, server_mapping_note='Use official test image metadata with package-acdc after inference.')

def prepare_labelled(args):
    if not args.images or not args.annotations or not args.output: raise ValueError('Labelled preparation needs --images, --annotations, --output')
    doc = read_json(args.annotations)
    validate_labels(doc)
    cats = [c for c in doc['categories'] if c['name'] in CLASSES]
    if len(cats) != 6 or {c['name'] for c in cats} != set(CLASSES): raise ValueError('Ground truth must define all six classes')
    keys, hashes, sources = exclusion_audit(args.dataset, args.data_root, args.exclude_json)
    records = [image_record(Path(args.images)/safe_relative(im['file_name']), im['file_name'], im['id'], args.dataset) for im in sorted(doc['images'],key=lambda x:str(x['id']))]
    for record, im in zip(records, sorted(doc['images'],key=lambda x:str(x['id']))):
        if (record['width'],record['height']) != (im['width'],im['height']): raise ValueError('Ground-truth image dimensions differ')
    check_overlap(records, keys, hashes)
    return save_manifest(args.output, args.dataset, args.images, records, cats, sources,
                         benchmark='User-supplied labelled holdout', server_mapping_verified=False,
                         ground_truth_available=True, annotations=str(Path(args.annotations).resolve()), annotations_sha256=digest(args.annotations))

def validate_labels(doc):
    if 'annotations' not in doc or not doc['annotations']: raise ValueError('Labelled evaluation needs real nonempty ground-truth annotations')
    image_ids=[im['id'] for im in doc['images']];category_ids=[c['id'] for c in doc['categories']]
    annotation_ids=[a['id'] for a in doc['annotations']]
    if any(len(set(ids))!=len(ids) for ids in [image_ids,category_ids,annotation_ids]):raise ValueError('Duplicate ground-truth IDs')
    image_ids=set(image_ids);category_ids=set(category_ids)
    for a in doc['annotations']:
        if a['image_id'] not in image_ids or a['category_id'] not in category_ids:raise ValueError('Unknown ground-truth image/category ID')
        box=a.get('bbox',[])
        if len(box)!=4 or not all(math.isfinite(float(v)) for v in box) or box[2]<0 or box[3]<0:raise ValueError('Invalid ground-truth box')
        if 'area' not in a or not math.isfinite(float(a['area'])) or a['area']<0:raise ValueError('Ground truth needs the original valid area field')

def prepare_weather(args):
    import cv2
    import numpy as np
    from vendor.original_weather import weather_once
    source = Path(args.base_manifest or Path(args.test_root)/'coco_nw_test.json'); base = read_json(source)
    verify_manifest(base)
    if base['dataset'] != 'coco_nw': raise ValueError('Weather input must be the frozen clear COCO test manifest')
    condition = args.condition; dest = Path(args.test_root)/'COCO'/('weather_'+condition+'_seed'+str(args.weather_seed))
    dest.mkdir(parents=True,exist_ok=True); records=[]
    for i,im in enumerate(base['images']):
        seed = int(signature([args.weather_seed, im['id'], condition])[:8],16); np.random.seed(seed)
        chosen = ['fog','rain','snow'][seed % 3] if condition == 'mixed' else condition
        frame = cv2.imread(str(Path(base['image_root'])/im['file_name']))
        if frame is None: raise ValueError('Unreadable source weather image')
        result = weather_once(frame, chosen)
        filename = f"{Path(im['file_name']).stem}_{chosen}.jpg"; path = dest/filename
        ok, encoded = cv2.imencode('.jpg',result)
        if not ok: raise IOError('Weather encoding failed')
        data = encoded.tobytes()
        if path.exists() and path.read_bytes() != data: raise ValueError('Existing weather image differs from the frozen generator recipe')
        if not path.exists(): path.write_bytes(data)
        records.append(image_record(path, filename, im['id'], 'coco_weather', chosen))
        if (i+1)%1000==0: print('WEATHER',i+1,'/',len(base['images']),flush=True)
    labelled = {}
    if base['ground_truth_available']:
        gt = read_json(base['annotations']); filenames = {r['id']:r['file_name'] for r in records}
        for im in gt['images']: im['file_name'] = filenames[im['id']]
        gt_path = dest/'annotations.json';write_json(gt_path,gt)
        labelled = {'ground_truth_available':True,'annotations':str(gt_path.resolve()),'annotations_sha256':digest(gt_path)}
    return save_manifest(args.output or Path(args.test_root)/f'coco_weather_{condition}_test.json', 'coco_weather', dest, records,
                         base['categories'], base['exclusion_sources'], benchmark='Synthetic weather on '+base['benchmark'],
                         server_mapping_verified=base['server_mapping_verified'], official_class_count=base.get('official_class_count'),
                         weather_recipe={'condition':condition,'seed':args.weather_seed,'seed_rule':'sha256(seed,image_id,condition), first 32 bits',
                         'generator_sha256':digest(ROOT/'vendor/original_weather.py'),'base_manifest_sha256':digest(source)},
                         submission_scope='Custom weather evaluation; do not describe as the standard clear-image benchmark', **labelled)

def find_acdc_info(args):
    doc=read_json(args.manifest)
    wanted={Path(im['file_name']).name for im in doc['images']}
    candidates=[]
    for p in Path(args.metadata_root).rglob('*.json'):
        if p.stat().st_size>250*1024*1024:continue
        try:
            value=read_json(p)
            if not isinstance(value,dict) or not isinstance(value.get('images'),list):continue
            found={Path(im['file_name']).name for im in value['images']}
            overlap=len(wanted&found)
            if overlap:candidates.append({'path':str(p.resolve()),'matching_test_images':overlap,'contains_categories':bool(value.get('categories')),
                                         'complete':wanted<=found,'provenance':'Candidate only; verify this file came from the ACDC provider.'})
        except (ValueError,KeyError,TypeError):continue
    print(json.dumps(candidates,indent=2))
    if not candidates:print('No matching test image metadata found. Obtain public test image IDs and category IDs from ACDC; test bounding-box labels are not needed.')

def verify_manifest(doc):
    if doc.get('schema') != SCHEMA: raise ValueError('Unsupported test manifest')
    if signature({k:v for k,v in doc.items() if k not in ['image_root','content_sha256']}) != doc['content_sha256']:
        raise ValueError('Test manifest changed')
    for i,im in enumerate(doc['images']):
        p = Path(doc['image_root'])/safe_relative(im['file_name'])
        if digest(p) != im['sha256']: raise ValueError('Test image changed: '+im['file_name'])
        if (i+1)%1000==0: print('VERIFY TEST',i+1,'/',len(doc['images']),flush=True)
    if doc['ground_truth_available'] and digest(doc['annotations']) != doc['annotations_sha256']:
        raise ValueError('Test annotations changed')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['download','prepare','weather','prepare-labelled','verify','find-acdc-info'])
    p.add_argument('--dataset',choices=['coco','acdc','coco_nw','coco_weather'],default='coco')
    p.add_argument('--test-root',default=DEFAULT_TEST_ROOT);p.add_argument('--data-root',default=DEFAULT_DATA_ROOT)
    p.add_argument('--archive',type=Path);p.add_argument('--url-file',type=Path);p.add_argument('--md5')
    p.add_argument('--metadata-only',action='store_true');p.add_argument('--images',type=Path);p.add_argument('--annotations',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--base-manifest',type=Path)
    p.add_argument('--metadata-root',type=Path)
    p.add_argument('--exclude-json',type=Path,action='append',default=[])
    p.add_argument('--condition',choices=['mixed','fog','rain','snow'],default='mixed');p.add_argument('--weather-seed',type=int,default=42)
    args=p.parse_args()
    if args.action=='download':
        if args.dataset not in ['coco','acdc']: raise ValueError('Download dataset must be coco or acdc')
        download_action(args)
    elif args.action=='prepare':
        if args.dataset in ['coco','coco_nw']: prepare_coco(args)
        elif args.dataset=='acdc': prepare_acdc(args)
        else: raise ValueError('Use the weather action for synthetic weather')
    elif args.action=='weather': prepare_weather(args)
    elif args.action=='prepare-labelled':
        if args.dataset=='coco':args.dataset='coco_nw'
        prepare_labelled(args)
    elif args.action=='verify': verify_manifest(read_json(args.manifest));print('TEST MANIFEST PASS')
    elif args.action=='find-acdc-info':
        if not args.manifest or not args.metadata_root:raise ValueError('Supply --manifest and --metadata-root')
        find_acdc_info(args)

if __name__=='__main__': main()
