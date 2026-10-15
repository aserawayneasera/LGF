#!/usr/bin/env python3
"""Paper 1: YOLO11m / RT-DETR-L baselines, shared COCO evaluation, and comparison.

Train/evaluate newer detectors with one script. Evaluate original LGF, baseline,
or UGS checkpoints through the previously supplied, unmodified UGS package.
Run --help, or see REVIEWER34_README.md. No datasets or trained weights included.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
from dataclasses import asdict, dataclass
import datetime
import faulthandler
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import time
import traceback

VERSION = '1.0.1'
PREPARED_DATA_VERSION = '1.0.0'  # Label conversion is unchanged; reuse audited prepared data.
ULTRALYTICS_VERSION = '8.3.253'
DEFAULT_ROOT = '/tmp/asera/paper1_original/reviewer34'
CLASSES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}
MODELS = {'yolo11m': ('yolo11m.pt', 'yolo11m.yaml', 'YOLO11m (Ultralytics)'),
          'rtdetr-l': ('rtdetr-l.pt', 'rtdetr-l.yaml', 'RT-DETR-L (Ultralytics, HGNetv2)')}
PRETRAINED_SHA256 = {'yolo11m':'d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95',
                    'rtdetr-l':'6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f'}
LEGACY = ['lgf_original', 'baseline', 'ugs_reimpl']
METRICS = ['mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_json(path): return json.loads(Path(path).read_text())


def save_torch(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); torch.save(value, tmp); os.replace(tmp, path)


def environment():
    import torch, torchvision
    import ultralytics
    packages = {}
    for name in ['torch', 'torchvision', 'ultralytics', 'numpy', 'pycocotools', 'albumentations']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {'python': sys.version.split()[0], 'python_executable': sys.executable,
            'module_paths': {m.__name__: str(m.__file__) for m in [torch, torchvision, ultralytics]},
            'packages': packages, 'cuda': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@contextlib.contextmanager
def fixed_inference_settings():
    """Use identical settings during selection, after training, and in fresh evaluation.

    Ultralytics 8.3.253 calls unset_deterministic() after training. Scope this
    policy to inference and restore the caller's training settings even on error.
    Do not reseed, round predictions, or weaken the reload equality check.
    """
    import torch
    old = (torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
           os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.deterministic = old[2]
        torch.backends.cudnn.benchmark = old[3]
        if old[4] is None: os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else: os.environ['CUBLAS_WORKSPACE_CONFIG'] = old[4]


def setup(args):
    root = Path(args.work_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(root / 'ultralytics_settings')
    os.environ.setdefault('TORCH_HOME', '/tmp/asera/paper1_original/torch_cache')
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch, torchvision, ultralytics
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f'Requires ultralytics=={ULTRALYTICS_VERSION}; found {ultralytics.__version__}')
    if not torch.__version__.startswith('2.10.') or not torchvision.__version__.startswith('0.25.'):
        raise RuntimeError('Requires your existing torch 2.10.x / torchvision 0.25.x environment')
    if args.device != 'cpu' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if args.device != 'cpu' and (not args.device.isdigit()): raise ValueError('Use one GPU index, e.g. --device 0')
    if os.environ.get('MAX_TRAIN_IMAGES'): raise RuntimeError('Unset MAX_TRAIN_IMAGES')
    if min(args.batch, args.eval_batch, args.train_size, args.eval_min, args.eval_max, args.epochs, args.threads) < 1:
        raise ValueError('Invalid size/batch/epoch argument')
    if args.workers < 0 or args.eval_min > args.eval_max: raise ValueError('Invalid workers/evaluation sizes')
    torch.set_num_threads(args.threads)
    # Isolated application settings; no external experiment logging.
    ultralytics.settings.update({k: False for k in ['sync', 'clearml', 'comet', 'dvc', 'hub',
        'mlflow', 'neptune', 'raytune', 'tensorboard', 'wandb'] if k in ultralytics.settings})
    ultralytics.settings.update({'weights_dir': str(root/'pretrained'), 'datasets_dir': str(root/'prepared'),
                                'runs_dir': str(root/'runs')})
    # Plots are disabled; avoid an unrelated font download during dataset checks.
    import ultralytics.data.utils as du
    du.check_font = lambda *a, **k: None
    return root


def dataset_paths(dataset, root):
    root = Path(root)
    if dataset in ['coco_nw', 'coco_weather']:
        tag = 'non_weather' if dataset == 'coco_nw' else 'weather'
        folder = root/'COCO'/f'{tag}-mini'
        return {f'{s}_images': folder/f'images/{s}2017_{tag}-{n}k6c' for s,n in [('train',2400),('val',500)]} | {
            f'{s}_annotations': folder/f'annotations/mini_{s}2017_{tag}-{n}k6c.json' for s,n in [('train',2400),('val',500)]}
    folder = root/'ACDC-1/ACDC-1-NEW'
    return {f'{s}_images': folder/f'images/{s}' for s in ['train','val']} | {
            f'{s}_annotations': folder/f'annotations/mini_{s}.json' for s in ['train','val']}


def inspect_split(images, annotations):
    import cv2
    doc = read_json(annotations)
    cats = sorted(doc['categories'], key=lambda c: c['id'])
    if len(cats) != 6 or {c['name'] for c in cats} != CLASSES or len({c['id'] for c in cats}) != 6:
        raise ValueError('Expected six unique category IDs and the original six classes')
    records = []; ids = set(); by_image = {}; ann_ids = set()
    for im in sorted(doc['images'], key=lambda x:x['id']):
        if im['id'] in ids: raise ValueError('Duplicate image ID')
        ids.add(im['id']); p = Path(images)/im['file_name']
        if not p.is_file(): raise FileNotFoundError(p)
        array = cv2.imread(str(p))
        if array is None: raise ValueError(f'Unreadable image: {p}')
        if array.shape[:2] != (im['height'], im['width']): raise ValueError(f'COCO/image dimensions differ: {p}')
        records.append({'image_id': im['id'], 'file_name': im['file_name'], 'sha256': sha256(p)})
        by_image[im['id']] = []
    if not ids: raise ValueError('Empty split')
    categories = {c['id'] for c in cats}
    for a in doc['annotations']:
        if a['id'] in ann_ids: raise ValueError('Duplicate annotation ID')
        ann_ids.add(a['id'])
        if a['image_id'] not in ids or a['category_id'] not in categories: raise ValueError('Unknown image/category')
        box = a['bbox']
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box): raise ValueError('Invalid box')
        by_image[a['image_id']].append(a)
    info = {'image_count': len(ids), 'annotation_count': len(ann_ids),
            'categories': [{'id':c['id'],'name':c['name']} for c in cats],
            'annotations_sha256': sha256(annotations), 'image_records': records,
            'content_sha256': signature({'annotations': sha256(annotations), 'images': records})}
    return doc, by_image, info


def audit(args, evaluation_only=False):
    paths = dataset_paths(args.dataset, args.data_root)
    if evaluation_only and (args.annotations or args.images):
        if not (args.annotations and args.images): raise ValueError('Supply both --annotations and --images')
        paths['val_annotations'], paths['val_images'] = Path(args.annotations), Path(args.images)
    result = {'dataset': args.dataset, 'paths': {k:str(Path(v).resolve()) for k,v in paths.items()}}
    for split in (['val'] if evaluation_only else ['train','val']):
        _, _, result[split] = inspect_split(paths[f'{split}_images'], paths[f'{split}_annotations'])
    if not evaluation_only:
        if result['train']['categories'] != result['val']['categories']: raise ValueError('Train/val categories differ')
        overlap = {r['sha256'] for r in result['train']['image_records']} & {r['sha256'] for r in result['val']['image_records']}
        if overlap: raise ValueError('Identical image bytes across train/validation')
    result['lineage_limit'] = 'Hashes detect identical bytes, not transformed versions of the same scene.'
    return result


def prepare(args):
    import yaml
    rec = audit(args); dest = Path(args.work_root)/'prepared'/args.dataset
    expected = {'train':rec['train']['content_sha256'], 'val':rec['val']['content_sha256'], 'version':PREPARED_DATA_VERSION}
    if (dest/'ready.json').is_file():
        if read_json(dest/'ready.json') != expected: raise ValueError('Prepared data differ; use a new --work-root')
        verify_prepared(dest)
        return dest/'dataset.yaml', rec
    if dest.exists(): raise FileExistsError(f'Incomplete prepared folder: {dest}; choose a new --work-root')
    dest.mkdir(parents=True)
    changes = {}; label_records = []
    for split in ['train','val']:
        doc, anns, _ = inspect_split(rec['paths'][f'{split}_images'], rec['paths'][f'{split}_annotations'])
        cat_to_label = {c['id']:i for i,c in enumerate(rec[split]['categories'])}
        (dest/'images'/split).mkdir(parents=True); (dest/'labels'/split).mkdir(parents=True)
        skipped = 0; crowd = 0
        for im in sorted(doc['images'],key=lambda x:x['id']):
            lines = []; seen = set()
            for a in anns[im['id']]:
                x,y,w,h = map(float,a['bbox'])
                if w <= 1 or h <= 1: skipped += 1; continue  # original LGF reader rule
                if min(x,y) < 0 or x+w > im['width']+1e-5 or y+h > im['height']+1e-5:
                    raise ValueError(f'Out-of-bounds training target: image {im["id"]}, annotation {a["id"]}')
                row = (cat_to_label[a['category_id']], (x+w/2)/im['width'], (y+h/2)/im['height'], w/im['width'], h/im['height'])
                if row in seen: raise ValueError('Duplicate box/class; Ultralytics would silently remove this target')
                seen.add(row); crowd += bool(a.get('iscrowd',0))
                lines.append(f'{row[0]} '+' '.join(f'{v:.12g}' for v in row[1:]))
            name = str(im['id']); source = Path(rec['paths'][f'{split}_images'])/im['file_name']
            link = dest/'images'/split/(name+source.suffix.lower()); link.symlink_to(source.resolve())
            label = dest/'labels'/split/(name+'.txt'); label.write_text('\n'.join(lines)+ ('\n' if lines else ''))
            label_records.append({'label':str(label.relative_to(dest)), 'label_sha256':sha256(label),
                                  'image':str(link.relative_to(dest)), 'image_sha256':sha256(link)})
        changes[split] = {'boxes_excluded_w_or_h_le_1_like_original':skipped,
                          'crowd_boxes_included_as_training_targets_like_original':crowd}
    names = {i:c['name'] for i,c in enumerate(rec['train']['categories'])}
    (dest/'dataset.yaml').write_text(yaml.safe_dump({'path':str(dest.resolve()), 'train':'images/train',
        'val':'images/val', 'names':names}, sort_keys=False))
    write_json(dest/'conversion.json', {'source':rec, 'target_handling':changes, 'files':label_records,
                                     'yaml_sha256':sha256(dest/'dataset.yaml')})
    write_json(dest/'ready.json', expected)
    return dest/'dataset.yaml', rec


def verify_prepared(dest):
    rec = read_json(dest/'conversion.json')
    if sha256(dest/'dataset.yaml') != rec['yaml_sha256']: raise ValueError('Prepared YAML changed')
    for item in rec['files']:
        for name in ['label','image']:
            if sha256(dest/item[name]) != item[name+'_sha256']: raise ValueError('Prepared files changed')


def coco_metrics(annotations, predictions):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    doc = read_json(annotations); doc.setdefault('info', {})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = doc; gt.createIndex()
        if predictions: dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO(); dt.dataset = {'images':copy.deepcopy(doc['images']),
                                      'categories':copy.deepcopy(doc['categories']), 'annotations':[]}; dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(im['id'] for im in doc['images'])
        ev.params.iouThrs = np.array([.5,.55,.6,.65,.7,.75,.8,.85,.9,.95])
        ev.params.maxDets = [1,10,100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    out = {k:float(v) for k,v in zip(METRICS,ev.stats)}
    out['detection_count'] = len(predictions)
    return out


def common_protocol(args):
    return {'min_size':args.eval_min, 'max_size':args.eval_max, 'pad_stride':32,
        'eval_batch':args.eval_batch, 'score_thresh':0. if getattr(args,'synthetic',False) else .05, 'max_dets':[1,10,100],
        'iou_thresholds':[.5,.55,.6,.65,.7,.75,.8,.85,.9,.95], 'precision':'FP32',
        'weights':'EMA', 'resize':'aspect-preserving, Torch bilinear, align_corners=False',
        'area_ranges':'COCO areas in original annotation coordinates', 'evaluation_split':args.split,
        'inference_settings':{'deterministic_algorithms':True,'warn_only':False,
            'cudnn_deterministic':True,'cudnn_benchmark':False,'cublas_workspace_config':':4096:8'}}


def read_images(items, folder, device):
    import cv2, torch
    return [torch.from_numpy(cv2.cvtColor(cv2.imread(str(Path(folder)/im['file_name'])),cv2.COLOR_BGR2RGB))
            .permute(2,0,1).float().div(255).to(device) for im in items]


def shared_resize(images, min_size, max_size):
    import torch
    import torch.nn.functional as F
    resized = []; shapes = []
    for im in images:
        h,w = im.shape[-2:]; scale = min(min_size/min(h,w), max_size/max(h,w))
        x = F.interpolate(im[None], scale_factor=scale, mode='bilinear', align_corners=False,
                          recompute_scale_factor=True)[0]
        resized.append(x); shapes.append((h,w,*x.shape[-2:]))
    ph = math.ceil(max(x.shape[-2] for x in resized)/32)*32
    pw = math.ceil(max(x.shape[-1] for x in resized)/32)*32
    # Native newer detectors consume RGB [0,1]; their normalizer differs from RetinaNet.
    batch = images[0].new_zeros((len(images),3,ph,pw))
    for i,x in enumerate(resized): batch[i,:,:x.shape[-2],:x.shape[-1]] = x
    return batch, shapes


def unscale_boxes(boxes, shape):
    h,w,rh,rw = shape; boxes = boxes.clone()
    boxes[:,[0,2]] *= w/rw; boxes[:,[1,3]] *= h/rh
    boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,w); boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,h)
    return boxes


def native_class(model):
    from ultralytics import YOLO, RTDETR
    return RTDETR if model == 'rtdetr-l' else YOLO


def load_detector(args, checkpoint):
    import torch
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:'+args.device)
    if args.model in MODELS:
        wrapper = native_class(args.model)(str(checkpoint))
        ck = torch.load(checkpoint,map_location='cpu',weights_only=False)
        meta = ck.get('reviewer34')
        if not meta: raise ValueError('Evaluation requires a selected checkpoint from this script')
        if meta['model'] != args.model or meta['dataset'] != args.dataset or meta['seed'] != args.seed:
            raise ValueError('Model/dataset/seed differ from checkpoint')
        model = wrapper.model; params = sum(p.numel() for p in model.parameters())
        return wrapper, device, meta, params
    package = Path(args.lgf_package).resolve()
    if not (package/'core_bridge.py').is_file(): raise FileNotFoundError('Supply --lgf-package with the earlier UGS package')
    sys.path.insert(0,str(package))
    from core_bridge import build_model
    from ugs import UGSConfig
    ck = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 'ema_state_dict' not in ck: raise ValueError('LGF/UGS checkpoint needs EMA weights')
    if ck.get('package_version'):
        if (ck['method'],ck['dataset'],ck['seed']) != (args.model,args.dataset,args.seed): raise ValueError('Checkpoint identity differs')
        model = build_model(args.model,6,ck['level'],UGSConfig(**ck['ugs_config']),pretrained=False)
        epoch = ck['completed_epochs']
    else:
        cfg = ck.get('config',{})
        if args.model == 'lgf_original' and (cfg.get('block_type'),cfg.get('gating_type')) != ('lgf','gated_spatial'):
            raise ValueError('Expected original spatial-gated LGF checkpoint')
        if args.model == 'baseline' and cfg.get('block_type') != 'none': raise ValueError('Expected original baseline')
        if args.model == 'ugs_reimpl': raise ValueError('Original notebook has no UGS mode')
        identity = re.fullmatch(r'(coco_nw|coco_weather|acdc)_(C[345])_.+_s(-?\d+)',str(ck.get('experiment_name','')))
        if identity and (identity[1],identity[2],int(identity[3])) != (args.dataset,args.level,args.seed):
            raise ValueError('Original checkpoint experiment identity differs')
        model = build_model('baseline',6,args.level,pretrained=False,original_config=cfg)
        epoch = ck['epoch']+1
    state = ck['ema_state_dict']
    if all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True); model.to(device).float().eval()
    if (list(model.transform.min_size),model.transform.max_size) != ([args.eval_min],args.eval_max):
        raise ValueError('Requested evaluation sizes differ from the original LGF transform')
    if (model.score_thresh,model.nms_thresh,model.detections_per_img) != (.05,.5,100): raise ValueError('Original prediction settings differ')
    label = {'lgf_original':'LGF (original spatial-gated model)', 'baseline':'RetinaNet (original baseline)',
             'ugs_reimpl':'UGS (our RetinaNet reimplementation)'}[args.model]
    meta={'model':args.model,'label':label,'dataset':args.dataset,'seed':args.seed,'epoch':epoch,
          'training_protocol':ck.get('protocol'), 'ugs_config':ck.get('ugs_config') if args.model=='ugs_reimpl' else None,
          'original_config':ck.get('config'), 'legacy_training_equivalence_verified':False}
    return model,device,meta,sum(p.numel() for p in model.parameters())


def infer_batch(detector, images, args):
    import torch
    with fixed_inference_settings(), torch.no_grad():
        if args.model in LEGACY:
            captured={}
            def capture_shape(module, inputs): captured['shape']=tuple(inputs[0].shape[-2:])
            handle=detector.backbone.register_forward_pre_hook(capture_shape)
            try: predictions=detector(images)
            finally: handle.remove()
            return predictions,[captured['shape'] for _ in images]
        tensor, shapes = shared_resize(images,args.eval_min,args.eval_max)
        out = detector.predict(source=tensor, device=args.device, imgsz=list(tensor.shape[-2:]),
            conf=0. if getattr(args,'synthetic',False) else .05, iou=.5, max_det=100, half=False, verbose=False, save=False, stream=False,
            augment=False, agnostic_nms=False)
        preds=[]
        for r,shape in zip(out,shapes):
            preds.append({'boxes':unscale_boxes(r.boxes.xyxy,shape),'scores':r.boxes.conf,'labels':r.boxes.cls.long()})
        return preds,[tuple(tensor.shape[-2:]) for _ in images]


def evaluate_checkpoint(args, checkpoint, rec=None, output=None, selection=None):
    import torch
    rec = rec or audit(args,evaluation_only=True)
    output = Path(output or args.output or Path(args.work_root)/'evaluations'/args.dataset/args.model/f'seed_{args.seed}')
    if output.exists(): raise FileExistsError(f'Evaluation directory exists: {output}')
    output.mkdir(parents=True)
    detector,device,meta,params = load_detector(args,checkpoint)
    selected=selection or args.selection
    if Path(checkpoint).stem in ['best_map','best_ap_small'] and selected != Path(checkpoint).stem:
        raise ValueError('Selection argument differs from the selected checkpoint filename')
    saved_protocol=meta.get('training_protocol') or {}
    saved_val=saved_protocol.get('val_content_sha256',saved_protocol.get('validation_content_sha256'))
    if args.split=='val' and saved_val and saved_val!=rec['val']['content_sha256']:
        raise ValueError('Validation files differ from the checkpoint training record')
    overlap_check=None
    if args.split=='test':
        source=audit(args)
        known={x['sha256'] for s in ['train','val'] for x in source[s]['image_records']}
        if known & {x['sha256'] for x in rec['val']['image_records']}:
            raise ValueError('Test images include identical bytes from training or selection validation')
        overlap_check={'identical_bytes_with_train_or_selection_val':0,'scene_independence_verified':False}
    cats=rec['val']['categories']; names={i:c['name'] for i,c in enumerate(cats)}
    if args.model in MODELS and {int(k):v for k,v in detector.names.items()} != names:
        raise ValueError('Checkpoint class names differ from COCO mapping')
    document=read_json(rec['paths']['val_annotations']); items=sorted(document['images'],key=lambda x:x['id'])
    detections=[]; shapes=[]; start=time.perf_counter()
    for offset in range(0,len(items),args.eval_batch):
        subset=items[offset:offset+args.eval_batch]
        images=read_images(subset,rec['paths']['val_images'],device)
        preds,dimensions=infer_batch(detector,images,args); shapes.extend([list(x) for x in dimensions])
        if len(preds)!=len(subset): raise RuntimeError('Prediction batch length differs')
        for im,pred in zip(subset,preds):
            boxes=pred['boxes'].detach().cpu().numpy(); scores=pred['scores'].detach().cpu().numpy(); labels=pred['labels'].detach().cpu().numpy()
            for box,score,label in zip(boxes,scores,labels):
                x1,y1,x2,y2=box
                if not all(math.isfinite(float(v)) for v in [*box,score]): raise FloatingPointError('Nonfinite prediction')
                detections.append({'image_id':im['id'],'category_id':cats[int(label)]['id'],
                    'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)})
        print(f'EVAL {args.model} {min(offset+args.eval_batch,len(items))}/{len(items)}',flush=True)
    metrics=coco_metrics(rec['paths']['val_annotations'],detections)
    write_json(output/'predictions.json',detections)
    protocol=common_protocol(args)
    result={**meta,'metrics':metrics,'selection':selected,'selection_provenance':args.selection_provenance,
        'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':sha256(checkpoint),
        'evaluation_protocol':protocol,'evaluation_signature':signature({'content':rec['val']['content_sha256'],'categories':cats,'protocol':protocol}),
        'evaluation_content_sha256':rec['val']['content_sha256'],'categories':cats,'parameter_count':params,
        'postprocessing':'native query confidence ranking, no NMS' if args.model=='rtdetr-l' else 'native class-aware NMS at IoU 0.5',
        'newer_detector_padding':'RGB zeros; native input normalization' if args.model in MODELS else 'original RetinaNet transform',
        'prediction_file':str(output/'predictions.json'),'prediction_sha256':sha256(output/'predictions.json'),
        'evaluation_seconds_including_IO':time.perf_counter()-start,'observed_shape_sample':shapes[:8],
        'environment':environment(),'script_sha256':sha256(__file__),'scientific_result':not getattr(args,'synthetic',False),
        'test_overlap_check':overlap_check,
        'evaluation_scope':'validation selected checkpoints are not independent test estimates' if args.split=='val' else 'separately supplied evaluation split'}
    write_json(output/'metrics.json',result)
    del detector
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def train(args, synthetic=False):
    import torch
    if args.model not in MODELS: raise ValueError('Train LGF/UGS using their original supplied runners')
    yaml_path,rec=prepare(args)
    run=Path(args.work_root)/'runs'/args.dataset/args.model/f'seed_{args.seed}'
    if run.exists() and not args.resume: raise FileExistsError(f'Run exists: {run}. Use --resume for an interrupted run')
    run.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(run/'run.lock','a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recipe={'epochs':args.epochs,'batch':args.batch,'nbs':16,'imgsz':args.train_size,'workers':args.workers,
        'device':args.device,'seed':args.seed,'amp':False,'optimizer':'AdamW' if args.model=='rtdetr-l' else 'SGD',
        'lr0':.0001 if args.model=='rtdetr-l' else .005,'lrf':.01,'momentum':.9,'weight_decay':.0001,
        'warmup_epochs':2.,'patience':0,'cos_lr':False,'deterministic':True,'freeze':None,
        'mosaic':0.,'mixup':0.,'copy_paste':0.,'degrees':0.,'translate':0.,'scale':0.,'shear':0.,
        'perspective':0.,'flipud':0.,'fliplr':.5,'hsv_h':0.,'hsv_s':0.,'hsv_v':0.,'close_mosaic':0,
        'plots':False,'val':True,'save':True,'save_period':-1,'cache':False,'rect':False,
        'project':str(run.resolve()),'name':'native','exist_ok':True,'pretrained':not synthetic}
    protocol={'version':VERSION,'model':args.model,'dataset':args.dataset,'seed':args.seed,'recipe':recipe,
        'train_content_sha256':rec['train']['content_sha256'],'val_content_sha256':rec['val']['content_sha256'],
        'common_evaluation':common_protocol(args),'evaluation_every_epochs':5,
        'environment':environment(),'script_sha256':sha256(__file__),'synthetic':synthetic,
        'training_scope':'separate cross-architecture recipe; not exact LGF or author-paper training reproduction'}
    protocol_id=signature(protocol); last=run/'native/weights/last.pt'
    if args.resume:
        if read_json(run/'protocol.json')['protocol_sha256']!=protocol_id: raise ValueError('Resume protocol changed')
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck.get('epoch',-1)<0 or ck.get('optimizer') is None: raise ValueError('Native checkpoint has no resumable state')
        last_epoch=ck['epoch']+1
        history=read_json(run/'selection_history.json') if (run/'selection_history.json').exists() else []
        if any(r['epoch']>last_epoch for r in history): raise ValueError('Selection history is ahead of native checkpoint')
        if last_epoch%5==0 and not any(r['epoch']==last_epoch for r in history):
            raise ValueError('Interrupted during shared evaluation. Finish that saved epoch evaluation before resuming; do not silently skip selection.')
        model=native_class(args.model)(str(last))
    else:
        history=[]; cache=Path(args.work_root)/'pretrained'; cache.mkdir(exist_ok=True)
        old=os.getcwd()
        try:
            os.chdir(cache)
            model=native_class(args.model)(MODELS[args.model][1] if synthetic else MODELS[args.model][0])
            weights=getattr(model,'ckpt_path',None)
            weights=Path(weights).resolve() if weights else None
        finally: os.chdir(old)
        if not synthetic and (not weights or not weights.is_file() or sha256(weights)!=PRETRAINED_SHA256[args.model]):
            raise ValueError('Unexpected pretrained weights. Use the pinned official asset in a fresh --work-root.')
        write_json(run/'initialization.json',{'source':'random YAML architecture' if synthetic else MODELS[args.model][0],
            'weights_sha256':sha256(weights) if weights and Path(weights).is_file() else None})
    write_json(run/'protocol.json',{**protocol,'protocol_sha256':protocol_id})
    best={k:read_json(run/(k+'.metrics.json'))['metrics'][m] if (run/(k+'.metrics.json')).exists() else -math.inf
          for k,m in [('best_ap_small','AP_small'),('best_map','mAP')]}
    seen={r['epoch'] for r in history}
    def on_epoch_end(trainer):
        epoch=trainer.epoch+1
        if epoch in seen or (epoch%5 and epoch!=args.epochs): return
        if trainer.ema is None: raise RuntimeError('EMA required')
        frozen=copy.deepcopy(trainer.ema.ema).cpu().float().eval()
        meta={'model':args.model,'label':MODELS[args.model][2],'dataset':args.dataset,'seed':args.seed,
              'epoch':epoch,'training_protocol':protocol,'protocol_sha256':protocol_id}
        checkpoint=run/f'candidate_epoch_{epoch:03d}.pt'
        save_torch(checkpoint,{'model':frozen,'ema':None,'train_args':vars(trainer.args),
            'version':ULTRALYTICS_VERSION,'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'reviewer34':meta,'epoch':trainer.epoch})
        del frozen
        result=evaluate_checkpoint(args,checkpoint,rec,run/'shared_evaluation'/f'epoch_{epoch:03d}',selection='epoch_candidate')
        keep=False
        for key,metric in [('best_ap_small','AP_small'),('best_map','mAP')]:
            value=result['metrics'][metric]
            if value>=0 and value>best[key]:
                best[key]=value; keep=True
                selected=run/(key+'.pt'); shutil.copyfile(checkpoint,selected)
                write_json(run/(key+'.metrics.json'),{**result,'selection':key,
                    'selection_provenance':'script selected maximum shared COCO validation metric',
                    'checkpoint':str(selected.resolve()),'checkpoint_sha256':sha256(selected)})
        history.append({'epoch':epoch,'metrics':result['metrics'],'candidate_checkpoint':str(checkpoint) if keep else None})
        write_json(run/'selection_history.json',history); seen.add(epoch)
        if not keep: checkpoint.unlink()
        print(f'SHARED EVALUATION PASS epoch={epoch} APsmall={result["metrics"]["AP_small"]:.6f}',flush=True)
    model.add_callback('on_fit_epoch_end',on_epoch_end)
    if synthetic:
        for event in ['on_train_epoch_end','on_val_start','on_val_end','on_model_save','on_train_end']:
            model.add_callback(event, lambda obj, name=event: print(f'SMOKE PHASE {name}', flush=True))
    start=time.perf_counter()
    try:
        model.train(data=str(yaml_path.resolve()),resume=str(last) if args.resume else False,**recipe)
        if args.epochs not in seen: raise RuntimeError('Final shared evaluation missing')
        write_json(run/'completed.json',{'epochs':args.epochs,'protocol_sha256':protocol_id,'training_seconds':time.perf_counter()-start,
            'best':{k:v if math.isfinite(v) else None for k,v in best.items()},'scientific_result':not synthetic})
        print(f'TRAINING COMPLETE {run}',flush=True)
    finally: lock.close()
    return run


def compare_reports(args):
    reports=[read_json(p) for p in args.reports]
    if not reports: raise ValueError('Supply --reports')
    if any(not r.get('scientific_result',False) for r in reports): raise ValueError('Synthetic results are not comparison results')
    if len({r['dataset'] for r in reports})!=1 or len({r['evaluation_signature'] for r in reports})!=1:
        raise ValueError('Dataset files or shared evaluation protocol differ')
    selections={r['selection'] for r in reports}
    if len(selections)!=1 or not selections<={'best_map','best_ap_small'}: raise ValueError('Use one verified checkpoint-selection rule')
    for r in reports:
        if sha256(r['checkpoint'])!=r['checkpoint_sha256']: raise ValueError('Checkpoint bytes changed')
        if sha256(r['prediction_file'])!=r['prediction_sha256']: raise ValueError('Prediction bytes changed')
    groups={}; rows=[]; seedsets=[]
    for r in reports: groups.setdefault(r['label'],[]).append(r)
    for label,items in groups.items():
        seeds=[r['seed'] for r in items]
        if len(seeds)!=len(set(seeds)): raise ValueError('Duplicate method/seed')
        seedsets.append(set(seeds))
        configs=[]
        for r in items:
            p=copy.deepcopy(r.get('training_protocol'))
            if p:
                p.pop('seed',None)
                if 'recipe' in p: p['recipe'].pop('seed',None); p['recipe'].pop('project',None)
            configs.append(signature({'training':p,'ugs_config':r.get('ugs_config'),'original_config':r.get('original_config')}))
        if len(set(configs))!=1: raise ValueError('Training configurations differ across seeds within one method')
        row={'method':label,'dataset':items[0]['dataset'],'n':len(items),'seeds':','.join(map(str,sorted(seeds))),
             'selection':items[0]['selection'],'parameters':items[0]['parameter_count']}
        for m in METRICS[:6]:
            values=[r['metrics'][m] for r in items]; valid=all(v>=0 for v in values)
            row[m+'_mean']=statistics.mean(values) if valid else None
            row[m+'_std']=statistics.stdev(values) if valid and len(values)>1 else None
        rows.append(row)
    if len(groups)<2 or any(s!=seedsets[0] for s in seedsets): raise ValueError('Supply at least two methods with matching seeds')
    output=Path(args.output or Path(args.work_root)/'comparison')
    if output.exists(): raise FileExistsError('Choose a new comparison output directory')
    output.mkdir(parents=True)
    write_json(output/'comparison.json',rows)
    with open(output/'comparison.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(output/'comparison_scope.json',{'evaluation_signature':reports[0]['evaluation_signature'],
        'three_seed_grid_complete':seedsets[0]=={42,1337,2025},'scale':'AP in 0-1 units',
        'claim':'complete-detector comparison under common evaluation; does not isolate the LGF module contribution',
        'training_equivalence':'not established across architectures; disclose each recorded recipe',
        'evaluation_split':reports[0]['evaluation_protocol']['evaluation_split'],
        'test_independence':'requires scene/split provenance; not established by a test label or byte hashes alone'})
    print(json.dumps(rows,indent=2))


def fixture(root):
    import numpy as np
    from PIL import Image
    paths=dataset_paths('coco_nw',root)
    names=['person','bicycle','car','motorcycle','bus','truck']
    for split,n in [('train',4),('val',2)]:
        paths[split+'_images'].mkdir(parents=True)
        doc={'images':[],'annotations':[],'categories':[{'id':i+1,'name':c} for i,c in enumerate(names)]}
        for i in range(n):
            idx=i+(100 if split=='val' else 0); name=f'{idx}.png'
            width=96 if split=='val' else 64
            Image.fromarray(np.random.default_rng(idx).integers(0,256,(64,width,3),dtype=np.uint8)).save(paths[split+'_images']/name)
            doc['images'].append({'id':idx,'file_name':name,'height':64,'width':width})
            doc['annotations'].append({'id':idx,'image_id':idx,'category_id':i%6+1,'bbox':[12,12,20,20],'area':400,'iscrowd':0})
        write_json(paths[split+'_annotations'],doc)


def reload_evidence(first, second):
    """Describe differences without accepting drift or altering saved predictions."""
    left, right = read_json(first['prediction_file']), read_json(second['prediction_file'])
    if sha256(first['checkpoint']) != first['checkpoint_sha256']:
        raise AssertionError('Selected checkpoint changed before reload')
    if second['checkpoint_sha256'] != first['checkpoint_sha256']:
        raise AssertionError('Reload used different checkpoint bytes')
    if first['evaluation_signature'] != second['evaluation_signature']:
        raise AssertionError('Reload evaluation data or protocol differs')
    identities = len(left) == len(right) and all(
        (a['image_id'], a['category_id']) == (b['image_id'], b['category_id'])
        for a, b in zip(left, right))
    metrics_equal = first['metrics'] == second['metrics']
    report = {'checkpoint_bytes_identical': True, 'evaluation_signatures_identical': True,
              'prediction_counts': [len(left), len(right)], 'row_identities_identical': identities,
              'predictions_identical': left == right, 'metrics_identical': metrics_equal,
              'passed': left == right and metrics_equal}
    if identities:
        report['max_box_absolute_difference_pixels'] = max(
            (abs(x-y) for a,b in zip(left,right) for x,y in zip(a['bbox'],b['bbox'])), default=0.)
        report['max_score_absolute_difference'] = max(
            (abs(a['score']-b['score']) for a,b in zip(left,right)), default=0.)
    return report


def smoke(args):
    # Keep evidence after Python exceptions and native crashes; never write real run folders.
    parent=Path(args.work_root)/'diagnostics'; parent.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix=f'smoke_{args.model}_',dir=parent))
    print(f'SMOKE ARTIFACTS {root.resolve()}',flush=True)
    write_json(root/'status.json',{'status':'running','model':args.model,'scientific_result':False})
    try:
        local=copy.deepcopy(args); fixture(root/'data')
        local.work_root=str(root/'work'); local.data_root=str(root/'data'); local.dataset='coco_nw'
        size=128 if args.model=='rtdetr-l' else 64  # RT-DETR needs >=300 encoder proposals.
        local.epochs=1; local.batch=2; local.eval_batch=2; local.train_size=size
        local.eval_min=size; local.eval_max=size*3//2; local.workers=0; local.resume=False; local.synthetic=True
        run=train(local,synthetic=True)
        result=read_json(run/'best_map.metrics.json')
        if result['metrics']['detection_count']==0: raise AssertionError('Smoke must exercise nonempty predictions')
        if not (run/'completed.json').is_file(): raise RuntimeError('Smoke training did not complete')
        again=evaluate_checkpoint(local,run/'best_map.pt',output=root/'reloaded',selection='best_map')
        evidence=reload_evidence(result,again)
        write_json(root/'reload_comparison.json',evidence)
        if not evidence['passed']:
            raise AssertionError(f'Checkpoint reload predictions or metrics differ; see {root}/reload_comparison.json')
        report={'status':'passed','model':args.model,'environment':environment(),'full_architecture':True,
                'training_epochs':1,'training_size':size,'shared_evaluation_sizes':[size,size*3//2],
                'checkpoint_reload_predictions_identical':True,'scientific_result':False,
                'artifacts':str(root.resolve()),'reload_comparison':evidence}
        write_json(root/'status.json',report)
        if args.output: write_json(args.output,report)
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        report={'status':'failed','model':args.model,'error':str(exc),'traceback':traceback.format_exc(),
                'artifacts':str(root.resolve()),'scientific_result':False}
        write_json(root/'status.json',report)
        if args.output: write_json(args.output,report)
        raise


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','prepare','train','evaluate','compare','smoke'])
    p.add_argument('--model',choices=list(MODELS)+LEGACY,default='yolo11m')
    p.add_argument('--dataset',choices=['coco_nw','coco_weather','acdc'],default='coco_nw')
    p.add_argument('--data-root',default='/tmp/asera/paper1_original/DATASET')
    p.add_argument('--work-root',default=DEFAULT_ROOT)
    p.add_argument('--device',default='0',help='One GPU index, or cpu')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--eval-batch',type=int,default=4)
    p.add_argument('--train-size',type=int,default=1280);p.add_argument('--eval-min',type=int,default=800)
    p.add_argument('--eval-max',type=int,default=1333);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--lgf-package',default='/tmp/asera/NEW/paper1_ugs_comparison_v1')
    p.add_argument('--level',choices=['C3','C4','C5'],default='C3')
    p.add_argument('--selection',choices=['unverified','best_map','best_ap_small'],default='unverified')
    p.add_argument('--selection-provenance',default='user must verify original selection history')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--annotations',type=Path);p.add_argument('--images',type=Path)
    p.add_argument('--reports',nargs='+',default=[])
    return p.parse_args()


def main():
    faulthandler.enable(all_threads=True)
    args=arguments()
    if args.action=='compare': compare_reports(args);return
    setup(args)
    if args.action=='check': print(json.dumps(environment(),indent=2));print('ENVIRONMENT PASS')
    elif args.action=='prepare': path,_=prepare(args);print(f'DATA PREPARE PASS {path}')
    elif args.action=='train':
        if args.split!='val' or args.annotations or args.images: raise ValueError('Training uses the prepared train/validation split only')
        train(args)
    elif args.action=='smoke': smoke(args)
    elif args.action=='evaluate':
        if not args.checkpoint: raise ValueError('--checkpoint required')
        if args.split=='test' and not (args.annotations and args.images): raise ValueError('Test evaluation needs separately supplied test annotations and images')
        result=evaluate_checkpoint(args,args.checkpoint)
        print(json.dumps(result['metrics'],indent=2))


if __name__=='__main__': main()
