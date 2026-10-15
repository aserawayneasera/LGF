"""Read-only adapters for the previously delivered detector implementations."""
from pathlib import Path
import importlib.util
import os
import re
import sys
from types import SimpleNamespace

METHODS = ['baseline','lgf_original','se','cbam','ugs_reimpl','yolo11m','rtdetr-l']
LEGACY = METHODS[:5]

def load_recent(package):
    path = Path(package)/'reviewer34_compare.py'
    spec=importlib.util.spec_from_file_location('paper1_recent_unmodified',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def configure(args):
    os.environ.setdefault('TORCH_HOME','/tmp/asera/paper1_original/torch_cache')
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch,torchvision
    if not torch.__version__.startswith('2.10.') or not torchvision.__version__.startswith('0.25.'):
        raise RuntimeError('Use torch 2.10.x and torchvision 0.25.x, as in the delivered runners')
    if args.device!='cpu' and not args.device.isdigit():raise ValueError('Use --device 0 or --device cpu')
    if args.device!='cpu' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if os.environ.get('MAX_TRAIN_IMAGES'):raise ValueError('Unset MAX_TRAIN_IMAGES')
    if args.batch_size<1 or args.threads<1:raise ValueError('Batch/threads must be positive')
    torch.set_num_threads(args.threads)

def load_model(args, class_names):
    import torch
    configure(args)
    device=torch.device('cpu' if args.device=='cpu' else 'cuda:'+args.device)
    if args.method not in LEGACY:
        recent=load_recent(args.recent_package)
        native=SimpleNamespace(model=args.method,dataset=args.train_dataset,seed=args.seed,device=args.device,
            work_root=args.runtime_root,threads=args.threads,batch=args.batch_size,eval_batch=args.batch_size,
            train_size=1280,eval_min=800,eval_max=1333,epochs=80,workers=0,lgf_package=args.lgf_package,level=args.level)
        recent.setup(native)
        model,device,meta,params=recent.load_detector(native,args.checkpoint)
        if {int(k):v for k,v in model.names.items()} != dict(enumerate(class_names)):
            raise ValueError('Checkpoint classes differ from the original training JSON')
        return model,device,meta,params,lambda images:recent.infer_batch(model,images,native)[0]
    sys.path.insert(0,str(Path(args.lgf_package).resolve()))
    from core_bridge import build_model
    from ugs import UGSConfig
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if 'ema_state_dict' not in ck:raise ValueError('EMA state is required for original-method evaluation')
    if ck.get('package_version'):
        if (ck['method'],ck['dataset'],ck['seed'],ck['level']) != (args.method,args.train_dataset,args.seed,args.level):
            raise ValueError('Checkpoint method/dataset/seed/level differs')
        build_options={'pretrained':False}
        if args.method in ('se','cbam'):
            # Extension checkpoints retain the archived attention configuration.
            # The native bridge otherwise selects its baseline fallback for both
            # method names, producing a different state schema at strict reload.
            cfg=ck.get('config',{})
            if cfg.get('block_type')!=args.method:
                raise ValueError('SE/CBAM checkpoint configuration differs from --method')
            build_options['original_config']=cfg
        model=build_model(args.method,6,args.level,UGSConfig(**ck['ugs_config']),**build_options)
        epoch=ck['completed_epochs']
    else:
        if args.method=='ugs_reimpl':raise ValueError('UGS requires a UGS-package checkpoint')
        cfg=ck.get('config',{})
        expected={'baseline':'none','lgf_original':'lgf','se':'se','cbam':'cbam'}[args.method]
        if cfg.get('block_type')!=expected:raise ValueError('Original checkpoint architecture differs from --method')
        if args.method=='lgf_original' and cfg.get('gating_type')!='gated_spatial':raise ValueError('Expected the proposed original spatial-gated LGF')
        identity=re.fullmatch(r'(coco_nw|coco_weather|acdc)_(C[345])_.+_s(-?\d+)',str(ck.get('experiment_name','')))
        if not identity or (identity[1],identity[2],int(identity[3]))!=(args.train_dataset,args.level,args.seed):
            raise ValueError('Original checkpoint needs matching experiment_name metadata')
        model=build_model('baseline',6,args.level,pretrained=False,original_config=cfg)
        epoch=ck.get('epoch',-1)+1
    state=ck['ema_state_dict']
    if state and all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model.to(device).float().eval()
    if (list(model.transform.min_size),model.transform.max_size)!=( [800],1333):raise ValueError('Original transform differs from 800/1333')
    if (model.score_thresh,model.nms_thresh,model.detections_per_img)!=(.05,.5,100):raise ValueError('Original prediction settings differ')
    labels={'baseline':'RetinaNet baseline','lgf_original':'LGF (original spatial-gated model)','se':'RetinaNet + SE',
            'cbam':'RetinaNet + CBAM','ugs_reimpl':'UGS (our RetinaNet reimplementation)'}
    meta={'model':args.method,'label':labels[args.method],'dataset':args.train_dataset,'seed':args.seed,'epoch':epoch,
          'training_protocol':ck.get('protocol'),'original_config':ck.get('config'),'ugs_config':ck.get('ugs_config')}
    def infer(images):
        with torch.no_grad():return model(images)
    return model,device,meta,sum(p.numel() for p in model.parameters()),infer

def rgb_tensors(records, image_root, device):
    import cv2,numpy as np,torch
    images=[]
    for im in records:
        frame=cv2.imread(str(Path(image_root)/im['file_name']))
        if frame is None:raise ValueError('Unreadable test image: '+im['file_name'])
        if frame.shape[:2]!=(im['height'],im['width']):raise ValueError('Test image dimensions changed')
        rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).transpose(2,0,1))
        images.append(torch.from_numpy(rgb).to(device=device,dtype=torch.float32)/255.)
    return images
