"""Opt-in real-architecture, full-evaluation-size adapter checks."""
import copy
from dataclasses import asdict
import gc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import models
import test_data as td

@unittest.skipUnless(os.environ.get('P1_REAL_MODEL_TESTS')=='1','Set P1_REAL_MODEL_TESTS=1 for full-size CPU adapter checks')
class RealModelTests(unittest.TestCase):
    def test_all_seven_checkpoint_adapters_at_800_1333(self):
        import torch
        from PIL import Image
        torch.set_num_threads(4)
        package=Path(__file__).resolve().parents[2]/'paper1_ugs_comparison_v1'
        recent_package=package.parent/'paper1_reviewer34_v1'
        sys.path.insert(0,str(package));import core_bridge
        from ugs import UGSConfig
        core=core_bridge.load_core();core.args.img_size=64
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'image.png';Image.new('RGB',(64,40),(150,60,20)).save(path)
            record=td.image_record(path,path.name,1,'coco_nw','clear')
            image=models.rgb_tensors([record],root,torch.device('cpu'))
            for method in models.METHODS:
                with self.subTest(method=method):
                    torch.manual_seed(42)
                    args=SimpleNamespace(method=method,train_dataset='coco_nw',seed=42,device='cpu',level='C3',threads=4,batch_size=1,
                        lgf_package=package,recent_package=recent_package,runtime_root=root/'runtime',checkpoint=root/f'{method}.pt')
                    if method in models.LEGACY:
                        cfg=copy.deepcopy(core.CONFIG_MAP[method]) if method in ['se','cbam'] else None
                        model=core_bridge.build_model(method,6,original_config=cfg,pretrained=False).eval()
                        with torch.no_grad():model.head.classification_head.cls_logits.bias.fill_(2.)
                        if method=='ugs_reimpl':
                            ck={'package_version':'1.0.0','method':method,'dataset':'coco_nw','seed':42,'level':'C3',
                                'ugs_config':asdict(UGSConfig()),'completed_epochs':5,'ema_state_dict':model.state_dict()}
                        else:
                            ck={'epoch':4,'config':copy.deepcopy(core.CURRENT_CONFIG),'ema_state_dict':model.state_dict(),
                                'experiment_name':f'coco_nw_C3_{method}_s42'}
                        with torch.no_grad():expected=model(image)
                        torch.save(ck,args.checkpoint);del model,ck;gc.collect()
                    else:
                        os.environ['YOLO_CONFIG_DIR']=str(root/'ultra_settings')
                        from ultralytics.nn.tasks import DetectionModel,RTDETRDetectionModel
                        native=DetectionModel('yolo11m.yaml',nc=6,verbose=False) if method=='yolo11m' else RTDETRDetectionModel('rtdetr-l.yaml',nc=6,verbose=False)
                        native.names=dict(enumerate(td.CLASSES));native.eval()
                        with torch.no_grad():
                            if method=='yolo11m':
                                for branch in native.model[-1].cv3:branch[-1].bias.fill_(2.)
                            else:
                                for head in native.model[-1].dec_score_head:head.bias.fill_(2.)
                                native.model[-1].enc_score_head.bias.fill_(2.)
                        torch.save({'model':native,'ema':None,'train_args':{'task':'detect','imgsz':1280},'version':'8.3.253',
                            'reviewer34':{'model':method,'dataset':'coco_nw','seed':42,'epoch':5,'label':method}},args.checkpoint)
                        del native;gc.collect();expected=None
                    model,device,meta,params,infer=models.load_model(args,td.CLASSES)
                    predictions=infer(image)
                    self.assertGreater(len(predictions[0]['boxes']),0)
                    self.assertLessEqual(len(predictions[0]['boxes']),100)
                    self.assertEqual(meta['epoch'],5);self.assertGreater(params,1_000_000)
                    if expected is not None:
                        for key in ['boxes','scores','labels']:torch.testing.assert_close(predictions[0][key],expected[0][key],rtol=0,atol=0)
                    print('REAL ADAPTER PASS',method,'detections=',len(predictions[0]['boxes']),flush=True)
                    del model,infer,predictions,expected;gc.collect();args.checkpoint.unlink()

if __name__=='__main__':unittest.main()
