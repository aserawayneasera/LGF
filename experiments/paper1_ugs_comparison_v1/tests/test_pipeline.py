import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from core_bridge import build_model, load_core, factory_without_download
from data import audit, dataset_paths, make_loaders
from evaluation import evaluate
from ugs import UGSConfig
from utils import save_checkpoint
import run


def fixture(root):
    paths=dataset_paths('coco_nw',root)
    classes=['person','bicycle','car','motorcycle','bus','truck']
    for split,n in [('train',4),('val',2)]:
        folder=paths[f'{split}_images'];folder.mkdir(parents=True)
        doc={'images':[],'annotations':[],'categories':[{'id':i+1,'name':c} for i,c in enumerate(classes)]}
        for i in range(n):
            idx=i+(100 if split=='val' else 0)
            array=np.random.default_rng(idx).integers(0,256,(64,64,3),dtype=np.uint8)
            name=f'{idx}.png';Image.fromarray(array).save(folder/name)
            doc['images'].append({'id':idx,'file_name':name,'width':64,'height':64})
            doc['annotations'].append({'id':idx,'image_id':idx,'category_id':1,'bbox':[12,12,16,16],
                                       'area':256,'iscrowd':0})
        ann=paths[f'{split}_annotations'];ann.parent.mkdir(parents=True,exist_ok=True);ann.write_text(json.dumps(doc))
    return paths


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_original_baseline_and_lgf_forward_are_preserved(self):
        core=load_core()
        for method,key in [('baseline','baseline'),('lgf_original','lgf_gated_spatial')]:
            torch.manual_seed(19)
            bridged=build_model(method,6,pretrained=False,smoke_size=64).eval()
            core.CURRENT_CONFIG=copy.deepcopy(core.CONFIG_MAP[key]);core.args.img_size=64
            with factory_without_download(core):native=core.build_model(6,'C3')
            core.convert_bn_to_gn(native,num_groups=32,convert_frozen=False)
            native.load_state_dict(bridged.state_dict(),strict=True)
            native.transform.min_size=(64,);native.transform.max_size=64;native.eval()
            bridged.score_thresh=native.score_thresh=0.0
            images=[torch.rand(3,64,64)]
            with torch.no_grad():a,b=bridged(images),native(images)
            self.assertGreater(len(a[0]['boxes']),0)
            for k in ['boxes','scores','labels']:torch.testing.assert_close(a[0][k],b[0][k],rtol=0,atol=0)
            core.args.img_size=640

    def test_audit_rejects_identical_images_across_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths=fixture(root)
            first=paths['train_images']/'0.png';target=paths['val_images']/'100.png'
            target.write_bytes(first.read_bytes())
            with self.assertRaisesRegex(ValueError,'identical'):audit('coco_nw',root)

    def test_audit_rejects_category_mapping_difference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths=fixture(root)
            doc=json.loads(paths['val_annotations'].read_text());doc['categories'][0]['id']=99
            for a in doc['annotations']:a['category_id']=99
            paths['val_annotations'].write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'category IDs'):audit('coco_nw',root)

    def test_exported_predictions_match_native_perfect_detection_ap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths=fixture(root);rec=audit('coco_nw',root)
            _,loader,_=make_loaders(rec,2,0,4,42,val_only=True)
            model=build_model('baseline',6,pretrained=False,smoke_size=64)
            def perfect(images):
                return [{'boxes':torch.tensor([[12.,12.,28.,28.]]),
                         'labels':torch.tensor([0]),'scores':torch.tensor([.9])} for _ in images]
            model.forward=perfect
            with contextlib.redirect_stdout(io.StringIO()):
                r=evaluate(model,loader,paths['val_annotations'],torch.device('cpu'),root/'eval',rec,{})
            self.assertAlmostEqual(r['metrics']['mAP'],1.,places=6)
            self.assertAlmostEqual(r['metrics']['AP_small'],1.,places=6)
            preds=json.loads((root/'eval/predictions.json').read_text())
            self.assertEqual([p['image_id'] for p in preds],[100,101])
            self.assertEqual(preds[0]['category_id'],1)
            self.assertEqual(preds[0]['bbox'],[12.,12.,16.,16.])

    def test_resnet_training_interruption_resume_and_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root/'data')
            args=SimpleNamespace(run_root=str(root/'runs'),dataset='coco_nw',method='ugs_reimpl',seed=42,
                level='C3',data_root=str(root/'data'),resume=None,batch_size=2,num_workers=0,prefetch_factor=4,
                accum_steps=2,amp=False,epochs=2,base_lr=.0001)
            core=load_core();core.set_seed(42);cfg=UGSConfig();device=torch.device('cpu')
            original_builder=build_model
            def small_builder(*a,**kw):
                kw['pretrained']=False;kw['smoke_size']=64
                return original_builder(*a,**kw)
            class Interrupted(Exception):pass
            def interrupt_after_commit(path,state):
                save_checkpoint(path,state)
                if Path(path).name=='last.pth' and state['completed_epochs']==1:raise Interrupted()
            with patch.object(run,'build_model',small_builder),patch.object(run,'save_checkpoint',interrupt_after_commit):
                with self.assertRaises(Interrupted),contextlib.redirect_stdout(io.StringIO()):
                    run.run_train(args,core,cfg,device)
            folder=root/'runs/coco_nw/ugs_reimpl/seed_42'
            args.resume=folder/'last.pth'
            with patch.object(run,'build_model',small_builder),contextlib.redirect_stdout(io.StringIO()):
                run.run_train(args,core,cfg,device)
            self.assertTrue((folder/'completed.json').is_file())
            self.assertTrue((folder/'best_ap_small.pth').is_file())
            self.assertTrue((folder/'best_map.pth').is_file())
            history=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in history],[1,2])
            ck=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
            self.assertEqual(ck['completed_epochs'],2)
            self.assertIn('UGS reimplementation',json.loads((folder/'best_map.metrics.json').read_text())['method_label'])
            model=small_builder('ugs_reimpl',6,'C3',cfg)
            model.load_state_dict(ck['ema_state_dict'],strict=True)
            model.eval()
            with torch.no_grad():out=model([torch.rand(3,64,64)])
            self.assertTrue(torch.isfinite(out[0]['boxes']).all())
            # Evaluation must use the checkpoint's recorded seed, not the CLI default.
            eval_args=SimpleNamespace(checkpoint=folder/'best_map.pth',dataset='coco_nw',
                data_root=str(root/'data'),batch_size=2,num_workers=0,prefetch_factor=4,
                seed=999,original_checkpoint=False,output=root/'reevaluated',run_root=str(root/'runs'))
            with patch.object(run,'build_model',small_builder),contextlib.redirect_stdout(io.StringIO()):
                run.run_evaluate(eval_args,cfg,device)
            self.assertEqual(json.loads((root/'reevaluated/metrics.json').read_text())['seed'],42)


if __name__=='__main__':unittest.main()
