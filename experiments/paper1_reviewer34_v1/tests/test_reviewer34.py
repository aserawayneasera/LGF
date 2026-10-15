"""Synthetic software checks. Fixture scores are not research results."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import reviewer34_compare as r


def args(root):
    return SimpleNamespace(dataset='coco_nw',data_root=str(root/'data'),work_root=str(root/'work'),
        annotations=None,images=None,eval_min=64,eval_max=96,eval_batch=2,split='val',model='yolo11m',
        device='cpu',seed=42,selection='best_map',selection_provenance='synthetic fixture',output=None,
        batch=2,epochs=2,train_size=64,workers=0,threads=2,resume=False,synthetic=True)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.a=args(self.root)
        r.fixture(self.root/'data');self.paths=r.dataset_paths('coco_nw',self.root/'data')
    def tearDown(self):self.tmp.cleanup()

    def test_noncontiguous_category_ids_and_box_roundtrip(self):
        for split in ['train','val']:
            path=self.paths[split+'_annotations'];doc=r.read_json(path)
            for c in doc['categories']:c['id']*=10
            for a in doc['annotations']:a['category_id']*=10
            r.write_json(path,doc)
        yaml,rec=r.prepare(self.a)
        values=list(map(float,(yaml.parent/'labels/train/0.txt').read_text().split()))
        self.assertEqual(values[0],0)
        cx,cy,w,h=values[1:]
        self.assertAlmostEqual((cx-w/2)*64,12);self.assertAlmostEqual(w*64,20)
        self.assertEqual(rec['val']['categories'][0]['id'],10)
        self.assertTrue((yaml.parent/'images/train/0.png').is_symlink())

    def test_original_small_box_filter_and_crowd_inclusion(self):
        path=self.paths['train_annotations'];doc=r.read_json(path)
        doc['annotations'][0]['iscrowd']=1
        doc['annotations'].append({'id':999,'image_id':0,'category_id':1,'bbox':[1,1,1,10],'area':10,'iscrowd':0})
        r.write_json(path,doc);yaml,_=r.prepare(self.a)
        record=r.read_json(yaml.parent/'conversion.json')['target_handling']['train']
        self.assertEqual(record['boxes_excluded_w_or_h_le_1_like_original'],1)
        self.assertEqual(record['crowd_boxes_included_as_training_targets_like_original'],1)

    def test_duplicate_targets_rejected(self):
        path=self.paths['train_annotations'];doc=r.read_json(path)
        item=copy.deepcopy(doc['annotations'][0]);item['id']=999;doc['annotations'].append(item);r.write_json(path,doc)
        with self.assertRaisesRegex(ValueError,'Duplicate box'):r.prepare(self.a)

    def test_outside_targets_rejected(self):
        path=self.paths['train_annotations'];doc=r.read_json(path);doc['annotations'][0]['bbox'][0]=-1;r.write_json(path,doc)
        with self.assertRaisesRegex(ValueError,'Out-of-bounds'):r.prepare(self.a)

    def test_changed_labels_rejected_on_reuse(self):
        yaml,_=r.prepare(self.a);(yaml.parent/'labels/train/0.txt').write_text('0 0.5 0.5 0.1 0.1\n')
        with self.assertRaisesRegex(ValueError,'Prepared files changed'):r.prepare(self.a)

    def test_identical_scene_bytes_rejected(self):
        doc=r.read_json(self.paths['val_annotations']);doc['images'][0]['width']=64;r.write_json(self.paths['val_annotations'],doc)
        (self.paths['val_images']/'100.png').write_bytes((self.paths['train_images']/'0.png').read_bytes())
        with self.assertRaisesRegex(ValueError,'Identical image'):r.audit(self.a)

    def test_dimension_disagreement_rejected(self):
        doc=r.read_json(self.paths['val_annotations']);doc['images'][0]['width']=99;r.write_json(self.paths['val_annotations'],doc)
        with self.assertRaisesRegex(ValueError,'dimensions differ'):r.audit(self.a)

    def test_perfect_and_empty_predictions_use_coco(self):
        doc=r.read_json(self.paths['val_annotations'])
        preds=[{'image_id':a['image_id'],'category_id':a['category_id'],'bbox':a['bbox'],'score':.9} for a in doc['annotations']]
        result=r.coco_metrics(self.paths['val_annotations'],preds)
        self.assertAlmostEqual(result['mAP'],1);self.assertAlmostEqual(result['AP_small'],1)
        empty=r.coco_metrics(self.paths['val_annotations'],[])
        self.assertEqual(empty['mAP'],0);self.assertEqual(empty['AP_small'],0)
        self.assertEqual(empty['AP_large'],-1)


class GeometryTests(unittest.TestCase):
    def test_resize_matches_torchvision_geometry_and_interpolation(self):
        images=[torch.rand(3,67,103),torch.rand(3,111,63)]
        batch,shapes=r.shared_resize(images,64,96)
        transform=GeneralizedRCNNTransform(64,96,[0,0,0],[1,1,1]).eval()
        expected,_=transform(images)
        torch.testing.assert_close(batch,expected.tensors,rtol=0,atol=0)
        self.assertEqual([s[2:] for s in shapes],list(expected.image_sizes))

    def test_boxes_return_to_original_coordinates(self):
        original=torch.tensor([[10.,20.,70.,80.]])
        scaled=original.clone();scaled[:,[0,2]]*=96/150;scaled[:,[1,3]]*=64/100
        decoded=r.unscale_boxes(scaled,(100,150,64,96))
        torch.testing.assert_close(decoded,original)


class EvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);r.fixture(self.root/'data')
        self.a=args(self.root);self.rec=r.audit(self.a,evaluation_only=True)
        self.ck=self.root/'best_map.pt';self.ck.write_bytes(b'fixture')
    def tearDown(self):self.tmp.cleanup()
    def call(self,meta):
        with patch.object(r,'load_detector',return_value=(None,torch.device('cpu'),meta,10)):
            return r.evaluate_checkpoint(self.a,self.ck,self.rec,self.root/'out')
    def test_wrong_checkpoint_selection_label_rejected(self):
        self.a.selection='best_ap_small'
        with self.assertRaisesRegex(ValueError,'Selection argument differs'):self.call({})
    def test_checkpoint_validation_data_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'Validation files differ'):
            self.call({'training_protocol':{'val_content_sha256':'different'}})
    def test_validation_images_cannot_be_relabelled_as_test(self):
        self.a.split='test'
        with self.assertRaisesRegex(ValueError,'Test images include identical bytes'):self.call({})


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.rows=[]
        checkpoint=self.root/'fixture.pt';checkpoint.write_bytes(b'fixture checkpoint')
        preds=self.root/'predictions.json';preds.write_text('[]')
        for model in ['LGF fixture','YOLO fixture']:
            for i,seed in enumerate([42,1337,2025]):
                self.rows.append({'label':model,'dataset':'coco_nw','seed':seed,'scientific_result':True,
                    'checkpoint':str(checkpoint),'checkpoint_sha256':r.sha256(checkpoint),'prediction_file':str(preds),
                    'prediction_sha256':r.sha256(preds),'evaluation_signature':'synthetic-common-signature',
                    'selection':'best_map','parameter_count':10,'evaluation_protocol':{'evaluation_split':'val'},
                    'training_protocol':{'seed':seed,'recipe':{'seed':seed,'project':str(seed),'epochs':80}},
                    'metrics':{m:.1*(i+1) for m in r.METRICS[:6]}})
    def tearDown(self):self.tmp.cleanup()
    def call(self):
        paths=[]
        for i,item in enumerate(self.rows):
            p=self.root/f'report_{i}.json';r.write_json(p,item);paths.append(str(p))
        a=SimpleNamespace(reports=paths,output=self.root/'out',work_root=str(self.root))
        with contextlib.redirect_stdout(io.StringIO()):r.compare_reports(a)
        return r.read_json(a.output/'comparison.json')
    def test_paired_seed_summary(self):
        for row in self.call():
            self.assertEqual(row['n'],3);self.assertAlmostEqual(row['mAP_mean'],.2);self.assertAlmostEqual(row['mAP_std'],.1)
    def test_mismatched_selection_rejected(self):
        self.rows[-1]['selection']='best_ap_small'
        with self.assertRaisesRegex(ValueError,'selection rule'):self.call()
    def test_mismatched_evaluation_rejected(self):
        self.rows[-1]['evaluation_signature']='different'
        with self.assertRaisesRegex(ValueError,'protocol differ'):self.call()
    def test_mismatched_seeds_rejected(self):
        self.rows[-1]['seed']=77
        with self.assertRaisesRegex(ValueError,'matching seeds'):self.call()
    def test_changed_prediction_bytes_rejected(self):
        Path(self.rows[0]['prediction_file']).write_text('[1]')
        with self.assertRaisesRegex(ValueError,'Prediction bytes changed'):self.call()
    def test_synthetic_smoke_not_accepted_as_research(self):
        self.rows[-1]['scientific_result']=False
        with self.assertRaisesRegex(ValueError,'Synthetic results'):self.call()


class LegacyAdapterTests(unittest.TestCase):
    def test_original_lgf_checkpoint_prediction_export_matches_direct_forward(self):
        package=Path(os.environ.get('P1_UGS_PACKAGE','/tmp/asera/NEW/paper1_ugs_comparison_v1'))
        if not (package/'core_bridge.py').is_file():self.skipTest('Set P1_UGS_PACKAGE to the earlier package')
        sys.path.insert(0,str(package));import core_bridge
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r.fixture(root/'data');a=args(root);a.synthetic=False;a.model='lgf_original';a.level='C3';a.lgf_package=str(package)
            builder=core_bridge.build_model
            def small(*xs,**kw):kw['smoke_size']=64;return builder(*xs,**kw)
            model=small('lgf_original',6,pretrained=False).eval()
            with torch.no_grad():model.head.classification_head.cls_logits.bias.fill_(2.)
            checkpoint=root/'original.pth'
            r.save_torch(checkpoint,{'epoch':4,'config':copy.deepcopy(core_bridge.load_core().CURRENT_CONFIG),
                'ema_state_dict':model.state_dict(),'experiment_name':'coco_nw_C3_lgf_gated_spatial_s42'})
            # The synthetic constructor uses max_size=64; evaluation must match.
            a.eval_max=64
            rec=r.audit(a,evaluation_only=True);doc=r.read_json(rec['paths']['val_annotations'])
            items=sorted(doc['images'],key=lambda x:x['id']);images=r.read_images(items,rec['paths']['val_images'],torch.device('cpu'))
            with torch.no_grad():direct=model(images)
            expected=[]
            for im,p in zip(items,direct):
                for b,s,c in zip(p['boxes'].numpy(),p['scores'].numpy(),p['labels'].numpy()):
                    expected.append({'image_id':im['id'],'category_id':int(c)+1,'bbox':[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],'score':float(s)})
            self.assertGreater(len(expected),0)
            with patch.object(core_bridge,'build_model',small),contextlib.redirect_stdout(io.StringIO()):
                result=r.evaluate_checkpoint(a,checkpoint,rec,root/'eval')
            self.assertEqual(r.read_json(result['prediction_file']),expected)
            self.assertEqual(result['epoch'],5)


class ResumeTests(unittest.TestCase):
    def test_native_epoch_checkpoint_resumes_and_selects_shared_coco_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r.fixture(root/'data');a=args(root);r.setup(a)
            cls=r.native_class
            class Interrupted(Exception):pass
            def constructor(which):
                def build(path):
                    model=cls(which)(path)
                    def stop(trainer):
                        if trainer.epoch==0:raise Interrupted()
                    model.add_callback('on_fit_epoch_end',stop)
                    return model
                return build
            with patch.object(r,'native_class',constructor),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(Interrupted):r.train(a,synthetic=True)
            a.resume=True
            with contextlib.redirect_stdout(io.StringIO()):folder=r.train(a,synthetic=True)
            self.assertEqual(r.read_json(folder/'completed.json')['epochs'],2)
            self.assertEqual(r.read_json(folder/'best_map.metrics.json')['epoch'],2)
            self.assertEqual(len(r.read_json(folder/'selection_history.json')),1)


if __name__=='__main__':unittest.main()
