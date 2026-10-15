"""Regression checks for the trainer teardown / checkpoint reload boundary."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
import torch
from ultralytics.utils.torch_utils import unset_deterministic
import reviewer34_compare as r


class InferenceSettingsTests(unittest.TestCase):
    def state(self):
        return (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
                os.environ.get('CUBLAS_WORKSPACE_CONFIG'))

    def test_trainer_teardown_does_not_change_inference_policy(self):
        initial=self.state()
        try:
            torch.use_deterministic_algorithms(True,warn_only=True)
            torch.backends.cudnn.deterministic=True
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            with r.fixed_inference_settings(): during_training=self.state()
            unset_deterministic()
            after_teardown=self.state()
            with r.fixed_inference_settings(): after_training=self.state()
            self.assertEqual(during_training,after_training)
            self.assertEqual(after_training,(True,False,True,False,':4096:8'))
            self.assertEqual(self.state(),after_teardown)
        finally:
            torch.use_deterministic_algorithms(initial[0],warn_only=initial[1])
            torch.backends.cudnn.deterministic=initial[2]
            torch.backends.cudnn.benchmark=initial[3]
            if initial[4] is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
            else:os.environ['CUBLAS_WORKSPACE_CONFIG']=initial[4]

    def test_exception_restores_caller_settings(self):
        initial=self.state()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with r.fixed_inference_settings():raise RuntimeError('fixture')
        self.assertEqual(self.state(),initial)


class ReloadGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        ck=self.root/'model.pt';ck.write_bytes(b'fixture checkpoint bytes')
        pred=[{'image_id':1,'category_id':1,'bbox':[1.,2.,3.,4.],'score':.9}]
        self.a={'prediction_file':str(self.root/'a.json'),'checkpoint':str(ck),
                'checkpoint_sha256':r.sha256(ck),'evaluation_signature':'same',
                'metrics':{'mAP':.25,'detection_count':1}}
        self.b=copy.deepcopy(self.a);self.b['prediction_file']=str(self.root/'b.json')
        r.write_json(self.a['prediction_file'],pred);r.write_json(self.b['prediction_file'],pred)

    def tearDown(self):self.tmp.cleanup()

    def test_identical_predictions_and_metrics_pass(self):
        self.assertTrue(r.reload_evidence(self.a,self.b)['passed'])

    def test_tiny_drift_is_reported_and_not_silently_accepted(self):
        pred=r.read_json(self.b['prediction_file']);pred[0]['score']+=1e-8
        r.write_json(self.b['prediction_file'],pred)
        result=r.reload_evidence(self.a,self.b)
        self.assertFalse(result['passed'])
        self.assertGreater(result['max_score_absolute_difference'],0.)

    def test_changed_class_rejected(self):
        pred=r.read_json(self.b['prediction_file']);pred[0]['category_id']=2
        r.write_json(self.b['prediction_file'],pred)
        result=r.reload_evidence(self.a,self.b)
        self.assertFalse(result['passed']);self.assertFalse(result['row_identities_identical'])

    def test_different_checkpoint_rejected(self):
        self.b['checkpoint_sha256']='changed'
        with self.assertRaisesRegex(AssertionError,'different checkpoint'):r.reload_evidence(self.a,self.b)

    def test_changed_evaluation_protocol_rejected(self):
        self.b['evaluation_signature']='changed'
        with self.assertRaisesRegex(AssertionError,'protocol differs'):r.reload_evidence(self.a,self.b)

    def test_changed_metrics_rejected(self):
        self.b['metrics']['mAP']=.3
        self.assertFalse(r.reload_evidence(self.a,self.b)['passed'])


if __name__=='__main__':unittest.main()
