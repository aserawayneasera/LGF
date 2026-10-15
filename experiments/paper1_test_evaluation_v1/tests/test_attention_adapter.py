"""Regression coverage for archived SE/CBAM checkpoint reconstruction.

Mocks isolate adapter arguments and strict state reload. No torch installation,
weights, downloaded data, or inference are required for these checks.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_adapter():
    path = Path(__file__).resolve().parents[1] / 'models.py'
    spec = importlib.util.spec_from_file_location('attention_adapter_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttentionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.adapter = load_adapter()
        self.args = SimpleNamespace(
            method='se', train_dataset='coco_nw', seed=42, level='C3',
            device='cpu', checkpoint=Path('synthetic.pth'), lgf_package=Path('synthetic-package'))
        self.checkpoint = {
            'package_version': '1.0.0', 'method': 'se', 'dataset': 'coco_nw',
            'seed': 42, 'level': 'C3', 'completed_epochs': 5,
            'ugs_config': {}, 'config': {'block_type': 'se', 'reduction': 16},
            'ema_state_dict': {'attention.weight': 'synthetic tensor'},
        }
        self.model = Mock()
        self.model.to.return_value.float.return_value.eval.return_value = self.model
        self.model.transform = SimpleNamespace(min_size=[800], max_size=1333)
        self.model.score_thresh, self.model.nms_thresh, self.model.detections_per_img = .05, .5, 100
        self.model.parameters.return_value = [SimpleNamespace(numel=lambda: 123)]
        self.build = Mock(return_value=self.model)
        torch = ModuleType('torch')
        torch.load = Mock(side_effect=lambda *args, **kwargs: self.checkpoint)
        torch.device = lambda value: value
        bridge = ModuleType('core_bridge')
        bridge.build_model = self.build
        ugs = ModuleType('ugs')
        ugs.UGSConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        self.modules = {'torch': torch, 'core_bridge': bridge, 'ugs': ugs}

    def load(self):
        old_path = sys.path[:]
        try:
            with patch.dict(sys.modules, self.modules), patch.object(self.adapter, 'configure'):
                return self.adapter.load_model(self.args, [])
        finally:
            sys.path[:] = old_path

    def test_extension_attention_config_reaches_archived_builder(self):
        for method in ('se', 'cbam'):
            with self.subTest(method=method):
                self.args.method = self.checkpoint['method'] = method
                self.checkpoint['config']['block_type'] = method
                self.build.reset_mock()
                model, device, metadata, params, _ = self.load()
                self.assertIs(model, self.model)
                self.assertEqual(device, 'cpu')
                self.assertEqual(metadata['epoch'], 5)
                self.assertEqual(params, 123)
                self.assertEqual(self.build.call_args.kwargs,
                                 {'pretrained': False, 'original_config': self.checkpoint['config']})
                self.model.load_state_dict.assert_called_with(self.checkpoint['ema_state_dict'], strict=True)

    def test_attention_config_must_match_declared_method(self):
        for config in ({}, {'block_type': 'none'}, {'block_type': 'cbam'}):
            with self.subTest(config=config):
                self.checkpoint['config'] = config
                self.build.reset_mock()
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    self.load()
                self.build.assert_not_called()

    def test_checkpoint_identity_is_checked_before_construction(self):
        self.checkpoint['seed'] = 2025
        with self.assertRaisesRegex(ValueError, 'method/dataset/seed/level differs'):
            self.load()
        self.build.assert_not_called()

    def test_other_packaged_models_keep_original_builder_arguments(self):
        for method in ('baseline', 'lgf_original', 'ugs_reimpl'):
            with self.subTest(method=method):
                self.args.method = self.checkpoint['method'] = method
                self.build.reset_mock()
                self.load()
                self.assertEqual(self.build.call_args.kwargs, {'pretrained': False})


if __name__ == '__main__':
    unittest.main()
