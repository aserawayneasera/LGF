"""Numerical and integration checks. No research dataset or trained weights used."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import controls
from gate_controls import apply_control, native_weights, names, parameter_cost
controls.bootstrap_package(ROOT/'source_runner')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from core_bridge import build_model, load_core
from data import audit, dataset_paths
from evaluation import evaluate, model_protocol


class ControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.core = load_core()

    def block(self):
        torch.manual_seed(7)
        b = self.core.StableLGFBlock(64, gating_type='gated_spatial').eval()
        with torch.no_grad():
            b.gate_expand.weight.normal_(0, 0.08)
            b.gate_expand.bias.normal_(0, 0.08)
            b.gamma.fill_(0.17)
        return b

    def test_native_is_exact_and_restores_after_exception(self):
        b = self.block()
        x = torch.randn(2, 64, 9, 11)
        state = controls.state_digest(b)
        with torch.no_grad():
            expected = b(x)
            with apply_control(b, 'native', verify_native=True) as evidence:
                torch.testing.assert_close(b(x), expected, rtol=0, atol=0)
            self.assertEqual(evidence['verified_blocks'], [''])
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                with apply_control(b, 'uniform'):
                    raise RuntimeError('interruption')
            torch.testing.assert_close(b(x), expected, rtol=0, atol=0)
        self.assertEqual(state, controls.state_digest(b))
        self.assertNotIn('forward', b.__dict__)

    def test_paper_fixed_controls_use_exact_equations_and_skip_gate_head(self):
        b = self.block()
        x = torch.randn(2, 64, 9, 11)
        with torch.no_grad():
            local, global_ = b.local(x), b.global_branch(x).expand_as(x)
            controls_expected = {'uniform': 0.5*local + 0.5*global_, 'no_gate': local+global_}
            controls_expected.update({f'alpha_{a}': float(a)*local+global_ for a in ('0.25', '0.5', '2.0')})
            def forbidden(*_):
                raise AssertionError('Fixed controls must not execute learned gate head')
            hook = b.gate_reduce.register_forward_pre_hook(forbidden)
            try:
                for control, fusion in controls_expected.items():
                    with apply_control(b, control):
                        torch.testing.assert_close(b(x), x+b.gamma*fusion, rtol=0, atol=0)
            finally:
                hook.remove()

    def test_swap_and_shuffle_match_independent_tensor_formulas(self):
        b = self.block()
        x = torch.randn(2, 64, 9, 11)
        with torch.no_grad():
            l, g = b.local(x), b.global_branch(x).expand_as(x)
            wl, wg = native_weights(b, x)
            with apply_control(b, 'swap'):
                torch.testing.assert_close(b(x), x+b.gamma*(wg*l+wl*g), rtol=0, atol=0)
            perm = torch.randperm(128, generator=torch.Generator().manual_seed(123))
            pl = wl.reshape(128, 9, 11)[perm].reshape_as(wl)
            pg = wg.reshape(128, 9, 11)[perm].reshape_as(wg)
            expected = x+b.gamma*(pl*l+pg*g)
            with apply_control(b, 'shuffle'):
                torch.testing.assert_close(b(x), expected, rtol=0, atol=0)
            with apply_control(b, 'shuffle'):
                torch.testing.assert_close(b(x), expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(expected, b(x)))

    def test_cost_counts_gate_head_only(self):
        b = self.block()
        cost = parameter_cost(b)
        branch_count = sum(p.numel() for p in b.local.parameters())
        branch_count += sum(p.numel() for p in b.global_branch.parameters()) + b.gamma.numel()
        self.assertEqual(cost['parameters_if_gate_head_deleted'], branch_count)
        self.assertEqual(cost['gate_parameters'], 64*16+16*128+128+2*128+1)

    def test_reject_other_architectures_and_wrong_suite(self):
        with self.assertRaises(ValueError):
            with apply_control(nn.Linear(3, 3), 'native'):
                pass
        with self.assertRaises(ValueError):
            with apply_control(self.block(), 'no_module', 'paper'):
                pass
        wrong = self.core.StableLGFBlock(64, gating_type='sum')
        with self.assertRaises(ValueError):
            with apply_control(wrong, 'native'):
                pass

    def test_notebook_matches_archived_functions(self):
        # Functions extracted verbatim from the user-supplied original notebook.
        path = ROOT/'tests/legacy_reference.py'
        namespace = {'torch': torch, 'F': F, 'types': types}
        exec(compile(path.read_text(), str(path), 'exec'), namespace)
        x = torch.randn(2, 64, 9, 11)
        for control in names('notebook'):
            a = self.block()
            b = copy.deepcopy(a)
            kwargs = {'override': {'swap': 'swap_lg'}.get(control, control), 'gate_seed': 123}
            if control.startswith('alpha_'):
                kwargs = {'override': 'native', 'alpha_local': float(control.split('_')[1]), 'gate_seed': 123}
            namespace['_patch_gate_override'](a, **kwargs)
            with torch.no_grad(), apply_control(b, control, 'notebook'):
                torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)

    def test_statistics_preserve_negative_deltas_and_missing_metrics(self):
        avg, sd, n = controls.mean_std([-1., 0., 1.])
        self.assertEqual((avg, sd, n), (0., 1., 3))
        self.assertEqual(controls.mean_std([None, 0.2]), (0.2, None, 1))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        paths = dataset_paths('coco_nw', root)
        images = paths['val_images']
        images.mkdir(parents=True)
        classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        doc = {'images': [], 'annotations': [],
               'categories': [{'id': i+1, 'name': name} for i, name in enumerate(classes)]}
        for i in range(2):
            Image.fromarray(np.random.default_rng(i).integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(images/f'{i}.png')
            doc['images'].append({'id': i, 'file_name': f'{i}.png', 'height': 64, 'width': 80})
            doc['annotations'].append({'id': i, 'image_id': i, 'category_id': 1,
                                       'bbox': [8, 8, 16, 16], 'area': 256, 'iscrowd': 0})
        ann = paths['val_annotations']
        ann.parent.mkdir(parents=True)
        ann.write_text(json.dumps(doc))
        return audit('coco_nw', root, val_only=True)

    def test_real_retinanet_checkpoint_grid_resume_and_rejections(self):
        import core_bridge
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = self.fixture(root/'data')
            model = build_model('lgf_original', 6, 'C3', pretrained=False, smoke_size=64).eval()
            with torch.no_grad():
                model.head.classification_head.cls_logits.bias.fill_(0.0)
                for _, b in __import__('gate_controls').blocks(model):
                    b.gate_expand.weight.normal_(0, .03)
                    b.gate_expand.bias.normal_(0, .03)
            original = copy.deepcopy(load_core().CURRENT_CONFIG)
            checkpoint = root/'runs/coco_nw/lgf_original/seed_42/best_ap_small.pth'
            checkpoint.parent.mkdir(parents=True)
            ck = {'package_version': '1.0.0', 'method': 'lgf_original', 'dataset': 'coco_nw',
                  'seed': 42, 'level': 'C3', 'config': original, 'completed_epochs': 5,
                  'ema_state_dict': model.state_dict(), 'improved_selection_keys': ['best_ap_small'],
                  'protocol': {'validation_content_sha256': record['val']['content_sha256'],
                               'model_protocol': model_protocol(model), 'batch_size': 2}}
            torch.save(ck, checkpoint)
            weights_hash = controls.sha256(checkpoint)
            args = ['run', '--run-root', str(root/'runs'), '--data-root', str(root/'data'),
                    '--output-root', str(root/'out'), '--datasets', 'coco_nw', '--seeds', '42',
                    '--device', 'cpu', '--batch-size', '2', '--threads', '2']
            real_build = core_bridge.build_model
            def small_build(*a, **kw):
                kw['smoke_size'] = 64
                return real_build(*a, **kw)
            with patch.object(core_bridge, 'build_model', small_build), contextlib.redirect_stdout(io.StringIO()):
                controls.main(['check', *args[1:]])
                controls.main(args)
                # Completed controls must be reused without another evaluation.
                with patch('evaluation.evaluate', side_effect=AssertionError('Should resume')):
                    controls.main(args)
            self.assertEqual(controls.sha256(checkpoint), weights_hash)
            summary = json.loads((root/'out/summary_status.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['rows'], 8)
            pred = root/'out/coco_nw/seed_42/native/predictions.json'
            self.assertGreater(len(json.loads(pred.read_text())), 0)
            controls.inspect_checkpoint(checkpoint, 'coco_nw', 42, 'C3', 'best_ap_small')
            with self.assertRaises(ValueError):
                controls.inspect_checkpoint(checkpoint, 'acdc', 42, 'C3', 'best_ap_small')
            bad = dict(ck, method='ugs_reimpl')
            badpath = root/'bad/best_ap_small.pth'
            badpath.parent.mkdir()
            torch.save(bad, badpath)
            with self.assertRaises(ValueError):
                controls.inspect_checkpoint(badpath, 'coco_nw', 42, 'C3', 'best_ap_small')
            # Refuse changed prediction files on aggregation and resume.
            pred.write_text('[]')
            with self.assertRaises(ValueError):
                controls.summarize(root/'out')

    def test_original_notebook_checkpoint_requires_its_selection_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'BEST_coco_nw_C3_lgf_gated_spatial_s42_epoch_5_map_0.2.pth'
            cfg = dict(load_core().CONFIG_MAP['lgf_gated_spatial'])
            torch.save({'config': cfg, 'experiment_name': 'coco_nw_C3_lgf_gated_spatial_s42',
                        'ema_state_dict': {'dummy': torch.tensor(1.)}, 'epoch': 4}, path)
            _, _, meta = controls.inspect_checkpoint(path, 'coco_nw', 42, 'C3', 'best_map')
            self.assertEqual(meta['epoch'], 5)
            with self.assertRaises(ValueError):
                controls.inspect_checkpoint(path, 'coco_nw', 42, 'C3', 'best_ap_small')

    def test_reference_guard_catches_native_metric_difference(self):
        ck = {'protocol': {'batch_size': 4}, 'epoch_result': {'metrics': {'mAP': .2, 'AP_small': .1}}}
        result = {'metrics': {'mAP': .3, 'AP_small': .1}}
        with self.assertRaises(RuntimeError):
            controls.check_reference(ck, result, 4, 1e-4)

    def test_three_seed_summary_pairs_each_control_with_its_native(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = [{'dataset': 'coco_nw', 'seed': seed, 'checkpoint_sha256': f'weights-{seed}'}
                    for seed in (42, 1337, 2025)]
            controls.write_json(root/'experiment.json', {'settings': {'suite': 'paper'}, 'jobs': jobs})
            for job, ap, delta in zip(jobs, (.2, .3, .4), (-.01, .02, .05)):
                run = root/'coco_nw'/f'seed_{job["seed"]}'
                controls.write_json(run/'completed.json', {'run_signature': str(job['seed'])})
                for name in names('paper'):
                    directory = run/name
                    controls.write_json(directory/'predictions.json', [])
                    value = ap if name == 'native' else ap + delta
                    metrics = {k: value for k in controls.METRICS}
                    differences = {k: 0. if name == 'native' else 100*delta for k in controls.METRICS}
                    controls.write_json(directory/'control_result.json', {
                        **job, 'suite': 'paper', 'control': name, 'epoch': 5,
                        'selection': 'best_ap_small', 'run_signature': str(job['seed']),
                        'group_signature': 'same-protocol', 'metrics': metrics,
                        'delta_vs_native_ap_points': differences,
                        'predictions_sha256': controls.sha256(directory/'predictions.json')})
            status = controls.summarize(root)
            self.assertTrue(status['complete'])
            with (root/'controls_mean_std.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            row = next(row for row in rows if row['control'] == 'no_gate')
            self.assertEqual(int(row['n_seeds']), 3)
            self.assertAlmostEqual(float(row['AP_small_mean']), .32)
            self.assertAlmostEqual(float(row['delta_AP_small_points_mean']), 2.)
            self.assertAlmostEqual(float(row['delta_AP_small_points_std']), 3.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
