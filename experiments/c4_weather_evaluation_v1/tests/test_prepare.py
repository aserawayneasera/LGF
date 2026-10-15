"""Synthetic fixtures only; no access to Wayne's server or checkpoints."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import prepare_coco_weather_variants as builder


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'DATASET/COCO/weather-mini'
        self.images = self.dataset / 'images/val2017_weather-500k6c'
        self.images.mkdir(parents=True)
        (self.dataset / 'annotations').mkdir()
        self.ann = self.dataset / 'annotations/mini_val2017_weather-500k6c.json'
        self.train = self.dataset / 'annotations/mini_train2017_weather-2400k6c.json'
        self.audit = self.root / 'source_audit.json'
        self.out = self.root / 'new_evaluation/manifests'
        categories = [{'id': i + 1, 'name': name} for i, name in enumerate(
            ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])]
        self.doc = {'info': {'description': 'Synthetic fixture'}, 'categories': categories,
            'images': [{'id': 10, 'file_name': '000010_fog.jpg', 'width': 64, 'height': 48},
                       {'id': 71, 'file_name': '000071_fog.jpg', 'width': 64, 'height': 48},
                       {'id': 83, 'file_name': '000083_fog.jpg', 'width': 64, 'height': 48}],
            'annotations': [{'id': 2, 'image_id': 10, 'category_id': 1, 'bbox': [1, 2, 10, 20],
                             'area': 170, 'iscrowd': 0, 'segmentation': []},
                            {'id': 30, 'image_id': 71, 'category_id': 3, 'bbox': [0, 0, 32, 32],
                             'area': 1024, 'iscrowd': 0}]}
        for im in self.doc['images']:
            for n, c in enumerate(builder.CONDITIONS):
                name = im['file_name'].replace('_fog', '_' + c)
                Image.new('RGB', (64, 48), (n * 70 + 20, im['id'], 33)).save(self.images / name)
        self.train.write_text(json.dumps({'images': [{'id': 999, 'file_name': '000999_fog.jpg'}]}))
        self.save()
        self.args = argparse.Namespace(audit=self.audit, output_dir=self.out,
            confirm_aligned_variants=True, expect_sources=3, train_annotations=None)
        self.source_before = self.snapshot()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.ann.write_text(json.dumps(self.doc))
        self.audit.write_text(json.dumps({'dataset': 'coco_weather', 'evaluation_split': 'validation',
            'annotations': str(self.ann), 'annotations_sha256': builder.digest(self.ann.read_bytes()),
            'image_root': str(self.images), 'training_annotation_checked': str(self.train)}))

    def snapshot(self):
        return {str(p): builder.digest(p.read_bytes()) for p in self.dataset.rglob('*') if p.is_file()}

    def run_prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return builder.prepare(self.args)

    def combined(self):
        return builder.load(self.out / builder.COMBINED_NAME)

    def test_complete_manifest_geometry_categories_and_source_immutability(self):
        result = self.run_prepare()
        doc = self.combined()
        self.assertEqual((len(doc['images']), len(doc['annotations'])), (9, 6))
        self.assertEqual(doc['categories'], self.doc['categories'])
        self.assertEqual(self.source_before, self.snapshot())
        originals = {a['id']: a for a in self.doc['annotations']}
        for a in doc['annotations']:
            original = originals[a['source_annotation_id']]
            for field in ('bbox', 'area', 'category_id', 'iscrowd', 'segmentation'):
                self.assertEqual(field in a, field in original)
                self.assertEqual(a.get(field), original.get(field))
        for c in builder.CONDITIONS:
            self.assertEqual(result['condition_counts'][c]['images'], 3)
            self.assertEqual(result['condition_counts'][c]['small_objects_non_crowd'], 2)
        self.assertFalse(result['independent_test_evidence'])

    def test_ids_groups_and_original_fog_ids(self):
        self.run_prepare()
        doc = self.combined()
        self.assertEqual(len({im['id'] for im in doc['images']}), 9)
        self.assertEqual(len({a['id'] for a in doc['annotations']}), 6)
        self.assertEqual({im['id'] for im in doc['images'] if im['weather'] == 'fog'}, {10, 71, 83})
        for group in {im['source_group_id'] for im in doc['images']}:
            self.assertEqual({im['weather'] for im in doc['images'] if im['source_group_id'] == group},
                             set(builder.CONDITIONS))

    def test_empty_annotation_image_retained_in_all_conditions(self):
        self.run_prepare()
        self.assertEqual(sum(im['source_image_id'] == 83 for im in self.combined()['images']), 3)

    def test_repeat_is_read_only(self):
        self.run_prepare()
        before = {str(p): p.read_bytes() for p in self.out.iterdir()}
        self.run_prepare()
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.out.iterdir()})

    def test_existing_output_tamper_refused(self):
        self.run_prepare()
        out_file = self.out / builder.COMBINED_NAME
        out_file.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'different/incomplete'):
            self.run_prepare()
        self.assertEqual(out_file.read_text(), '{}')

    def test_missing_variant_refused_before_output(self):
        (self.images / '000010_rain.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing matching variant'):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_wrong_dimensions_refused(self):
        Image.new('RGB', (20, 48)).save(self.images / '000010_rain.jpg')
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_unreadable_image_refused(self):
        (self.images / '000010_rain.jpg').write_bytes(b'not an image')
        with self.assertRaises(OSError):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_identical_variant_bytes_refused(self):
        (self.images / '000010_rain.jpg').write_bytes((self.images / '000010_fog.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'identical file bytes'):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_training_source_overlap_refused(self):
        self.train.write_text(json.dumps({'images': [{'id': 999, 'file_name': '000010_rain.jpg'}]}))
        with self.assertRaisesRegex(ValueError, 'Source overlap'):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_source_changed_after_audit_refused(self):
        self.ann.write_text(self.ann.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed after the audit'):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_duplicate_source_and_unsafe_name_refused(self):
        self.doc['images'][1]['file_name'] = '000010_rain.jpg'
        self.save()
        with self.assertRaisesRegex(ValueError, 'More than one source'):
            self.run_prepare()
        self.doc['images'][1]['file_name'] = '../000071_fog.jpg'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.run_prepare()

    def test_old_weather_paths_not_copied_as_new_weather_evidence(self):
        self.doc['images'][0].update(original_file_name='000010_fog.jpg',
            metadata={'weather': 'fog', 'other': 'preserved'}, weather='fog')
        self.save()
        self.run_prepare()
        rain = next(im for im in self.combined()['images'] if im['source_image_id'] == 10 and im['weather'] == 'rain')
        self.assertNotIn('original_file_name', rain)
        self.assertEqual(rain['metadata'], {'other': 'preserved'})
        report = builder.load(self.out / 'preparation_report.json')
        self.assertEqual(report['source_image_records'][0]['original_file_name'], '000010_fog.jpg')

    def test_requires_alignment_assertion_and_valid_source_count(self):
        self.args.confirm_aligned_variants = False
        with self.assertRaisesRegex(ValueError, 'confirm-aligned'):
            self.run_prepare()
        self.args.confirm_aligned_variants = True
        self.args.expect_sources = 500
        with self.assertRaisesRegex(ValueError, 'Expected 500'):
            self.run_prepare()

    def test_per_condition_manifests_preserve_annotations(self):
        self.run_prepare()
        for c in builder.CONDITIONS:
            doc = builder.load(self.out / ('mini_val2017_weather_%s.json' % c))
            builder.validate_source(doc)
            self.assertEqual((len(doc['images']), len(doc['annotations'])), (3, 2))
            self.assertEqual({im['weather'] for im in doc['images']}, {c})

    def test_output_inside_original_dataset_refused(self):
        self.args.output_dir = self.dataset / 'new_manifests'
        with self.assertRaisesRegex(ValueError, 'OUTSIDE'):
            self.run_prepare()

    def test_actual_existing_weather_evaluator_accepts_manifest(self):
        evaluator_path = Path(os.environ.get('WEATHER', '/mnt/data/paper1_weather_eval.py'))
        if not evaluator_path.is_file():
            self.skipTest('Existing evaluator path is not present in this local test environment.')
        self.run_prepare()
        spec = importlib.util.spec_from_file_location('original_weather_eval', evaluator_path)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        evaluator.validate_gt(self.combined())
        args = argparse.Namespace(data_root=self.root / 'DATASET', annotations=self.out / builder.COMBINED_NAME,
            images=self.images, condition_map=None, split='validation', output_root=self.root / 'eval_audit')
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluator.audit_one(args, 'coco_weather')
        self.assertEqual(report['unknown_images'], 0)
        self.assertEqual(report['conflicts'], [])
        self.assertEqual({k: v['images'] for k, v in report['condition_counts'].items()},
                         {'fog': 3, 'rain': 3, 'snow': 3})
        self.assertEqual(report['source_frame_overlap_with_current_training'], [])

    def test_shell_check_with_actual_evaluator_and_placeholder_checkpoint_paths(self):
        evaluator_path = Path(os.environ.get('WEATHER', '/mnt/data/paper1_weather_eval.py'))
        if not evaluator_path.is_file():
            self.skipTest('Existing evaluator not present for local shell preflight test.')
        self.run_prepare()
        package = self.root / 'package'
        (package / 'paper1_test_evaluation_v1').mkdir(parents=True)
        (package / 'paper1_test_evaluation_v1/models.py').write_text('# Placeholder, never imported by --check.\n')
        run_root = self.root / 'training_runs'
        for method in ('baseline', 'lgf_original'):
            for seed in (42, 1337, 2025):
                path = run_root / 'coco_weather' / method / ('seed_%s' % seed) / 'best_map.pth'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'Placeholder for path checks only, not model weights.')
        env = dict(os.environ, P1PY=sys.executable, P34PY=sys.executable, WEATHER=str(evaluator_path),
                   PKG=str(package), DATA=str(self.root / 'DATASET'), OUT3=str(self.out.parent),
                   UGS_RUNS=str(run_root), BASELINE_SOURCE='ugs', LGF_SOURCE='ugs')
        command = ['bash', str(Path(__file__).parent / 'run_coco_three_weather.sh'), 'check']
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PATH/LABEL CHECK PASSED', result.stdout)
        self.assertEqual(len(list((self.out.parent / 'jobs').glob('*.json'))), 6)
        for path in (self.out.parent / 'jobs').glob('*.json'):
            job = builder.load(path)
            self.assertEqual(job['train_dataset'], 'coco_weather')
            self.assertEqual(job['dataset'], 'coco_weather')
            self.assertEqual(job['selection'], 'best_map')


if __name__ == '__main__':
    unittest.main(verbosity=2)
