import contextlib
import copy
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

import c4_weather_eval as runner
import test_prepare


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = test_prepare.ManifestTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.output = self.fixture.out.parent
        self.training = self.fixture.train
        train = json.loads(self.training.read_text())
        train['categories'] = copy.deepcopy(self.fixture.doc['categories'])
        self.training.write_text(json.dumps(train))
        with contextlib.redirect_stdout(io.StringIO()):
            test_prepare.builder.prepare(self.fixture.args)
        self.runs = self.root / 'saved_runs'
        self.args = runner.arguments(['check', '--data-root', str(self.root / 'DATASET'),
            '--output', str(self.output), '--runs-base', str(self.runs),
            '--source-annotations', str(self.fixture.ann), '--images', str(self.fixture.images),
            '--train-annotations', str(self.training), '--expect-sources', '3'])
        for entry in runner.entries(self.args):
            entry['checkpoint'].parent.mkdir(parents=True, exist_ok=True)
            entry['checkpoint'].write_bytes(b'Synthetic path-check fixture, not model weights')
            entry['metrics'].write_text(json.dumps({'seed': entry['seed'], 'dataset': 'coco_weather',
                'selection': 'best_ap_small', 'checkpoint_sha256': runner.sha256(entry['checkpoint'])}))

    def tearDown(self):
        self.fixture.tearDown()

    def validate(self):
        return runner.validate_bundle(self.fixture.out, self.fixture.images, self.training, 3)

    def mutate_manifest(self, change):
        path = self.fixture.out / test_prepare.builder.COMBINED_NAME
        doc = runner.read_json(path)
        change(doc)
        path.write_text(json.dumps(doc))
        report_path = self.fixture.out / 'preparation_report.json'
        report = runner.read_json(report_path)
        report['output_sha256'][path.name] = runner.sha256(path)
        report_path.write_text(json.dumps(report))

    def test_complete_paired_dataset_and_fog_ids(self):
        doc, audit, fingerprint = self.validate()
        self.assertEqual(len(doc['images']), 9)
        self.assertEqual(fingerprint['source_scenes'], 3)
        self.assertEqual({row['condition'] for row in audit['rows']}, set(runner.CONDITIONS))
        fog_ids = {im['id'] for im in doc['images'] if im['weather'] == 'fog'}
        self.assertEqual(fog_ids, {10, 71, 83})

    def test_changed_image_refused(self):
        file = self.fixture.images / '000010_rain.jpg'
        file.write_bytes(file.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'image changed'):
            self.validate()

    def test_different_geometry_across_conditions_refused(self):
        def change(doc):
            rain_id = next(im['id'] for im in doc['images'] if im['weather'] == 'rain')
            next(ann for ann in doc['annotations'] if ann['image_id'] == rain_id)['bbox'][0] += 2
        self.mutate_manifest(change)
        with self.assertRaisesRegex(ValueError, 'geometry differs'):
            self.validate()

    def test_duplicate_weather_in_scene_refused(self):
        def change(doc):
            next(im for im in doc['images'] if im['weather'] == 'rain')['weather'] = 'fog'
        self.mutate_manifest(change)
        with self.assertRaisesRegex(ValueError, 'Duplicate condition'):
            self.validate()

    def test_different_training_category_order_refused(self):
        doc = runner.read_json(self.training)
        doc['categories'][0]['id'] = 80
        self.training.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'category IDs'):
            self.validate()

    def test_check_loads_no_model_and_preserves_sources(self):
        snapshots = {path: path.read_bytes() for path in [self.fixture.ann, self.training]}
        with contextlib.redirect_stdout(io.StringIO()):
            checked = runner.preflight(self.args)
        self.assertEqual(len(checked[-1]), 3)
        self.assertEqual([entry['placement'] for entry in checked[-1]], ['C4'] * 3)
        for path, data in snapshots.items():
            self.assertEqual(path.read_bytes(), data)

    def test_cli_check_with_real_manifest_and_placeholder_weights(self):
        args = ['check', '--data-root', str(self.root / 'DATASET'), '--output', str(self.output),
                '--runs-base', str(self.runs), '--source-annotations', str(self.fixture.ann),
                '--images', str(self.fixture.images), '--train-annotations', str(self.training),
                '--expect-sources', '3']
        result = subprocess.run([sys.executable, str(runner.ROOT / 'c4_weather_eval.py'), *args],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('CHECK PASSED', result.stdout)

    def test_prepare_and_check_through_new_shell_launcher(self):
        output = self.root / 'launcher_evaluation'
        common = ['--data-root', str(self.root / 'DATASET'), '--output', str(output),
                  '--runs-base', str(self.runs), '--expect-sources', '3']
        env = dict(os.environ, C4_WEATHER_PY=sys.executable)
        for action in ['prepare', 'check']:
            result = subprocess.run(['bash', str(runner.ROOT / 'run_c4_weather.sh'), action, *common],
                                    capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((output / 'manifests/mini_val2017_weather_fog_rain_snow.json').is_file())

    def test_checkpoint_hash_and_selection_mismatches_refused(self):
        entry = runner.entries(self.args)[0]
        value = runner.read_json(entry['metrics'])
        value['selection'] = 'best_map'
        entry['metrics'].write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'selection'):
            runner.selected_metrics(entry, 'best_ap_small', runner.sha256(entry['checkpoint']))
        value['selection'] = 'best_ap_small'
        value['checkpoint_sha256'] = 'different'
        entry['metrics'].write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'different checkpoint'):
            runner.selected_metrics(entry, 'best_ap_small', runner.sha256(entry['checkpoint']))

    def test_checkpoint_data_audit_must_match_original_annotations(self):
        entry = runner.entries(self.args)[0]
        _, _, fingerprint = self.validate()
        path = entry['checkpoint'].parent / 'data_audit.json'
        runner.write_json(path, {'val': {'annotations_sha256': fingerprint['source_annotations_sha256'],
                                        'categories': self.fixture.doc['categories']}})
        self.assertEqual(runner.checkpoint_data_audit(entry, fingerprint), runner.sha256(path))
        runner.write_json(path, {'val': {'annotations_sha256': 'different-source'}})
        with self.assertRaisesRegex(ValueError, 'different source validation annotations'):
            runner.checkpoint_data_audit(entry, fingerprint)

    def test_output_cannot_target_existing_run_folder(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.arguments(['run', '--runs-base', str(self.runs),
                              '--output', str(self.runs / 'runs_C4' / 'unsafe')])


def synthetic_results(placements=('C3', 'C4'), seeds=(42, 1337, 2025)):
    records = []
    for placement in placements:
        for i, seed in enumerate(seeds):
            base = .20 + i * .01 + (.03 if placement == 'C4' else 0)
            rows = [{'condition': condition, 'images': 1500 if condition == 'ALL' else 500,
                     **{metric: base for metric in runner.METRICS}}
                    for condition in (*runner.CONDITIONS, 'ALL')]
            records.append({'placement': placement, 'seed': seed, 'source_scenes': 500,
                            'dataset_sha256': 'same-paired-scenes',
                            'evaluation_protocol': {'min_size': [800], 'weights': 'EMA'},
                            'results': rows})
    return records


class ReportingChecks(unittest.TestCase):
    def test_saved_metric_edit_is_detected_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.write_json(path / 'predictions.json', [])
            result = {'protocol': {'selection': 'best_ap_small'},
                      'protocol_sha256': runner.signature({'selection': 'best_ap_small'}),
                      'predictions_sha256': runner.sha256(path / 'predictions.json'),
                      'AP_small': .2}
            result['result_sha256'] = runner.signature(result)
            runner.write_json(path / 'metrics.json', result)
            self.assertEqual(runner.verify_result(path / 'metrics.json')['AP_small'], .2)
            result['AP_small'] = .9
            runner.write_json(path / 'metrics.json', result)
            with self.assertRaisesRegex(ValueError, 'Saved metrics changed'):
                runner.verify_result(path / 'metrics.json')

    def test_three_seed_sample_sd_and_paired_percentage_points(self):
        per_seed, summary, deltas = runner.summarize_results(synthetic_results(), ['C3', 'C4'], [42, 1337, 2025])
        self.assertEqual(len(per_seed), 24)
        self.assertEqual(len(summary), 8)
        self.assertEqual(len(deltas), 4)
        fog = next(row for row in summary if row['placement'] == 'C3' and row['condition'] == 'fog')
        self.assertAlmostEqual(fog['AP_small_mean'], .21)
        self.assertAlmostEqual(fog['AP_small_sd'], .01)
        self.assertEqual(fog['n_seeds'], 3)
        self.assertAlmostEqual(deltas[0]['AP_small_difference_pp'], 3.)
        self.assertAlmostEqual(deltas[0]['AP_small_paired_seed_sd_pp'], 0.)

    def test_c4_only_and_pooled_image_count(self):
        _, summary, deltas = runner.summarize_results(synthetic_results(('C4',)), ['C4'], [42, 1337, 2025])
        self.assertEqual(len(summary), 4)
        self.assertEqual(deltas, [])
        pooled = next(row for row in summary if row['condition'] == 'ALL')
        self.assertEqual(pooled['images'], 1500)
        self.assertEqual(pooled['source_scenes'], 500)

    def test_missing_seed_refused(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            runner.summarize_results(synthetic_results()[:-1], ['C3', 'C4'], [42, 1337, 2025])

    def test_different_data_or_inference_protocol_refused(self):
        records = synthetic_results()
        records[-1]['dataset_sha256'] = 'other-scenes'
        with self.assertRaisesRegex(ValueError, 'different evaluation datasets'):
            runner.summarize_results(records, ['C3', 'C4'], [42, 1337, 2025])
        records = synthetic_results()
        records[-1]['evaluation_protocol']['min_size'] = [640]
        with self.assertRaisesRegex(ValueError, 'different inference protocols'):
            runner.summarize_results(records, ['C3', 'C4'], [42, 1337, 2025])

    def test_fog_reference_matches_and_detects_changed_accuracy(self):
        rows = synthetic_results(('C4',))[0]['results']
        selected = {'metrics': {metric: rows[0][metric] for metric in runner.METRICS[:6]}}
        self.assertEqual(runner.fog_reference(rows, selected, .0001)['status'], 'match')
        selected['metrics']['AP_small'] += .02
        self.assertEqual(runner.fog_reference(rows, selected, .0001)['status'], 'different')


if __name__ == '__main__':
    unittest.main()
