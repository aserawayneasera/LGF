import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

launcher = load('launcher', 'run.py')
inputs = load('input_audit', 'tools/audit_inputs.py')

class ReleaseToolsTests(unittest.TestCase):
    def test_portable_default_and_explicit_override(self):
        command, _ = launcher.command('detectors', ['train'], 'python3')
        self.assertIn(str(launcher.EXPERIMENTS / 'paper1_ugs_comparison_v1'), command)
        command, _ = launcher.command('detectors', ['train', '--lgf-package=/custom'], 'python3')
        self.assertNotIn('--lgf-package', command)

    def test_help_does_not_inject_experiment_parameters(self):
        command, _ = launcher.command('weather', ['--help'], 'python3')
        self.assertEqual(command[-1], '--help')
        self.assertNotIn('--package-root', command)

    def test_weather_path_only_for_evaluate(self):
        command, _ = launcher.command('weather', ['audit'], 'python3')
        self.assertNotIn('--package-root', command)
        command, _ = launcher.command('weather', ['evaluate'], 'python3')
        self.assertIn('--package-root', command)

    def test_failure_run_roots_are_not_source_packages(self):
        command, _ = launcher.command('failures', [], 'python3')
        self.assertNotIn('--ugs-root', command)

    def test_forwarded_argument_is_not_a_shell_command(self):
        value = '$(touch never-created)'
        command, _ = launcher.command('original', ['--output', value], 'python3')
        self.assertIn(value, command)

    def test_annotation_ids_and_class_names(self):
        data = {'images': [{'id': 1, 'file_name': '000000000001_fog.jpg'}],
                'categories': [{'id': i+1, 'name': name} for i, name in enumerate(sorted(inputs.CLASSES))],
                'annotations': [{'id': 0, 'image_id': 1, 'category_id': 1, 'bbox': [0,0,10,10], 'area': 100}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'annotation.json'; path.write_text(json.dumps(data))
            report, _ = inputs.audit(path, positive_ids=True)
            self.assertTrue(report['errors'])
            data['annotations'][0]['id'] = 1; path.write_text(json.dumps(data))
            report, _ = inputs.audit(path, positive_ids=True)
            self.assertEqual(report['errors'], [])

    def test_source_scene_matching_and_unknown_lineage(self):
        scenes, unknown = inputs.coco_scenes({'images': [
            {'file_name': '000000000123_fog.jpg'}, {'file_name': '000000000123_snow.jpg'},
            {'file_name': 'unknown.png'}]})
        self.assertEqual(scenes, {'000000000123'})
        self.assertEqual(unknown, ['unknown.png'])

if __name__ == '__main__':
    unittest.main()
