"""Synthetic software tests. These are not detector performance experiments."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper1_weather_eval as w


def document(names=None):
    names = names or ['000000000001_fog.jpg', '000000000002_rain.jpg', '000000000003_snow.jpg']
    return {
        'images': [{'id': i+1, 'file_name': name, 'width': 64, 'height': 64} for i, name in enumerate(names)],
        'annotations': [{'id': i+1, 'image_id': i+1, 'category_id': 1, 'bbox': [2, 2, 10, 10], 'area': 100, 'iscrowd': 0} for i in range(len(names))],
        'categories': [{'id': i+1, 'name': name} for i, name in enumerate(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])],
    }


def setup_data(tmp_path, doc, dataset='coco_weather'):
    image_root, ann, training = w.dataset_paths(tmp_path/'data', dataset)
    image_root.mkdir(parents=True)
    for im in doc['images']:
        p = image_root / im['file_name']
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'synthetic path-existence fixture, not a real image')
    w.write_json(ann, doc)
    args = SimpleNamespace(data_root=tmp_path/'data', output_root=tmp_path/'out', condition_map=None,
                           annotations=None, images=None, split='validation', action='audit')
    return args


def test_coco_conditions_and_no_input_changes(tmp_path):
    doc = document()
    for im, cond in zip(doc['images'], ('fog', 'rain', 'snow')):
        im['weather'] = cond
    args = setup_data(tmp_path, doc)
    ann = w.dataset_paths(args.data_root, 'coco_weather')[1]
    before = w.sha256(ann)
    result = w.audit_one(args, 'coco_weather')
    assert result['unknown_images'] == 0
    assert {k: v['images'] for k, v in result['condition_counts'].items()} == dict(fog=1, rain=1, snow=1)
    assert w.sha256(ann) == before


def test_single_present_weather_not_fabricated(tmp_path):
    args = setup_data(tmp_path, document(['001_fog.jpg', '002_fog.jpg']))
    result = w.audit_one(args, 'coco_weather')
    assert set(result['condition_counts']) == {'fog'}


def test_unknown_not_silently_clear(tmp_path):
    args = setup_data(tmp_path, document(['000000000001.jpg']))
    with pytest.raises(ValueError, match='incomplete'):
        w.audit_one(args, 'coco_weather')
    report = w.read_json(args.output_root/'audits/coco_weather/condition_audit.json')
    assert report['unknown_images'] == 1
    assert report['condition_counts'] == {}


def test_conflicting_labels_rejected(tmp_path):
    doc = document()
    doc['images'][0]['weather'] = 'rain'
    args = setup_data(tmp_path, doc)
    with pytest.raises(ValueError, match='contradictory'):
        w.audit_one(args, 'coco_weather')


def test_acdc_archive_exact_flattened_frame_join(tmp_path):
    seqs = ['GOPR0122', 'GP010376', 'GOPR0399', 'GOPR0999']
    conditions = ['fog', 'night', 'rain', 'snow']
    archive = tmp_path/'rgb.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for seq, condition in zip(seqs, conditions):
            z.writestr(f'rgb_anon/{condition}/val/{seq}/{seq}_frame_000001_rgb_anon.png', b'not decoded')
        z.writestr('rgb_anon/fog/val_ref/GOPR0122/GOPR0122_frame_000001_rgb_anon.png', b'reference ignored')
    names = [f'{seq}_frame_000001_rgb_anon_png.rf.' + 'a'*32 + '.jpg' for seq in seqs]
    args = setup_data(tmp_path, document(names), 'acdc')
    index, descriptors = w.build_acdc_index(archive)
    report = w.audit_one(args, 'acdc', index, descriptors)
    assert [r['condition'] for r in report['rows']] == conditions
    assert report['official_split_counts'] == {'val': 4}
    assert len(index) == 4
    assert not (tmp_path/'rgb_anon').exists()  # No image extraction.


def test_exact_acdc_join_does_not_infer_from_sequence(tmp_path):
    name = 'GOPR0122_frame_000002_rgb_anon_png.rf.' + 'a'*32 + '.jpg'
    args = setup_data(tmp_path, document([name]), 'acdc')
    index = {'gopr0122_frame_000001_rgb_anon.png': {'condition': 'fog', 'official_split': 'val', 'source_path': 'known'}}
    with pytest.raises(ValueError, match='incomplete'):
        w.audit_one(args, 'acdc', index, [])


def test_acdc_conflicting_archive_entries_rejected(tmp_path):
    path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for condition in ('fog', 'snow'):
            z.writestr(f'rgb_anon/{condition}/val/GOPR0122/GOPR0122_frame_000001_rgb_anon.png', b'x')
    with pytest.raises(ValueError, match='Ambiguous'):
        w.build_acdc_index(path)


def test_official_detection_zip_metadata(tmp_path):
    path = tmp_path/'gt_detection_trainval.zip'
    doc = document(['fog/val/GOPR0122/GOPR0122_frame_000001_rgb_anon.png'])
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('gt_detection/instancesonly_val_gt_detection.json', json.dumps(doc))
    index, _ = w.build_acdc_index(path)
    assert index['gopr0122_frame_000001_rgb_anon.png']['condition'] == 'fog'


def test_missing_labels_do_not_accept_hidden_test_info():
    doc = document()
    del doc['annotations']
    with pytest.raises(ValueError, match='labelled'):
        w.validate_gt(doc)


def test_duplicate_annotation_ids_rejected():
    doc = document()
    doc['annotations'][1]['id'] = 1
    with pytest.raises(ValueError, match='unique integer IDs'):
        w.validate_gt(doc)


def test_current_train_source_overlap_reported(tmp_path):
    doc = document(['000000000123_snow.jpg'])
    args = setup_data(tmp_path, doc)
    training = w.dataset_paths(args.data_root, 'coco_weather')[2]
    w.write_json(training, document(['000000000123_rain.jpg']))
    result = w.audit_one(args, 'coco_weather')
    assert result['source_frame_overlap_with_current_training'] == ['123']
    assert result['complete_historical_scene_independence_verified'] is False


def test_notebook_best_epoch_ties_and_selection(tmp_path):
    for epoch, score in [(10, '.21'), (20, '.22'), (30, '.22')]:
        (tmp_path/f'BEST_acdc_C3_baseline_s42_epoch_{epoch}_map_0{score}.pth').touch()
    cp = w.notebook_checkpoint(tmp_path, 'acdc', 'baseline', 42, 'C3', 'best_map')
    assert 'epoch_30_' in cp.name
    with pytest.raises(ValueError, match='not APsmall'):
        w.notebook_checkpoint(tmp_path, 'acdc', 'baseline', 42, 'C3', 'best_ap_small')


def test_notebook_mixed_repeats_rejected(tmp_path):
    for epoch, score in [(10, .22), (20, .21)]:
        (tmp_path/f'BEST_acdc_C3_baseline_s42_epoch_{epoch}_map_{score}.pth').touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        w.notebook_checkpoint(tmp_path, 'acdc', 'baseline', 42, 'C3', 'best_map')


def test_explicit_checkpoint_selection_mismatch(tmp_path):
    p = tmp_path/'best_ap_small.pth'
    p.touch()
    with pytest.raises(ValueError, match='disagree'):
        w.checkpoint_for(SimpleNamespace(checkpoint=p, selection='best_map'), 'acdc', 'lgf_original', 42)


def test_unsafe_path_rejected(tmp_path):
    with pytest.raises(ValueError, match='safe relative'):
        w.safe_image(tmp_path, '../image.png')


def test_small_counts_use_annotation_area_and_non_crowd():
    doc = document()
    doc['annotations'][0]['area'] = 1024
    doc['annotations'][1]['area'] = 1025
    doc['annotations'][2]['iscrowd'] = 1
    counts = w.support_counts(doc, [1, 2, 3])
    assert counts['objects_non_crowd'] == 2
    assert counts['small_objects_non_crowd'] == 1


def test_summary_sd_missing_metrics_and_paired_deltas(tmp_path):
    for method, values in [('baseline', [.1, .2, .3]), ('lgf_original', [.11, .22, .33])]:
        for seed, value in zip([42, 1337, 2025], values):
            row = {'condition': 'fog', 'images': 3, 'objects_non_crowd': 3, 'small_objects_non_crowd': 3,
                   'classes_with_objects': 1, 'classes_with_small_objects': 1,
                   **dict.fromkeys(w.METRICS, value)}
            row['AP_large'] = None
            report = {'dataset': 'coco_weather', 'train_dataset': 'coco_weather', 'selection': 'best_map',
                      'evaluation_split': 'validation', 'evaluation_signature': 'same', 'method': method,
                      'seed': seed, 'rows': [row]}
            w.write_json(tmp_path/'runs'/method/str(seed)/'weather_metrics.json', report)
    means = w.summarize(tmp_path)
    baseline = next(r for r in means if r['method'] == 'baseline')
    assert baseline['AP_small_mean'] == pytest.approx(.2)
    assert baseline['AP_small_std'] == pytest.approx(.1)
    assert baseline['AP_large_mean'] is None
    import csv
    delta = list(csv.DictReader(open(tmp_path/'summary/lgf_minus_baseline.csv')))[0]
    assert float(delta['AP_small_delta_points_mean']) == pytest.approx(2)
    assert float(delta['AP_small_delta_points_std']) == pytest.approx(1)


def test_summary_different_evaluation_sets_rejected(tmp_path):
    for seed in [42, 1337]:
        r = {'dataset': 'acdc', 'train_dataset': 'acdc', 'selection': 'best_map', 'evaluation_split': 'validation',
             'evaluation_signature': str(seed), 'method': 'baseline', 'seed': seed, 'rows': []}
        w.write_json(tmp_path/'runs'/str(seed)/'weather_metrics.json', r)
    with pytest.raises(ValueError, match='different images'):
        w.summarize(tmp_path)


def test_loader_source_adapter_interface():
    # Source-level contract check, not a real checkpoint load.
    import ast
    source = (Path(__file__).parents[1]/'paper1_weather_eval.py').read_text()
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert any(n.func.attr == 'load_model' and len(n.args) == 2 for n in calls)
    assert any(n.func.attr == 'rgb_tensors' and len(n.args) == 3 for n in calls)


def test_coco_perfect_and_empty_predictions():
    pytest.importorskip('pycocotools')
    doc = document()
    predictions = [{'image_id': i, 'category_id': 1, 'bbox': [2, 2, 10, 10], 'score': .9} for i in (1, 2, 3)]
    perfect = w.coco_metrics(doc, predictions, [1, 2, 3])
    assert perfect['AP_small'] == pytest.approx(1)
    assert perfect['AP_large'] is None
    zero = w.coco_metrics(doc, [], [1, 2, 3])
    assert zero['AP_small'] == pytest.approx(0)


def test_coco_subset_keeps_images_without_detections():
    pytest.importorskip('pycocotools')
    doc = document()
    pred = [{'image_id': 1, 'category_id': 1, 'bbox': [2, 2, 10, 10], 'score': .9}]
    one = w.coco_metrics(doc, pred, [1])
    all_images = w.coco_metrics(doc, pred, [1, 2, 3])
    assert one['AP_small'] == pytest.approx(1)
    assert all_images['AP_small'] < .5
    assert w.coco_metrics(doc, pred, [2])['AP_small'] == pytest.approx(0)


def test_cli_evaluate_check_builds_job_without_loading_checkpoint(tmp_path):
    import subprocess
    args = setup_data(tmp_path, document())
    pkg = tmp_path/'package'
    (pkg/'paper1_test_evaluation_v1').mkdir(parents=True)
    (pkg/'paper1_test_evaluation_v1/models.py').write_text('# intentionally not importable as a model adapter\n')
    cp = tmp_path/'ugs/coco_weather/lgf_original/seed_42/best_map.pth'
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b'not a real checkpoint; check must not load this')
    cmd = [sys.executable, str(Path(w.__file__)), 'evaluate', '--datasets', 'coco_weather', '--methods', 'lgf_original',
           '--seeds', '42', '--data-root', str(args.data_root), '--output-root', str(args.output_root),
           '--package-root', str(pkg), '--ugs-root', str(tmp_path/'ugs'), '--legacy-python', sys.executable, '--check']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'No checkpoint was loaded' in proc.stdout
    assert len(list((args.output_root/'jobs').glob('*.json'))) == 1
    assert not (args.output_root/'runs').exists()
