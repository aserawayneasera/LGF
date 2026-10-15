#!/usr/bin/env python3
"""Portable dispatch to the archived PRL experiment implementations."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / 'experiments'
WORKFLOWS = {
    'original': ('paper1_ugs_comparison_v1/run.py', 'Definition-preserving RetinaNet/LGF/UGS runner'),
    'ugs': ('paper1_ugs_comparison_v1/run.py', 'Alias for the original/UGS runner'),
    'detectors': ('paper1_reviewer34_v1/reviewer34_compare.py', 'YOLO11m and RT-DETR-L comparisons'),
    'official-export': ('paper1_test_evaluation_v1/evaluate_test.py', 'Frozen checkpoint export and labelled holdout scoring'),
    'se-cbam': ('paper1_se_cbam_extension_nas_v1_0_2/run_se_cbam.py', 'SE/CBAM extension'),
    'controls': ('lgf_original_controls_v1/controls.py', 'Eight frozen-checkpoint gate interventions'),
    'bootstrap': ('paper1_paired_bootstrap/paper1_paired_bootstrap.py', 'Paired seed/image AP bootstrap'),
    'gate-maps': ('paper1_gate_maps_four_conditions_v2/paper1_gate_visualization_conditions_v2.py', 'Matched clear/fog/rain/snow gate figures'),
    'acdc-gate-maps': ('paper1_acdc_gate_maps_by_condition_v1/paper1_acdc_gate_visualization.py', 'Optional ACDC gate inspection'),
    'failures': ('paper1_balanced_failures_v1/paper1_failure_analysis.py', 'Object-level detection outcomes and examples'),
    'weather': ('paper1_weather_eval_v1/paper1_weather_eval.py', 'Checkpoint-frozen condition evaluation'),
    'prepare-weather': ('paper1_coco_three_weather_v1/prepare_coco_weather_variants.py', 'Audit/materialize paired synthetic weather variants'),
    'multilevel': ('paper1_ugs_multilevel_v1/run.py', 'Archived placement training runner'),
    'profile': ('paper1_placement_profile_v1/profile_placements.py', 'Archived reference-image and timing profile'),
    'c4-weather': ('c4_weather_evaluation_v1/c4_weather_eval.py', 'Full pooled C4 synthetic-weather evaluation'),
    'legacy-train': ('../legacy/train_lgf.py', 'Historical GitHub script, retains original path defaults'),
}

def has_flag(args, flag):
    return any(arg == flag or arg.startswith(flag + '=') for arg in args)

def command(workflow, forwarded, python):
    script = (EXPERIMENTS / WORKFLOWS[workflow][0]).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    forwarded = list(forwarded)
    # Help remains the native parser's help. No experiment parameters are changed.
    if not any(x in forwarded for x in ('-h', '--help')):
        defaults = {}
        if workflow == 'detectors':
            defaults['--lgf-package'] = EXPERIMENTS / 'paper1_ugs_comparison_v1'
        elif workflow == 'se-cbam':
            defaults['--package'] = EXPERIMENTS / 'paper1_ugs_comparison_v1'
        elif workflow == 'weather' and forwarded[:1] == ['evaluate']:
            defaults['--package-root'] = EXPERIMENTS
        elif workflow in ('gate-maps', 'acdc-gate-maps'):
            defaults['--package-root'] = EXPERIMENTS / 'lgf_original_controls_v1/source_runner'
        for flag, value in defaults.items():
            if not has_flag(forwarded, flag):
                forwarded += [flag, str(value)]
    return [python, str(script), *forwarded], Path.cwd()

def doctor():
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'pycocotools', 'albumentations',
                 'ultralytics', 'tensorboard', 'opencv-python-headless', 'matplotlib', 'tqdm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    versions_match = (packages['torch'] or '').startswith('2.10.') and (packages['torchvision'] or '').startswith('0.25.')
    report = {'python': sys.version.split()[0], 'root': str(ROOT), 'packages': packages,
              'archived_torch_versions_match': versions_match,
              'workflow_scripts_exist': all((EXPERIMENTS / path).is_file() for path, _ in WORKFLOWS.values())}
    print(json.dumps(report, indent=2))
    return 0 if report['workflow_scripts_exist'] else 1

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable, help='Interpreter for this workflow')
    parser.add_argument('--print-command', action='store_true', help='Print dispatch without running an experiment')
    parser.add_argument('workflow', choices=['list', 'doctor', *WORKFLOWS])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.workflow == 'list':
        for name, (_, description) in WORKFLOWS.items():
            print(f'{name:18s} {description}')
        return 0
    if args.workflow == 'doctor':
        return doctor()
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    cmd, cwd = command(args.workflow, forwarded, args.python)
    if args.print_command:
        print(json.dumps({'argv': cmd, 'cwd': str(cwd)}, indent=2))
        return 0
    env = os.environ.copy()
    env.setdefault('TORCH_HOME', str(ROOT / 'workspace/torch_cache'))
    env.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    print('Running: ' + shlex.join(cmd), flush=True)
    # Execution uses an argument vector. User-provided arguments never enter a shell.
    return subprocess.run(cmd, cwd=cwd, env=env).returncode

if __name__ == '__main__':
    raise SystemExit(main())
