#!/usr/bin/env python3
"""Verify release hashes, archived model identities, Python syntax, and workflow paths."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check():
    errors, counts = [], {}
    for p in ROOT.rglob('*.py'):
        if '__pycache__' in p.parts or 'workspace' in p.parts:
            continue
        try:
            ast.parse(p.read_text(), filename=str(p))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'{p.relative_to(ROOT)}: {exc}')
    counts['python_files'] = len(list(ROOT.rglob('*.py')))
    identities = list(ROOT.rglob('source_identity.json'))
    for identity in identities:
        data = json.loads(identity.read_text())
        core = identity.parent / 'original_core.py'
        if core.exists() and hashlib.sha256(core.read_bytes()).hexdigest() != data.get('source_sha256'):
            errors.append(f'Archived model identity failed: {core.relative_to(ROOT)}')
    counts['archived_model_identities'] = len(identities)
    spec = importlib.util.spec_from_file_location('release_launcher', ROOT / 'run.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for workflow in mod.WORKFLOWS:
        try:
            mod.command(workflow, ['--help'], 'python3')
        except FileNotFoundError as exc:
            errors.append(f'Missing workflow {workflow}: {exc}')
    counts['workflows'] = len(mod.WORKFLOWS)
    sums = ROOT / 'CHECKSUMS.sha256'
    if sums.exists():
        for row in sums.read_text().splitlines():
            expected, relative = row.split('  ', 1)
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f'Release checksum failed: {relative}')
        counts['release_hashes'] = len(sums.read_text().splitlines())
    print(json.dumps({'counts': counts, 'errors': errors, 'passed': not errors}, indent=2))
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(check())
