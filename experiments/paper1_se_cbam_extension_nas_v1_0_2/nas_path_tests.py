#!/usr/bin/env python3
"""Storage-routing checks. No model imports, GPU use, or dataset access."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import run_se_cbam as entry
import train_se_cbam_grid as grid
import evaluate_test_se_cbam as test_entry

class NASPaths(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(str(entry.PROJECT_ROOT), '/nas.dbms/asera')
        self.assertEqual(grid.DEFAULT_RUNS, entry.DEFAULT_RUN_ROOT)
        for p in (entry.DEFAULT_DATA_ROOT, entry.DEFAULT_RUN_ROOT, entry.DEFAULT_RUNTIME_ROOT):
            self.assertTrue(str(p).startswith('/nas.dbms/asera/'))

    def test_unset_cache(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            entry.configure_storage()
            self.assertEqual(os.environ['TORCH_HOME'], '/nas.dbms/asera/paper1_original/torch_cache')

    def test_moved_cache(self):
        with mock.patch.dict(os.environ, {'TORCH_HOME': '/tmp/asera/paper1_original/torch_cache'}):
            entry.configure_storage()
            self.assertEqual(os.environ['TORCH_HOME'], '/nas.dbms/asera/paper1_original/torch_cache')

    def test_custom_cache_preserved(self):
        with mock.patch.dict(os.environ, {'TORCH_HOME': '/other/cache'}):
            entry.configure_storage()
            self.assertEqual(os.environ['TORCH_HOME'], '/other/cache')

    def test_explicit_cli_paths_preserved(self):
        argv = ['train', '--data-root', '/custom/data', '--run-root=/custom/runs']
        self.assertEqual(entry.with_storage_defaults(argv, {'--data-root': entry.DEFAULT_DATA_ROOT,
                         '--run-root': entry.DEFAULT_RUN_ROOT}), argv)

    def test_default_injection_is_idempotent(self):
        defaults = {'--data-root': entry.DEFAULT_DATA_ROOT, '--run-root': entry.DEFAULT_RUN_ROOT}
        once = entry.with_storage_defaults(['train'], defaults)
        self.assertEqual(entry.with_storage_defaults(once, defaults), once)

    def test_direct_entry_overrides_native_old_defaults(self):
        seen = []
        def arguments():
            p = argparse.ArgumentParser()
            p.add_argument('action')
            p.add_argument('--method')
            p.add_argument('--data-root', default='/tmp/asera/old_data')
            p.add_argument('--run-root', default='/tmp/asera/old_runs')
            return p.parse_args()
        runner = SimpleNamespace(arguments=arguments, summarize=seen.append)
        with mock.patch.object(entry, 'resolve_package', return_value=Path('/unused')), \
             mock.patch.object(entry, 'load_runner', return_value=(runner, None)), \
             contextlib.redirect_stdout(io.StringIO()):
            entry.main(['summarize'])
        self.assertEqual(seen[0].data_root, str(entry.DEFAULT_DATA_ROOT))
        self.assertEqual(seen[0].run_root, str(entry.DEFAULT_RUN_ROOT))

    def test_grid_defaults(self):
        seen = []
        def job(args, *rest):
            seen.append(args)
            return 'SKIP COMPLETED', ['python', 'unused.py'], args.log_root/'unused.log', Path('/unused/protocol.json')
        with mock.patch.object(sys, 'argv', ['grid', '--methods', 'se', '--datasets', 'coco_nw', '--seeds', '42', '--check']), \
             mock.patch.object(grid, 'resolve_package', return_value=Path('/unused')), \
             mock.patch.object(grid, 'original_hashes', return_value={}), \
             mock.patch.object(grid, 'make_job', side_effect=job), \
             contextlib.redirect_stdout(io.StringIO()):
            grid.main()
        self.assertEqual(seen[0].data_root, entry.DEFAULT_DATA_ROOT)
        self.assertEqual(seen[0].run_root, entry.DEFAULT_RUN_ROOT)
        self.assertEqual(seen[0].log_root, entry.DEFAULT_RUN_ROOT.parent/'logs')

    def test_test_export_overrides_native_old_defaults(self):
        seen = []
        def arguments():
            p = argparse.ArgumentParser()
            p.add_argument('action')
            p.add_argument('--method')
            p.add_argument('--data-root', default='/tmp/asera/old_data')
            p.add_argument('--runtime-root', default='/tmp/asera/old_runtime')
            return p.parse_args()
        module = SimpleNamespace(arguments=arguments, export=seen.append, score=seen.append,
                                 compare=seen.append, package_acdc=seen.append)
        with tempfile.TemporaryDirectory() as temp:
            pkg = Path(temp)/'paper1_ugs_comparison_v1'
            ev = Path(temp)/'paper1_test_evaluation_v1'; ev.mkdir()
            for name in ('evaluate_test.py', 'models.py', 'test_data.py'):
                (ev/name).touch()
            with mock.patch.object(sys, 'argv', ['exporter', 'export', '--method', 'se']), \
                 mock.patch.object(test_entry, 'resolve_package', return_value=pkg), \
                 mock.patch.object(test_entry, 'load_exporter', return_value=module):
                test_entry.main()
        self.assertEqual(seen[0].data_root, str(entry.DEFAULT_DATA_ROOT))
        self.assertEqual(seen[0].runtime_root, str(entry.DEFAULT_RUNTIME_ROOT))

if __name__ == '__main__':
    unittest.main(verbosity=2)
