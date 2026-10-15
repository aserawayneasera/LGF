"""Synthetic report fixtures exercise comparison guards, not detector performance."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import compare


def reports():
    rows=[]
    for method in ['lgf_original','ugs_reimpl']:
        for i,seed in enumerate([42,1337,2025]):
            rows.append({'dataset':'coco_nw','evaluation_signature':'same synthetic validation',
                'method':method,'method_label':method,'seed':seed,'epoch':10,
                'selection':'best_map','checkpoint':'synthetic fixture','checkpoint_sha256':'synthetic fixture',
                'ugs_config':{'alpha':5,'ur_weight':.1} if method=='ugs_reimpl' else None,
                'metrics':{m:.1*(i+1) for m in ['AP_small','mAP','AP50','AP75','AP_medium','AP_large']}})
    return rows


class ComparisonTests(unittest.TestCase):
    def run_comparison(self,rows):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);files=[]
            for i,r in enumerate(rows):
                p=root/f'{i}.json';p.write_text(json.dumps(r));files.append(str(p))
            argv=['compare.py','--reports',*files,'--output',str(root/'out')]
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):compare.main()
            return json.loads((root/'out/comparison.json').read_text())

    def test_three_seed_mean_and_sample_standard_deviation(self):
        result=self.run_comparison(reports())
        self.assertEqual(len(result),2)
        for row in result:
            self.assertEqual(row['n'],3)
            self.assertAlmostEqual(row['AP_small_mean'],.2)
            self.assertAlmostEqual(row['AP_small_std'],.1)

    def test_different_evaluation_rejected(self):
        rows=reports();rows[-1]['evaluation_signature']='different'
        with self.assertRaisesRegex(ValueError,'identical validation'):self.run_comparison(rows)

    def test_different_seed_sets_rejected(self):
        rows=reports();rows[-1]['seed']=2026
        with self.assertRaisesRegex(ValueError,'matching seed'):self.run_comparison(rows)

    def test_different_checkpoint_selection_rejected(self):
        rows=reports();rows[-1]['selection']='best_ap_small'
        with self.assertRaisesRegex(ValueError,'selection rule'):self.run_comparison(rows)

    def test_different_ugs_configurations_rejected(self):
        rows=reports();rows[-1]['ugs_config']['alpha']=2
        with self.assertRaisesRegex(ValueError,'configurations differ'):self.run_comparison(rows)


if __name__=='__main__':unittest.main()
