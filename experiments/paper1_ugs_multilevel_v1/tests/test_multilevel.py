"""Runtime unit tests for simultaneous pre-FPN LGF placement."""
import copy
import hashlib
from pathlib import Path
import unittest

import torch

from core_bridge import build_model, load_core
from multilevel import LEVEL_SETS


class MultiLevelLGFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_architecture_inserts_each_lgf_exactly_once_and_reloads(self):
        core = load_core()
        identity = __import__('json').loads((Path(__file__).parents[1] / 'vendor/source_identity.json').read_text())
        actual_sha = hashlib.sha256((Path(__file__).parents[1] / 'vendor/original_core.py').read_bytes()).hexdigest()
        self.assertEqual(identity['source_sha256'], actual_sha)
        for level,expected in [('C3_C4', [('0',512), ('1',1024)]),
                               ('C3_C4_C5', [('0',512), ('1',1024), ('2',2048)])]:
            with self.subTest(level=level):
                torch.manual_seed(12)
                model = build_model('lgf_original', 6, level, pretrained=False, smoke_size=64).eval()
                actual = [(k, model.backbone.block_fpn_in[k].local[0].in_channels)
                          for k in model.backbone.block_fpn_in]
                self.assertEqual(actual, expected)
                self.assertEqual(tuple(model.backbone.lgf_semantic_levels), LEVEL_SETS[level])
                blocks = list(model.backbone.block_fpn_in.values())
                self.assertEqual(len({id(m) for m in blocks}), len(expected))
                self.assertTrue(all(isinstance(m, core.StableLGFBlock) for m in blocks))
                with torch.no_grad():
                    out = model([torch.rand(3,64,64)])
                self.assertEqual(len(out), 1)
                self.assertTrue({'boxes','labels','scores'} <= set(out[0]))
                saved = copy.deepcopy(model.state_dict())
                clone = build_model('lgf_original', 6, level, pretrained=False, smoke_size=64)
                clone.load_state_dict(saved, strict=True)
                self.assertTrue(all(torch.equal(clone.state_dict()[k], v) for k,v in saved.items()))
                del model, clone, saved
        self.assertEqual(core.CustomBackboneBlockBeforeFPN.__name__,
                         'CustomBackboneBlockBeforeFPN')

    def test_backward_reaches_all_selected_modules(self):
        torch.manual_seed(18)
        model = build_model('lgf_original',6,'C3_C4_C5',pretrained=False,smoke_size=64).train()
        imgs=[torch.rand(3,64,64), torch.rand(3,64,64)]
        targets=[{'boxes':torch.tensor([[10.,10.,25.,28.]]), 'labels':torch.tensor([i])}
                 for i in (0,1)]
        losses=model(imgs,targets)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        sum(losses.values()).backward()
        for k,block in model.backbone.block_fpn_in.items():
            grads=[p.grad for p in block.parameters() if p.grad is not None]
            self.assertTrue(grads, f'No gradient for LGF at feature key {k}')
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(g.abs().sum().item()>0 for g in grads),
                            f'Zero gradient for LGF at feature key {k}')

    def test_multi_levels_reject_other_methods(self):
        with self.assertRaises(ValueError):
            build_model('baseline',6,'C3_C4',pretrained=False,smoke_size=64)


if __name__=='__main__': unittest.main()
