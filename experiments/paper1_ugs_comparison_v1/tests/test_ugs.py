import copy
import unittest

import torch
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection._utils import BoxCoder

from ugs import UGSConfig, UGSHead, nonuniform_grid, two_hot, distribution_terms, expectation, perturbation

torch.set_num_threads(2)


class QuantizationTests(unittest.TestCase):
    def test_grid_is_symmetric_monotone_and_has_exact_endpoints(self):
        grid = nonuniform_grid(UGSConfig(alpha=5, beta=1, intervals=10))
        self.assertEqual(float(grid[0]), -5)
        self.assertEqual(float(grid[-1]), 5)
        self.assertEqual(float(grid[5]), 0)
        self.assertTrue(torch.all(grid[1:] > grid[:-1]))
        torch.testing.assert_close(grid, -grid.flip(0))
        self.assertLess(float(grid[6]-grid[5]), float(grid[-1]-grid[-2]))

    def test_uniform_limit(self):
        grid = nonuniform_grid(UGSConfig(alpha=2,beta=0,intervals=10))
        torch.testing.assert_close(grid,torch.linspace(-2,2,11))

    def test_two_hot_normalizes_and_reconstructs_targets(self):
        grid = nonuniform_grid(UGSConfig())
        t = torch.cat([grid, torch.linspace(-5,5,123)])
        l,r,wl,wr=two_hot(t,grid)
        torch.testing.assert_close(wl+wr,torch.ones_like(t))
        self.assertTrue(torch.all((wl>=0)&(wr>=0)))
        torch.testing.assert_close(wl*grid[l]+wr*grid[r],t,rtol=1e-6,atol=1e-6)

    def test_out_of_range_clips_to_boundary(self):
        grid=nonuniform_grid(UGSConfig())
        l,r,wl,wr=two_hot(torch.tensor([-20.,20.]),grid)
        torch.testing.assert_close(wl*grid[l]+wr*grid[r],torch.tensor([-5.,5.]))

    def test_ce_gradient_is_probability_minus_normalized_target(self):
        grid=nonuniform_grid(UGSConfig())
        logits=torch.linspace(-1,1,44).reshape(1,4,11).requires_grad_()
        targets=torch.tensor([[0.2,-0.4,1.5,-3.]])
        ce,_=distribution_terms(logits,targets,grid)
        gradient,=torch.autograd.grad(ce.sum(),logits)
        l,r,wl,wr=two_hot(targets,grid)
        expected=logits.softmax(-1).detach()
        expected.scatter_add_(-1,l.unsqueeze(-1),-wl.unsqueeze(-1))
        expected.scatter_add_(-1,r.unsqueeze(-1),-wr.unsqueeze(-1))
        torch.testing.assert_close(gradient,expected,rtol=1e-5,atol=1e-7)
        self.assertLessEqual(float(gradient.abs().max()),1)
        # Summing 11 FP32 terms can leave a residual of one machine epsilon.
        torch.testing.assert_close(gradient.sum(-1),torch.zeros_like(targets),atol=2e-7,rtol=0)

    def test_entropy_gradient_has_softmax_chain_rule(self):
        grid=nonuniform_grid(UGSConfig())
        logits=torch.linspace(-2,3,44).reshape(1,4,11).requires_grad_()
        _,entropy=distribution_terms(logits,torch.zeros(1,4),grid)
        grad,=torch.autograd.grad(entropy.sum(),logits)
        p=logits.softmax(-1)
        expected=-p*(logits.log_softmax(-1)+entropy.unsqueeze(-1))
        torch.testing.assert_close(grad,expected,atol=1e-7,rtol=1e-5)

    def test_nonuniform_expectation_decodes_known_boxes(self):
        coder=BoxCoder((1.,1.,1.,1.))
        anchors=torch.tensor([[0.,0.,32.,32.],[10.,5.,26.,37.]])
        boxes=torch.tensor([[4.,8.,24.,28.],[9.,8.,29.,33.]])
        encoded=coder.encode_single(boxes,anchors)
        grid=nonuniform_grid(UGSConfig())
        l,r,wl,wr=two_hot(encoded,grid)
        probs=torch.zeros(2,4,len(grid))
        probs.scatter_add_(-1,l.unsqueeze(-1),wl.unsqueeze(-1))
        probs.scatter_add_(-1,r.unsqueeze(-1),wr.unsqueeze(-1))
        logits=probs.clamp_min(1e-30).log()
        decoded=coder.decode_single(expectation(logits,grid),anchors)
        torch.testing.assert_close(decoded,boxes,atol=1e-5,rtol=1e-6)

    def test_perturbation_norm_and_zero_gradient(self):
        g=torch.arange(48,dtype=torch.float32).reshape(2,3,2,4)
        g[1].zero_()
        e=perturbation(g.requires_grad_(),0.05)
        self.assertFalse(e.requires_grad)
        torch.testing.assert_close(torch.linalg.vector_norm(e.flatten(1),dim=1),torch.tensor([0.05,0.]),atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(e,perturbation(g*100,0.05),atol=1e-7,rtol=1e-5)

    def test_invalid_hyperparameters_rejected(self):
        for values in [{"alpha":0},{"beta":-1},{"rho":float('nan')},{"intervals":2.5}]:
            with self.assertRaises(ValueError):UGSConfig(**values)


class HeadTests(unittest.TestCase):
    def setup_problem(self, *, empty=False, config=None, gradients=True):
        torch.manual_seed(42)
        head=UGSHead(RetinaNetHead(8,3,6),config or UGSConfig(feature_reg_weight=0.01))
        head.train()
        # Avoid an exactly uniform initialization in the directional-gradient test.
        torch.nn.init.normal_(head.regression_head.bbox_distribution.weight,std=0.1)
        features=[torch.randn(2,8,8,8,requires_grad=gradients),torch.randn(2,8,4,4,requires_grad=gradients)]
        targets=[{"boxes":torch.tensor([[2.,3.,12.,13.]]) if not empty else torch.zeros(0,4),
                  "labels":torch.tensor([0]) if not empty else torch.zeros(0,dtype=torch.long)} for _ in range(2)]
        anchors=[torch.tensor([[0.,0.,16.,16.]]).repeat(240,1) for _ in range(2)]
        matched=[torch.full((240,),-1,dtype=torch.long) for _ in range(2)]
        if not empty:
            for m in matched:m[torch.tensor([0,5,191,192,239])]=0
        return head,features,targets,anchors,matched

    def test_full_loss_backpropagates_to_features_and_distribution_layer(self):
        head,f,t,a,m=self.setup_problem()
        losses=head.compute_loss(t,head(f),a,m)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        sum(losses.values()).backward()
        self.assertGreater(float(head.regression_head.bbox_distribution.weight.grad.abs().sum()),0)
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in f))
        self.assertIsNone(head._features)
        self.assertGreater(head.last_stats['nonzero_perturbation_directions'],0)
        self.assertLessEqual(head.last_stats['max_perturbation_l2'],0.050001)

    def test_frozen_backbone_features_support_refinement(self):
        head,f,t,a,m=self.setup_problem(gradients=False)
        losses=head.compute_loss(t,head(f),a,m)
        sum(losses.values()).backward()
        self.assertTrue(all(x.grad is None for x in f))
        self.assertGreater(head.last_stats['nonzero_perturbation_directions'],0)

    def test_empty_targets_have_zero_localization_loss(self):
        head,f,t,a,m=self.setup_problem(empty=True)
        losses=head.compute_loss(t,head(f),a,m)
        for k in ['ugs_ce','ugs_um','ugs_ur']:self.assertEqual(float(losses[k].detach()),0)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(losses['classification']))

    def test_ugs_does_not_change_classification_loss(self):
        head,f,t,a,m=self.setup_problem()
        outputs=head(f)
        native=head.classification_head.compute_loss(t,outputs,m)
        losses=head.compute_loss(t,outputs,a,m)
        torch.testing.assert_close(native,losses['classification'],rtol=0,atol=0)

    def test_refinement_has_no_inference_branch(self):
        head,f,t,a,m=self.setup_problem()
        head.eval()
        with torch.no_grad():outputs=head(f)
        self.assertEqual(set(outputs),{'cls_logits','bbox_regression'})
        self.assertIsNone(head._features)
        self.assertEqual(outputs['bbox_regression'].shape,(2,240,4))

    def test_ablation_is_labelled_and_removes_auxiliary_losses(self):
        cfg=UGSConfig(um_weight=0,ur_weight=0)
        head,f,t,a,m=self.setup_problem(config=cfg)
        losses=head.compute_loss(t,head(f),a,m)
        self.assertEqual(float(losses['ugs_um'].detach()),0)
        self.assertEqual(float(losses['ugs_ur'].detach()),0)
        self.assertIn('; CE)',cfg.label)


if __name__=='__main__':unittest.main()
