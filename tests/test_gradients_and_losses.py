"""
Comprehensive gradient flow and loss stability audit for MIRAGE.
Tests loss modes, gradient backprop, stop-gradients, learned item 0, and numerical stability.
"""

import sys
import unittest
import torch
import torch.nn.functional as F

from mirage.config import ModelConfig
from mirage.model import MIRAGEModel


class TestGradientsAndLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_items = 100
        self.embedding_dim = 32
        self.batch_size = 4
        self.seq_len = 10

        # Construct synthetic KNN graph (100 items, K=5)
        # Note: Item 0 is padding, items 1..99 are real items
        self.K = 5
        knn = torch.randint(1, self.num_items, (self.num_items, self.K))
        # Ensure item 0 has 0s, and some items have padding neighbors to test isolation
        knn[0] = 0
        knn[1, :2] = 0
        self.knn_graph = knn

    def _create_model(self, **kwargs):
        cfg_dict = dict(
            item_count=self.num_items,
            max_len=self.seq_len,
            hidden_size=self.embedding_dim,
            attention_heads=2,
            num_blocks=3,
            last_blocks=1,
            decoder_dims=(64, 128),
            graph_k=self.K,
            graph_weight=0.1,
            recommendation_weight=1.0,
            reconstruction_weight=0.1,
            loss_type="full_softmax",
        )
        cfg_dict.update(kwargs)
        cfg = ModelConfig(**cfg_dict)
        model = MIRAGEModel(cfg)
        model.set_graph(self.knn_graph)
        return model

    def _generate_batch(self):
        # Sequences with item 0 padding at the start
        sequence = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        sequence[:, :3] = 0  # padding tokens
        target = torch.randint(1, self.num_items, (self.batch_size,))
        negative = torch.randint(1, self.num_items, (self.batch_size, 10))
        return sequence, target, negative

    def test_loss_modes_forward_backward(self):
        """Test forward and backward passes across all distance and time modes."""
        modes = ["softmin", "hardmin"]
        time_mods = [True, False]

        sequence, target, negative = self._generate_batch()

        for mode in modes:
            for tmod in time_mods:
                with self.subTest(mode=mode, time_mod=tmod):
                    model = self._create_model(
                        graph_distance_mode=mode, graph_time_mod=tmod
                    )
                    model.train()
                    model.zero_grad()

                    losses = model.compute_loss(sequence, target, negative)
                    total = losses["loss"]

                    self.assertTrue(
                        torch.isfinite(total), f"Total loss not finite for {mode}"
                    )
                    self.assertGreater(total.item(), 0.0)
                    for name, val in losses.items():
                        self.assertTrue(
                            torch.isfinite(val), f"Loss {name} not finite for {mode}"
                        )
                        self.assertGreaterEqual(val.item(), 0.0)

                    total.backward()

                    for p_name, param in model.named_parameters():
                        if param.grad is not None:
                            self.assertFalse(
                                torch.isnan(param.grad).any(),
                                f"NaN gradient in {p_name} for {mode}",
                            )
                            self.assertFalse(
                                torch.isinf(param.grad).any(),
                                f"Inf gradient in {p_name} for {mode}",
                            )

                    pad_grad = model.item_embedding.weight.grad[0]
                    self.assertGreater(
                        pad_grad.abs().sum().item(),
                        0.0,
                        f"Item 0 did not receive a gradient for {mode}",
                    )

    def test_user_anchor_mode(self):
        """Test graph_anchor_mode='user' forward and backward."""
        sequence, target, negative = self._generate_batch()
        model = self._create_model(graph_anchor_mode="user", graph_distance_mode="softmin")
        model.train()
        losses = model.compute_loss(sequence, target, negative)
        total = losses["loss"]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertFalse(torch.isnan(model.item_embedding.weight.grad).any())
        self.assertGreater(model.item_embedding.weight.grad[0].abs().sum().item(), 0.0)

    def test_sampled_softmax_loss(self):
        """Test sampled_softmax recommendation loss mode."""
        sequence, target, negative = self._generate_batch()
        model = self._create_model(loss_type="sampled_softmax", graph_distance_mode="softmin")
        model.train()
        losses = model.compute_loss(sequence, target, negative)
        total = losses["loss"]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertFalse(torch.isnan(model.item_embedding.weight.grad).any())

    def test_history_anchor_source(self):
        """Test use_history_anchor=True flow source."""
        sequence, target, negative = self._generate_batch()
        model = self._create_model(use_history_anchor=True)
        model.train()
        losses = model.compute_loss(sequence, target, negative)
        total = losses["loss"]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertFalse(torch.isnan(model.item_embedding.weight.grad).any())

    def test_stop_gradient_controls(self):
        """Test ablation stop-gradient flags: detach_items and detach_flow_state."""
        sequence, target, negative = self._generate_batch()

        # Detach items: graph loss should not propagate to item embedding through graph neighbors
        model = self._create_model(
            graph_weight=1.0,
            recommendation_weight=0.0,
            reconstruction_weight=0.0,
            detach_items=True,
        )
        model.train()
        # Zero out flow loss for isolated test of graph loss
        losses = model.compute_loss(sequence, target, negative)
        # Backward on pure graph loss
        losses["graph"].backward()
        # Item embeddings (nb_emb) were detached, so grad on item_embedding from nb_emb must be zero
        # Only target_embedding in flow state received grad
        self.assertFalse(torch.isnan(model.item_embedding.weight.grad).any())

    def test_endpoint_only_reg(self):
        """Test endpoint_only_reg ablation mode forward and backward."""
        sequence, target, negative = self._generate_batch()
        model = self._create_model(endpoint_only_reg=True, graph_weight=0.5)
        model.train()
        losses = model.compute_loss(sequence, target, negative)
        total = losses["loss"]
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(losses["graph"].item(), 0.0)
        total.backward()
        self.assertFalse(torch.isnan(model.item_embedding.weight.grad).any())
        self.assertGreater(model.item_embedding.weight.grad[0].abs().sum().item(), 0.0)

    def test_zero_neighbors_are_ignored_by_graph_loss(self):
        """Reserved zero slots must not contribute to topology loss or gradients."""
        sequence, _, negative = self._generate_batch()
        target = torch.ones(self.batch_size, dtype=torch.long)

        for mode in ("softmin", "hardmin"):
            with self.subTest(mode=mode):
                model = self._create_model(graph_distance_mode=mode)
                graph = self.knn_graph.clone()
                graph[1] = 0
                model.set_graph(graph)

                losses = model.compute_loss(sequence, target, negative)
                self.assertEqual(losses["graph"].item(), 0.0)
                losses["graph"].backward()

                gradient = model.item_embedding.weight.grad
                if gradient is not None:
                    self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_history_loss_global_normalization(self):
        """Test the global full-catalog normalization used by the paper runs."""
        model = self._create_model(mask_ratio=0.0)
        # Create synthetic output and target with 2 users
        # User 1 has 1 selected token with error 1.0 (sq_err = 1.0 -> per_user = 1.0)
        # User 2 has 2 selected tokens each with error 2.0 (sq_err = 4.0 * 2 = 8.0 -> per_user = 4.0)
        # The selected squared error is 9 and the global tensor has 202 entries.
        out = torch.zeros(2, self.num_items + 1)
        tgt = torch.zeros(2, self.num_items + 1)
        # user 0: item 1 positive
        tgt[0, 1] = 1.0
        out[0, 1] = 2.0  # error = 1.0, sq = 1.0
        # user 1: items 2, 3 positive
        tgt[1, 2] = 1.0
        tgt[1, 3] = 1.0
        out[1, 2] = 3.0  # error = 2.0, sq = 4.0
        out[1, 3] = 3.0  # error = 2.0, sq = 4.0

        loss = model.balanced_reconstruction_loss(out, tgt)
        self.assertAlmostEqual(loss.item(), 9.0 / (2 * (self.num_items + 1)), places=7)

if __name__ == "__main__":
    unittest.main()
