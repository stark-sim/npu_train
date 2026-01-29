"""
Tests for MoE Tensor Parallelism

Tests cover:
- AllToAll communication
- Expert sharding
- Router top-k selection
- Full MoE layer forward/backward
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npu_parallel.tp_moe import (
    AllToAllFromTensor,
    TPMoEExperts,
    TPMoERouter,
    TPMoELayer,
    TPDeepSeekMoE,
)


class TestAllToAll(unittest.TestCase):
    """Test AllToAll communication primitive"""

    def test_alltoall_autograd_function(self):
        """Test AllToAllFromTensor autograd function"""
        # This test doesn't require distributed setup
        # We just verify the function structure is correct
        self.assertTrue(hasattr(AllToAllFromTensor, 'forward'))
        self.assertTrue(hasattr(AllToAllFromTensor, 'backward'))
        self.assertTrue(callable(AllToAllFromTensor.forward))
        self.assertTrue(callable(AllToAllFromTensor.backward))


class TestMoERouter(unittest.TestCase):
    """Test MoE Router"""

    def setUp(self):
        self.hidden_size = 128
        self.num_experts = 8
        self.top_k = 2
        self.device = torch.device("cpu")

        self.router = TPMoERouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=torch.float32,
            device=self.device,
        )

    def test_router_forward(self):
        """Test router forward pass"""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, self.hidden_size, device=self.device)

        router_logits, routing_weights, expert_indices = self.router(x)

        # Check output shapes
        self.assertEqual(router_logits.shape, (batch, seq_len, self.num_experts))
        self.assertEqual(routing_weights.shape, (batch * seq_len, self.top_k))
        self.assertEqual(expert_indices.shape, (batch * seq_len, self.top_k))

        # Check top-k selection
        self.assertTrue(torch.all(expert_indices >= 0))
        self.assertTrue(torch.all(expert_indices < self.num_experts))

        # Check weights sum to approximately 1 (per token)
        weight_sums = routing_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))

    def test_aux_loss_computation(self):
        """Test auxiliary loss computation"""
        batch, seq_len = 4, 16
        x = torch.randn(batch, seq_len, self.hidden_size, device=self.device)

        router_logits, routing_weights, expert_indices = self.router(x)
        router_logits_flat = router_logits.view(-1, self.num_experts)

        aux_loss = self.router.compute_aux_loss(router_logits_flat, expert_indices)

        # Aux loss should be a scalar
        self.assertEqual(aux_loss.shape, ())
        self.assertTrue(aux_loss >= 0)

    def test_router_parameters(self):
        """Test router has correct parameters"""
        self.assertEqual(self.router.gate.in_features, self.hidden_size)
        self.assertEqual(self.router.gate.out_features, self.num_experts)
        self.assertEqual(self.router.top_k, self.top_k)


class TestMoEExperts(unittest.TestCase):
    """Test MoE Experts"""

    def setUp(self):
        self.hidden_size = 128
        self.intermediate_size = 256
        self.num_experts = 8
        self.tp_size = 2
        self.rank = 0
        self.device = torch.device("cpu")

        self.experts = TPMoEExperts(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            tp_size=self.tp_size,
            rank=self.rank,
            dtype=torch.float32,
            device=self.device,
        )

    def test_experts_initialization(self):
        """Test experts are initialized correctly"""
        self.assertEqual(self.experts.num_local_experts, self.num_experts // self.tp_size)
        self.assertEqual(len(self.experts.experts), self.num_experts // self.tp_size)

        # Check each expert has the required layers
        for expert in self.experts.experts:
            self.assertIn('gate_proj', expert)
            self.assertIn('up_proj', expert)
            self.assertIn('down_proj', expert)

    def test_expert_forward(self):
        """Test expert forward pass"""
        batch_tokens = 16
        x = torch.randn(batch_tokens, self.hidden_size, device=self.device)

        # Assign all tokens to first expert
        expert_indices = torch.full((batch_tokens,), self.experts.expert_start_idx, device=self.device)

        output = self.experts(x, expert_indices)

        # Check output shape
        self.assertEqual(output.shape, (batch_tokens, self.hidden_size))

    def test_multiple_experts(self):
        """Test processing tokens assigned to different experts"""
        batch_tokens = 32
        x = torch.randn(batch_tokens, self.hidden_size, device=self.device)

        # Assign tokens to different local experts
        num_local = self.experts.num_local_experts
        expert_indices = torch.tensor(
            [self.experts.expert_start_idx + (i % num_local) for i in range(batch_tokens)],
            device=self.device
        )

        output = self.experts(x, expert_indices)

        # Check output shape
        self.assertEqual(output.shape, (batch_tokens, self.hidden_size))


class TestMoELayer(unittest.TestCase):
    """Test Complete MoE Layer"""

    def setUp(self):
        self.hidden_size = 128
        self.intermediate_size = 256
        self.num_experts = 8
        self.top_k = 2
        self.tp_size = 1
        self.rank = 0
        self.device = torch.device("cpu")

        self.moe = TPMoELayer(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            tp_size=self.tp_size,
            rank=self.rank,
            dtype=torch.float32,
            device=self.device,
        )

    def test_moe_forward(self):
        """Test MoE layer forward pass"""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, self.hidden_size, device=self.device)

        output = self.moe(x, return_aux_loss=False)

        # Check output shape
        self.assertEqual(output.shape, (batch, seq_len, self.hidden_size))

    def test_moe_forward_with_aux_loss(self):
        """Test MoE layer forward pass with auxiliary loss"""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, self.hidden_size, device=self.device)

        output, aux_loss = self.moe(x, return_aux_loss=True)

        # Check output shape
        self.assertEqual(output.shape, (batch, seq_len, self.hidden_size))

        # Check aux loss is a scalar
        self.assertEqual(aux_loss.shape, ())

    def test_moe_backward(self):
        """Test MoE layer backward pass"""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, self.hidden_size, device=self.device, requires_grad=True)

        output, aux_loss = self.moe(x, return_aux_loss=True)

        # Compute loss
        loss = output.sum() + aux_loss
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # Check router gradients exist
        self.assertIsNotNone(self.moe.router.gate.weight.grad)

        # Check expert gradients exist
        for expert in self.moe.experts.experts:
            if expert is not None:
                # Check that at least some parameters have gradients
                grad_params = [p for p in expert.parameters() if p.grad is not None]
                self.assertGreater(len(grad_params), 0)


class TestDeepSeekMoE(unittest.TestCase):
    """Test DeepSeek-style MoE with shared and routed experts"""

    def setUp(self):
        self.hidden_size = 128
        self.intermediate_size = 256
        self.num_shared_experts = 2
        self.num_routed_experts = 8
        self.top_k = 2
        self.device = torch.device("cpu")

        self.moe = TPDeepSeekMoE(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            top_k=self.top_k,
            tp_size=1,
            rank=0,
            dtype=torch.float32,
            device=self.device,
        )

    def test_forward(self):
        """Test forward pass"""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, self.hidden_size, device=self.device)

        output = self.moe(x, return_aux_loss=False)

        # Check output shape
        self.assertEqual(output.shape, (batch, seq_len, self.hidden_size))

    def test_forward_with_aux_loss(self):
        """Test forward pass with auxiliary loss"""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, self.hidden_size, device=self.device)

        output, aux_loss = self.moe(x, return_aux_loss=True)

        # Check output shape
        self.assertEqual(output.shape, (batch, seq_len, self.hidden_size))
        self.assertEqual(aux_loss.shape, ())


class TestMoEIntegration(unittest.TestCase):
    """Integration tests for MoE with TP"""

    def test_moe_tp_sharding(self):
        """Test that experts are correctly sharded across TP ranks"""
        hidden_size = 128
        intermediate_size = 256
        num_experts = 8
        tp_size = 4

        for rank in range(tp_size):
            experts = TPMoEExperts(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                tp_size=tp_size,
                rank=rank,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

            # Each rank should have num_experts // tp_size experts
            expected_local = num_experts // tp_size
            self.assertEqual(experts.num_local_experts, expected_local)

            # Expert range should be correct
            expected_start = rank * expected_local
            expected_end = expected_start + expected_local
            self.assertEqual(experts.expert_start_idx, expected_start)
            self.assertEqual(experts.expert_end_idx, expected_end)

            # All experts should be in the correct range
            total_experts_covered = sum(
                TPMoEExperts(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    tp_size=tp_size,
                    rank=r,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                ).num_local_experts
                for r in range(tp_size)
            )
            self.assertEqual(total_experts_covered, num_experts)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAllToAll))
    suite.addTests(loader.loadTestsFromTestCase(TestMoERouter))
    suite.addTests(loader.loadTestsFromTestCase(TestMoEExperts))
    suite.addTests(loader.loadTestsFromTestCase(TestMoELayer))
    suite.addTests(loader.loadTestsFromTestCase(TestDeepSeekMoE))
    suite.addTests(loader.loadTestsFromTestCase(TestMoEIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_tests())
