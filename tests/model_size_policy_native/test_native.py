"""Tiny CPU construction only: no target training, generation or large allocation."""
import unittest
from unittest.mock import patch

import torch
from trinite.contracts import ContractError, ModelConfig
from trinite.model import Decoder
from model_size_policy_v1 import build_native_decoder, require_matching_count


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(blocks=1, width=4, heads=1,
                                  feed_forward_width=8, context_length=8)

    def test_lanes_match_native_without_global_rng_change(self):
        before = torch.get_rng_state().clone()
        for lane in ('dense', 'ternary', 'four-state'):
            model = build_native_decoder(self.config, lane=lane, seed=17)
            reference = Decoder(self.config, lane=lane, seed=17)
            self.assertIs(model.output_weight, model.token_embedding)
            self.assertEqual(sum(p.numel() for p in model.parameters()),
                             self.config.parameter_count)
            for (name, value), (other_name, other) in zip(
                    model.named_parameters(), reference.named_parameters(), strict=True):
                self.assertEqual(name, other_name)
                self.assertTrue(torch.equal(value, other))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_frozen_learned_parameters_and_tied_alias(self):
        model = build_native_decoder(self.config)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        actual = sum(p.numel() for p in model.parameters())
        self.assertEqual(require_matching_count(self.config.parameter_count, actual), actual)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 0)
        all_names = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
        self.assertEqual(all_names - actual, model.token_embedding.numel())

    def test_existing_5m_limit_and_per_run_budget_not_relaxed(self):
        for budget in (5_000_001, 8_000_000_000, True,
                       self.config.parameter_count - 1):
            with patch('torch.empty', side_effect=AssertionError('allocation forbidden')):
                with self.assertRaises(ContractError):
                    build_native_decoder(self.config, parameter_budget=budget)
        large = ModelConfig(blocks=24, width=1024, heads=4, feed_forward_width=4096)
        with patch('torch.empty', side_effect=AssertionError('allocation forbidden')):
            with self.assertRaises(ContractError):
                build_native_decoder(large, parameter_budget=5_000_000)

    def test_postconstruction_mismatch_is_rejected(self):
        model = Decoder(self.config)
        # Break the alias with a separately allocated learned matrix.
        model.output_weight = torch.nn.Parameter(model.token_embedding.detach().clone())
        with patch('trinite.model.Decoder', return_value=model):
            with self.assertRaisesRegex(ContractError, 'disagrees'):
                build_native_decoder(self.config)
