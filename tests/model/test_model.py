from pathlib import Path
import unittest

import torch

from trinite.contracts import ContractError, ModelConfig, parse_json, identity
from trinite.model import CaptureSpec, Decoder
from oracle import forward, hand_parameters

ROOT = Path(__file__).resolve().parents[2]
SMALL = ModelConfig(context_length=8, blocks=1, width=4, heads=2, feed_forward_width=8)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class ModelTests(unittest.TestCase):
    def test_reference_parameter_inventory_and_matched_initialization(self):
        rng = torch.get_rng_state().clone()
        dense, ternary = Decoder(lane="dense", seed=37), Decoder(lane="ternary", seed=37)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in dense.parameters()), 1247232)
        self.assertEqual(sum(p.numel() for p in ternary.parameters()), 1247232)
        self.assertIs(dense.output_weight, dense.token_embedding)
        for (a, x), (b, y) in zip(dense.named_parameters(), ternary.named_parameters()):
            self.assertEqual(a, b)
            self.assertEqual(x.dtype, torch.float32)
            self.assertTrue(torch.equal(x, y))
        ids = torch.tensor([[256, 49, 257]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            a, b = dense(ids, mask), ternary(ids, mask)
        self.assertEqual(a.logits.shape, (1, 3, 259))
        self.assertTrue(bool(torch.isfinite(b.logits).all()))
        self.assertFalse(torch.equal(a.logits, b.logits))

    def test_independent_dense_and_ternary_forward_fixtures(self):
        parameters = hand_parameters()
        fixture = parse_json((ROOT/"fixtures/model-v0/forward.json").read_bytes(), canonical=True)
        self.assertEqual(fixture["oracle_identity"], identity((ROOT/"tests/model/oracle.py").read_bytes()))
        self.assertEqual(fixture["absolute_tolerance"], "0.000002")
        self.assertEqual(fixture["relative_tolerance"], "0.00001")
        ids, mask = fixture["ids"], fixture["mask"]
        for lane in ("dense", "ternary"):
            model = Decoder(SMALL, lane=lane)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(torch.tensor(parameters[name]))
            output = model(torch.tensor([ids]), torch.tensor([mask]),
                           capture=CaptureSpec(("embedding", "block.0", "final"), (0, 1, 2, 3)))
            expected, states = forward(parameters, ids, mask, lane)
            retained = [[float.fromhex(n) for n in row] for row in fixture[lane]["logits"]]
            self.assertTrue(torch.allclose(torch.tensor(expected, dtype=torch.float64),
                                           torch.tensor(retained, dtype=torch.float64),
                                           atol=2e-6, rtol=1e-5))
            self.assertTrue(torch.allclose(output.logits[0], torch.tensor(retained), atol=2e-6, rtol=1e-5))
            for name, state in states.items():
                self.assertTrue(torch.allclose(output.captures[name][0], torch.tensor(state),
                                               atol=2e-6, rtol=1e-5), (lane, name))

    def test_causality_padding_batch_isolation_and_prefix_parity(self):
        for lane in ("dense", "ternary"):
            model = Decoder(SMALL, lane=lane, seed=7)
            ids = torch.tensor([[256, 65, 66, 258, 258], [256, 70, 71, 72, 73]])
            mask = torch.tensor([[True, True, True, False, False], [True]*5])
            original = model(ids, mask).logits
            changed = ids.clone(); changed[0, 2:] = torch.tensor([99, 12, 21]); changed[1, :] = 42
            altered = model(changed, mask).logits
            self.assertTrue(torch.equal(original[0, :2], altered[0, :2]))
            padding = ids.clone(); padding[0, 3:] = torch.tensor([12, 21])
            self.assertTrue(torch.equal(original, model(padding, mask).logits))
            prefix = model(ids[:1, :3], mask[:1, :3]).logits
            self.assertTrue(torch.allclose(original[:1, :3], prefix, atol=2e-6, rtol=1e-5))
            self.assertEqual(int(torch.count_nonzero(original[0, 3:])), 0)

    def test_backward_reaches_all_named_parameters_without_masked_loss(self):
        for lane in ("dense", "ternary"):
            model = Decoder(SMALL, lane=lane)
            ids = torch.tensor([[256, 65, 257, 258]])
            mask = torch.tensor([[True, True, True, False]])
            logits = model(ids, mask).logits
            torch.nn.functional.cross_entropy(logits[0, :2], ids[0, 1:3]).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertEqual(int(torch.count_nonzero(model.position_embedding.grad[3:])), 0)

    def test_capture_is_detached_selected_and_does_not_change_forward(self):
        model = Decoder(SMALL)
        ids = torch.tensor([[256, 65, 257]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        rng = torch.get_rng_state().clone()
        plain = model(ids, mask)
        captured = model(ids, mask, capture=CaptureSpec(("block.0", "final"), (2, 0), 64))
        self.assertTrue(torch.equal(plain.logits, captured.logits))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for state in captured.captures.values():
            self.assertEqual(state.shape, (1, 2, 4))
            self.assertFalse(state.requires_grad)
            self.assertIsNone(state.grad_fn)
        captured.captures["final"].fill_(999)
        self.assertTrue(torch.equal(plain.logits, model(ids, mask).logits))
        plain.logits.sum().backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad()
        model(ids, mask, capture=CaptureSpec(("final",), (0,))).logits.sum().backward()
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(gradients[name], p.grad), name)

    def test_invalid_inputs_capture_and_allocation_budgets(self):
        model = Decoder(SMALL)
        ids = torch.tensor([[256, 65, 257]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        invalid = [(ids.float(), mask), (ids, mask.int()), (ids, mask[:, :2]),
                   (torch.tensor([[-1, 0, 0]]), mask), (torch.tensor([[259, 0, 0]]), mask),
                   (ids, torch.tensor([[True, False, True]])), (ids, ~mask),
                   (ids.repeat(9, 1), mask.repeat(9, 1)),
                   (ids.repeat(1, 3), mask.repeat(1, 3)),
                   (ids.to("meta"), mask)]
        for a, b in invalid:
            with self.assertRaises(ContractError): model(a, b)
        for capture in (CaptureSpec(("missing",), (0,)), CaptureSpec(("final", "final"), (0,)),
                        CaptureSpec(("final",), (3,)), CaptureSpec(("final",), (True,)),
                        CaptureSpec(("final",), (0, 0)), CaptureSpec(("final",), (0,), 15)):
            with self.assertRaises(ContractError): model(ids, mask, capture=capture)
        with self.assertRaises(ContractError): Decoder(parameter_budget=100)
        with self.assertRaises(ContractError): Decoder(lane="fallback")
        oversized_attention = Decoder(ModelConfig(context_length=256, blocks=1, width=128,
                                                  heads=128, feed_forward_width=1))
        with self.assertRaises(ContractError):
            oversized_attention(torch.zeros((8, 256), dtype=torch.int64),
                                torch.ones((8, 256), dtype=torch.bool))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaises(ContractError): model(ids, mask)
        with torch.no_grad(): model.token_embedding[0, 0] = float("inf")
        with self.assertRaises(ContractError): model(ids, mask)

    def test_modified_shape_dtype_sharing_or_lane_is_rejected(self):
        ids = torch.tensor([[256, 257]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        mutations = [lambda m: setattr(m.blocks[0].q, "weight", torch.nn.Parameter(torch.ones(1, 1))),
                     lambda m: m.double(),
                     lambda m: setattr(m, "output_weight", torch.nn.Parameter(m.token_embedding.clone())),
                     lambda m: setattr(m.blocks[0].q, "lane", "dense"),
                     lambda m: setattr(m.blocks[0].attention_norm, "epsilon", 1.),
                     lambda m: setattr(m.blocks[0], "heads", 1),
                     lambda m: setattr(m, "config", ModelConfig())]
        for mutate in mutations:
            model = Decoder(SMALL)
            mutate(model)
            with self.assertRaises(ContractError): model(ids, mask)
