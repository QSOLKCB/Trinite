"""CPU engineering conformance; no optimizer, research run or learned claim."""
from dataclasses import replace
import unittest

import torch
from torch import nn

from trinite.contracts import ContractError, ModelConfig
from trinite.model import CaptureSpec, Decoder, Linear
from position_coupling_v1.model import PositionDecoder
from position_coupling_v1.positions import prefix_positions

SMALL = ModelConfig(context_length=16, blocks=2, width=8, heads=2, feed_forward_width=16)
SEEDS = (0, 7, 37)


class PositionDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def inputs(self):
        ids = torch.tensor([[256, 49, 43, 50, 61, 258], [256, 51, 43, 52, 61, 55]])
        mask = torch.tensor([[True] * 5 + [False], [True] * 6])
        positions = torch.arange(6).expand_as(ids)
        return ids, mask, positions

    def test_native_sequence_exact_initialization_logits_gradients_and_captures(self):
        ids, mask, positions = self.inputs()
        capture = CaptureSpec(('embedding', 'block.0', 'block.1', 'final'), (4, 0, 5))
        for seed in SEEDS:
            native = Decoder(SMALL, lane='dense', seed=seed)
            adapter = PositionDecoder(SMALL, seed=seed)
            self.assertEqual(list(native.state_dict()), list(adapter.state_dict()))
            for name, tensor in native.state_dict().items():
                self.assertTrue(torch.equal(tensor, adapter.state_dict()[name]), name)
            a = native(ids, mask, capture=capture)
            b = adapter(ids, mask, position_ids=positions, capture=capture)
            self.assertTrue(torch.equal(a.logits, b.logits))
            for name in a.captures:
                self.assertTrue(torch.equal(a.captures[name], b.captures[name]), name)
            # The same shifted next-token objective reaches the same native tensors.
            for output in (a, b):
                torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 259),
                                                   ids[:, 1:].reshape(-1)).backward()
            for (name, x), (other, y) in zip(native.named_parameters(), adapter.named_parameters()):
                self.assertEqual(name, other)
                self.assertIsNotNone(y.grad)
                self.assertTrue(torch.equal(x.grad, y.grad), name)

    def test_small_target_inventory_and_tied_output(self):
        config = ModelConfig(context_length=64, blocks=2, width=64, heads=4, feed_forward_width=128)
        model = PositionDecoder(config)
        self.assertEqual(config.parameter_count, 86528)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 86528)
        self.assertIs(model.output_weight, model.token_embedding)
        self.assertTrue(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(m.lane == 'dense' for m in model.modules() if isinstance(m, Linear)))
        with self.assertRaises(ContractError):
            PositionDecoder(config, parameter_budget=86527)

    def test_repeated_and_descending_positions_never_leak_future_tokens(self):
        ids, mask, _ = self.inputs()
        for seed in SEEDS:
            model = PositionDecoder(SMALL, seed=seed)
            for positions in (torch.zeros_like(ids), torch.tensor([[5, 4, 3, 2, 1, 0]]).expand_as(ids)):
                original = model(ids, mask, position_ids=positions).logits
                for cut in range(1, ids.shape[1]):
                    changed = ids.clone(); changed[:, cut:] = 99
                    shifted = positions.clone(); shifted[:, cut:] = 15
                    actual = model(changed, mask, position_ids=shifted).logits
                    self.assertTrue(torch.equal(original[:, :cut], actual[:, :cut]), (seed, cut))

    def test_padding_batch_and_prefix_invariance_with_explicit_positions(self):
        ids, mask, _ = self.inputs()
        positions = torch.tensor([[0, 1, 0, 1, 0, 15], [0, 4, 0, 4, 0, 5]])
        capture = CaptureSpec(('embedding', 'block.0', 'block.1', 'final'), tuple(range(6)))
        for seed in SEEDS:
            model = PositionDecoder(SMALL, seed=seed)
            original = model(ids, mask, position_ids=positions, capture=capture)
            changed = ids.clone(); changed[0, 5] = 33; changed[1] = 99
            shifted = positions.clone(); shifted[0, 5] = 0; shifted[1] = 15
            actual = model(changed, mask, position_ids=shifted, capture=capture)
            self.assertTrue(torch.equal(original.logits[0], actual.logits[0]))
            self.assertEqual(torch.count_nonzero(original.logits[0, 5]).item(), 0)
            for name in original.captures:
                self.assertTrue(torch.equal(original.captures[name][0], actual.captures[name][0]))
                self.assertEqual(torch.count_nonzero(original.captures[name][0, 5]).item(), 0)
            for cut in range(1, 6):
                prefix = model(ids[:1, :cut], mask[:1, :cut], position_ids=positions[:1, :cut]).logits
                torch.testing.assert_close(prefix, original.logits[:1, :cut], atol=2e-6, rtol=1e-5)

    def test_capture_and_forward_preserve_rng_inputs_and_gradients(self):
        ids, mask, positions = self.inputs()
        inputs = [x.clone() for x in (ids, mask, positions)]
        rng = torch.get_rng_state().clone()
        model = PositionDecoder(SMALL, seed=7)
        plain = model(ids, mask, position_ids=positions)
        captured = model(ids, mask, position_ids=positions,
                         capture=CaptureSpec(('embedding', 'final'), (4, 0)))
        self.assertTrue(torch.equal(plain.logits, captured.logits))
        plain.logits.sum().backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad()
        captured.logits.sum().backward()
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(gradients[name], parameter.grad), name)
        for value in captured.captures.values():
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            value.fill_(999)
        self.assertTrue(torch.equal(plain.logits, model(ids, mask, position_ids=positions).logits))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for before, after in zip(inputs, (ids, mask, positions)):
            self.assertTrue(torch.equal(before, after))

    def test_embedding_uses_current_input_slot_and_gradient_accumulates_addresses(self):
        prompt = '1+2='
        ids = torch.tensor([[256, *prompt.encode(), 51, 48, 257]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.tensor([prefix_positions(prompt, 3, mode='coupled')])
        model = PositionDecoder(SMALL)
        output = model(ids, mask, position_ids=positions,
                       capture=CaptureSpec(('embedding',), tuple(range(ids.shape[1]))))
        expected = model.token_embedding[ids] + model.position_embedding[positions]
        self.assertTrue(torch.equal(output.captures['embedding'], expected))
        self.assertEqual(positions.tolist(), [[0, 1, 0, 1, 0, 1, 2, 0]])
        output.logits.sum().backward()
        self.assertEqual(torch.count_nonzero(model.position_embedding.grad[3:]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.position_embedding.grad[:3]).item(), 0)

    def test_invalid_position_tensors_are_rejected(self):
        ids, mask, positions = self.inputs()
        model = PositionDecoder(SMALL)
        invalid = [None, positions.tolist(), positions.float(), positions.int(), positions.bool(),
                   positions[0], positions[:, :-1], positions.unsqueeze(-1), positions.to('meta'),
                   positions.to_sparse(), torch.full_like(positions, -1), torch.full_like(positions, 16)]
        for value in invalid:
            with self.subTest(value=str(value)), self.assertRaises(ContractError):
                model(ids, mask, position_ids=value)
        with self.assertRaises(TypeError):
            model(ids, mask)

    def test_native_input_capture_and_autocast_guards_are_retained(self):
        ids, mask, positions = self.inputs()
        model = PositionDecoder(SMALL)
        invalid = [(ids.float(), mask), (ids, mask.int()), (ids, mask[:, :-1]),
                   (ids.to('meta'), mask), (ids, mask.to('meta')), (ids.to_sparse(), mask),
                   (torch.full_like(ids, -1), mask), (torch.full_like(ids, 259), mask),
                   (ids, torch.zeros_like(mask)), (ids[:, :0], mask[:, :0]),
                   (ids[:0], mask[:0]), (ids.repeat(5, 1), mask.repeat(5, 1))]
        gap = mask.clone(); gap[0, 1] = False; invalid.append((ids, gap))
        for a, b in invalid:
            with self.assertRaises(ContractError):
                model(a, b, position_ids=positions)
        for capture in ('final', CaptureSpec(('bad',), (0,)), CaptureSpec(('final',), (6,)),
                        CaptureSpec(('final',), (0, 0)), CaptureSpec(('final',), (True,)),
                        CaptureSpec(('final', 'final'), (0,)), CaptureSpec(('final',), (0,), 1)):
            with self.assertRaises(ContractError):
                model(ids, mask, position_ids=positions, capture=capture)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with self.assertRaises(ContractError):
                model(ids, mask, position_ids=positions)

    def test_state_corruption_is_rejected_not_silently_substituted(self):
        ids, mask, positions = self.inputs()
        mutations = [lambda m: m.double(), lambda m: m.to('meta'),
                     lambda m: setattr(m, 'output_weight', nn.Parameter(m.token_embedding.detach().clone())),
                     lambda m: setattr(m, 'position_embedding', nn.Parameter(torch.zeros(1, 8))),
                     lambda m: setattr(m, 'lane', 'ternary'),
                     lambda m: setattr(m.blocks[0].q, 'lane', 'ternary'),
                     lambda m: setattr(m.blocks[0], 'heads', 1),
                     lambda m: setattr(m.final_norm, 'epsilon', 0.1),
                     lambda m: setattr(m, 'config', replace(SMALL, context_length=15)),
                     lambda m: m.register_parameter('unexpected', nn.Parameter(torch.zeros(1))),
                     lambda m: m.position_embedding.data.fill_(float('nan'))]
        for mutation in mutations:
            model = PositionDecoder(SMALL); mutation(model)
            with self.assertRaises(ContractError):
                model(ids, mask, position_ids=positions)
        # Even consistently changed native lanes must not activate a hidden alternate lane.
        model = PositionDecoder(SMALL); model.lane = 'ternary'
        for module in model.modules():
            if isinstance(module, Linear):
                module.lane = 'ternary'
        with self.assertRaises(ContractError):
            model(ids, mask, position_ids=positions)

    def test_future_token_and_position_embeddings_have_zero_prefix_gradient(self):
        # Unique suffix addresses make gradient leakage observable separately
        # from repeated prefix IDs. Token embeddings are tied to the output, so
        # use final hidden captures via a hook rather than output vocabulary loss.
        ids = torch.tensor([[256, 49, 43, 50, 61, 57]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.tensor([[3, 3, 2, 2, 14, 15]])
        for seed in SEEDS:
            model = PositionDecoder(SMALL, seed=seed)
            states = []
            handle = model.final_norm.register_forward_hook(lambda module, inputs, output: states.append(output))
            try:
                model(ids, mask, position_ids=positions)
                states[0][:, :4].square().sum().backward()
            finally:
                handle.remove()
            self.assertEqual(torch.count_nonzero(model.position_embedding.grad[14:]).item(), 0)
            self.assertEqual(torch.count_nonzero(model.token_embedding.grad[[61, 57]]).item(), 0)
            self.assertGreater(torch.count_nonzero(model.position_embedding.grad[[2, 3]]).item(), 0)

    def test_native_construction_and_attention_allocation_bounds_are_retained(self):
        for kwargs in ({'seed': True}, {'seed': -1}, {'seed': 2**32},
                       {'parameter_budget': True}, {'parameter_budget': 5_000_001}):
            with self.assertRaises(ContractError):
                PositionDecoder(SMALL, **kwargs)
        with self.assertRaises(ContractError):
            PositionDecoder({})
        model = PositionDecoder(ModelConfig(context_length=256, blocks=1, width=128,
                                            heads=128, feed_forward_width=1))
        ids = torch.zeros((8, 256), dtype=torch.int64)
        with self.assertRaises(ContractError):
            model(ids, torch.ones_like(ids, dtype=torch.bool), position_ids=ids)
        model = PositionDecoder(SMALL)
        ids = torch.zeros((1, 17), dtype=torch.int64)
        with self.assertRaises(ContractError):
            model(ids, torch.ones_like(ids, dtype=torch.bool), position_ids=ids)

    def test_native_answer_plus_eos_shifted_masked_target_loss_exact_parity(self):
        from trinite.tokenizer import ByteTokenizer
        from trinite.training import batch, target_loss
        tokenizer = ByteTokenizer(SMALL.context_length)
        examples = []
        for prompt, answer in (('1+2=', '30'), ('12+34=', '640')):
            encoded = tokenizer.encode(prompt + answer, answer_start=len(prompt))
            examples.append({'ids': encoded.input_ids, 'mask': encoded.attention_mask,
                             'loss': encoded.loss_mask})
        ids, mask, scored = batch(examples)
        weights = scored[:, 1:]
        self.assertEqual(int(weights.sum()), 7)  # 2+EOS and 3+EOS, not prompt/PAD.
        self.assertEqual(ids[:, 1:][weights].tolist(), [51, 48, 257, 54, 52, 48, 257])
        self.assertFalse(bool((weights & ~mask[:, 1:]).any()))
        positions = torch.arange(ids.shape[1], dtype=torch.int64).expand_as(ids)
        for seed in SEEDS:
            native = Decoder(SMALL, lane='dense', seed=seed)
            adapter = PositionDecoder(SMALL, seed=seed)
            expected, tokens = target_loss(native, examples)
            # IDs, masks AND addresses slice actual inputs together; only targets
            # and scoring masks shift right. This is test composition, no trainer.
            logits = adapter(ids[:, :-1], mask[:, :-1], position_ids=positions[:, :-1]).logits
            losses = torch.nn.functional.cross_entropy(logits.reshape(-1, 259),
                                                       ids[:, 1:].reshape(-1), reduction='none')
            actual = (losses.reshape(weights.shape) * weights).sum() / int(weights.sum())
            self.assertEqual(tokens, 7)
            self.assertTrue(torch.equal(expected, actual))
            expected.backward(); actual.backward()
            for (name, x), (other, y) in zip(native.named_parameters(), adapter.named_parameters()):
                self.assertEqual(name, other)
                self.assertTrue(torch.equal(x.grad, y.grad), name)
