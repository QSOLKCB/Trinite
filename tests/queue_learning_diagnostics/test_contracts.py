"""Synthetic engineering tests. AdamW.step is forbidden throughout this suite."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import queue_evaluation_v1 as gate
from queue_learning_v1 import codec, numerical
from queue_learning_v1.producer import episodes as old_episodes
from queue_learning_v2 import diagnostics, protocol
from trinite.comparison import greedy
from trinite.contracts import ContractError
from trinite.model import Decoder
from trinite.tokenizer import BOS, EOS, PAD
from trinite.training import batch, target_loss, model_identity, create_optimizer, require_optimizer


def public():
    return gate.encode({'schema': 'trinite.queue-observation.v1', 'capacity': 2,
        'observe_ticks': 1, 'horizon': 8, 'rows': [{'tick': 0, 'outage': False,
        'values': [0, 1, 1, 1, 0, 0], 'observed': [True] * 6}]})


def row(answer, name='synthetic'):
    enc = codec.supervised(public(), answer)
    return {'identity': name, 'split': 'train', 'answer': answer, 'observation': gate.decode(public()),
            'ids': list(enc.input_ids), 'mask': list(enc.attention_mask), 'loss': list(enc.loss_mask)}


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numerical.configure()

    def setUp(self):
        self.guard = patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('optimizer steps forbidden'))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_decimal_byte_ids_boundary_and_eos(self):
        self.assertEqual((BOS, EOS, PAD), (256, 257, 258))
        for n in (0, 7, 42, 256):
            r = row(n)
            boundary = 1 + len(codec.prompt(public()))
            self.assertEqual(r['ids'][boundary:], [*str(n).encode('ascii'), EOS])
            self.assertEqual(r['loss'], [0] * boundary + [1] * (len(str(n)) + 1))
            self.assertEqual(codec.output(r['ids'][boundary:], public()), {'status': 'prediction', 'value': n})

    def test_native_shift_mask_denominator_and_gradient_support(self):
        rows = [row(0), row(256)]
        ids, mask, scored = batch(rows)
        logits = torch.linspace(-1, 1, (ids.shape[1] - 1) * 259).reshape(1, -1, 259).repeat(2, 1, 1).requires_grad_()
        calls = []
        def forward(x, m):
            calls.append((x.clone(), m.clone()))
            return SimpleNamespace(logits=logits)
        loss, count = target_loss(forward, rows)
        self.assertEqual(count, 6)  # Four decimal bytes, two EOS; prompt/padding excluded.
        self.assertTrue(torch.equal(calls[0][0], ids[:, :-1]))
        self.assertTrue(torch.equal(calls[0][1], mask[:, :-1]))
        selected = scored[:, 1:]
        expected = F.cross_entropy(logits[selected], ids[:, 1:][selected])
        self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=6)
        loss.backward()
        self.assertEqual(int(logits.grad[~selected].count_nonzero()), 0)
        self.assertTrue(bool((logits.grad[selected].abs().sum(-1) > 0).all()))
        # First scored output is the prompt's '=' position, not first answer or EOS.
        self.assertEqual(int(ids[0, torch.where(selected[0])[0][0]]), ord('='))

    def test_causality_padding_and_generation_first_logit(self):
        model = Decoder(protocol.plan(17).model, lane='dense', seed=17)
        r = row(42)
        ids, mask, _ = batch([r])
        boundary = 1 + len(codec.prompt(public()))
        altered = ids.clone(); altered[:, boundary:] = 90
        with torch.no_grad():
            full = model(ids, mask).logits
            changed = model(altered, mask).logits
            prefix = model(ids[:, :boundary], mask[:, :boundary]).logits
            self.assertTrue(torch.allclose(full[:, :boundary], changed[:, :boundary], atol=1e-6, rtol=0))
            self.assertTrue(torch.allclose(full[:, boundary-1], prefix[:, -1], atol=1e-6, rtol=0))
            padded = torch.cat((ids, torch.tensor([[PAD, PAD]])), 1)
            pmask = torch.cat((mask, torch.tensor([[False, False]])), 1)
            out = model(padded, pmask).logits
            self.assertTrue(torch.allclose(full, out[:, :ids.shape[1]], atol=1e-6, rtol=0))
            self.assertEqual(int(out[:, -2:].count_nonzero()), 0)

    def test_greedy_eos_stopping_cap_and_strict_parse(self):
        for emitted, expected in (([EOS], [EOS]), ([48, EOS, 49], [48, EOS]), ([49]*8, [49]*5)):
            seen = []
            def forward(ids, mask):
                seen.append(ids.tolist()[0])
                logits = torch.zeros((1, ids.shape[1], 259))
                logits[0, -1, emitted[len(seen)-1]] = 10
                return SimpleNamespace(logits=logits)
            fake = SimpleNamespace(config=SimpleNamespace(context_length=256), __call__=forward)
            class Model:
                config = fake.config
                def __call__(self, ids, mask):
                    return forward(ids, mask)
            self.assertEqual(greedy(Model(), codec.prompt(public()), 5), expected)
            self.assertEqual(seen[0], [BOS, *codec.prompt(public()).encode('ascii')])
        for tokens in ([EOS], [48, 48, EOS], [48], [PAD, EOS], [255, EOS], [50, 53, 55, EOS]):
            self.assertEqual(codec.output(tokens, public())['status'], 'malformed')
        self.assertEqual(codec.output([63, EOS], public()), {'status': 'abstain'})

    def test_split_supervision_diagnostics_preserve_model_and_raw(self):
        model = Decoder(protocol.plan(17).model, lane='dense', seed=17)
        rows = [row(0), row(256)]
        before = model_identity(model)
        value = diagnostics.supervision(model, rows)
        self.assertEqual([value[k]['targets'] for k in ('answer', 'eos', 'first_answer', 'native')], [4, 2, 2, 6])
        summed = sum(float.fromhex(value[k]['nll_sum_hex']) for k in ('answer', 'eos')) / 6
        self.assertAlmostEqual(summed, float.fromhex(value['native']['mean_nll_hex']), places=5)
        raw = diagnostics.generation(model, rows)
        self.assertEqual(raw['planned'], 2)
        self.assertEqual(sum(raw['statuses'].values()), 2)
        self.assertTrue(all('tokens' in r and 'parsed' in r for r in raw['records']))
        self.assertEqual(model_identity(model), before)
        rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ContractError, 'training rows only'):
            diagnostics.supervision(model, rows)

    def test_optimizer_unique_parameters_and_exact_plan_no_step(self):
        config = protocol.plan(17)
        model = Decoder(config.model, lane='dense', seed=17)
        optimizer = create_optimizer(model, config)
        require_optimizer(SimpleNamespace(config=config, model=model, optimizer=optimizer))
        ids = [id(p) for p in optimizer.param_groups[0]['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIs(model.output_weight, model.token_embedding)
        self.assertEqual(optimizer.state, {})

    def test_forensic_reader_is_versioned_and_anchor_bound(self):
        from run_queue_diagnostics_v2 import forensic_restore
        from trinite.training import TrainingState, evaluate
        from trinite.contracts import parse_json, json_bytes
        def state(*args):
            config = protocol.plan(17)
            model = Decoder(config.model, lane='dense', seed=17)
            rows = [row(0)]
            return TrainingState(config, 'dense', model, create_optimizer(model, config),
                                 {'train': rows, 'validation': rows}, 'synthetic-data', 'synthetic-binding',
                                 model_identity(model), evaluate(model, rows, 8))
        original = state()
        payload, metadata = numerical.snapshot(original, 'synthetic-binding')
        value = parse_json(metadata, canonical=True)
        value['environment']['python'] = 'foreign-exposed-engineering-host'
        metadata = json_bytes(value)
        expected = [gate.digest(payload), gate.digest(metadata)]
        with patch.object(numerical, 'state_for', side_effect=state):
            restored = forensic_restore(payload, metadata, [], 17, expected)
            self.assertEqual(model_identity(restored.model), model_identity(original.model))
            self.assertEqual(restored.optimizer.state, {})
            with self.assertRaisesRegex(ValueError, 'anchors'):
                forensic_restore(payload, metadata, [], 17, ['0'*64, expected[1]])
            value['plan']['learning_rate'] = '0.002'
            bad = json_bytes(value)
            with self.assertRaisesRegex(ValueError, 'source/plan/data/init'):
                forensic_restore(payload, bad, [], 17, [gate.digest(payload), gate.digest(bad)])
            with self.assertRaisesRegex(ValueError, 'context mismatch'):
                numerical.restore(payload, metadata, [], 17, 'synthetic-binding', expected)

    def test_fit_subset_ignores_labels_and_rejects_heldout(self):
        rows = []
        for i in range(10):
            r = row(0, f'synthetic-{i}')
            r['observation']['capacity'] = i + 2
            enc = codec.supervised(gate.encode(r['observation']), 0)
            r.update(ids=list(enc.input_ids), mask=list(enc.attention_mask), loss=list(enc.loss_mask))
            rows.append(r)
        first = protocol.fit_rows(rows)
        self.assertEqual(len(first), 8)
        for r in rows:
            enc = codec.supervised(gate.encode(r['observation']), 1)
            r.update(answer=1, ids=list(enc.input_ids), mask=list(enc.attention_mask), loss=list(enc.loss_mask))
        self.assertEqual([r['identity'] for r in first], [r['identity'] for r in protocol.fit_rows(rows)])
        measured = protocol.fit_report(Decoder(protocol.plan(17).model, lane='dense', seed=17), rows)
        self.assertEqual(measured['generation']['planned'], 8)
        self.assertEqual(len(measured['generation']['records']), 8)
        self.assertEqual(measured['meets_fit_criterion'], measured['generation']['exact'] == 8)
        rows[0]['split'] = 'validation'
        with self.assertRaises(ContractError):
            protocol.fit_rows(rows)

    def test_fresh_observations_and_related_variant_overlap(self):
        rows, old = protocol.episodes(), old_episodes()
        def observed(items):
            return {r['id']: gate.run('observe', gate.encode(r['request']))[0] for r in items}
        obs, old_obs = observed(rows), observed(old)
        value = protocol.audit(rows, obs, old_obs)
        self.assertEqual(value['episodes'], 384)
        root = Path(__file__).resolve().parents[2]
        evidence = root / 'docs/evidence/queue-eos-diagnostics-v2'
        self.assertEqual(json.loads((evidence / 'requests.json').read_bytes()), rows)
        self.assertEqual(json.loads((evidence / 'proposal.json').read_bytes()), protocol.PROTOCOL)
        self.assertEqual(json.loads((evidence / 'overlap.json').read_bytes()), value)
        for line in (evidence / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(hashlib.sha256((evidence / name).read_bytes()).hexdigest(), digest)
        report = json.loads((evidence / 'training-forward.json').read_bytes())
        for name, digest in report['diagnostic_sources'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        fresh = next(r for r in rows if r['split'] != 'train')
        bad = dict(obs); bad[fresh['id']] = next(iter(old_obs.values()))
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            protocol.audit(rows, bad, old_obs)
        other = next(r for r in rows if r['split'] not in ('train', fresh['split']))
        bad = dict(obs); bad[other['id']] = obs[fresh['id']]
        with self.assertRaisesRegex(ValueError, 'cross v2 splits'):
            protocol.audit(rows, bad, old_obs)
        changed = copy.deepcopy(fresh)
        changed['request']['capacity'] += 1
        changed['request']['initial_queue'] = 1
        changed['request']['observed'] = [False]*6
        changed['request']['arrivals'][-1] += 1
        self.assertEqual(protocol.related_key(fresh), protocol.related_key(changed))
        self.assertEqual(protocol.PROTOCOL['limits']['total_optimizer_updates'], 128 + 256*2)
        self.assertFalse(protocol.PROTOCOL['execution_enabled'])

    def test_queue_ci_has_no_scientific_entrypoint(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/queue-learning-v1.yml').read_text()
        for forbidden in ('run_queue_diagnostics_v2.py --evidence', 'run_queue_learning_v1.py run',
                          'run_queue_learning_v1.py freeze'):
            self.assertNotIn(forbidden, workflow)


if __name__ == '__main__':
    unittest.main()
