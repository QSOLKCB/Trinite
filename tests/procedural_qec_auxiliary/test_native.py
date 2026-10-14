"""Actual bounded CPU conformance, never a scientific learning experiment."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.procedural_qec_auxiliary import experiment as x
from trinite.procedural_qec_auxiliary.inputs import materialize_inputs
from trinite.training import batch, model_identity, run_steps, target_loss

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = Path(os.environ['TRINITE_HISTORICAL_SOURCE'])
ENV = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


class AuxiliaryNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.inputs = materialize_inputs(HISTORICAL)
        cls.request_id = identity(json_bytes(x.request_core('engineering', cls.inputs)))

    def state(self, seed=0, arm='extra-procedural'):
        return x.state_for(seed, arm, 'engineering', self.inputs)

    def snap(self, state, seed=0, arm='extra-procedural'):
        return x.snapshot(state, seed, arm, 'engineering', self.request_id, self.inputs)

    def load(self, pair, seed=0, arm='extra-procedural'):
        return x.restore(*pair, seed, arm, 'engineering', self.request_id,
                         tuple(map(identity, pair)), self.inputs)

    def test_research_rejected_before_model_initialization(self):
        with self.assertRaisesRegex(ContractError, 'disabled'):
            x.state_for(0, 'extra-procedural', 'research', self.inputs)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'request.json'
            with self.assertRaisesRegex(ContractError, 'disabled'):
                x.prepare_request(path, 'research', self.inputs)
            self.assertFalse(path.exists())

    def test_every_cell_padded_batch_and_odd_even_restart_full_adamw_replay(self):
        initial = {}
        for seed, arm in x.cells():
            with self.subTest(seed=seed, arm=arm):
                uninterrupted = self.state(seed, arm)
                initial[seed, arm] = uninterrupted.initialization_identity
                self.assertEqual(len(uninterrupted.data['train']), 16 * 8)
                for start in range(0, len(uninterrupted.data['train']), 8):
                    ids, mask, loss = batch(uninterrupted.data['train'][start:start+8])
                    self.assertEqual(tuple(ids[:, :-1].shape), (8, 63))
                    self.assertEqual(tuple(mask.shape), (8, 64))
                    self.assertEqual(tuple(loss.shape), (8, 64))
                run_steps(uninterrupted)
                expected = self.snap(uninterrupted, seed, arm)
                for stop in (1, 2):
                    interrupted = self.state(seed, arm)
                    run_steps(interrupted, stop_after=stop)
                    checkpoint = self.snap(interrupted, seed, arm)
                    resumed = self.load(checkpoint, seed, arm)
                    self.assertEqual(resumed.cursor, stop * 8)
                    self.assertEqual(resumed.target_tokens,
                        sum(sum(r['loss']) for r in resumed.data['train'][:stop*8]))
                    run_steps(resumed)
                    self.assertEqual(expected, self.snap(resumed, seed, arm))
                self.assertEqual(uninterrupted.step, 16)
        for seed in x.SEEDS:
            self.assertEqual(initial[seed, 'extra-final'], initial[seed, 'extra-procedural'])

    def test_padded_tokens_forward_loss_and_gradient_equivalence(self):
        state = self.state()
        # Final, both prefix and both trace kinds exercise distinct target spans.
        for offset in (0, 8, 24, 40, 56):
            with self.subTest(offset=offset):
                state.optimizer.zero_grad(set_to_none=True)
                padded = state.data['train'][offset:offset+8]
                unpadded = []
                for row in padded:
                    count = sum(row['mask'])
                    unpadded.append({**row, 'ids': row['ids'][:count],
                                    'mask': row['mask'][:count], 'loss': row['loss'][:count]})
                    self.assertEqual(sum(row['loss']), sum(unpadded[-1]['loss']))
                    self.assertTrue(all(t == 258 for t in row['ids'][count:]))
                    self.assertFalse(any(row['mask'][count:]) or any(row['loss'][count:]))
                short_ids, short_mask, _ = batch(unpadded)
                long_ids, long_mask, _ = batch(padded)
                a = state.model(short_ids[:, :-1], short_mask[:, :-1]).logits
                b = state.model(long_ids[:, :-1], long_mask[:, :-1]).logits
                # CPU conformance tolerance, not a historical trajectory guarantee.
                torch.testing.assert_close(a, b[:, :a.shape[1]], rtol=1e-5, atol=1e-6)
                short_loss, short_n = target_loss(state.model, unpadded)
                short_loss.backward()
                grads = {n: p.grad.detach().clone() for n, p in state.model.named_parameters()}
                state.optimizer.zero_grad(set_to_none=True)
                long_loss, long_n = target_loss(state.model, padded)
                long_loss.backward()
                self.assertEqual(short_n, long_n)
                torch.testing.assert_close(short_loss, long_loss, rtol=1e-5, atol=1e-6)
                for name, parameter in state.model.named_parameters():
                    torch.testing.assert_close(grads[name], parameter.grad, rtol=1e-5, atol=1e-6)

    def test_resealed_checkpoint_context_progress_and_tensor_tampering_rejected(self):
        state = self.state()
        run_steps(state, stop_after=1)
        payload, raw = self.snap(state)
        mutations = [
            lambda r: r.update(arm='extra-final'),
            lambda r: r.update(seed=1),
            lambda r: r.update(cursor=16),
            lambda r: r.update(target_tokens=r['target_tokens']+1),
            lambda r: r['plan'].update(loss='combined-objective'),
            lambda r: r['inputs'].update(mapping_receipt_identity='sha256:'+'0'*64),
            lambda r: r['source'].update({'src/trinite/training.py': 'sha256:'+'0'*64}),
            lambda r: r['history'][0].update(examples=[]),
        ]
        for mutation in mutations:
            record = parse_json(raw, canonical=True)
            mutation(record)
            with self.assertRaises((ContractError, ValueError)):
                self.load((payload, json_bytes(record)))
        with self.assertRaises(ContractError):
            self.load((payload[:-1], raw))
        with self.assertRaises(ContractError):
            x.restore(payload, raw, 0, 'extra-procedural', 'engineering', self.request_id,
                      ('sha256:'+'0'*64, identity(raw)), self.inputs)

    def test_verification_request_must_be_outside_output_even_with_dot_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'alias').mkdir()
            for verifier in (root/'run'/'verification.json',
                             root/'alias'/'..'/'run'/'verification.json'):
                # Failure-path controller test; successful custody uses actual replay below.
                with patch.object(x, 'read_request', return_value=({'run_profile': 'engineering'}, b'{}')):
                    with self.assertRaisesRegex(ContractError, 'outside output'):
                        x.run_matrix(root/'run', root/'request.json', self.request_id,
                                     verifier, self.inputs)
                self.assertFalse((root/'run').exists())

    def test_mutated_schedule_is_not_snapshot_certified(self):
        state = self.state()
        state.data['train'][0] = copy.deepcopy(state.data['train'][0])
        state.data['train'][0]['ids'] = (0,) + tuple(state.data['train'][0]['ids'][1:])
        with self.assertRaisesRegex(ContractError, 'schedule/encoding'):
            self.snap(state)


class AuxiliaryFourCellSmoke(unittest.TestCase):
    def call(self, script, *args, timeout=900):
        proc = subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str,args)],
            cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
        detail = ''
        if proc.returncode and '--output' in args:
            candidate = Path(args[args.index('--output') + 1]) / 'summary.json'
            if candidate.is_file():
                summary = json.loads(candidate.read_bytes())
                detail = json.dumps({'worker_errors': summary.get('worker_errors'),
                    'pairing_errors': summary.get('pairing_errors'),
                    'failed_cells': [c for c in summary.get('cells', [])
                                     if c.get('outcome') != 'completed']}, sort_keys=True)
        self.assertEqual(proc.returncode, 0, proc.stdout[-4000:]+'\n'+proc.stderr[-4000:]+'\n'+detail)
        return json.loads(proc.stdout)

    def test_four_cells_closed_custody_three_replays_and_lossless_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, verifier, run = root/'request.json', root/'verification.json', root/'run'
            runner = 'run_procedural_qec_auxiliary.py'
            prepared = self.call(runner, '--prepare-request', request,
                '--run-profile', 'engineering', '--historical-source', HISTORICAL)
            self.assertFalse(prepared['scientific_protocol_frozen'])
            measured = self.call(runner, '--run', '--request', request,
                '--request-identity', prepared['request_identity'], '--verification-request', verifier,
                '--output', run, '--historical-source', HISTORICAL)
            self.assertEqual(measured['outcome'], 'completed')
            self.assertFalse(measured['training_gate_pass'])
            before = {str(p.relative_to(root)): p.read_bytes()
                      for p in root.rglob('*') if p.is_file()}
            checked = self.call(runner, '--verify-only', '--request', request,
                '--request-identity', prepared['request_identity'], '--verification-request', verifier,
                '--verification-request-identity', measured['verification_request_identity'],
                '--output', run, '--historical-source', HISTORICAL)
            self.assertEqual(checked['replayed_cells'], 4)
            summary = json.loads((run/'summary.json').read_bytes())
            self.assertEqual(len(summary['cells']), 4)
            for cell in summary['cells']:
                self.assertTrue(all(cell['checks'].values()))
                self.assertEqual(cell['report']['steps'], 16)
                self.assertFalse(cell['report']['held_out_scored'])
                self.assertFalse(cell['report']['training_gate_evaluated'])
            # Archive-time full replay is the third pass, not byte verification.
            archive = root/'evidence.zip'
            archived = self.call('archive_procedural_qec_auxiliary.py', '--archive',
                '--output', archive, '--run', run, '--request', request,
                '--request-identity', prepared['request_identity'],
                '--verification-request', verifier,
                '--verification-request-identity', measured['verification_request_identity'],
                '--historical-source', HISTORICAL)
            self.assertEqual(archived['outcome'], 'passed')
            transport = self.call('archive_procedural_qec_auxiliary.py', '--verify-archive',
                '--archive-file', archive, '--archive-identity', archived['archive_identity'],
                '--inventory-identity', archived['inventory_identity'])
            self.assertEqual(transport['outcome'], 'passed')
            self.assertFalse(transport['numerical_replay_performed'])
            with tempfile.TemporaryDirectory() as restoration:
                restored_source = Path(restoration)/'historical'
                restored = self.call('archive_procedural_qec_auxiliary.py',
                    '--restore-historical-source', '--archive-file', archive,
                    '--archive-identity', archived['archive_identity'],
                    '--inventory-identity', archived['inventory_identity'],
                    '--output', restored_source)
                self.assertEqual(restored['outcome'], 'passed')
                admitted = materialize_inputs(restored_source)
                self.assertEqual(identity(json_bytes(admitted.receipt)),
                    json.loads(request.read_bytes())['inputs']['mapping_receipt_identity'])
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob('*') if p.is_file() and p != archive})
            # An independently committed file change must fail read-only replay.
            report = run/x.cell_name(0, 'extra-final')/'report.json'
            report.write_bytes(report.read_bytes()+b' ')
            proc = subprocess.run([sys.executable, str(ROOT/'scripts'/runner), '--verify-only',
                '--request',str(request), '--request-identity', prepared['request_identity'],
                '--verification-request',str(verifier), '--verification-request-identity',
                measured['verification_request_identity'], '--output',str(run),
                '--historical-source',str(HISTORICAL)], cwd=ROOT, env=ENV,
                capture_output=True, text=True, timeout=900)
            self.assertNotEqual(proc.returncode, 0)


if __name__ == '__main__':
    unittest.main()
