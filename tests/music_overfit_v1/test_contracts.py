"""Train-only admission/controller contracts; no model construction or target run."""
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes
from music_learning_v1.codec import encode_example
from music_overfit_v1 import spec, request, controller


class Contracts(unittest.TestCase):
    def test_fixed_subsets_schedule_and_original_membership(self):
        for subset, count in [('one', 1), ('four', 4)]:
            rows = spec.subset_rows(subset)
            self.assertEqual(rows, spec.subset_rows(subset))
            self.assertEqual(len(rows), count)
            self.assertEqual(rows, spec.validate_rows(rows, subset))
            self.assertTrue(all(r['split'] == 'train' for r in rows))
            receipt = spec.schedule_receipt(subset)
            self.assertEqual(receipt['scheduled_visits'], 512)
            self.assertEqual(receipt['visits_per_example'], 512 // count)
            self.assertEqual(receipt['target_tokens'], sum(r['answer_token_count'] for r in rows) * (512 // count))
        rows = spec.subset_rows('four')
        for task in ('attacks', 'duration'):
            self.assertEqual(len({r['answer'] for r in rows if r['task'] == task}), 2)
        self.assertEqual(spec.CELLS, ((17, 'one'), (29, 'one'), (17, 'four'), (29, 'four')))

    def test_mislabeled_heldout_and_changed_train_rows_rejected(self):
        _, _, all_rows = spec.admitted()
        for split in ('validation', 'test'):
            row = deepcopy(next(r for r in all_rows if r['split'] == split))
            row['split'] = 'train'
            with self.assertRaises(ContractError):
                spec.validate_rows([row])
        rows = spec.subset_rows('one')
        for field, value in [('answer', '999'), ('prompt', 'hidden'), ('example_identity', 'foreign')]:
            changed = deepcopy(rows); changed[0][field] = value
            with self.assertRaises(ContractError):
                spec.validate_rows(changed)
        with self.assertRaises(ContractError):
            spec.validate_rows(rows + rows)
        with self.assertRaises(ContractError):
            spec.validate_rows(spec.subset_rows('four'), 'one')

    def test_exact_answer_eos_shift_and_no_prompt_loss(self):
        for row in spec.subset_rows('four'):
            encoded = encode_example({k: row[k] for k in ('task', 'prompt', 'answer', 'text', 'answer_start')})
            p = len(row['prompt'].encode('ascii')); a = len(row['answer'].encode('ascii'))
            self.assertEqual(encoded.input_ids[p+1:], (*row['answer'].encode('ascii'), 257))
            self.assertEqual(encoded.loss_mask, (0,) * (p+1) + (1,) * (a+1))
            self.assertEqual([i for i, bit in enumerate(encoded.loss_mask[1:]) if bit], list(range(p, p+a+1)))

    def test_requests_are_exact_and_source_imports_bound(self):
        value = {'test': 'request contract only'}; raw = json_bytes(value)
        with patch.object(request, 'core', return_value=value):
            self.assertEqual(request.validate(raw, identity(raw)), value)
            with self.assertRaises(ContractError):
                request.validate(raw, 'sha256:' + '0'*64)
            with self.assertRaises(ContractError):
                request.validate(raw + b' ', identity(raw + b' '))
        with patch.object(request, 'core', return_value={'changed': True}):
            with self.assertRaises(ContractError):
                request.validate(raw, identity(raw))
        with patch.object(spec, '__file__', '/tmp/foreign/spec.py'):
            with self.assertRaises(ContractError):
                request.sources()

    def test_authorization_precedes_runtime_and_output_is_separate(self):
        for action in ('run', '_worker'):
            with patch.object(controller, 'configure', side_effect=AssertionError('must not configure')):
                with redirect_stderr(StringIO()) as errors:
                    self.assertEqual(controller.main([action]), 1)
                self.assertIn('explicit owner authorization', errors.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); req = root / 'request.json'; req.write_bytes(b'{}')
            for output in (request.ROOT, request.ROOT / 'new', root):
                with self.assertRaises(ContractError):
                    controller.protect(output, req)
            linked = root / 'linked'; linked.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ContractError):
                controller.protect(linked / 'new', req)
            self.assertEqual(controller.protect(root / 'fresh', req), (root / 'fresh', req.resolve()))

    def test_preflight_never_imports_numerical_backend(self):
        code = ('import sys; from music_overfit_v1.controller import main; '
                'assert main(["preflight"]) == 0; '
                'assert not any(n in sys.modules for n in ("torch", "numpy", "trinite.model", "trinite.training"))')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"model_executed":false', result.stdout)

    def test_custody_tamper_rejected_before_numerical_interpretation(self):
        from music_overfit_v1 import workflow
        raw, receipt, _ = spec.admitted()
        request_value = {'admission_identity': identity(receipt), 'test_scope': 'synthetic custody contract'}
        request_raw = json_bytes(request_value); request_id = identity(request_raw)
        def fake_trial(seed, subset, admission_identity, request_identity, write):
            for name in sorted(workflow.artifact_names() - {'report.json'}):
                write(name, json_bytes({'synthetic': name, 'not_numerical_evidence': True}))
            return {'synthetic': True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'fresh'; root.mkdir(); info = root.stat()
            with patch.object(workflow, 'validate', return_value=request_value), \
                 patch('music_overfit_v1.numerical.run_trial', side_effect=fake_trial):
                commitment = workflow.run(root, request_raw, request_id, 17, 'one', (info.st_dev, info.st_ino))
            self.assertEqual(identity((root / 'completion.json').read_bytes()), commitment)
            artifact = root / 'step-000.safetensors'; artifact.write_bytes(b'tampered')
            with patch.object(workflow, 'validate', return_value=request_value), \
                 patch.object(workflow, 'subset_rows', side_effect=AssertionError('must fail before diagnostics')):
                with self.assertRaisesRegex(ContractError, 'artifact identity'):
                    workflow.audit(root, request_raw, request_id, 17, 'one', commitment)
            self.assertEqual(artifact.read_bytes(), b'tampered')

    def test_worker_rejects_source_destination(self):
        from music_overfit_v1 import workflow
        info = request.ROOT.stat()
        with patch.object(workflow, 'validate', side_effect=AssertionError('must fail before validation')):
            with self.assertRaises(ContractError):
                workflow.run(request.ROOT, b'{}', 'synthetic', 17, 'one', (info.st_dev, info.st_ino))

    def test_timeout_and_interruption_keep_every_expected_cell(self):
        for failure, outcomes in [
                (subprocess.TimeoutExpired('synthetic-worker', 900), ['timed_out'] * 4),
                (KeyboardInterrupt(), ['interrupted'] + ['not_attempted'] * 3)]:
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); req = root / 'request.json'; raw = json_bytes({'synthetic': True})
                req.write_bytes(raw)
                with patch.object(controller, 'validate', return_value={'synthetic': True}), \
                     patch.object(controller, 'launch', side_effect=failure):
                    result = controller.run_matrix(root / 'attempt', req, identity(raw))
                self.assertEqual(result['outcome'], 'incomplete')
                names = [request.cell_name(seed, subset) for seed, subset in spec.CELLS]
                self.assertEqual([result['cells'][name]['outcome'] for name in names], outcomes)
                self.assertTrue((root / 'attempt' / 'matrix.json').is_file())
                for name, outcome in zip(names, outcomes):
                    self.assertEqual((root / 'attempt' / name / 'attempt.json').is_file(), outcome != 'not_attempted')


if __name__ == '__main__':
    unittest.main()
