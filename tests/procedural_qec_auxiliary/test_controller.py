"""Controller failure fixtures; no synthetic receipt certifies native custody."""
from contextlib import ExitStack
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes
from trinite.procedural_qec_auxiliary import experiment as x


class AuxiliaryControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.producer, self.historical = self.root/'producer', self.root/'history'
        for source in (self.producer, self.historical):
            (source/'src'/'nested').mkdir(parents=True)
        self.inputs = SimpleNamespace(repository=self.producer, historical_source=self.historical, receipt={})
        self.request = self.root/'request.json'
        self.request.write_bytes(b'{}\n')
        self.request_id = identity(self.request.read_bytes())
        self.req = {'run_profile': 'engineering'}
        self.run, self.verification = self.root/'run', self.root/'verification.json'

    def call(self):
        return x.run_matrix(self.run, self.request, self.request_id, self.verification, self.inputs)

    def mock_request(self):
        return patch.object(x, 'read_request', return_value=(self.req, self.request.read_bytes()))

    def summary(self, errors):
        core = x.verification_core(self.run, self.request_id)
        x.evidence.write(self.verification, core)
        return x.summary_for(self.run, self.req, self.request, self.request_id, self.verification,
                             identity(json_bytes(core)), self.inputs, errors)

    def roots(self, errors):
        self.run.mkdir()
        (self.run/'request.json').write_bytes(self.request.read_bytes())
        x.evidence.write(self.run/'inputs.json', self.inputs.receipt)
        x.evidence.write(self.run/'worker-errors.json', errors)

    def test_all_write_entrypoints_reject_source_paths_before_admission_or_model(self):
        # Patch the actual module root as well as supplying input repository roots.
        actual = self.root/'actual-producer'
        actual.mkdir()
        for source in (actual, self.producer, self.historical):
            for path in (source/'out', source/'src'/'out', source/'src'/'nested'/'..'/'out'):
                with self.subTest(path=path), patch.object(x, 'ROOT', actual), \
                        patch.object(x, 'read_request') as read, patch.object(x, 'request_core') as core, \
                        patch.object(x, 'state_for') as state:
                    actions = (
                        lambda: x.prepare_request(path, 'engineering', self.inputs),
                        lambda: x.train_cell(path, 0, 'extra-final', 'engineering', self.request,
                                             self.request_id, self.inputs),
                        lambda: x.run_matrix(path, self.request, self.request_id, self.verification, self.inputs),
                        lambda: x.run_matrix(self.run, self.request, self.request_id, path, self.inputs))
                    for action in actions:
                        with self.assertRaisesRegex(ContractError, 'outside producer source'):
                            action()
                    read.assert_not_called(); core.assert_not_called(); state.assert_not_called()
                    self.assertFalse(path.exists()); self.assertFalse(self.run.exists())

    def test_source_aliases_and_symlink_destinations_are_rejected(self):
        alias = self.root/'alias'
        alias.symlink_to(self.producer, target_is_directory=True)
        with self.mock_request(), patch.object(x, 'train_cell') as train:
            for destination in (alias/'out', alias/'..'/'out'):
                with self.assertRaisesRegex(ContractError, 'symlink'):
                    x.run_matrix(destination, self.request, self.request_id, self.verification, self.inputs)
            train.assert_not_called()
        self.assertFalse(self.run.exists())

    def test_failed_before_cell_directory_still_retains_four_bound_failure_receipts(self):
        with self.mock_request(), patch.object(x, 'train_cell', side_effect=RuntimeError('injected pre-mkdir')):
            result = self.call()
            self.assertEqual(result['outcome'], 'failed')
            checked = x.verify_only(self.run, self.request, self.request_id, self.verification,
                result['verification_request_identity'], self.inputs)
        self.assertEqual(checked['outcome'], 'failed')
        self.assertEqual(checked['replayed_cells'], 0)
        core = x.verification_core(self.run, self.request_id)
        self.assertTrue(core['schema'].endswith('.v2'))
        for cell in core['cells'].values():
            self.assertEqual(cell['outcome'], 'failed')
            self.assertEqual(set(cell['artifacts']), {'failure.json'})
        summary = x.parse_json((self.run/'summary.json').read_bytes(), canonical=True)
        self.assertEqual(len(summary['cells']), 4)
        self.assertFalse(summary['training_gate_pass'])

    def test_failure_and_partial_output_deletion_edit_addition_are_external_mismatches(self):
        def failure(root, *args):
            root.mkdir()
            (root/'steps.json').write_bytes(b'{"partial":true}\n')
            raise RuntimeError('injected partial')
        with self.mock_request(), patch.object(x, 'train_cell', side_effect=failure):
            result = self.call()
        directory = self.run/x.cell_name(0, 'extra-final')
        baseline = {p.name: p.read_bytes() for p in directory.iterdir()}
        before = {str(p.relative_to(self.run)): p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        with self.mock_request():
            x.verify_only(self.run, self.request, self.request_id, self.verification,
                          result['verification_request_identity'], self.inputs)
        self.assertEqual(before, {str(p.relative_to(self.run)): p.read_bytes() for p in self.run.rglob('*') if p.is_file()})
        for name, action in (('failure.json','remove'), ('failure.json','edit'), ('steps.json','remove'),
                             ('steps.json','edit'), ('metadata.json','add')):
            with self.subTest(name=name, action=action):
                path = directory/name
                if action == 'remove': path.unlink()
                else: path.write_bytes(b'{"changed":true}\n')
                with self.mock_request(), self.assertRaises(ContractError):
                    x.verify_only(self.run, self.request, self.request_id, self.verification,
                                  result['verification_request_identity'], self.inputs)
                if name in baseline: path.write_bytes(baseline[name])
                else: path.unlink()

    def test_failed_inventory_rejects_unknown_symlink_special_and_overbudget_members(self):
        directory = self.root/'failed'; directory.mkdir()
        x.retain_failure(directory, 'RuntimeError: fixture')
        for kind in ('unknown', 'symlink', 'directory', 'fifo', 'count', 'bytes'):
            with self.subTest(kind=kind), ExitStack() as stack:
                target = directory/('unexpected' if kind in ('unknown','directory') else 'steps.json')
                if kind == 'symlink': target.symlink_to(self.request)
                elif kind == 'directory': target.mkdir()
                elif kind == 'fifo': os.mkfifo(target)
                else: target.write_bytes(b'{}\n')
                if kind == 'count': stack.enter_context(patch.object(x, 'FAILED_MAX_FILES', 1))
                if kind == 'bytes': stack.enter_context(patch.object(x, 'FAILED_MAX_BYTES', 1))
                with self.assertRaises(ContractError): x.failed_commitments(directory, 'RuntimeError: fixture')
                if target.is_dir(): target.rmdir()
                else: target.unlink()

    def test_multibyte_and_escaped_backend_errors_fit_all_receipt_byte_bounds(self):
        message = ('\U0001f30d\x00\ud800' * 10000)
        with self.mock_request(), patch.object(x, 'train_cell', side_effect=RuntimeError(message)):
            result = self.call()
            x.verify_only(self.run, self.request, self.request_id, self.verification,
                          result['verification_request_identity'], self.inputs)
        self.assertEqual(result['outcome'], 'failed')
        self.assertLess((self.run/'worker-errors.json').stat().st_size, 64 * 1024)
        for cell in x.cells():
            raw = (self.run/x.cell_name(*cell)/'failure.json').read_bytes()
            self.assertLess(len(raw), 16 * 1024)
            self.assertIn(b'[truncated]', raw)

    def test_maximum_four_failed_inventories_fit_verification_byte_limit(self):
        errors = {x.cell_name(*c): 'RuntimeError: fixture' for c in x.cells()}
        self.roots(errors)
        for name, error in errors.items():
            directory = self.run/name
            x.retain_failure(directory, error)
            leaf = directory/'provenance'/'artifact_records'/'sha256'
            leaf.mkdir(parents=True)
            # Three directory entries + the failure receipt + longest permitted
            # partial names exactly fill the per-cell traversal ceiling.
            for index in range(x.FAILED_MAX_FILES - 4):
                (leaf/(format(index, '064x')+'.json')).write_bytes(b'{}\n')
        core = x.verification_core(self.run, self.request_id)
        raw = json_bytes(core)
        self.assertLess(len(raw), 128 * 1024)
        self.verification.write_bytes(raw)
        self.assertEqual(core, x.read_verification(self.verification, identity(raw),
                                                   self.run, self.request_id))

    def test_completed_cell_rejects_added_failure_receipt(self):
        self.roots({})
        directory = self.run/x.cell_name(0, 'extra-final'); directory.mkdir()
        x.retain_failure(directory, 'RuntimeError: undeclared')
        with self.assertRaisesRegex(ContractError, 'unexpected failure'):
            x.verification_core(self.run, self.request_id)

    def test_replay_runtime_error_retains_all_four_failed_cells_and_matrix_summary(self):
        def completed(root, *args):
            root.mkdir()
            # Controller-only stand-ins: they are never passed off as valid native artifacts.
            for name in x.evidence.ARTIFACTS:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'{}\n')
        with self.mock_request(), patch.object(x, 'train_cell', side_effect=completed), \
                patch.object(x, 'verify_cell', side_effect=RuntimeError('injected allocation failure')) as verify:
            result = self.call()
        self.assertEqual(verify.call_count, 4)
        self.assertEqual(result['outcome'], 'failed')
        summary = x.parse_json((self.run/'summary.json').read_bytes(), canonical=True)
        self.assertEqual(len(summary['cells']), 4)
        self.assertTrue(all(c['error'] == 'RuntimeError: injected allocation failure' for c in summary['cells']))
        self.assertFalse(summary['training_gate_pass'])

    def test_replay_memory_error_is_retained_but_process_cancellation_propagates(self):
        self.roots({})
        for exception in (MemoryError('injected allocation'), KeyboardInterrupt()):
            with patch.object(x, 'verify_cell', side_effect=exception) as verify:
                if isinstance(exception, Exception):
                    result = self.summary({})
                    self.assertEqual(result['outcome'], 'failed')
                    self.assertEqual(verify.call_count, 4)
                    self.assertTrue(all(c['error'] == 'MemoryError: injected allocation' for c in result['cells']))
                    self.verification.unlink()
                else:
                    with self.assertRaises(KeyboardInterrupt): self.summary({})

    def test_recheck_failed_artifacts_after_other_cells_replay(self):
        name = x.cell_name(0, 'extra-final')
        errors = {name: 'RuntimeError: fixture'}
        self.roots(errors)
        directory = self.run/name
        x.retain_failure(directory, errors[name])
        def replay(*args):
            (directory/'metadata.json').write_bytes(b'{}\n')
            raise RuntimeError('injected')
        with patch.object(x, 'verify_cell', side_effect=replay), self.assertRaises(ContractError):
            self.summary(errors)


if __name__ == '__main__':
    unittest.main()
