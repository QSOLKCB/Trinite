"""Controller/artifact doubles certify mechanics, never model fit or custody."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_exposure_v1 import controller, workflow
from music_exposure_v1.request import cell_name

RAW = b'synthetic request'
RID = identity(RAW)
CID = identity(b'synthetic completion')


class ControllerTests(unittest.TestCase):
    def test_authorization_required_before_backend(self):
        for action in ('run', '_worker'):
            with patch.object(controller, 'configure') as configure, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(controller.main([action]), 1)
                configure.assert_not_called()

    def test_preflight_never_configures_backend(self):
        output = io.StringIO()
        with patch.object(controller, 'configure') as configure, contextlib.redirect_stdout(output):
            self.assertEqual(controller.main(['preflight']), 0)
            configure.assert_not_called()
        result = parse_json(output.getvalue().encode(), canonical=True)
        self.assertIs(result['model_executed'], False)
        self.assertEqual(set(result['schedules']), {'0', '64', '640'})

    def test_no_heldout_or_gate_cli_route(self):
        for argv in (['heldout'], ['run', '--phase', 'heldout'], ['run', '--gate-identity', RID]):
            with patch.object(controller, 'configure') as configure, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    controller.main(argv)
                configure.assert_not_called()
        self.assertFalse(hasattr(workflow, 'require_gate'))
        self.assertFalse(hasattr(controller, '_gate'))

    def test_both_seeds_required_for_positive_outcome(self):
        records = {cell_name(seed): {'outcome': 'completed', 'completion_identity': CID} for seed in (17, 29)}
        for first, second in ((True, False), (False, True)):
            with patch.object(workflow, 'audit', side_effect=[{'report': {'training_fit': first}},
                                                           {'report': {'training_fit': second}}]) as audit:
                self.assertFalse(controller._fit(Path('/synthetic'), records, RAW, RID))
                self.assertEqual(audit.call_count, 2)
        self.assertEqual(controller._outcome(records, False), 'training_fit_failed')
        self.assertEqual(controller._outcome(records, True), 'training_fit_passed')
        for outcome in ('failed', 'timed_out', 'interrupted', 'not_attempted'):
            records['train-29']['outcome'] = outcome
            self.assertEqual(controller._outcome(records, False), 'incomplete')
            self.assertEqual(controller._outcome(records, True), 'incomplete')

    def test_success_and_negative_fit_have_only_two_train_cells(self):
        for fits in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); request = root/'request.json'; request.write_bytes(RAW)
                with patch.object(controller, 'validate'), \
                        patch.object(controller, '_cell', return_value={'outcome': 'completed', 'completion_identity': CID}) as cell, \
                        patch.object(controller, '_fit', return_value=fits):
                    result = controller.run_matrix(root/'run', request, RID)
                self.assertEqual(set(result['cells']), {'train-17', 'train-29'})
                self.assertEqual([call.args[4] for call in cell.call_args_list], [17, 29])
                self.assertEqual(result['both_seeds_fit'], fits)
                self.assertEqual(set(p.name for p in (root/'run').iterdir()), {'request.json', 'matrix.json'})

    def test_failure_does_not_skip_second_seed_but_interrupt_does(self):
        for outcome, calls in (('failed', 2), ('timed_out', 2), ('interrupted', 1)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); request = root/'request.json'; request.write_bytes(RAW)
                with patch.object(controller, 'validate'), \
                        patch.object(controller, '_cell', return_value={'outcome': outcome, 'completion_identity': None}) as cell:
                    result = controller.run_matrix(root/'run', request, RID)
                self.assertEqual(cell.call_count, calls)
                self.assertEqual(result['outcome'], 'incomplete')
                if outcome == 'interrupted':
                    self.assertEqual(result['cells']['train-29'], {'outcome': 'not_attempted',
                        'reason': 'earlier controller interruption'})

    def test_failed_worker_retains_partial_journal_and_missing_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def failed(child, *args):
                (child/'worker.log').write_bytes(b'synthetic failure\n')
                (child/'step-000-journal.jsonl').write_bytes(b'{"partial":true}\n')
                return 1
            with patch.object(controller, 'launch', side_effect=failed):
                result = controller._cell(root, root/'request.json', RAW, RID, 17)
            self.assertEqual(result['outcome'], 'failed')
            self.assertIsNone(result['completion_identity'])
            attempt = parse_json((root/'train-17/attempt.json').read_bytes(), canonical=True)
            self.assertEqual(attempt['missing_checkpoint_files'], workflow.missing_checkpoint_files(root/'train-17'))
            self.assertNotIn('step-000-journal.jsonl', attempt['missing_checkpoint_files']['0'])
            self.assertEqual(len(attempt['missing_checkpoint_files']['640']), 4)
            self.assertEqual((root/'train-17/step-000-journal.jsonl').read_bytes(), b'{"partial":true}\n')

    def test_timeout_and_interruption_are_retained_not_completed(self):
        for error, outcome in ((subprocess.TimeoutExpired('synthetic', 900), 'timed_out'),
                               (KeyboardInterrupt(), 'interrupted')):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(controller, 'launch', side_effect=error):
                    result = controller._cell(root, root/'request.json', RAW, RID, 17)
                self.assertEqual(result['outcome'], outcome)
                self.assertIsNone(result['completion_identity'])
                self.assertTrue((root/'train-17/attempt.json').is_file())

    def test_incomplete_matrix_read_only_audit_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root/'request.json'; request.write_bytes(RAW)
            with patch.object(controller, 'validate'), patch.object(controller, 'launch', return_value=1):
                result = controller.run_matrix(root/'run', request, RID)
                before = {str(p): p.read_bytes() for p in (root/'run').rglob('*') if p.is_file()}
                audited = controller.audit_matrix(root/'run', RAW, RID, result['matrix_identity'])
                self.assertTrue(audited['verified'])
                self.assertEqual(audited['outcome'], 'incomplete')
                self.assertEqual(before, {str(p): p.read_bytes() for p in (root/'run').rglob('*') if p.is_file()})
                (root/'run/train-17/injected.json').write_bytes(b'{}\n')
                with self.assertRaisesRegex(ContractError, 'retained cell membership'):
                    controller.audit_matrix(root/'run', RAW, RID, result['matrix_identity'])

    def test_resealed_incomplete_matrix_cannot_claim_negative_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root/'request.json'; request.write_bytes(RAW)
            with patch.object(controller, 'validate'), patch.object(controller, 'launch', return_value=1):
                controller.run_matrix(root/'run', request, RID)
                matrix = parse_json((root/'run/matrix.json').read_bytes(), canonical=True)
                for forged in ('training_fit_failed', 'training_fit_passed'):
                    matrix['outcome'] = forged
                    raw = json_bytes(matrix); (root/'run/matrix.json').write_bytes(raw)
                    with self.assertRaisesRegex(ContractError, 'matrix outcome'):
                        controller.audit_matrix(root/'run', RAW, RID, identity(raw))

    def test_paths_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root/'request.json'; request.write_bytes(RAW)
            with self.assertRaises(ContractError):
                controller.protect(controller.ROOT/'run', request)
            with self.assertRaises(ContractError):
                controller.protect(root, request)
            (root/'run').mkdir()
            with patch.object(controller, 'validate'):
                with self.assertRaises(FileExistsError):
                    controller.run_matrix(root/'run', request, RID)

    def test_completion_requires_all_three_fixed_artifact_groups(self):
        self.assertEqual(len(workflow.names()), 13)
        self.assertEqual(workflow.names(), {'report.json'} | set().union(*(workflow.checkpoint_names(s) for s in (0,64,640))))
        for value in ('nan', '-0x1.0000000000000p+0', 'inf', None, 1, (121.0).hex()):
            with self.assertRaises(ContractError):
                workflow.finite(value, 120)


if __name__ == '__main__':
    unittest.main()
