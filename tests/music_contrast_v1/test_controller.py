"""Controller/gate fakes certify mechanics only, never target learning/custody."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_contrast_v1 import controller, workflow
from music_contrast_v1.request import cell_name

RID = identity(b'synthetic request')
CID = identity(b'synthetic completion')


class GateTests(unittest.TestCase):
    def test_requires_both_bound_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = {'schema': 'trinite.music-contrast-gate.v1', 'request_identity': RID,
                    'passed': True, 'training_completions': {'train-17': CID, 'train-29': CID}}
            raw = json_bytes(gate); (root/'gate.json').write_bytes(raw)
            with patch.object(workflow, 'audit', return_value={'report': {'training_fit': True}}) as audit:
                checked, reports = workflow.require_gate(root, b'synthetic request', RID, identity(raw))
                self.assertEqual(checked, gate)
                self.assertEqual(set(reports), {'train-17','train-29'})
                self.assertEqual(audit.call_count, 2)
                self.assertEqual([call.args[3] for call in audit.call_args_list], [17,29])

    def test_one_seed_fit_cannot_unlock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = json_bytes({'schema': 'trinite.music-contrast-gate.v1', 'request_identity': RID,
                'passed': True, 'training_completions': {'train-17':CID, 'train-29':CID}})
            (root/'gate.json').write_bytes(raw)
            with patch.object(workflow, 'audit', side_effect=[{'report':{'training_fit':True}},
                                                             {'report':{'training_fit':False}}]):
                with self.assertRaisesRegex(ContractError, 'both seeds'):
                    workflow.require_gate(root, b'synthetic request', RID, identity(raw))

    def test_gate_bytes_are_independently_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'gate.json').write_bytes(b'{}\n')
            with patch.object(workflow, 'audit') as audit:
                with self.assertRaises(ContractError):
                    workflow.require_gate(root, b'synthetic request', RID, CID)
                audit.assert_not_called()

    def test_failed_gate_and_integer_true_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for passed in (False, 1):
                raw = json_bytes({'schema':'trinite.music-contrast-gate.v1', 'request_identity':RID,
                    'passed':passed, 'training_completions': {'train-17':CID,'train-29':CID}})
                (root/'gate.json').write_bytes(raw)
                with patch.object(workflow, 'audit') as audit:
                    with self.assertRaises(ContractError):
                        workflow.require_gate(root, b'synthetic request', RID, identity(raw))
                    audit.assert_not_called()

    def test_checkpoint_substitution_after_gate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original_payload=b'synthetic safe tensor placeholder'
            metadata=json_bytes({'synthetic':True})
            completion=json_bytes({'artifacts':{
                'step-064.safetensors':identity(original_payload),
                'step-064-checkpoint.json':identity(metadata)}})
            (root/'completion.json').write_bytes(completion)
            (root/'step-064.safetensors').write_bytes(b'substituted bytes')
            (root/'step-064-checkpoint.json').write_bytes(metadata)
            with self.assertRaisesRegex(ContractError,'checkpoint bytes changed'):
                workflow.bound_checkpoint(root,identity(completion))
            (root/'step-064.safetensors').write_bytes(original_payload)
            (root/'step-064-checkpoint.json').write_bytes(json_bytes({'synthetic':False}))
            with self.assertRaisesRegex(ContractError,'checkpoint bytes changed'):
                workflow.bound_checkpoint(root,identity(completion))
            (root/'completion.json').write_bytes(json_bytes({'artifacts':{}}))
            with self.assertRaisesRegex(ContractError,'completion changed'):
                workflow.bound_checkpoint(root,identity(completion))


    def test_unrecognised_cells_rejected(self):
        for seed, phase in ((True,'train'), (0,'train'), (17,'test')):
            with self.assertRaises(ContractError): cell_name(seed,phase)


class ControllerTests(unittest.TestCase):
    def test_target_run_requires_explicit_flag_before_configure(self):
        with patch.object(controller, 'configure') as configure, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(controller.main(['run']), 1)
            configure.assert_not_called()

    def test_preflight_never_configures_model(self):
        with patch.object(controller,'configure') as configure, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(controller.main(['preflight']), 0)
            configure.assert_not_called()

    def test_no_holdout_calls_when_training_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); request=root/'request.json'; request.write_bytes(b'synthetic request')
            gate={'schema':'trinite.music-contrast-gate.v1','request_identity':RID,'passed':False,
                  'training_completions': {'train-17':CID,'train-29':CID}}
            with patch.object(controller,'validate'), patch.object(controller,'_cell',
                return_value={'outcome':'completed','completion_identity':CID}) as cell, \
                patch.object(controller,'_gate',return_value=gate):
                result=controller.run_matrix(root/'run',request,RID)
            self.assertEqual(result['outcome'],'training_fit_failed')
            self.assertEqual(cell.call_count,2)
            self.assertTrue(all(c.args[5]=='train' for c in cell.call_args_list))
            self.assertFalse(result['training_gate_passed'])
            self.assertEqual(result['cells']['heldout-17']['outcome'],'not_attempted')
            self.assertEqual(result['cells']['heldout-29']['outcome'],'not_attempted')

    def test_successful_gate_is_written_before_heldout_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); request=root/'request.json'; request.write_bytes(b'synthetic request')
            gate={'schema':'trinite.music-contrast-gate.v1','request_identity':RID,'passed':True,
                  'training_completions': {'train-17':CID,'train-29':CID}}
            seen=[]
            def fake_cell(output, request_path, raw, rid, seed, phase, gate_id):
                seen.append((seed,phase))
                if phase=='heldout':
                    self.assertEqual((output/'gate.json').read_bytes(),json_bytes(gate))
                    self.assertEqual(gate_id,identity(json_bytes(gate)))
                return {'outcome':'completed','completion_identity':CID}
            with patch.object(controller,'validate'),patch.object(controller,'_cell',side_effect=fake_cell), \
                    patch.object(controller,'_gate',return_value=gate):
                result=controller.run_matrix(root/'run',request,RID)
            self.assertEqual(seen,[(17,'train'),(29,'train'),(17,'heldout'),(29,'heldout')])
            self.assertEqual(result['outcome'],'completed')

    def test_interruption_preserves_unattempted_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); request=root/'request.json'; request.write_bytes(b'synthetic request')
            gate={'schema':'trinite.music-contrast-gate.v1','request_identity':RID,'passed':False,
                  'training_completions': {'train-17':None,'train-29':None}}
            with patch.object(controller,'validate'),patch.object(controller,'_cell',return_value={
                    'outcome':'interrupted','completion_identity':None}) as cell, \
                    patch.object(controller,'_gate',return_value=gate):
                result=controller.run_matrix(root/'run',request,RID)
            self.assertEqual(cell.call_count,1)
            self.assertEqual(result['outcome'],'incomplete')
            self.assertEqual(result['cells']['train-29']['outcome'],'not_attempted')

    def test_fresh_destination_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); request=root/'request.json';request.write_bytes(b'synthetic request')
            output=root/'run';output.mkdir();(output/'retained').write_bytes(b'keep')
            with patch.object(controller,'validate'):
                with self.assertRaises(FileExistsError):controller.run_matrix(output,request,RID)
            self.assertEqual((output/'retained').read_bytes(),b'keep')

    def test_output_cannot_enclose_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';request.write_bytes(b'x')
            with self.assertRaises(ContractError):controller.protect(root,request)

    def test_symlink_output_parent_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'real').mkdir();(root/'link').symlink_to(root/'real',target_is_directory=True)
            request=root/'request.json';request.write_bytes(b'x')
            with self.assertRaises(ContractError):controller.protect(root/'link'/'run',request)

if __name__=='__main__':unittest.main()
