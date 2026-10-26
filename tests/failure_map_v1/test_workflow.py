"""Real tiny producer -> frozen request -> observational custody -> read-only audit.

Four-update engineering models only, not the owner's six target checkpoints.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.model import Decoder
from addition_development_v1 import controller as original_controller
from failure_map_v1 import request, workflow
from frozen_token_diagnostic_v1 import producer
# Reuse actual engineering producer/custody fixture, not a mocked success receipt.
import importlib.util
_spec = importlib.util.spec_from_file_location('failure_map_toy_producer',
    Path(__file__).resolve().parents[1]/'frozen_token_diagnostic_v1/test_producer.py')
_toy = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_toy)
toy_matrix, toy_patches = _toy.toy_matrix, _toy.toy_patches


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_controller.configure()
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.patches = toy_patches(); cls.patches.__enter__()
        try:
            cls.source = cls.root/'producer'
            rid,mid,models = toy_matrix(cls.source)
            cls.raw = json_bytes(request.core(cls.source,rid,mid,models)); cls.rid = identity(cls.raw)
            cls.output = cls.root/'observation'; cls.output.mkdir(); st = cls.output.stat()
            with patch.object(torch.optim,'AdamW',side_effect=AssertionError('no optimizer in diagnostic')):
                cls.cid = workflow.run(cls.output,cls.raw,cls.rid,cls.source,'control',0,(st.st_dev,st.st_ino))
        except BaseException:
            cls.patches.close(); cls.temp.cleanup(); raise

    @classmethod
    def tearDownClass(cls):
        cls.patches.close(); cls.temp.cleanup()

    def test_real_closed_custody_all_rows_no_audit_forward(self):
        before = producer.snapshot_matrix(self.source)
        with patch.object(Decoder,'forward',side_effect=AssertionError('audit must not infer')), \
             patch.object(torch.optim,'AdamW',side_effect=AssertionError('audit must not construct optimizer')):
            result = workflow.audit(self.output,self.raw,self.rid,self.source,'control',0,self.cid)
        self.assertTrue(result['verified']); self.assertFalse(result['audit_replays_inference'])
        self.assertEqual(result['summary']['rows'],338)
        # This actual four-update engineering model gets every raw answer wrong;
        # matching retained wrong answers still close custody (no accuracy gate).
        self.assertEqual(result['summary']['raw_correct'],0)
        self.assertEqual(result['summary']['fail'],338)
        report = parse_json((self.output/'report.json').read_bytes(),canonical=True)
        self.assertTrue(report['target_inference_executed'])
        self.assertEqual(report['historical_raw_predictions_compared'],318)
        self.assertEqual(result['summary']['reproducibility'].get('mismatch',0),0)
        self.assertEqual((self.output/'original-training.json').read_bytes(),
                         (self.source/'control-0'/'training-4.json').read_bytes())
        for stratum in producer.STRATA:
            self.assertEqual((self.output/f'original-raw-{stratum}.json').read_bytes(),
                             (self.source/'control-0'/f'development-{stratum}.json').read_bytes())
        producer.unchanged_matrix(self.source,before)

    def test_wrong_completion_rejected_and_untouched(self):
        from unseen_evaluation_v1.packet import inventory, unchanged
        before = inventory(self.output)
        with self.assertRaises(ContractError):
            workflow.audit(self.output,self.raw,self.rid,self.source,'control',0,'sha256:'+'0'*64)
        unchanged(self.output,before)

    def test_chunk_and_resource_contracts(self):
        self.assertEqual(len(workflow.chunk_names()),11)
        completion = parse_json((self.output/'completion.json').read_bytes(),canonical=True)
        self.assertEqual(set(completion['artifacts']),workflow.artifact_names())
        resources = parse_json((self.output/'resources.json').read_bytes(),canonical=True)
        for field,bad in [('forward_calls_upper_bound',True),('forward_calls_upper_bound',1691),('kv_payload_bytes',0),
                          ('wall_seconds_hex',(901.0).hex()),('peak_rss_kib',-1)]:
            with self.assertRaises(ContractError): workflow.resources({**resources,field:bad})

    def test_replay_mismatch_retained_without_completion(self):
        # Inject a retained-output mismatch, not a numerical success certificate.
        # Admission, producer import, report checker and failure writes remain real.
        from failure_map_v1.measurements import _build_report
        from trinite.tokenizer import EOS
        original = {s: (self.source/'control-0'/f'development-{s}.json').read_bytes()
                    for s in producer.STRATA}
        training = (self.source/'control-0'/'training-4.json').read_bytes()
        good = parse_json((self.output/'measurements.json').read_bytes(), canonical=True)
        predictions = [r['prediction'] for r in good['rows']]
        row = predictions[0]
        row['generated_tokens'] = [EOS] if row['generated_tokens'] != [EOS] else [48, EOS]
        row['correct'] = row['generated_tokens'] == row['expected_tokens']
        bad = _build_report(predictions,'control',0,good['model_identity'],original,training)
        self.assertGreater(bad['summary']['reproducibility']['mismatch'],0)
        output = self.root/'replay-mismatch'; output.mkdir(); st = output.stat()
        def measurement(*args, progress=None):
            for row in bad['rows']: progress(row)
            return bad
        with patch.object(workflow,'measure',side_effect=measurement), \
             self.assertRaisesRegex(ContractError,'reproducibility/plumbing mismatch'):
            workflow.run(output,self.raw,self.rid,self.source,'control',0,(st.st_dev,st.st_ino))
        self.assertEqual((output/'measurements.json').read_bytes(),json_bytes(bad))
        self.assertTrue((output/'original-training.json').is_file())
        self.assertTrue((output/'progress-index.json').is_file())
        self.assertFalse((output/'completion.json').exists())
        self.assertFalse((output/'provenance').exists())

    def test_request_reseal_does_not_change_protocol(self):
        record = parse_json(self.raw,canonical=True)
        record['protocol']['checkpoint_step'] = 1000
        encoded = json_bytes(record)
        with self.assertRaisesRegex(ContractError,'source/environment/producer/protocol'):
            request.validate(encoded,identity(encoded),self.source)


if __name__ == '__main__': unittest.main()
