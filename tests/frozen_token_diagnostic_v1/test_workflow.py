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
from frozen_token_diagnostic_v1 import request, workflow, producer
from test_producer import toy_matrix, toy_patches


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
        self.assertTrue(result['verified']); self.assertFalse(result['predictions_replayed'])
        self.assertEqual(sum(r['gold_prefix_diagnostic']['examples'] for r in result['summaries'].values()),298)
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
        for field,bad in [('forward_calls',True),('forward_calls',597),('kv_payload_bytes',0),
                          ('wall_seconds_hex',(901.0).hex()),('peak_rss_kib',-1)]:
            with self.assertRaises(ContractError): workflow.resources({**resources,field:bad})

    def test_request_reseal_does_not_change_protocol(self):
        record = parse_json(self.raw,canonical=True)
        record['protocol']['checkpoint_step'] = 1000
        encoded = json_bytes(record)
        with self.assertRaisesRegex(ContractError,'source/environment/producer/protocol'):
            request.validate(encoded,identity(encoded),self.source)


if __name__ == '__main__': unittest.main()
