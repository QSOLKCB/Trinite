"""Retained transport/source checks, not a substitute for numerical replay."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from scale_refit_training_v1.spec import PROTOCOL_ID, sources, specification


def identity(raw):return 'sha256:'+hashlib.sha256(raw).hexdigest()


class RefitPilotEvidenceTests(unittest.TestCase):
    def test_original_report_source_inputs_and_all_negative_cells_retained(self):
        root=ROOT/'fixtures/scale-refit-training-v1'
        commitment=json.loads((root/'commitment.json').read_bytes())
        compressed=(root/'report.json.gz').read_bytes()
        self.assertEqual(identity(compressed),commitment['compressed_report_identity'])
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw=stream.read(16_777_217)
        self.assertLessEqual(len(raw),16_777_216)
        self.assertEqual(len(raw),commitment['uncompressed_bytes'])
        self.assertEqual(identity(raw),commitment['report_identity'])
        self.assertEqual(commitment['report_identity'],'sha256:5e3614ea4a34248afe656b867cc59fca2ea64b8198d372e8d58bf7aa6e4e04c0')
        report=json.loads(raw)
        request=(root/'request.json').read_bytes()
        self.assertEqual(identity(request),commitment['request_identity'])
        requested=json.loads(request)
        self.assertEqual(report['source'],sources())
        self.assertEqual(requested['source'],report['source'])
        self.assertEqual(requested['environment'],report['environment'])
        self.assertEqual(requested['protocol'],specification())
        self.assertEqual(report['protocol_identity'],PROTOCOL_ID)
        self.assertFalse(report['test_evaluated'])
        self.assertFalse(report['speed_claim'])
        self.assertFalse(report['format_selected'])
        self.assertFalse(report['release_ready'])
        self.assertFalse(commitment['closed_custody'])
        self.assertEqual([(c['seed'],c['arm']) for c in report['cells']],[(0,'absmean'),(0,'refit'),(1,'absmean'),(1,'refit')])
        for cell in report['cells']:
            self.assertEqual(cell['verified_resume_stops'],[0,1,30,60])
            final=cell['final_checkpoint']
            self.assertEqual((final['step'],final['cursor'],final['target_tokens']),(60,240,480))
            self.assertEqual(final['inventory']['parameters'],6752)
            self.assertEqual(final['source'],report['source'])
            self.assertEqual(set(cell['quality']),{'train','validation'})
            for split,count in (('train',36),('validation',6)):
                q=cell['quality'][split]
                self.assertEqual(q['examples'],count)
                self.assertEqual(len(q['predictions']),count)
                correct=sum(p['generated_tokens']==p['expected_tokens'] for p in q['predictions'])
                self.assertEqual(q['correct'],correct)
                self.assertEqual(correct,0)  # Preserve the measured negative exact-answer outcome.
            for d in cell['same_weight_diagnostics'].values():
                self.assertTrue(d['codes_preserved_at_same_weight'])
                baseline,selected=map(int,d['weight_sse_numerators'])
                self.assertLessEqual(selected,baseline)
                self.assertIn('fixed_input_output_sse_numerators',d)
        for seed in (0,1):
            a,b=[c for c in report['cells'] if c['seed']==seed]
            self.assertEqual(a['initial_checkpoint']['model_hex'],b['initial_checkpoint']['model_hex'])
            self.assertEqual([h['examples'] for h in a['final_checkpoint']['history']],
                             [h['examples'] for h in b['final_checkpoint']['history']])

if __name__=='__main__':unittest.main()
