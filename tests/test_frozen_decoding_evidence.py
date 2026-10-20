"""Read-only byte and result audit; not a substitute for pinned CPU inference replay."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class FrozenDecodingEvidenceTests(unittest.TestCase):
    def test_source_bound_full_matrix_and_constant_answer_limits(self):
        root=ROOT/'fixtures/frozen-decoding-v1'
        commitment=json.loads((root/'commitment.json').read_bytes())
        for name,expected in commitment['files'].items():
            self.assertEqual('sha256:'+hashlib.sha256((root/name).read_bytes()).hexdigest(),expected)
        raw=gzip.decompress((root/'report.json.gz').read_bytes())
        self.assertEqual('sha256:'+hashlib.sha256(raw).hexdigest(),commitment['report_identity'])
        report=json.loads(raw);request=json.loads((root/'request.json').read_bytes())
        self.assertEqual(report['request'],request)
        for path,expected in request['source'].items():
            self.assertEqual('sha256:'+hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected,path)
        self.assertEqual([(c['seed'],c['arm']) for c in report['cells']],[(0,'absmean'),(0,'refit'),(1,'absmean'),(1,'refit')])
        self.assertEqual(report['training_updates'],0);self.assertFalse(report['test_evaluated'])
        for c in report['cells']:
            for mode in ('greedy','syntax'):
                rows=c['modes'][mode]['predictions'];self.assertEqual(len(rows),42)
                self.assertEqual(len({r['example_identity'] for r in rows}),42)
                for row in rows:
                    self.assertEqual(row['generated_tokens'],[257] if mode=='greedy' else [50 if c['seed']==0 else 48,257])
                    self.assertEqual(row['forward_calls'],1 if mode=='greedy' else 2)
                counts=c['modes'][mode]['splits']
                self.assertEqual(counts['train']['correct'],0 if mode=='greedy' else 12)
                self.assertEqual(counts['validation']['correct'],0 if mode=='greedy' else (4 if c['seed']==0 else 2))
