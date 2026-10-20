"""Read-only v2 transport packet audit, preserving the original v1 record."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

def digest(raw):return 'sha256:'+hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()

class TransportEvidenceTests(unittest.TestCase):
    def test_packet_binds_new_transport_and_unchanged_inference(self):
        root=ROOT/'fixtures/frozen-decoding-transport-v2';c=json.loads((root/'commitment.json').read_bytes())
        for name,expected in c['files'].items():self.assertEqual(digest((root/name).read_bytes()),expected)
        raw=gzip.decompress((root/'report.json.gz').read_bytes());self.assertEqual(digest(raw),c['report_identity'])
        report=json.loads(raw);request=json.loads((root/'request.json').read_bytes());self.assertEqual(report['request'],request)
        for paths in (request['transport_source'],request['inference_request']['source']):
            for path,expected in paths.items():self.assertEqual(digest((ROOT/path).read_bytes()),expected,path)
        old=gzip.decompress((ROOT/'fixtures/frozen-decoding-v1/report.json.gz').read_bytes())
        self.assertEqual(canonical(report['inference_report']),old)
        self.assertEqual(digest(old),c['unchanged_inference_identity'])
        sidecars=list(root.glob('report.json.attempt-*.cells.jsonl.gz'));self.assertEqual(len(sidecars),1)
        records=[json.loads(line) for line in gzip.decompress(sidecars[0].read_bytes()).splitlines()]
        self.assertEqual(records,report['inference_report']['cells'])
