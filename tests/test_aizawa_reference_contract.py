"""Standard-library protocol freeze and bounded reference source closure."""
from pathlib import Path
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'cb904c512d45d8abba55a258c6b649e320c2ca37a7c04c9464d4c5623ca4857f'


class AizawaProtocolContract(unittest.TestCase):
    def test_immutable_protocol_canonical_json_and_combined_runner(self):
        raw = (ROOT/'scripts/aizawa_reference/protocol.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED)
        self.assertEqual(raw,(json.dumps(json.loads(raw),sort_keys=True,ensure_ascii=False,
                            separators=(',',':'))+'\n').encode())
        contract = json.loads(raw)
        self.assertEqual(contract['schema'], 'trinite.aizawa-reference-protocol.v1')
        self.assertEqual(contract['numerics']['samples'],201)
        self.assertEqual(contract['numerics']['coarse_steps'],20000)
        self.assertEqual(contract['numerics']['fine_steps'],40000)
        self.assertEqual(contract['evidence_class'],'SIMULATION')
        self.assertEqual(contract['analysis']['alignment_mode'],
                         'error (matching 201 sampled positions)')
        self.assertTrue((ROOT/'scripts/run_aizawa_reference.py').is_file())
        self.assertFalse((ROOT/'src/trinite/aizawa_reference.py').exists())
        runner = (ROOT/'scripts/check_research_cpu.py').read_text()
        self.assertIn("'aizawa_reference'",runner.split('RESEARCH_SUITES = ',1)[1].split('\n',1)[0])


if __name__ == '__main__':
    unittest.main()
