"""Stdlib-only freeze and scope test; no model/Torch imports in root discovery."""
from pathlib import Path
import hashlib
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_SHA='14261ecc9c6bf5f9487a71a1def5ce9115298de6cefd9fd9b702eb91a06bbebf'

class TernaryDynamicsContracts(unittest.TestCase):
    def test_frozen_protocol_and_separate_observer_source(self):
        data=(ROOT/'scripts/ternary_dynamics/protocol.json').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),PROTOCOL_SHA)
        self.assertEqual(data,(json.dumps(json.loads(data),sort_keys=True,
                         ensure_ascii=False,separators=(',',':'))+'\n').encode())
        doc=json.loads(data)
        self.assertEqual(doc['quantizer'],'trinite.absmean-ste.v0')
        self.assertIn('no update/replay/causal',doc['scope'])
        self.assertTrue((ROOT/'scripts/analyze_ternary_dynamics.py').is_file())
        self.assertFalse((ROOT/'src/trinite/ternary_dynamics.py').exists())
        runner=(ROOT/'scripts/check_research_cpu.py').read_text()
        self.assertIn("'ternary_dynamics'", runner.split('RESEARCH_SUITES = ', 1)[1].split('\n', 1)[0])


if __name__=='__main__':unittest.main()
