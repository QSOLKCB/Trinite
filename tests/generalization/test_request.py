from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.generalization import freeze_request, read_request, analyze_run, protocol


class DiagnosticRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    def test_request_binds_source_environment_curriculum_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'request.json'; frozen = freeze_request(path)
            record, raw = read_request(path,frozen['request_identity'])
            for key in ('source','environment','curricula','protocol','protocol_commit'):
                changed = parse_json(raw); changed[key] = 'altered'; path.write_bytes(json_bytes(changed))
                with self.subTest(key=key), self.assertRaises(ContractError):
                    read_request(path,identity(path.read_bytes()))
            path.write_bytes(raw)
            with patch('trinite.generalization.sources',return_value={'changed':'source'}):
                with self.assertRaises(ContractError): read_request(path,frozen['request_identity'])
            self.assertEqual(record['protocol']['parent'],protocol()['parent'])

    def test_wrong_parent_rejects_before_model_verification_or_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root/'request.json'; frozen = freeze_request(path)
            run = root/'run'; run.mkdir(); (run/'request.json').write_bytes(b'{}\n')
            with patch('trinite.learning.summarize') as verify, patch('trinite.generalization.diagnose') as diagnostic:
                with self.assertRaisesRegex(ContractError,'parent request.json identity mismatch'):
                    analyze_run(run,path,frozen['request_identity'])
                verify.assert_not_called(); diagnostic.assert_not_called()


if __name__ == '__main__': unittest.main()
