import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from check_pcgrad import main
from pcgrad_v1.evidence import report
from pcgrad_v1.spec import ROOT, PROTOCOL_ID, sources
from trinite.contracts import identity, json_bytes, parse_json


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_report_full_cells_and_fresh_repeat(self):
        first = report()
        self.assertEqual(json_bytes(first), json_bytes(report()))
        self.assertEqual(first['source'], sources())
        self.assertEqual(first['protocol_identity'], PROTOCOL_ID)
        self.assertEqual({(c['arm'], c['seed']) for c in first['cells']},
                         {(a, s) for a in ('sum', 'pcgrad') for s in (0, 1)})
        for key in ('scientific_execution', 'training_gate_evaluated', 'learning_claim',
                    'release_ready', 'closed_custody', 'native_checkpoint_replay'):
            self.assertIs(first[key], False)
        for cell in first['cells']:
            self.assertEqual(identity(json_bytes(cell['final_checkpoint'])), cell['final_checkpoint_identity'])
            self.assertEqual([r['stop'] for r in cell['interruptions']], [0, 3, 7, 8])
            self.assertTrue(all(r['exact_match'] for r in cell['interruptions']))

    def test_cli_exclusive_readonly_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            path = Path(folder)/'report.json'
            args = ['--output', str(path)]
            self.assertEqual(main(args), 0)
            raw = path.read_bytes(); digest = identity(raw)
            before = path.stat().st_mtime_ns
            self.assertEqual(main(args+['--verify-only', '--expected-identity', digest]), 0)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertEqual(main(args), 1)
            self.assertEqual(main(args+['--verify-only']), 1)
            self.assertEqual(main(args+['--expected-identity', digest]), 1)
            path.write_bytes(raw+b' ')
            self.assertEqual(main(args+['--verify-only', '--expected-identity', identity(path.read_bytes())]), 1)
            path.unlink(); path.symlink_to(Path(folder)/'missing')
            self.assertEqual(main(args), 1)
            self.assertEqual(main(args+['--verify-only', '--expected-identity', digest]), 1)

    def test_cli_research_rejects_before_report_or_output(self):
        with tempfile.TemporaryDirectory() as folder, patch('check_pcgrad.report', side_effect=AssertionError('executed')), contextlib.redirect_stderr(io.StringIO()):
            path = Path(folder)/'no.json'
            self.assertEqual(main(['--output', str(path), '--profile', 'research']), 1)
            self.assertFalse(path.exists())

    def test_retained_source_binding_and_scope(self):
        # Retained fixture is source-bound, not rerun under a different host.
        root = ROOT/'fixtures/pcgrad-v1'
        raw = (root/'report.json').read_bytes()
        commitment = parse_json((root/'commitment.json').read_bytes(), canonical=True)
        self.assertEqual(identity(raw), commitment['report_identity'])
        retained = parse_json(raw, canonical=True)
        self.assertEqual(retained['source'], sources())
        self.assertEqual(retained['protocol_identity'], PROTOCOL_ID)
        self.assertFalse(retained['learning_claim'])
        self.assertEqual(len(retained['cells']), 4)
        for cell in retained['cells']:
            self.assertEqual(cell['final_checkpoint']['source'], sources())
            self.assertEqual(identity(json_bytes(cell['final_checkpoint'])), cell['final_checkpoint_identity'])
            self.assertEqual(cell['final_checkpoint']['step'], 8)


if __name__ == '__main__': unittest.main()
