"""Immutable data-only proposal records; these are not research execution tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/procedural-qec-nondisplacing-proposal'
RECEIPT_SHA256 = 'e8cc15200ed9bf6ae3820674f4383d3c67b54ab608048b1309d861598b232555'
AUDIT_SHA256 = '00894610c3ebb2d61590a6320aeedd029fdfaae7b66344c137632d097a1dfa53'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def retained(raw, sidecar):
    if type(raw) is not bytes or type(sidecar) is not bytes:
        raise ValueError('immutable planning bytes required')
    if (digest(raw) != RECEIPT_SHA256 or sidecar !=
            (RECEIPT_SHA256 + '  planning-counts.json\n').encode('ascii')):
        raise ValueError('independent planning commitment mismatch')
    value = json.loads(raw)
    canonical = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'), allow_nan=False) + '\n').encode()
    if canonical != raw:
        raise ValueError('noncanonical planning receipt')
    return value


class NonDisplacingProposalTests(unittest.TestCase):
    def record(self):
        return retained((EVIDENCE / 'planning-counts.json').read_bytes(),
                        (EVIDENCE / 'planning-counts.sha256').read_bytes())

    def test_original_planning_bindings_and_limits(self):
        record = self.record()
        self.assertEqual(record['planning_script_identity'],
                         'sha256:' + digest((EVIDENCE / 'planning_counts.py').read_bytes()))
        self.assertEqual(record['proposal_identity'], 'sha256:' + digest(
            (ROOT / 'docs/PROCEDURAL-QEC-NONDISPLACING-PROPOSAL.md').read_bytes()))
        self.assertEqual(record['audit_script_identity'], 'sha256:' + AUDIT_SHA256)
        audit = ROOT / 'docs/evidence/procedural-qec-control-order'
        self.assertEqual(digest((audit / 'audit.py').read_bytes()), AUDIT_SHA256)
        self.assertEqual(record['original_comparison_receipt_identity'],
                         'sha256:' + digest((audit / 'comparison.json').read_bytes()))
        self.assertEqual(record['historical_producer_commit'],
                         '93522bf8f18fe38970ceafdd2ace792f8922c382')
        self.assertEqual(record['implementation_status'],
                         'proposal only; no protocol or request frozen')
        self.assertIn('not research outcomes', record['classification'])
        self.assertIn('No model calls, training, workflow launch, native replay or new freeze',
                      record['limitations'])

    def test_complete_recipe_counts_and_historical_primary_commitments(self):
        record = self.record()
        historical = json.loads((ROOT / 'docs/evidence/procedural-qec-control-order/comparison.json').read_bytes())
        expected_targets = {(0, 'extra-final'): 152429, (0, 'extra-procedural'): 272510,
                            (1, 'extra-final'): 152430, (1, 'extra-procedural'): 272493}
        self.assertEqual(len(record['cells']), 4)
        self.assertEqual({(c['seed'], c['arm']) for c in record['cells']}, set(expected_targets))
        for cell in record['cells']:
            key = cell['seed'], cell['arm']
            self.assertEqual(cell['native_updates'], 8192)
            self.assertEqual(cell['primary']['totals']['visits'], 32768)
            self.assertEqual(cell['extra']['totals']['visits'], 32768)
            self.assertEqual(cell['combined']['totals']['visits'], 65536)
            self.assertEqual(cell['combined']['totals']['target_tokens'], expected_targets[key])
            for field in ('visits', 'target_tokens', 'input_tokens'):
                self.assertEqual(cell['combined']['totals'][field],
                                 cell['primary']['totals'][field] + cell['extra']['totals'][field])
            self.assertEqual(cell['model_input_positions'], 8192 * 8 * 63)
            self.assertEqual(cell['padded_encoded_positions'], 8192 * 8 * 64)
            self.assertLess(cell['combined']['totals']['target_tokens'], 1048576)
            self.assertEqual(cell['original_audit_primary_sequence_identity'],
                             historical['cells'][cell['seed'] * 2]['historical_sequence_identity'])
        self.assertEqual(record['aggregate'], {'native_training_updates': 32768,
                         'training_row_visits': 262144, 'training_target_tokens': 849862})

    def test_mutated_or_coordinately_resealed_receipts_reject(self):
        original = (EVIDENCE / 'planning-counts.json').read_bytes()
        sidecar = (EVIDENCE / 'planning-counts.sha256').read_bytes()
        changed = original + b' '
        resealed = (digest(changed) + '  planning-counts.json\n').encode('ascii')
        for raw, seal in ((changed, sidecar), (changed, resealed), (original, resealed),
                          (bytearray(original), sidecar)):
            with self.subTest(mutation=type(raw).__name__, seal=seal), self.assertRaises(ValueError):
                retained(raw, seal)
        self.assertEqual((EVIDENCE / 'planning-counts.json').read_bytes(), original)

    def test_changed_audit_or_git_bytes_reject_before_execution(self):
        # Process fakes here test controller failure only, never certify data.
        path = EVIDENCE / 'planning_counts.py'
        module = types.ModuleType('planning_negative_test')
        module.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        original = (ROOT / 'docs/evidence/procedural-qec-control-order/audit.py').read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / 'docs/evidence/procedural-qec-control-order/audit.py'
            audit.parent.mkdir(parents=True)
            marker = root / 'executed'
            changed = original + ('\nfrom pathlib import Path\nPath(' + repr(str(marker))
                                  + ').write_text("executed")\n').encode()
            audit.write_bytes(changed)
            for git_bytes in (original, changed):
                argv = [str(path), '--repository', str(root),
                        '--historical-source', str(root / 'unused-historical')]
                with self.subTest(git_changed=git_bytes != original), \
                     patch.object(module.sys, 'argv', argv), \
                     patch.object(module.subprocess, 'run',
                                  return_value=types.SimpleNamespace(stdout=git_bytes)), \
                     self.assertRaisesRegex(RuntimeError, 'independently pinned baseline'):
                    module.main()
                self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
