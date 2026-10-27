"""Exact source-specific admission conformance; never runs a target model."""
from copy import deepcopy
from pathlib import Path
import os
from importlib import import_module
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_learning_v1.admission import (
    ROOT, SOURCES, MAX_DATASET_BYTES, MAX_ADMISSION_BYTES,
    dataset, audit_bytes, validate_admission, source_receipt,
)
from music_learning_v1.codec import TASKS, encode_example


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.admission = dataset()

    def test_exact_replay_counts_split_and_masks(self):
        raw, admission = dataset()
        self.assertEqual((raw, admission), (self.raw, self.admission))
        self.assertTrue(audit_bytes(raw, admission)['verified'])
        rows = parse_json(raw)['examples']
        self.assertEqual(len(rows), 960)
        for split, expected, families in [('train', 480, 4), ('validation', 240, 2), ('test', 240, 2)]:
            selected = [r for r in rows if r['split'] == split]
            self.assertEqual(len(selected), expected)
            self.assertEqual(len({r['family_id'] for r in selected}), families)
            self.assertEqual({r['task'] for r in selected}, set(TASKS))
        for row in rows:
            view = {k: row[k] for k in ('task', 'prompt', 'answer', 'text', 'answer_start')}
            encoding = encode_example(view)
            self.assertEqual(sum(encoding.loss_mask), row['answer_token_count'])
            self.assertNotIn(row['family_id'], row['prompt'])
        self.assertEqual(validate_admission(admission)['outcome'], 'admitted')

    def test_contradictory_policy_and_self_resealed_input_rejected(self):
        original = parse_json(self.admission)
        for key, value in [('outcome', 'rejected'), ('scope', 'all music'),
                           ('rights_basis', 'permissive software license'),
                           ('reviewed_on', '2026-10-28'), ('author', 'anonymous'),
                           ('execution_authorization', 'run automatically')]:
            bad = deepcopy(original)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_admission(bad)
        changed = parse_json(self.raw)
        changed['examples'][0]['answer'] = '999'
        raw = json_bytes(changed)
        bad = deepcopy(original)
        bad['dataset_identity'] = identity(raw)
        with self.assertRaises(ContractError):
            audit_bytes(raw, json_bytes(bad))

    def test_injection_split_or_lineage_not_admitted(self):
        original = parse_json(self.raw)
        for key, value in [('split', 'test'), ('family_id', 'new-family'),
                           ('seed', True), ('generator_revision', 'sha256:'+'0'*64),
                           ('formal', {}), ('prompt', 'attacks=999')]:
            bad = deepcopy(original)
            # Ensure split actually changes even if first parent is test.
            if key == 'split' and bad['examples'][0][key] == value:
                value = 'train'
            bad['examples'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                audit_bytes(json_bytes(bad), self.admission)
        with self.assertRaises(ContractError):
            audit_bytes(self.raw + b' ', self.admission)
        with self.assertRaises(ContractError):
            audit_bytes(self.raw, self.admission.replace(b'"outcome":', b'"outcome":"admitted","outcome":'))

    def test_source_change_and_foreign_root_rejected(self):
        original = source_receipt
        def changed_receipt():
            result = original()
            result['scripts/music_learning_v1/codec.py'] = 'sha256:' + '0' * 64
            return result
        with patch('music_learning_v1.admission.source_receipt', side_effect=changed_receipt):
            with self.assertRaises(ContractError):
                audit_bytes(self.raw, self.admission)
        with tempfile.TemporaryDirectory() as directory:
            with patch('music_learning_v1.admission.ROOT', Path(directory)):
                with self.assertRaises(ContractError):
                    source_receipt()

    def test_byte_caps_and_types_precede_json_parsing(self):
        with patch('music_learning_v1.admission.parse_json', side_effect=AssertionError('must not parse')):
            for raw, receipt in [(b'x' * (MAX_DATASET_BYTES + 1), self.admission),
                                 (self.raw, b'x' * (MAX_ADMISSION_BYTES + 1)),
                                 (bytearray(self.raw), self.admission), (self.raw, 'not bytes')]:
                with self.assertRaises(ContractError):
                    audit_bytes(raw, receipt)
            for receipt in (b'x' * (MAX_ADMISSION_BYTES + 1), bytearray(b'{}'), None):
                with self.assertRaises(ContractError):
                    validate_admission(receipt)

    def test_foreign_imported_runtime_rejected(self):
        for name in ('music_learning_v1.codec', 'symbolic_music_v1.oracle', 'trinite.tokenizer'):
            module = import_module(name)
            with patch.object(module, '__file__', '/tmp/foreign/' + name + '.py'):
                with self.subTest(module=name), self.assertRaisesRegex(ContractError, 'foreign music admission import'):
                    source_receipt()

    def test_actual_copied_source_replay_and_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in SOURCES:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / name).read_bytes())
            (root / 'dataset.json').write_bytes(self.raw)
            (root / 'admission.json').write_bytes(self.admission)
            code = ('from pathlib import Path; from music_learning_v1.admission import audit_bytes; '
                    'audit_bytes(Path("dataset.json").read_bytes(), Path("admission.json").read_bytes())')
            env = dict(os.environ, PYTHONPATH=str(root / 'src') + os.pathsep + str(root / 'scripts'))
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            path = root / 'scripts/music_learning_v1/codec.py'
            path.write_bytes(path.read_bytes() + b'\n# changed executing source\n')
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('exact source/input admission replay mismatch', result.stderr)

    def test_no_model_geometry_or_audio_import(self):
        code = ('import sys; from music_learning_v1.admission import dataset; dataset(); '
                'assert not any(n in sys.modules for n in '
                '["torch", "numpy", "trinite.model", "trinite.training", '
                '"trinite.geometry", "symbolic_music_v1.render"])')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
