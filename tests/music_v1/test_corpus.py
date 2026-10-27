import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from symbolic_music_v1.corpus import (
    generate, validate_rows, family_split, artifacts, write_bundle, verify_bundle,
)

ROOT = Path(__file__).resolve().parents[2]


class CorpusTests(unittest.TestCase):
    def test_deterministic_complete_disjoint(self):
        rows = generate()
        self.assertEqual(rows, generate())
        self.assertEqual(len(rows), 160)
        self.assertEqual({s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')},
                         {'train': 80, 'validation': 40, 'test': 40})
        self.assertEqual(len({r['id'] for r in rows}), 160)
        for family in range(8):
            selected = [r for r in rows if r['family'] == f'rest-mask-{family}']
            self.assertEqual(len(selected), 20)
            self.assertEqual({r['split'] for r in selected}, {family_split()[family]})
            self.assertEqual({(r['input']['score']['meter']['numerator'],
                               r['input']['score']['meter']['denominator']) for r in selected},
                             {(3,4), (4,4), (5,8), (7,8), (8,16)})

    def test_mutations_fail_closed(self):
        for mutate in (
            lambda r: r[0].update(split='test' if r[0]['split'] != 'test' else 'train'),
            lambda r: r[0]['target'].update(attacks=999),
            lambda r: r[0]['input'].update(target=r[0]['target']),
            lambda r: r[0]['lineage'].update(rest_mask=True),
            lambda r: r[0].update(family='rest-mask-7'),
            lambda r: r.__setitem__(1, deepcopy(r[0])),
            lambda r: r[0]['input']['score'].update(tempo_qpm=91),
        ):
            rows = generate()
            mutate(rows)
            with self.assertRaises(ValueError):
                validate_rows(rows)

    def test_separate_input_views(self):
        files = artifacts(ROOT)
        self.assertEqual(len(files), 8)
        for name, data in files.items():
            if name.endswith('.inputs.jsonl'):
                for line in data.splitlines():
                    self.assertEqual(set(json.loads(line)), {'task', 'score'})
                for forbidden in (b'"target"', b'"family"', b'"split"', b'"lineage"', b'"attacks"'):
                    self.assertNotIn(forbidden, data)

    def test_source_bound_read_only_replay_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'corpus'
            commitment = write_bundle(ROOT, destination)
            before = {p.name: p.read_bytes() for p in destination.iterdir()}
            self.assertTrue(verify_bundle(ROOT, destination, commitment)['verified'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in destination.iterdir()})
            with self.assertRaises(FileExistsError):
                write_bundle(ROOT, destination)
            with self.assertRaises(ValueError):
                verify_bundle(ROOT, destination, '0' * 64)
            path = destination / 'train.inputs.jsonl'
            path.write_bytes(path.read_bytes().replace(b'120', b'121', 1))
            with self.assertRaises(ValueError):
                verify_bundle(ROOT, destination, commitment)
            path.write_bytes(before[path.name])
            (destination / 'sound.wav').write_bytes(b'not symbolic truth')
            with self.assertRaises(ValueError):
                verify_bundle(ROOT, destination, commitment)

    def test_source_change_rejected(self):
        from symbolic_music_v1.corpus import SOURCES
        with tempfile.TemporaryDirectory() as temporary:
            other = Path(temporary) / 'source'
            for name in SOURCES:
                target = other / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            destination = Path(temporary) / 'corpus'
            commitment = write_bundle(ROOT, destination)
            with self.assertRaises(ValueError):
                artifacts(other)
            with self.assertRaises(ValueError):
                write_bundle(other, Path(temporary) / "false-source")
            (other / SOURCES[0]).write_text('# changed source\n')
            with self.assertRaises(ValueError):
                verify_bundle(other, destination, commitment)

    def test_optional_render_not_imported_by_symbolic_path(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys; from symbolic_music_v1.corpus import generate; generate(); '
            'assert not any(n in sys.modules for n in '
            '["torch", "numpy", "symbolic_music_v1.render", "trinite.geometry", "trinite.training"])'],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        allowed = {'__future__', 'dataclasses', 'fractions', 'hashlib', 'json', 'pathlib', 'math', 'core', 'oracle'}
        for name in ('core.py', 'corpus.py', 'oracle.py'):
            tree = ast.parse((ROOT / 'scripts/symbolic_music_v1' / name).read_text())
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(a.name.split('.')[0] for a in node.names)
                if isinstance(node, ast.ImportFrom):
                    imports.add(node.module.split('.')[0])
            self.assertLessEqual(imports, allowed)


if __name__ == '__main__':
    unittest.main()
