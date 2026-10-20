"""Independent adversarial tests for the isolated pretrained reference boundary."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('ncm_review_core', REPO/'scripts/ncm_reference_v1/core.py')
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class SudokuReview(unittest.TestCase):
    def setUp(self):
        # Independent conventional 3x3-block solution, not the production formula.
        self.grid = [int(x) for x in '534678912672195348198342567859761423426853791713924856961537284287419635345286179']
        self.clues = self.grid.copy()
        self.clues[0] = 0

    def test_all_units_required(self):
        self.assertTrue(core.grid_valid(self.grid))
        latin = [(r+c)%9+1 for r in range(9) for c in range(9)]
        self.assertFalse(core.grid_valid(latin))  # Rows/columns valid, boxes invalid.
        for cells in [self.grid[:-1], self.grid+[1], [True]+self.grid[1:], [0]+self.grid[1:], [10]+self.grid[1:]]:
            self.assertFalse(core.grid_valid(cells))

    def test_exact_token_shape_and_separator_positions(self):
        tokens = core.encode_board(self.grid)
        self.assertEqual(core.decode_board(tokens), self.grid)
        malformed = [tokens[:-1], tokens+[1], [True]+tokens[1:], [0]+tokens[1:]]
        bad = tokens.copy(); bad[9] = 1; malformed.append(bad)
        bad = tokens.copy(); bad[0] = 10; malformed.append(bad)
        bad = tokens.copy(); bad[0] = 11; malformed.append(bad)
        for candidate in malformed:
            with self.subTest(candidate=candidate[:10]):
                with self.assertRaises(ValueError):
                    core.decode_board(candidate)
                result = core.score(candidate, self.clues, self.grid)
                self.assertFalse(result['solved'])
                self.assertFalse(result['structurally_valid'])
                self.assertEqual(result['matching_cells'], 0)

    def test_valid_grid_does_not_imply_clue_solution(self):
        alternative = [x%9+1 for x in self.grid]
        self.assertTrue(core.grid_valid(alternative))
        result = core.score(core.encode_board(alternative), self.clues, self.grid)
        self.assertTrue(result['valid_grid'])
        self.assertFalse(result['clues_preserved'])
        self.assertFalse(result['solved'])
        # With no clues, a valid alternative is a solution but not reference exact.
        result = core.score(core.encode_board(alternative), [0]*81, self.grid)
        self.assertTrue(result['solved'])
        self.assertFalse(result['target_agreement'])

    def test_invalid_reference_is_not_scored(self):
        with self.assertRaises(ValueError):
            core.score(core.encode_board(self.grid), [1]+self.clues[1:], self.grid)
        with self.assertRaises(ValueError):
            core.score(core.encode_board(self.grid), self.clues, [1]*81)
        with self.assertRaises(ValueError):
            core.score(core.encode_board(self.grid), [False]+self.clues[1:], self.grid)

    def test_formal_inputs_have_independently_unique_completions(self):
        rows = core.inputs()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r['id'] for r in rows}), 4)
        self.assertEqual({r['family'] for r in rows}, {'cyclic-latin-3x3-v1'})
        self.assertEqual(len({tuple(r['clues']) for r in rows}), 4)
        for row in rows:
            self.assertTrue(core.grid_valid(row['target']))
            self.assertEqual(row['clues'].count(0), 9)
            completed = row['clues'].copy()
            for r in range(9):
                segment = completed[9*r:9*r+9]
                self.assertEqual(segment.count(0), 1)
                completed[9*r+segment.index(0)] = 45-sum(segment)
            self.assertEqual(completed, row['target'])
            self.assertTrue(core.grid_valid(completed))
            self.assertEqual(len([11]+core.encode_board(row['clues'])+[11]), 91)


class FilesystemReview(unittest.TestCase):
    def test_publish_does_not_clobber_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'evidence.json'
            core.publish(output, b'original')
            with self.assertRaises(FileExistsError): core.publish(output, b'replacement')
            self.assertEqual(output.read_bytes(), b'original')
            link = root/'alias'; link.symlink_to(output)
            with self.assertRaises(FileExistsError): core.publish(link, b'replacement')
            self.assertEqual(output.read_bytes(), b'original')
            dangling = root/'dangling'; dangling.symlink_to(root/'absent')
            with self.assertRaises(FileExistsError): core.publish(dangling, b'replacement')
            self.assertFalse((root/'absent').exists())
            self.assertEqual({p.name for p in root.iterdir()}, {'evidence.json','alias','dangling'})

    def test_reader_rejects_symlinks_and_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root/'target'; target.write_bytes(b'ab')
            alias = root/'alias'; alias.symlink_to(target)
            for path, limit in [(alias,10),(target,1),(root,10),(root/'missing',10)]:
                with self.assertRaises(ValueError): core.read_bytes(path,limit)

    def test_upstream_closure_rejects_changed_extra_missing_sources(self):
        core.verify_upstream()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'reference'
            shutil.copytree(core.ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            with patch.object(core,'ROOT',root):
                core.verify_upstream()
                victim = root/'vendor/models/common.py'; old = victim.read_bytes()
                victim.write_bytes(old+b'\n# changed\n')
                with self.assertRaises(ValueError): core.verify_upstream()
                victim.write_bytes(old)
                extra = root/'vendor/extra.py'; extra.write_text('')
                with self.assertRaises(ValueError): core.verify_upstream()
                extra.unlink(); victim.unlink()
                with self.assertRaises(ValueError): core.verify_upstream()

    def test_upstream_metadata_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'reference'
            shutil.copytree(core.ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            with patch.object(core,'ROOT',root):
                for name in ['config.yaml','checkpoint-card.md']:
                    path=root/name; old=path.read_bytes(); path.write_bytes(old+b'\n')
                    with self.assertRaises(ValueError): core.verify_upstream()
                    path.write_bytes(old)


if __name__ == '__main__':
    unittest.main()
