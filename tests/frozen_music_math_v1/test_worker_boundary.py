"""Fresh-process refusal and isolation checks; no target tensors or inference."""
import ast
import py_compile
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'scripts/frozen_music_math_v1/worker.py'


class WorkerBoundary(unittest.TestCase):
    def test_isolated_help_needs_no_native_import(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(WORKER), '--help'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--authorize-inference', result.stdout)

    def test_internal_infer_refuses_missing_authorization_before_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-B', str(WORKER), 'infer',
                '--original-source', directory, '--original-run', directory,
                '--original-request-identity', 'sha256:' + '0'*64,
                '--original-matrix-identity', 'sha256:' + '0'*64],
                capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('separate inference authorization', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unchecked_hash_bytecode_rejected_before_original_import(self):
        from frozen_music_math_v1.worker import reject_import_caches
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir(); (root / 'scripts').mkdir()
            source = root / 'src' / 'synthetic.py'
            source.write_text("VALUE = 'synthetic stale cache'\n")
            cached = Path(py_compile.compile(str(source), doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH))
            self.assertEqual(int.from_bytes(cached.read_bytes()[4:8], 'little'), 1)
            source.write_text("VALUE = 'different source'\n")
            with self.assertRaisesRegex(ValueError, 'caches'):
                reject_import_caches(root)
            # No cached or source module is ever imported/executed by this test.

    def test_extension_and_linked_import_roots_rejected(self):
        from frozen_music_math_v1.worker import reject_import_caches
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir(); (root / 'scripts').mkdir()
            shadow = root / 'scripts' / 'module.so'; shadow.write_bytes(b'synthetic')
            with self.assertRaisesRegex(ValueError, 'shadows'):
                reject_import_caches(root)
            shadow.unlink()
            (root / 'scripts' / 'linked').symlink_to(root / 'src', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'linked directories'):
                reject_import_caches(root)

    def test_worker_has_no_gold_scoring_or_training_calls(self):
        tree = ast.parse(WORKER.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
        attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
        self.assertFalse(names & {'gold_answers', 'score_records', 'run_steps', 'evaluate', 'state_for'})
        self.assertFalse(attributes & {'backward', 'step'})
        greedy = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'greedy']
        self.assertEqual(len(greedy), 1)
        self.assertEqual([ast.unparse(arg) for arg in greedy[0].args], ['model', 'prompt', 'CAP'])
        self.assertEqual(greedy[0].keywords, [])


if __name__ == '__main__':
    unittest.main()
