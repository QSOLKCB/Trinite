"""Actual isolated CLI and safe filesystem rejection; no target checkpoints."""
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from frozen_music_eval_v1 import artifacts as a
from frozen_music_eval_v1 import workflow as w

class Boundaries(unittest.TestCase):
    def test_cli_has_no_training_resume_selection_or_expansion(self):
        launcher = a.ROOT / 'scripts/run_frozen_music_eval.py'
        for arguments in (['train'], ['resume'], ['run','--seed','17'],
                          ['run','--checkpoint','arbitrary'], ['run','--steps','641'],
                          ['run','--generation-cap','65'], ['run','--authorize-run']):
            result = subprocess.run([sys.executable,'-I','-B',str(launcher),*arguments],
                capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0, arguments)
            self.assertIn(b'error:', result.stderr)

    def test_launcher_rejects_missing_flags_before_imports_or_cache_writes(self):
        # A disposable copy also makes the pre-fix failure safe to reproduce:
        # the old missing-B entrypoint writes caches only inside this directory.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / 'checkout'; checkout.mkdir()
            for name in ('src', 'scripts'):
                shutil.copytree(a.ROOT / name, checkout / name,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
            ambient = root / 'ambient'; ambient.mkdir()
            sentinel = ambient / 'sentinel-executed'
            (ambient / 'pathlib.py').write_text(
                "open(" + repr(str(sentinel)) + ", 'w').write('benign sentinel')\n"
                "raise RuntimeError('BENIGN_AMBIENT_PATHLIB_EXECUTED')\n")
            env = dict(os.environ, PYTHONPATH=str(ambient))
            env.pop('PYTHONDONTWRITEBYTECODE', None)
            launcher = checkout / 'scripts/run_frozen_music_eval.py'
            for flags in (('-I',), ('-B',), ()):
                result = subprocess.run([sys.executable, *flags, str(launcher), 'preflight'],
                    env=env, cwd=root, capture_output=True, timeout=30)
                with self.subTest(flags=flags):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'Use python -I -B', result.stderr)
                    self.assertNotIn(b'BENIGN_AMBIENT_PATHLIB_EXECUTED', result.stderr)
                    self.assertFalse(sentinel.exists())
                    self.assertFalse(list(checkout.rglob('__pycache__')))
            result = subprocess.run([sys.executable, '-I', '-B', str(launcher), 'preflight'],
                env=env, cwd=root, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertFalse(sentinel.exists())
            self.assertFalse(list(checkout.rglob('__pycache__')))

    def test_safe_reads_no_links_noncanonical_json_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/'original'; original.write_bytes(b'unchanged')
            link = root/'link'; link.symlink_to(original)
            with self.assertRaises(ValueError): a.read(link)
            with self.assertRaises(ValueError): a.read(original, 2)
            with self.assertRaises(FileExistsError): a.publish(original, b'replace')
            self.assertEqual(original.read_bytes(), b'unchanged')
            for raw in (b'{"a":1,"a":2}\n', b'{ "a":1}\n', b'{"a":NaN}\n'):
                with self.assertRaises(ValueError): a.parse(raw)
            with self.assertRaises(ValueError): a.disjoint(root/'output', root)

    def test_completion_requires_full_frozen_budget_and_replay(self):
        rows = [{'generated_tokens':[55,257]} for _ in range(320)]
        value = dict(decodes=320,forward_calls=640,model_identities_unchanged=True,
                     original_inventory_unchanged=True,training_performed=False,
                     train_replay_decodes=80,heldout_decodes=240,
                     replay_gate_identity=a.digest(a.encoded(w.replay_gate())))
        w.validate_completion(value, rows)
        for key, bad in [('decodes',240),('training_performed',True),
                         ('train_replay_decodes',False),('heldout_decodes',120),
                         ('forward_calls',20481),('replay_gate_identity',a.digest(b'fake'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.validate_completion({**value,key:bad}, rows)

if __name__ == '__main__': unittest.main()
