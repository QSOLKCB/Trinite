"""Actual isolated CLI and safe filesystem rejection; no target checkpoints."""
import os
from pathlib import Path
import subprocess
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
