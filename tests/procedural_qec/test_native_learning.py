"""Locked CPU integration: actual native updates, complete 4-cell smoke and replay."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_procedural_qec.py"
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


class NativeProceduralSmoke(unittest.TestCase):
    def command(self, *args, expected=0):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
            env=ENV, text=True, capture_output=True, timeout=900)
        self.assertEqual(completed.returncode, expected,
                         completed.stdout[-3000:] + "\n" + completed.stderr[-3000:])
        return json.loads(completed.stdout)

    def test_experiment_both_arms_two_seeds_and_read_only_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, verification, output = (root / "request.json",
                                             root / "verification.json",
                                             root / "results")
            frozen = self.command("--freeze-request", request, "--run-profile", "smoke")
            self.assertEqual(frozen["planned_cells"], 4)
            measured = self.command("--run", "--request", request,
                "--request-identity", frozen["request_identity"],
                "--verification-request", verification, "--output", output)
            self.assertEqual(measured["outcome"], "completed")
            self.assertFalse(measured["training_gate_pass"])
            self.assertFalse(measured["phase5_ready"])
            before = {str(p.relative_to(root)): p.read_bytes()
                      for p in root.rglob("*") if p.is_file()}
            replay = self.command("--verify-only", "--request", request,
                "--request-identity", frozen["request_identity"],
                "--verification-request", verification,
                "--verification-request-identity", measured["verification_request_identity"],
                "--output", output)
            self.assertEqual(replay["replayed_cells"], 4)
            self.assertFalse(replay["learning_adequacy"])
            after = {str(p.relative_to(root)): p.read_bytes()
                     for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after, "verify-only modified retained evidence")
            summary = json.loads((output / "summary.json").read_text())
            for cell in summary["cells"]:
                self.assertTrue(cell["checks"]["complete_optimizer_replay"])
                self.assertTrue(cell["report"]["scores"]["final"]["tasks"])
                self.assertTrue(cell["report"]["scores"]["prefix"]["tasks"])
                self.assertTrue(cell["report"]["scores"]["trace"]["tasks"])
                self.assertFalse(cell["report"]["held_out_scored"])
            # External verification-request identity is not optional or inferred.
            rejected = subprocess.run([sys.executable, str(SCRIPT), "--verify-only",
                "--request", str(request), "--request-identity", frozen["request_identity"],
                "--verification-request", str(verification),
                "--verification-request-identity", "sha256:" + "0" * 64,
                "--output", str(output)], cwd=ROOT, env=ENV,
                capture_output=True, text=True, timeout=60)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
