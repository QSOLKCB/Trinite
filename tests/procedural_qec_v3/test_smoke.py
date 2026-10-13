"""Actual CPU smoke for source-bound v3, not a research learning claim."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT/"scripts/run_procedural_qec_v3.py"
ENV = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONDONTWRITEBYTECODE="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


class V3NativeSmoke(unittest.TestCase):
    def call(self, *args):
        p = subprocess.run([sys.executable, str(RUNNER), *map(str, args)],
                           cwd=ROOT, env=ENV, text=True, capture_output=True, timeout=900)
        self.assertEqual(p.returncode, 0, p.stdout[-2500:]+"\n"+p.stderr[-2500:])
        return json.loads(p.stdout)

    def test_four_native_cells_evidence_diagnostics_and_complete_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            request, verifier, run = root/"request.json",root/"verifier.json",root/"run"
            frozen=self.call("--freeze-request", request, "--run-profile", "smoke")
            self.assertEqual(frozen["planned_cells"],4)
            measured=self.call("--run","--request",request,
                               "--request-identity",frozen["request_identity"],
                               "--verification-request",verifier,"--output",run)
            self.assertEqual(measured["outcome"],"completed")
            self.assertFalse(measured["phase5_ready"])
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
            checked=self.call("--verify-only","--request",request,
                              "--request-identity",frozen["request_identity"],
                              "--verification-request",verifier,
                              "--verification-request-identity",measured["verification_request_identity"],
                              "--output",run)
            self.assertEqual(checked["replayed_cells"],4)
            self.assertFalse(checked["learning_adequacy"])
            self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes()
                                     for p in root.rglob("*") if p.is_file()})
            source=json.loads(request.read_bytes())["source"]
            for dep in ("src/trinite/foundations_data.py","src/trinite/inspection.py",
                        "src/trinite/observation.py","src/provenance_core/__init__.py"):
                self.assertIn(dep,source)
            record=json.loads((run/"summary.json").read_bytes())
            for cell in record["cells"]:
                self.assertEqual(cell["outcome"],"completed")
                self.assertTrue(all(cell["checks"].values()))
                report=cell["report"]
                self.assertIn("mixed_all_modes_loss_hex",report)
                self.assertIn("final_only_loss_hex",report)
                self.assertIn("loss_by_mode_and_task",report)
                self.assertIn("auxiliary_visited_unvisited",report)
                self.assertIn("schedule_coverage",report)
                self.assertNotIn("final_train_loss_hex",report)
                self.assertTrue(all(t in report["scores"]["final"]["tasks"] for t in
                                    ("add","subtract","multiply","divide","fold-sum","gf3")))
                self.assertFalse(report["held_out_scored"])


if __name__=="__main__":
    unittest.main()
