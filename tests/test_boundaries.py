import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "contracts": set(), "tokenizer": {"contracts"},
    "data": {"contracts", "tokenizer"},
    "cli": {"contracts", "tokenizer", "data", "model", "inspection", "training", "experiment", "observation", "geometry_run", "comparison", "foundations"},
    "quantizer": {"contracts"}, "model": {"contracts", "quantizer"},
    "inspection": {"contracts", "model", "quantizer"},
    "training": {"contracts", "data", "inspection", "model", "tokenizer"},
    "checkpoint": {"contracts", "inspection", "training"},
    "observation": {"contracts"},
    "experiment": {"contracts", "data", "checkpoint", "observation", "training", "inspection"},
    "capture": {"contracts", "inspection", "model", "tokenizer"},
    "geometry": {"contracts", "_geo_reference"}, "_geo_reference": set(),
    "geometry_run": {"contracts", "data", "geometry", "observation", "training", "checkpoint", "capture", "inspection"},
    "qec_source": {"contracts"},
    "comparison_data": {"contracts", "qec_source", "tokenizer"},
    "comparison": {"contracts", "comparison_training", "model", "quantizer",
                   "tokenizer", "inspection", "training", "geometry", "observation", "comparison_checkpoint"},
    "comparison_training": {"contracts", "data", "comparison_data", "qec_source", "model", "tokenizer", "training"},
    "comparison_checkpoint": {"contracts", "training", "comparison_training"},
    "foundations_oracle": {"contracts"},
    "foundations_data": {"contracts", "tokenizer", "foundations_oracle"},
    "foundations_plan": {"contracts"},
    "foundations_training": {"contracts", "foundations_data", "foundations_plan", "foundations_oracle",
                             "comparison_training", "model", "tokenizer", "training"},
    "foundations_checkpoint": {"contracts", "comparison_checkpoint", "foundations_training", "foundations_data", "foundations_plan", "training"},
    "foundations": {"contracts", "foundations_data", "foundations_training", "foundations_plan",
                    "foundations_checkpoint", "comparison", "training", "tokenizer", "observation"},
    "convergence": {"contracts", "foundations_training", "foundations_plan", "foundations_data",
                    "foundations_checkpoint", "foundations", "training", "observation"},
    "game_oracle": {"contracts"},
    "game_data": {"contracts", "game_oracle", "tokenizer"},
    "budget_plan": {"contracts", "foundations_plan"},
    "budget_checkpoint": {"contracts", "comparison_checkpoint", "training", "budget_plan", "budget_training", "foundations_data"},
    "budget_training": {"contracts", "foundations_training", "budget_plan", "foundations_data", "foundations", "training"},
    "budget": {"contracts", "budget_training", "foundations_data",
               "budget_checkpoint", "convergence", "foundations", "training", "observation"},
    "__main__": {"cli"}, "__init__": set(),
}


class BoundaryTests(unittest.TestCase):
    def test_no_accelerator_network_or_undeclared_dependency(self):
        for path in (ROOT/"src/trinite").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        for name in ([node.module] if node.module else [a.name for a in node.names]):
                            self.assertIn(name, ALLOWED[path.stem], str(path))
                        continue
                    imports = [node.module.split(".")[0]]
                elif isinstance(node, ast.Import):
                    imports = [a.name.split(".")[0] for a in node.names]
                else:
                    continue
                for name in imports:
                    optional = ({"torch"} if path.stem in {
                        "model", "quantizer", "inspection", "cli", "training", "checkpoint", "capture", "geometry_run", "comparison", "comparison_training", "comparison_checkpoint",
                        "foundations", "foundations_checkpoint"} else set())
                    if path.stem == "checkpoint": optional |= {"numpy", "safetensors"}
                    if path.stem == "comparison_checkpoint": optional |= {"safetensors"}
                    if path.stem == "foundations_checkpoint": optional |= {"safetensors"}
                    if path.stem == "cli": optional |= {"numpy"}
                    if path.stem == "geometry_run": optional |= {"numpy"}
                    if path.stem == "observation": optional |= {"provenance_core", "provenance_verify"}
                    self.assertIn(name, sys.stdlib_module_names | optional)
                    self.assertNotIn(name, {"socket", "urllib", "http", "ssl", "subprocess"})

    def test_foundation_imports_without_optional_site_packages(self):
        import os
        import subprocess
        environment = dict(os.environ, PYTHONPATH=str(ROOT/"src"))
        result = subprocess.run([sys.executable, "-S", "-c",
                                 "import sys; import trinite.cli; "
                                 "assert 'torch' not in sys.modules"],
                                env=environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-S", "-m", "trinite", "inspect-model"],
                                env=environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("requires the CPU dependencies", result.stderr)


if __name__ == "__main__":
    unittest.main()
