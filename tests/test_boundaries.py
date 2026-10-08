import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "contracts": set(), "tokenizer": {"contracts"},
    "data": {"contracts", "tokenizer"},
    "cli": {"contracts", "tokenizer", "data", "model", "inspection"},
    "quantizer": {"contracts"}, "model": {"contracts", "quantizer"},
    "inspection": {"contracts", "model", "quantizer"},
    "__main__": {"cli"}, "__init__": set(),
}


class BoundaryTests(unittest.TestCase):
    def test_no_accelerator_network_or_undeclared_dependency(self):
        for path in (ROOT/"src/trinite").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        self.assertIn(node.module, ALLOWED[path.stem], str(path))
                        continue
                    imports = [node.module.split(".")[0]]
                elif isinstance(node, ast.Import):
                    imports = [a.name.split(".")[0] for a in node.names]
                else:
                    continue
                for name in imports:
                    self.assertIn(name, sys.stdlib_module_names | (
                        {"torch"} if path.stem in {"model", "quantizer", "inspection", "cli"}
                        else set()))
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
