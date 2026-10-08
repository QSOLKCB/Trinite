import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "contracts": set(), "tokenizer": {"contracts"},
    "data": {"contracts", "tokenizer"},
    "cli": {"contracts", "tokenizer", "data"},
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
                    self.assertIn(name, sys.stdlib_module_names)
                    self.assertNotIn(name, {"socket", "urllib", "http", "ssl", "subprocess"})


if __name__ == "__main__":
    unittest.main()
