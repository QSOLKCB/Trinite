"""Standard-library-only checks for the additive encode-only package boundary."""
import ast
from pathlib import Path
import os
import subprocess
import sys
import unittest

from trinite.contracts import ContractError
from trinite.forward_export import ForwardExport
from trinite.forward_export.source import (MAX_SOURCE_BYTES, MAX_SOURCE_MEMBER_BYTES,
                                          SOURCE_FILES, SourceSnapshot, read_source_snapshot)

ROOT = Path(__file__).resolve().parents[1]


class ForwardExportBoundaryTests(unittest.TestCase):
    def test_package_inventory_and_explicit_dependency_direction(self):
        package = ROOT / "src/trinite/forward_export"
        self.assertEqual({p.name for p in package.glob("*.py")}, {"__init__.py", "source.py", "encoder.py"})
        allowed = {"__init__": {"contracts"}, "source": {"contracts"},
                   "encoder": {"contracts", "inspection", "model", "packed_ternary",
                               "quantizer", "tokenizer", "source"}}
        for path in package.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        if node.module:
                            self.assertIn(node.module, allowed[path.stem])
                        continue
                    names = [node.module.split(".")[0]]
                    if node.module == "safetensors.torch":
                        self.assertEqual([n.name for n in node.names], ["save"])
                elif isinstance(node, ast.Import):
                    names = [n.name.split(".")[0] for n in node.names]
                else:
                    continue
                for name in names:
                    self.assertIn(name, sys.stdlib_module_names | ({"torch", "numpy", "safetensors"} if path.stem == "encoder" else set()))
                    self.assertNotIn(name, {"socket", "urllib", "http", "ssl", "pickle", "subprocess"})
                    if path.stem != "source":
                        self.assertNotIn(name, {"pathlib", "os", "io", "importlib"})
        for name in ("model.py", "quantizer.py", "tokenizer.py", "contracts.py"):
            self.assertNotIn("forward_export", (ROOT/"src/trinite"/name).read_text())

    def test_stdlib_only_import_and_source_capture(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import sys; import trinite.forward_export; "
             "from trinite.forward_export.source import read_source_snapshot; "
             "assert read_source_snapshot().receipt()['members']; "
             "assert not any(n in sys.modules for n in ('torch','numpy','safetensors','trinite.training'))"],
            env=dict(os.environ, PYTHONPATH=str(ROOT/"src")),
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_snapshot_exact_types_bounded_members_and_inventory(self):
        for value in (bytearray(), [], "", None):
            with self.assertRaises(ContractError): ForwardExport(value, b"", b"")
            with self.assertRaises(ContractError): ForwardExport(b"", value, b"")
            with self.assertRaises(ContractError): ForwardExport(b"", b"", value)
        source = read_source_snapshot()
        self.assertEqual(tuple(n for n, raw in source.members), SOURCE_FILES)
        self.assertLessEqual(sum(len(raw) for _, raw in source.members), MAX_SOURCE_BYTES)
        self.assertEqual(read_source_snapshot(expected_identity=source.content_identity), source)
        name, raw = source.members[0]
        invalid = [[], list(source.members), (), source.members[:-1], source.members[::-1],
                   ((name, bytearray(raw)), *source.members[1:]),
                   ((name, b""), *source.members[1:]),
                   ((name, b"x"*(MAX_SOURCE_MEMBER_BYTES+1)), *source.members[1:]),
                   ((name+"/../other", raw), *source.members[1:]),
                   ([name, raw], *source.members[1:]),
                   ((True, raw), *source.members[1:]),
                   tuple((n, b"x"*MAX_SOURCE_MEMBER_BYTES) for n, _ in source.members)]
        class Bytes(bytes):
            pass
        invalid.append(((name, Bytes(raw)), *source.members[1:]))
        for members in invalid:
            with self.subTest(kind=str(type(members))):
                with self.assertRaises(ContractError): SourceSnapshot(members).receipt()


if __name__ == "__main__":
    unittest.main()
