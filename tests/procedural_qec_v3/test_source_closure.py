"""Regression for the actual v3 executable source inventory (native CI lane)."""
from pathlib import Path
from unittest import mock
import tempfile
import unittest

from trinite.contracts import ContractError, identity
from trinite.procedural_qec_v3 import experiment


class ProceduralSourceClosureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.module = self.root / "src/trinite/procedural_qec_v3/experiment.py"
        self.module.parent.mkdir(parents=True)
        self.module.write_bytes(b"# admitted module\n")
        (self.root / "src/trinite/base.py").write_bytes(b"# module\n")
        (self.root / "scripts").mkdir()
        for name in ("requirements.lock", "pyproject.toml",
                     "scripts/run_procedural_qec_v3.py"):
            path = self.root / name
            path.write_bytes(name.encode())
        patched = mock.patch.object(experiment, "__file__", str(self.module))
        patched.start()
        self.addCleanup(patched.stop)

    def test_clean_tree_complete_inventory_and_source_mutation(self):
        expected = {
            "src/trinite/procedural_qec_v3/experiment.py": identity(b"# admitted module\n"),
            "src/trinite/base.py": identity(b"# module\n"),
        }
        for name in ("requirements.lock", "pyproject.toml",
                     "scripts/run_procedural_qec_v3.py"):
            expected[name] = identity(name.encode())
        self.assertEqual(experiment.sources(), expected)
        (self.root / "src/trinite/base.py").write_bytes(b"# changed module\n")
        changed = experiment.sources()
        self.assertNotEqual(changed["src/trinite/base.py"],
                            expected["src/trinite/base.py"])

    def test_symlinked_python_module_cannot_disappear_from_receipt(self):
        source = self.root / "external.py"
        source.write_bytes(b"# apparently original scheduler\n")
        (self.module.parent / "scheduler.py").symlink_to(source)
        with self.assertRaisesRegex(ContractError, "symlinked procedural v3 source file"):
            experiment.sources()

    def test_symlinked_package_directory_cannot_disappear_from_receipt(self):
        outside = self.root / "external"
        outside.mkdir()
        (outside / "child.py").write_bytes(b"# executable source\n")
        (self.root / "src/trinite/hidden").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ContractError, "symlinked procedural v3 source directory"):
            experiment.sources()

    def test_packaging_and_bytecode_generation_do_not_change_receipt(self):
        original = experiment.sources()
        for name in ("src/trinite.egg-info/SOURCES.txt",
                     "src/tool.dist-info/METADATA.txt",
                     "src/trinite/__pycache__/module.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"generated")
        self.assertEqual(original, experiment.sources())

    def test_missing_required_launcher_rejects_instead_of_omitting(self):
        (self.root / "scripts/run_procedural_qec_v3.py").unlink()
        with self.assertRaises(ContractError):
            experiment.sources()


if __name__ == "__main__":
    unittest.main()
