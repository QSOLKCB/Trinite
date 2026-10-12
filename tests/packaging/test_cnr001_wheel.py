"""Exercise the real wheel boundary without importing the source checkout."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class CNR001Wheel(unittest.TestCase):
    def test_installed_protocol_checks_installed_baseline_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            project.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(ROOT / name, project / name)
            shutil.copytree(ROOT / "src", project / "src",
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            def run(*args):
                result = subprocess.run(args, cwd=root, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout)
                return result.stdout

            # Build and install a wheel, never an editable source tree or an
            # unpacked imitation. Build dependencies are supplied by the CI lane.
            run(sys.executable, "-m", "pip", "wheel", "--no-build-isolation",
                "--no-deps", "--no-index", "--wheel-dir", str(root / "wheels"), str(project))
            wheels = list((root / "wheels").glob("trinite-*.whl"))
            self.assertEqual(len(wheels), 1)
            installed = root / "installed"
            run(sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                "--no-compile", "--target", str(installed), str(wheels[0]))
            # Remove even the disposable checkout before testing isolated imports.
            shutil.rmtree(project)
            code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from trinite.cnr001 import core
from trinite.contracts import ContractError
installed = Path(sys.argv[1]).resolve()
assert Path(core.__file__).resolve().is_relative_to(installed)
assert not (core.ROOT / "src/trinite").exists()
if sys.argv[2] == "valid":
    protocol, digest = core.protocol()
    assert protocol["schema"] == "trinite.cnr-001-protocol.v4"
    assert len(digest) == 71
else:
    try:
        core.protocol()
    except ContractError as error:
        assert "v0.1.0 reference source changed: src/trinite/contracts.py" in str(error)
    else:
        raise AssertionError("tampered installed baseline was accepted")
'''
            run(sys.executable, "-I", "-c", code, str(installed), "valid")
            pinned = installed / "trinite/contracts.py"
            pinned.write_bytes(pinned.read_bytes() + b"\n# installed-source tamper\n")
            run(sys.executable, "-I", "-c", code, str(installed), "tampered")


if __name__ == "__main__":
    unittest.main()
