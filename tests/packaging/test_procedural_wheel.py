"""Real installed-wheel QEC resource test; never import the build checkout.

The numerical research runner remains checkout-only; this validates installed
standard-library data, its exact protocol resources and source-bound admission.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = (
    "procedural_qec/protocol.json",
    "procedural_qec/protocol-v2.json",
    "procedural_qec_v3/protocol.json",
)
FROZEN_V3_SHA256 = "be5e998e5e79d52ecae40bf679c8b0a686c70577c78ac4ff0af76b49d7f27b18"


class ProceduralInstalledWheel(unittest.TestCase):
    def test_real_wheel_exact_resources_and_installed_audit(self):
        with tempfile.TemporaryDirectory(prefix="trinite-procedural-wheel-") as temp:
            root = Path(temp)
            project = root / "build-project"
            project.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(ROOT / name, project / name)
            shutil.copytree(
                ROOT / "src", project / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.dist-info"))
            expected = {
                name: hashlib.sha256((ROOT / "src/trinite" / name).read_bytes()).hexdigest()
                for name in RESOURCES
            }
            self.assertEqual(expected["procedural_qec_v3/protocol.json"],
                             FROZEN_V3_SHA256)

            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            env.pop("PYTHONPATH", None)

            def run(*args):
                result = subprocess.run(
                    args, cwd=root, env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
                return result.stdout

            # A non-editable wheel, built with the CI-installed pinned backend.
            run(sys.executable, "-m", "pip", "wheel", "--no-build-isolation",
                "--no-deps", "--no-index", "--wheel-dir", str(root / "wheels"),
                str(project))
            wheels = list((root / "wheels").glob("trinite-*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as wheel:
                names = wheel.namelist()
                for rel, digest in expected.items():
                    member = "trinite/" + rel
                    self.assertEqual(names.count(member), 1, member)
                    self.assertEqual(hashlib.sha256(wheel.read(member)).hexdigest(),
                                     digest, member)

            installed = root / "installed"
            run(sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                "--no-compile", "--target", str(installed), str(wheels[0]))
            shutil.rmtree(project)  # No uninstalled source tree is available to imports.
            self.assertFalse(project.exists())
            # Independently regenerate the authoritative checkout evidence.
            checkout = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from trinite.procedural_qec import data
raw, manifest = data.dataset()
data.audit_bytes(raw, manifest)
assert all(name.startswith("src/trinite/") for name in data.source_receipt())
Path(sys.argv[2]).write_bytes(raw)
Path(sys.argv[3]).write_bytes(manifest)
'''
            expected_data = root / "checkout-data.json"
            expected_admission = root / "checkout-admission.json"
            run(sys.executable, "-I", "-S", "-c", checkout, str(ROOT / "src"),
                str(expected_data), str(expected_admission))
            self.assertGreater(expected_data.stat().st_size, 0)
            self.assertGreater(expected_admission.stat().st_size, 0)

            program = r'''
import hashlib
import json
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
expected = json.loads(sys.argv[3])
checkout_raw = Path(sys.argv[4]).read_bytes()
checkout_manifest = Path(sys.argv[5]).read_bytes()
sys.path.insert(0, str(target))
from trinite.contracts import ContractError
from trinite.procedural_qec import data
from trinite.procedural_qec_v3 import scheduler
assert Path(data.__file__).resolve().is_relative_to(target)
assert Path(scheduler.__file__).resolve().is_relative_to(target)
assert not (target.parent / "build-project").exists()
resources = target / "trinite"
for rel, digest in expected.items():
    protocol = resources / rel
    assert protocol.is_file(), rel
    actual = hashlib.sha256(protocol.read_bytes()).hexdigest()
    changed = (mode == "tampered-v2" and rel == "procedural_qec/protocol-v2.json") or (
        mode == "tampered-v3" and rel == "procedural_qec_v3/protocol.json")
    if changed:
        assert actual != digest, "tampered protocol unchanged"
    else:
        assert actual == digest, "installed protocol bytes differ: " + rel
if mode == "valid":
    assert len(expected) == 3
    assert all(name.startswith("src/trinite/") for name in data.source_receipt())
    raw, manifest = data.dataset()
    assert raw == checkout_raw, "installed dataset differs from checkout"
    assert manifest == checkout_manifest, "installed admission differs from checkout"
    data.audit_bytes(raw, manifest)
    data.audit_bytes(checkout_raw, checkout_manifest)
    assert scheduler.auxiliary_stream(0) == ("fold-sum", "prefix")
elif mode == "tampered-v2":
    try:
        data.audit_bytes(checkout_raw, checkout_manifest)
    except ContractError:
        pass
    else:
        raise AssertionError("changed installed v2 resource accepted checkout evidence")
else:
    assert mode == "tampered-v3"
'''
            def isolated(mode, target):
                run(sys.executable, "-I", "-S", "-c", program,
                    str(target), mode, json.dumps(expected, sort_keys=True),
                    str(expected_data), str(expected_admission))

            isolated("valid", installed)
            # Actual prefix relocation must preserve *both* complete streams.
            relocated = root / "relocated-wheel"
            installed.rename(relocated)
            self.assertFalse(installed.exists())
            isolated("valid", relocated)
            v2 = relocated / "trinite/procedural_qec/protocol-v2.json"
            original = v2.read_bytes()
            v2.write_bytes(original + b"\n")
            isolated("tampered-v2", relocated)
            v2.write_bytes(original)
            v3 = relocated / "trinite/procedural_qec_v3/protocol.json"
            v3.write_bytes(v3.read_bytes() + b"\n")
            isolated("tampered-v3", relocated)


if __name__ == "__main__":
    unittest.main()
