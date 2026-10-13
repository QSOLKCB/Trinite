"""Real wheel export/import-direction smoke, isolated from build and checkout."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]


class ForwardExportInstalledWheel(unittest.TestCase):
    def test_real_wheel_encode_relocated_and_source_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="trinite-export-wheel-") as temp:
            root = Path(temp)
            project = root / "build-project"
            project.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(ROOT/name, project/name)
            shutil.copytree(ROOT/"src", project/"src",
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.dist-info"))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            env.pop("PYTHONPATH", None)
            def run(program, *args, cwd=root):
                result = subprocess.run([sys.executable, "-I", "-c", program, *map(str, args)],
                                        cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout+"\n"+result.stderr)
                return result.stdout
            # No dependency installation/change: use the existing build backend.
            run("from setuptools.build_meta import build_wheel; build_wheel('dist')", cwd=project)
            wheels = list((project/"dist").glob("trinite-*.whl"))
            self.assertEqual(len(wheels), 1)
            installed = root/"installed"
            with zipfile.ZipFile(wheels[0]) as archive:
                for name in ("__init__.py", "encoder.py", "source.py"):
                    member = "trinite/forward_export/"+name
                    self.assertEqual(archive.namelist().count(member), 1)
                    self.assertEqual(archive.read(member), (ROOT/"src"/member).read_bytes())
                archive.extractall(installed)
            shutil.rmtree(project)
            program = r'''
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from trinite.contracts import ContractError, ModelConfig, parse_json
from trinite.forward_export.source import read_source_snapshot
from trinite.forward_export.encoder import export_forward
import trinite.forward_export.encoder as encoder
from trinite.model import Decoder
assert Path(encoder.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert not (Path(sys.argv[1]).parent/'build-project').exists()
assert not any(n.startswith(('trinite.training','trinite.checkpoint','trinite.comparison_checkpoint')) for n in sys.modules)
if sys.argv[2] == 'changed':
    try:
        read_source_snapshot(expected_identity=sys.argv[3])
    except ContractError:
        pass
    else:
        raise AssertionError('changed source accepted old expected identity')
else:
    source = read_source_snapshot()
    config = ModelConfig(blocks=1,width=3,heads=1,feed_forward_width=5,context_length=8)
    a = export_forward(Decoder(config),source=source,expected_source_identity=source.content_identity)
    m = parse_json(a.manifest, canonical=True)
    assert m['counts']['matrices'] == 6
    assert m['source_identity'] == source.content_identity
    assert not any(n.startswith(('trinite.training','trinite.checkpoint','trinite.comparison_checkpoint')) for n in sys.modules)
    print(source.content_identity)
'''
            original = run(program, installed, "valid").strip()
            relocated = root/"relocated"
            installed.rename(relocated)
            self.assertEqual(run(program, relocated, "valid").strip(), original)
            path = relocated/"trinite/model.py"
            path.write_bytes(path.read_bytes()+b"\n")
            run(program, relocated, "changed", original)


if __name__ == "__main__":
    unittest.main()
