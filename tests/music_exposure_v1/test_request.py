"""Source/import isolation contracts, with no target backend import or execution."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from trinite.contracts import ContractError, identity
from music_exposure_v1 import request


class RequestTests(unittest.TestCase):
    def test_only_frozen_seed_cells(self):
        self.assertEqual(request.cell_name(17), 'train-17')
        self.assertEqual(request.cell_name(29), 'train-29')
        for seed in (True, 0, 18, '17'):
            with self.assertRaises(ContractError):
                request.cell_name(seed)
        with self.assertRaises(TypeError):
            request.cell_name(17, 'heldout')

    def test_closure_binds_new_and_reused_authoritative_sources(self):
        sources = request.sources()
        for path in ('scripts/music_exposure_v1/spec.py', 'scripts/music_exposure_v1/request.py',
            'scripts/music_exposure_v1/workflow.py', 'scripts/music_exposure_v1/numerical.py',
            'scripts/run_music_exposure.py', 'docs/MUSIC-EXPOSURE.md',
            'scripts/music_contrast_v1/spec.py', 'scripts/music_contrast_v1/reporting.py',
            'src/trinite/training.py', 'src/trinite/comparison_checkpoint.py', 'src/trinite/foundations.py',
            'scripts/native_diagnostic_v1/artifacts.py', 'scripts/unseen_evaluation_v1/packet.py'):
            self.assertEqual(sources[path], identity((request.ROOT/path).read_bytes()))

    def test_foreign_and_unbound_package_imports_rejected(self):
        for name, path in (('music_exposure_v1.foreign', '/tmp/foreign.py'),
                           ('music_contrast_v1.numerical', str(request.ROOT/'scripts/music_contrast_v1/numerical.py'))):
            module = types.ModuleType(name); module.__file__ = path
            with patch.dict(sys.modules, {name: module}):
                with self.assertRaisesRegex(ContractError, 'foreign or unbound'):
                    request.sources()

    def test_cache_link_and_extension_shadow_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts/music_exposure_v1').mkdir(parents=True)
            (root/'src').mkdir()
            target = root/'outside'; target.write_text('untrusted')
            for name in ('__pycache__', 'shadow.pyc', 'shadow.pyo', 'shadow.so', 'shadow.pyd', 'linked.py'):
                path = root/'src'/name
                if name == '__pycache__':
                    path.mkdir()
                elif name == 'linked.py':
                    path.symlink_to(target)
                else:
                    path.write_bytes(b'untrusted')
                with patch.object(request, 'ROOT', root), patch.object(request, 'upstream_sources', return_value={}), \
                        patch.object(request.sys, 'modules', {}):
                    with self.assertRaisesRegex(ContractError, 'caches, links or extension shadows'):
                        request.sources()
                path.rmdir() if path.is_dir() and not path.is_symlink() else path.unlink()

    def test_size_identity_and_changed_request_rejected(self):
        for raw, expected in ((b'{}\n', identity(b'other')), ('not bytes', identity(b'other')),
                              (b'x'*(1024*1024+1), identity(b'x'*(1024*1024+1)))):
            with patch.object(request, 'core') as core:
                with self.assertRaises(ContractError):
                    request.validate(raw, expected)
                core.assert_not_called()
        with patch.object(request, 'core', return_value={'new': True}):
            with self.assertRaisesRegex(ContractError, 'source/environment'):
                request.validate(b'{}\n', identity(b'{}\n'))

    def test_imports_do_not_load_backend(self):
        code = ('import sys; import music_exposure_v1.spec, music_exposure_v1.request, '
                'music_exposure_v1.workflow, music_exposure_v1.reporting; '
                'raise SystemExit(int("torch" in sys.modules))')
        env = dict(os.environ, PYTHONPATH=str(request.ROOT/'src')+os.pathsep+str(request.ROOT/'scripts'))
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=request.ROOT,
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
