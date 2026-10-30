"""Source bootstrap rejection before any project/native import."""
from pathlib import Path
import os
import shutil
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
from music_placement_v1 import bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'root'
        self.root.mkdir()
        for name in ('src', 'scripts', 'docs'):
            (self.root / name).mkdir()
        (self.root / 'src/a.py').write_text('x=1\n')
        (self.root / 'scripts/a.py').write_text('x=1\n')
        (self.root / 'docs/MUSIC-PLACEMENT.md').write_text('contract\n')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'source')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.PIPE)

    def test_clean_read_only_bound_source(self):
        before = self.git('status', '--porcelain')
        result = bootstrap.check(self.root)
        self.assertEqual(len(result['files']), 3)
        self.assertEqual(before, self.git('status', '--porcelain'))

    def test_core_worktree_untracked_and_ignored_shadow(self):
        mirror = Path(self.tmp.name) / 'mirror'
        subprocess.check_call(['git', 'clone', '-q', str(self.root), str(mirror)])
        self.git('config', 'core.worktree', str(mirror))
        self.assertEqual(bootstrap.check(self.root)['tree'], bootstrap.check(mirror)['tree'])
        shadow = self.root / 'src/torch.py'
        shadow.write_text('raise RuntimeError("never imported")\n')
        with self.assertRaises(ValueError): bootstrap.check(self.root)
        (self.root / '.git/info/exclude').write_text('src/torch.py\n')
        with self.assertRaises(ValueError): bootstrap.check(self.root)

    def test_actual_bytes_checked_despite_assume_unchanged(self):
        self.git('update-index', '--assume-unchanged', 'src/a.py')
        (self.root / 'src/a.py').write_text('x=2\n')
        with self.assertRaisesRegex(ValueError, 'committed blob'): bootstrap.check(self.root)

    def test_ambient_git_selectors_do_not_redirect(self):
        with mock.patch.dict(os.environ, {'GIT_WORK_TREE': '/missing', 'GIT_DIR': '/missing'}):
            self.assertIn('src/a.py', bootstrap.check(self.root)['files'])

    def test_dirty_package_initializer_not_executed_before_bootstrap(self):
        package = self.root / 'scripts/music_placement_v1'
        package.mkdir()
        (package / '__init__.py').write_text('"""trusted empty init"""\n')
        shutil.copyfile(bootstrap.ROOT / 'scripts/music_placement_v1/bootstrap.py', package / 'bootstrap.py')
        launcher = self.root / 'scripts/run_music_placement.py'
        shutil.copyfile(bootstrap.ROOT / 'scripts/run_music_placement.py', launcher)
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'launcher')
        marker = Path(self.tmp.name) / 'executed'
        (package / '__init__.py').write_text('open(' + repr(str(marker)) + ', "w").write("sentinel")\n')
        result = subprocess.run([sys.executable, '-I', '-B', str(launcher), 'preflight'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('clean committed study checkout required', result.stderr)
        self.assertFalse(marker.exists())

    def test_links_caches_and_bounded_reads(self):
        linked = Path(self.tmp.name) / 'linked'; linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): bootstrap.check(linked)
        cache = self.root / 'src/__pycache__'; cache.mkdir()
        (self.root / '.git/info/exclude').write_text('src/__pycache__/\n')
        with self.assertRaises(ValueError): bootstrap.check(self.root)
        with self.assertRaises(ValueError): bootstrap.read(self.root / 'src/a.py', 1)
        with self.assertRaises(ValueError): bootstrap.read(linked / 'src/a.py')

if __name__ == '__main__': unittest.main()
