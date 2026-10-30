"""Read-only Git/worktree admission before project or native imports."""
import hashlib
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 32 * 1024 * 1024


def path(value):
    result = Path(os.path.abspath(value))
    for parent in result.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError('non-link directory ancestors required')
    return result


def read(value, limit=LIMIT):
    value = path(value)
    before = value.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError('bounded ordinary source/artifact required')
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)
    fd = os.open(value, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if signature(os.fstat(fd)) != signature(before):
            raise ValueError('source/artifact replaced at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        if (len(raw) > limit or signature(os.fstat(fd)) != signature(before)
                or signature(value.lstat()) != signature(before)):
            raise ValueError('source/artifact changed during read')
        return raw
    finally:
        os.close(fd)


def git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.check_output(['git', '--no-replace-objects', '--work-tree=' + str(root),
        '-c', 'core.fsmonitor=false', '-C', str(root), *args], env=env,
        stderr=subprocess.PIPE, timeout=30).decode().strip()


def check(root=ROOT):
    root = path(root)
    if not stat.S_ISDIR(root.lstat().st_mode) or git(root, 'rev-parse', '--show-toplevel') != str(root):
        raise ValueError('exact non-link effective source worktree required')
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('clean committed study checkout required')
    names = ('src', 'scripts', 'configs', 'AGENTS.md', 'requirements.lock', 'requirements.music-device.lock',
             'docs/MUSIC-DEVICE-TRIAL.md', 'docs/MUSIC-ROBUSTNESS-V2.md',
             'docs/MUSIC-PLACEMENT.md')
    files = {}
    for line in git(root, 'ls-tree', '-r', '--full-tree', 'HEAD', '--', *names).splitlines():
        metadata, name = line.split('\t', 1)
        mode, kind, expected = metadata.split()
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError('ordinary tracked study source required')
        raw = read(root / name)
        if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != expected:
            raise ValueError('actual study source differs from committed blob')
        files[name] = 'sha256:' + hashlib.sha256(raw).hexdigest()
    if not {'docs/MUSIC-DEVICE-TRIAL.md', 'docs/MUSIC-ROBUSTNESS-V2.md','docs/MUSIC-PLACEMENT.md','requirements.music-device.lock'} <= set(files):
        raise ValueError('committed prospective contracts and GPU lock required')
    for base in ('src', 'scripts'):
        if git(root, 'ls-files', '--others', '--ignored', '--exclude-standard', '--', base):
            raise ValueError('ignored runtime files forbidden')
        for current, directories, members in os.walk(root / base, followlinks=False):
            for name in directories:
                p = Path(current) / name
                if name == '__pycache__' or not stat.S_ISDIR(p.lstat().st_mode):
                    raise ValueError('runtime caches or linked directories forbidden')
            for name in members:
                p = Path(current) / name
                if (p.relative_to(root).as_posix() not in files or p.is_symlink()
                        or p.suffix in ('.pyc', '.pyo', '.so', '.pyd')):
                    raise ValueError('untracked runtime/cache/shadow forbidden')
    return {'commit': git(root, 'rev-parse', 'HEAD'), 'tree': git(root, 'rev-parse', 'HEAD^{tree}'),
            'files': dict(sorted(files.items()))}
