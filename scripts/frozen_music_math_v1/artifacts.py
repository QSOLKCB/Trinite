"""Standard-library bounded, non-link, no-clobber local evidence handling."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MAX_FILE = 32 * 1024 * 1024


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                       allow_nan=False) + '\n').encode()


def digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def identity(value):
    if (type(value) is not str or len(value) != 71 or not value.startswith('sha256:')
            or any(c not in '0123456789abcdef' for c in value[7:])):
        raise ValueError('canonical sha256 identity required')
    return value


def safe_path(path):
    path = Path(os.path.abspath(path))
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError('non-link directory ancestors required')
    return path


def read(path, limit=MAX_FILE):
    path = safe_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError('bounded regular file required')
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if signature(os.fstat(fd)) != signature(before):
            raise ValueError('file changed at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        if (len(raw) > limit or signature(os.fstat(fd)) != signature(before)
                or signature(path.lstat()) != signature(before)):
            raise ValueError('file changed during read')
        return raw
    finally:
        os.close(fd)


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    if encoded(value) != raw:
        raise ValueError('canonical JSON required')
    return value


def publish(path, raw):
    path = safe_path(path)
    if type(raw) is not bytes or len(raw) > MAX_FILE:
        raise ValueError('bounded bytes required')
    fd, staging = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path)  # Atomic visibility, never replace existing evidence.
    finally:
        os.unlink(staging)


def inventory(root, *, max_files=128, max_bytes=128 * 1024 * 1024):
    root = safe_path(root)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError('non-link run directory required')
    result = {}
    total = 0
    directory_count = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directory_count += 1
        if directory_count > 128 or len(Path(current).relative_to(root).parts) > 8:
            raise ValueError('inventory directory bound exceeded')
        for name in directories:
            if not stat.S_ISDIR((Path(current) / name).lstat().st_mode):
                raise ValueError('linked/special directory in inventory')
        for name in files:
            if len(result) >= max_files:
                raise ValueError('inventory file bound exceeded')
            path = Path(current) / name
            raw = read(path)
            total += len(raw)
            if total > max_bytes:
                raise ValueError('inventory byte bound exceeded')
            result[path.relative_to(root).as_posix()] = digest(raw)
    return dict(sorted(result.items()))


def disjoint(path, *protected):
    path = safe_path(path)
    for other in protected:
        other = safe_path(other)
        if path.is_relative_to(other) or other.is_relative_to(path):
            raise ValueError('new destination overlaps protected source/input')
    return path


def sources():
    paths = sorted((ROOT / 'scripts/frozen_music_math_v1').glob('*.py'))
    paths += [ROOT / p for p in ('scripts/run_frozen_music_math.py',
              'docs/FROZEN-MUSIC-MATH.md', 'AGENTS.md', 'requirements.lock')]
    return {p.relative_to(ROOT).as_posix(): digest(read(p)) for p in paths}
