"""Authenticate derivative report bytes, then reuse the frozen public scorer.

This is not an audit of original runs, weights or numerical execution. Run on a
quiescent checkout: mutation observations are not hostile-writer attestation.
"""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/frozen-music-math-result-20261028-v1'
LIMIT = 1024 * 1024
FIXTURE_HASHES = {'README.md': 'sha256:dbf143143620858382484d4c9a544f872624cc62da386096c4bc5fc6aadf03fd', 'predictions.json': 'sha256:a64536e545450855ae5f8970cefdcd6b8851e9ce091586287ddffa3721426802', 'provenance.json': 'sha256:5b4b33c0f40f772ca993b47b3292f6f14cf109a0bc446dd81af17db4bc10be15', 'summary.json': 'sha256:5a9cdd4126c310e63638a0ee184d43a7de23b89f699e59e702f55d9317678439'}
SOURCES = {'scripts/frozen_music_math_v1/__init__.py': 'sha256:b797051c8f3b6a13969858b2016f9a8704bd48551771e40aecc45343f5f72844', 'scripts/frozen_music_math_v1/artifacts.py': 'sha256:90ae610158bdee4b27e714e5bb95a23701594efb81df9a596746be4bacf314b6', 'scripts/frozen_music_math_v1/protocol.py': 'sha256:b063b84e9daa5a5279e5acb6e889566848663666119bffd9d2b396ca7fa0fb15'}


def digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def read(path):
    """Bounded no-follow read with ancestor and observed mutation rejection."""
    path = Path(os.path.abspath(path))
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in path.parents):
        raise ValueError('unsafe ancestor')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT:
        raise ValueError('bounded regular file required')
    signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if signature(os.fstat(fd)) != signature(before):
            raise ValueError('file replaced at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(LIMIT + 1)
        if (len(raw) > LIMIT or signature(os.fstat(fd)) != signature(before)
                or signature(path.lstat()) != signature(before)):
            raise ValueError('file changed during read')
        return raw
    finally:
        os.close(fd)


def snapshot(fixture):
    fixture = Path(fixture).absolute()
    if not stat.S_ISDIR(fixture.lstat().st_mode):
        raise ValueError('non-link fixture directory required')
    if set(p.name for p in fixture.iterdir()) != set(FIXTURE_HASHES):
        raise ValueError('derivative fixture membership differs')
    return {name: read(fixture / name) for name in FIXTURE_HASHES}


def sources():
    """Check scorer bytes and reject import caches/shadows before importing."""
    package = ROOT / 'scripts/frozen_music_math_v1'
    if not stat.S_ISDIR(package.lstat().st_mode):
        raise ValueError('non-link scorer package required')
    allowed = {'__init__.py', 'artifacts.py', 'protocol.py', 'receipt.py',
               'worker.py', 'workflow.py'}
    if set(p.name for p in package.iterdir()) != allowed:
        raise ValueError('scorer import cache or shadow membership')
    if any(not stat.S_ISREG(p.lstat().st_mode) for p in package.iterdir()):
        raise ValueError('scorer import shadow or non-regular member')
    for name, expected in SOURCES.items():
        if digest(read(ROOT / name)) != expected:
            raise ValueError('frozen scoring source changed')
    names = {'frozen_music_math_v1': 'scripts/frozen_music_math_v1/__init__.py',
             'frozen_music_math_v1.artifacts': 'scripts/frozen_music_math_v1/artifacts.py',
             'frozen_music_math_v1.protocol': 'scripts/frozen_music_math_v1/protocol.py'}
    for name, path in names.items():
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module, '__file__', '')).absolute() != ROOT / path:
            raise ValueError('foreign scoring module')
    # Resolve the package before submodules: find_spec on a submodule may import
    # its parent. The verified package initializer is empty and source-pinned.
    for name, path in names.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin != str(ROOT / path):
            raise ValueError('foreign scoring import resolution')


def verify(fixture=FIXTURE):
    before = snapshot(fixture)
    if {n: digest(raw) for n, raw in before.items()} != FIXTURE_HASHES:
        raise ValueError('derivative fixture bytes changed')
    sources()
    protocol = importlib.import_module('frozen_music_math_v1.protocol')
    sources()
    artifacts = importlib.import_module('frozen_music_math_v1.artifacts')
    records = artifacts.parse(before['predictions.json'])
    summary = artifacts.parse(before['summary.json'])
    provenance = artifacts.parse(before['provenance.json'])
    if protocol.score_records(records) != summary:
        raise ValueError('derived scores differ')
    if provenance['public_artifacts'] != {n: digest(before[n]) for n in ('predictions.json', 'summary.json')}:
        raise ValueError('derivative provenance links differ')
    if snapshot(fixture) != before:
        raise ValueError('derivative changed during verification')
    sources()
    return {'schema': 'trinite.frozen-music-math-report-check.v1',
            'verified': True, 'scope': 'pinned derivative bytes and report-token scoring only',
            'prediction_records': len(records),
            'initial_exact': sum(p['initial_correct'] for p in summary['paired'].values()),
            'final_exact': sum(p['final_correct'] for p in summary['paired'].values()),
            'original_raw_custody_verified': False, 'inference_replayed': False,
            'fresh_confirmation': False, 'learning_adequate': False,
            'phase5_ready': False, 'release_ready': False}


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(), sort_keys=True))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
