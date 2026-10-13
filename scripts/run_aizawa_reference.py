"""Freeze a source/environment request, measure, or verify Aizawa simulation."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from aizawa_reference.benchmark import (freeze_request, measure, protocol, ROOT as SOURCE_ROOT)


def _read(path: Path, limit: int) -> bytes:
    absolute = Path(os.path.abspath(str(path)))
    if (not absolute.is_file() or absolute.is_symlink() or
            any(parent.is_symlink() for parent in absolute.parents)):
        raise ContractError('missing or symlinked Aizawa request/report')
    with absolute.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ContractError('Aizawa input size budget exceeded')
    return raw


def safe_output(path: Path, request_path: Path | None = None) -> Path:
    result = Path(os.path.abspath(str(path)))
    if (result == SOURCE_ROOT or SOURCE_ROOT in result.parents or
        result == request_path or
        any(member.is_symlink() for member in (result, *result.parents)) or
        not result.parent.is_dir()):
        raise ContractError('Aizawa output must be a new file outside the source tree and symlinks')
    return result


def write_exclusive(path: Path, content: bytes) -> None:
    """Exclusive, no-follow, pinned-directory receipt with durability and cleanup."""
    components = path.parts
    fds = []
    created_inode = None
    try:
        fds.append(os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC))
        for name in components[1:-1]:
            fds.append(os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW |
                               os.O_CLOEXEC, dir_fd=fds[-1]))
        fd = os.open(components[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fds[-1])
        created_inode = os.fstat(fd)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(fds[-1])
        except BaseException:
            try:
                found = os.stat(components[-1], dir_fd=fds[-1], follow_symlinks=False)
                if (found.st_ino, found.st_dev) == (created_inode.st_ino, created_inode.st_dev):
                    os.unlink(components[-1], dir_fd=fds[-1])
                    os.fsync(fds[-1])
            except OSError:
                pass
            raise
    finally:
        for fd in reversed(fds):
            os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--freeze-request', action='store_true')
    selector.add_argument('--request', type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        p = protocol()
        source = Path(os.path.abspath(str(args.request))) if args.request is not None else None
        output = safe_output(args.output, source)
        if args.freeze_request:
            if args.verify_only or args.request_identity is not None:
                raise ContractError('Aizawa freeze takes no verification selectors')
            result = freeze_request()
        else:
            require_identity(args.request_identity)
            request_bytes = _read(args.request, p['limits']['max_file_bytes'])
            if identity(request_bytes) != args.request_identity:
                raise ContractError('external Aizawa request identity differs')
            request = parse_json(request_bytes, canonical=True)
            result = measure(request)
        payload = json_bytes(result)
        if len(payload) > p['limits']['max_report_bytes']:
            raise ContractError('Aizawa reference report exceeds size budget')
        if args.verify_only:
            if _read(output, p['limits']['max_file_bytes']) != payload:
                raise ContractError('retained Aizawa reference differs from recomputation')
        else:
            write_exclusive(output, payload)
        print(json_bytes({'outcome': 'completed', 'evidence_class': 'SIMULATION',
                          'identity': identity(payload), 'schema': result['schema']}).decode(), end='')
        return 0
    except (ContractError, OSError, ValueError, RuntimeError, KeyError, TypeError, OverflowError) as error:
        print('aizawa-reference: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
