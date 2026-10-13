"""Offline, source-bound, read-only analysis of native ternary checkpoint snapshots."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from ternary_dynamics.dynamics import (freeze_request, report_for, protocol,
                                        _bounded_file, selection)


def safe_output(path: Path, input_root: Path, *sources: Path) -> Path:
    result = Path(os.path.abspath(str(path)))
    root = Path(os.path.abspath(str(input_root)))
    if (result == root or root in result.parents or
            any(result == Path(os.path.abspath(str(s))) for s in sources)):
        raise ContractError('ternary dynamics must never write into input evidence')
    if any(p.is_symlink() for p in (result, *result.parents)):
        raise ContractError('ternary dynamics output symlinks forbidden')
    if not result.parent.is_dir():
        raise ContractError('ternary dynamics output parent must already exist')
    return result


def write_exclusive(path: Path, raw: bytes) -> None:
    """Exclusive receipt with pinned no-follow directory; retry safe on write errors."""
    segments = Path(os.path.abspath(str(path))).parts
    fds = []
    created = False
    saved_inode = None
    try:
        fds.append(os.open('/', os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC))
        for piece in segments[1:-1]:
            fds.append(os.open(piece, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,
                               dir_fd=fds[-1]))
        fd = os.open(segments[-1], os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,
                     0o600, dir_fd=fds[-1])
        created = True
        try:
            saved_inode = os.fstat(fd)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(fds[-1])
        except BaseException:
            if created and saved_inode is not None:
                try:
                    check = os.stat(segments[-1], dir_fd=fds[-1], follow_symlinks=False)
                    if (check.st_dev,check.st_ino)==(saved_inode.st_dev,saved_inode.st_ino):
                        os.unlink(segments[-1],dir_fd=fds[-1]);os.fsync(fds[-1])
                except OSError:
                    pass
            raise
    finally:
        for fd in reversed(fds):
            os.close(fd)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-selection',type=Path)
    group.add_argument('--request',type=Path)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--request-identity')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args(argv)
    try:
        protocol()
        src=args.freeze_selection or args.request
        output=safe_output(args.output,args.root,src)
        if args.freeze_selection:
            if args.verify_only or args.request_identity:
                raise ContractError('freeze does not accept request/verification selectors')
            raw=_bounded_file(src,262144)
            selected=selection(parse_json(raw,canonical=True))
            result=freeze_request(selected,args.root)
        else:
            require_identity(args.request_identity)
            raw=_bounded_file(args.request,262144)
            if identity(raw)!=args.request_identity:
                raise ContractError('ternary dynamics external request identity mismatch')
            request=parse_json(raw,canonical=True)
            result=report_for(request,args.root)
        payload=json_bytes(result)
        if len(payload)>protocol()['maximum_output_bytes']:
            raise ContractError('ternary dynamics report exceeds output bound')
        if args.verify_only:
            if args.freeze_selection:
                raise ContractError('cannot verify-only a freeze')
            if _bounded_file(output,protocol()['maximum_output_bytes'])!=payload:
                raise ContractError('retained dynamics report differs from fresh inspection')
        else:
            write_exclusive(output,payload)
        print(json_bytes({'outcome':'complete', 'identity':identity(payload),
                          'schema':result['schema'], 'scope':'read-only diagnostic; no research gate'}).decode(),end='')
        return 0
    except (ContractError,OSError,KeyError,TypeError,ValueError,RuntimeError) as err:
        print('ternary-dynamics: '+str(err),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
