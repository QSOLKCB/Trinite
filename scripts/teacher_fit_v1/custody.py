# SPDX-License-Identifier: MPL-2.0
"""Bounded immutable local artifacts. Hashes establish identity, not authority."""
import os
from pathlib import Path
import stat

from trinite.contracts import ContractError, identity, json_bytes, parse_json

MAX_FILE = 16 * 1024 * 1024
MAX_RUN = 128 * 1024 * 1024


def read(path, cap=MAX_FILE):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ContractError('symlink artifact path')
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > cap:
        raise ContractError('unsafe or oversized artifact')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ContractError('artifact exceeded byte bound')
    return raw


def write(path, raw):
    if type(raw) is not bytes or len(raw) > MAX_FILE:
        raise ContractError('invalid artifact payload')
    path = Path(path)
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def record(path, value):
    # Publish only complete fsynced canonical JSON. A killed writer leaves a
    # named pending artifact for failure custody, never a false completed seal.
    path=Path(path)
    pending=path.with_name('.pending-'+path.name)
    if path.exists() or path.is_symlink():
        raise ContractError('immutable record already exists')
    write(pending,json_bytes(value))
    if path.exists() or path.is_symlink():
        raise ContractError('immutable record destination appeared')
    os.rename(pending,path)
    fd=os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def load(path):
    return parse_json(read(path), canonical=True)


def anchored(path, anchor):
    from trinite.contracts import require_identity
    require_identity(anchor)
    raw = read(path)
    if identity(raw) != anchor:
        raise ContractError('external artifact anchor mismatch')
    return parse_json(raw, canonical=True)


def fresh(path):
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ContractError('fresh destination required; attempts cannot be retried')
    if any(p.is_symlink() for p in path.parents):
        raise ContractError('symlink destination parent')
    path.mkdir(mode=0o700)
    return path


def outside(path, *protected):
    target=Path(path).absolute()
    if target!=target.resolve() or any(target.is_relative_to(Path(p).resolve()) for p in protected):
        raise ContractError('output must be outside protected inputs/source')
    return target


def inventory(folder, exclude=()):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ContractError('invalid evidence directory')
    members = []
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():
            raise ContractError('linked evidence')
        if p.is_dir():
            continue
        name = p.relative_to(folder).as_posix()
        if name not in exclude:
            raw = read(p)
            members.append({'name': name, 'identity': identity(raw), 'bytes': len(raw)})
    if sum(m['bytes'] for m in members) > MAX_RUN:
        raise ContractError('evidence byte budget exceeded')
    return members


def seal(folder, context):
    value = dict(context, members=inventory(folder, ('manifest.json',)))
    record(Path(folder) / 'manifest.json', value)
    return identity(json_bytes(value))


def verify_seal(folder, anchor):
    manifest = anchored(Path(folder) / 'manifest.json', anchor)
    if manifest.get('members') != inventory(folder, ('manifest.json',)):
        raise ContractError('missing, duplicate, replaced or unexpected evidence')
    return manifest
