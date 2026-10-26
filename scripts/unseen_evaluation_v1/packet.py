"""Bounded local packet inventory and mutation guard, not a race-free filesystem snapshot.

Use only quiescent local packets. Pre/post inode metadata checks reject observed
replacement, links and edits; they cannot attest against a hostile concurrent writer.
"""
import os
from pathlib import Path
import stat

from trinite.contracts import ContractError

MAX_MEMBERS = 1024
MAX_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode)


def inventory(root):
    """Return bounded regular-file/directory inode inventory without following links."""
    root = Path(root).absolute()
    ancestors = {}
    for path in root.parents:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ContractError('packet ancestor is not a regular directory')
        # Parent directory contents may legitimately change; its identity may not.
        ancestors[str(path)] = (info.st_dev, info.st_ino, info.st_mode)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ContractError('packet root is not a regular directory')
    members = {'.': _signature(info)}
    pending = [root]
    total = 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if len(members) >= MAX_MEMBERS:
                    raise ContractError('packet member count exceeds bound')
                info = entry.stat(follow_symlinks=False)
                if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    raise ContractError('packet contains symlink or special member')
                path = Path(entry.path)
                members[path.relative_to(root).as_posix()] = _signature(info)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                else:
                    if info.st_size > MAX_FILE_BYTES:
                        raise ContractError('packet member byte bound exceeded')
                    total += info.st_size
                    if total > MAX_BYTES:
                        raise ContractError('packet physical payload bound exceeded')
    return {'root': str(root), 'ancestors': ancestors, 'members': members, 'bytes': total}


def unchanged(root, snapshot):
    """Raise on any observable packet/member/ancestor change since inventory."""
    if inventory(root) != snapshot:
        raise ContractError('packet changed during read-only verification')
