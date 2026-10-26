"""Bounded exclusive artifact writes and atomic no-clobber publication."""
import os
from pathlib import Path
import tempfile

from trinite.contracts import ContractError


def publish(path, raw):
    path = Path(path)
    if type(raw) is not bytes or len(raw) > 16 * 1024 * 1024:
        raise ContractError('artifact bytes exceed diagnostic bound')
    fd, staged = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(staged, path)
    finally:
        os.unlink(staged)


class Store:
    """Keep own artifact writes attached to the originally opened directory."""
    def __init__(self, root, expected_identity=None):
        self.root = Path(root)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ContractError('diagnostic output must be an existing safe attempt directory')
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.stat = os.fstat(self.fd)
        if expected_identity is not None and (self.stat.st_dev, self.stat.st_ino) != tuple(expected_identity):
            os.close(self.fd)
            raise ContractError('diagnostic attempt directory identity differs at handoff')
        self.bytes = 0

    def check(self):
        stat = self.root.lstat()
        if (stat.st_dev, stat.st_ino) != (self.stat.st_dev, self.stat.st_ino):
            raise ContractError('diagnostic attempt directory was replaced')

    def write(self, name, raw):
        if Path(name).name != name or not name or name in ('.', '..'):
            raise ContractError('artifact name must be a basename')
        if type(raw) is not bytes or len(raw) > 16 * 1024 * 1024 or self.bytes + len(raw) > 32 * 1024 * 1024:
            raise ContractError('diagnostic artifact/attempt budget exceeded')
        self.check()
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.bytes += len(raw)
        self.check()

    def close(self):
        os.close(self.fd)
