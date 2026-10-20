"""Same-directory atomic publication without replacing any existing destination."""
import os
from pathlib import Path
import tempfile


def sidecar(output, suffix):
    output=Path(output)
    return output.with_name(output.name+suffix)


def write_all(stream, raw):
    """Handle short writes explicitly; an interrupted staging write is unpublished."""
    view=memoryview(raw)
    while view:
        count=stream.write(view)
        if count is None or count<=0:
            raise OSError('artifact write made no progress')
        view=view[count:]


def publish(path, raw):
    """Publish complete bytes via hard-link create, never rename/replace overwrite.

    Readers see either no destination or all bytes. A competing destination, even
    a dangling symlink, wins without modification. Catchable failures/interrupts
    clean staging files; SIGKILL may leave a hidden temp, never a partial final.
    """
    path=Path(path)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            fd=None
            write_all(stream,raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp,path)
    finally:
        if fd is not None:os.close(fd)
        os.unlink(temp)
