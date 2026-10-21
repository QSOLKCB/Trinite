"""CPU software conformance with exclusive, failure-retaining attempts."""
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
from . import spec, checks


def publish(path, raw):
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def execute(req, folder):
    if sys.platform != 'linux':
        raise ValueError('checker requires Linux; pure reference remains portable')
    if spec.canonical(req) != spec.canonical(spec.request()):
        raise ValueError('request/source mismatch')
    folder.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'schema': 'trinite.cyclic-result.v1', 'status': 'failed',
              'request_identity': spec.identity(spec.canonical(req)),
              'claim': 'bounded exact CPU software conformance, not model quality',
              'environment': {'python': sys.version, 'platform': platform.platform()}}
    try:
        publish(folder/'request.json', spec.canonical(req))
        count = 0
        with (folder/'rows.jsonl').open('xb') as stream:
            for row in checks.rows(req):
                raw = spec.canonical(row)
                if stream.tell()+len(raw) > spec.LIMIT:
                    raise ValueError('retained rows exceed bound')
                stream.write(raw); stream.flush()
                count += 1
            os.fsync(stream.fileno())
        if spec.canonical(req) != spec.canonical(spec.request()):
            raise ValueError('sources changed during conformance')
        result.update(status='passed', cases=count, rows_identity=spec.identity((folder/'rows.jsonl').read_bytes()))
    except (OSError, ValueError, ArithmeticError) as error:
        result['error'] = str(error)[:1000]
    result['resources'] = {'elapsed_seconds_hex': (time.monotonic()-started).hex(),
                           'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           'peak_rss_before_kib': before,
                           'scope': 'Linux RUSAGE_SELF whole checker high-water, not isolated model usage'}
    # A single record is both result and controller outcome; no separate success status.
    # Missing result is incomplete, never success, including publication/storage failure.
    publish(folder/'result.json', spec.canonical(result))
    return result
