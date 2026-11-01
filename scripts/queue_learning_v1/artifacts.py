"""Small exclusive artifact operations, sharing the queue gate's path checks."""
import os

import queue_evaluation_v1 as gate
from trinite.contracts import json_bytes, parse_json

MAX_JSON = 4 * 1024 * 1024


def read_json(path, limit=MAX_JSON):
    return parse_json(gate.read(path, limit), canonical=True)


def write_json(path, value):
    raw = json_bytes(value)
    gate.require(len(raw) <= MAX_JSON, 'queue artifact exceeds JSON bound')
    gate.write(path, raw)
    return gate.digest(raw)


def append_json(path, value):
    # Journals are created exclusively by their owner before the first append.
    path = gate.safe_path(path)
    gate.read(path, 32 * 1024 * 1024)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    with os.fdopen(fd, 'ab') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def anchored_json(path, expected):
    gate.hash_arg(expected)
    raw = gate.read(path, MAX_JSON)
    gate.require(gate.digest(raw) == expected, 'independent artifact anchor mismatch')
    return parse_json(raw, canonical=True)


def inventory(directory):
    directory = gate.safe_path(directory)
    result = {}
    for path in sorted(directory.rglob('*')):
        gate.safe_path(path)
        name = str(path.relative_to(directory))
        if path.is_dir():
            result[name + '/'] = 'directory'
        else:
            result[name] = gate.digest(gate.read(path, 32 * 1024 * 1024))
    return result


def retain_failure(directory, error):
    gate.failure(directory, error)
    try:
        write_json(directory / 'failure-detail.json', {'type': type(error).__name__,
                   'message': str(error)[:2048], 'successful_result': False})
    except OSError:
        pass
