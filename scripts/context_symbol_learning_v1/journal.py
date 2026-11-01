"""Read a complete canonical journal prefix without repairing retained bytes."""
import queue_evaluation_v1 as gate
from trinite.contracts import parse_json


def prefix(path):
    raw=gate.read(path,4<<20)
    records=[]
    consumed=0
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break
        try:
            record=parse_json(line,canonical=True)
        except ValueError:
            break
        records.append(record)
        consumed+=len(line)
    # The entire suffix, including any later lines, remains bound by the final
    # inventory. Never skip malformed content or rewrite an interrupted journal.
    return records,raw[consumed:]
