"""Standard-library input, exact Sudoku scoring and immutable byte boundaries."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import stat

ROOT = Path(__file__).resolve().parent
LEVELS = (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path, limit):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('symlink input')
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    except OSError as error:
        raise ValueError('unreadable regular input') from error
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        os.close(fd)
        raise ValueError('nonregular or oversized input')
    with os.fdopen(fd, 'rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('oversized input')
    return data


def publish(path, data):
    """Atomic visibility, never replace an existing file or symlink."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            view = memoryview(data)
            while view:
                count = stream.write(view)
                if not count:
                    raise OSError('short write')
                view = view[count:]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def board(values, *, blanks=False):
    if not isinstance(values, list) or len(values) != 81:
        raise ValueError('exactly 81 cells required')
    low = 0 if blanks else 1
    if any(type(v) is not int or not low <= v <= 9 for v in values):
        raise ValueError('invalid cell')
    return values


def grid_valid(values):
    try:
        board(values)
    except ValueError:
        return False
    units = [values[r*9:(r+1)*9] for r in range(9)]
    units += [values[c::9] for c in range(9)]
    units += [[values[(r+i)*9+c+j] for i in range(3) for j in range(3)]
              for r in (0, 3, 6) for c in (0, 3, 6)]
    return all(set(unit) == set(range(1, 10)) for unit in units)


def encode_board(values):
    board(values, blanks=True)
    return [token for row in range(9)
            for token in (values[row*9:(row+1)*9] + ([10] if row < 8 else []))]


def decode_board(tokens):
    if not isinstance(tokens, list) or len(tokens) != 89 or any(type(v) is not int for v in tokens):
        raise ValueError('exactly 89 integer output tokens required')
    separators = {9 + 10*i for i in range(8)}
    if any(tokens[i] != 10 for i in separators):
        raise ValueError('invalid row separators')
    return board([v for i, v in enumerate(tokens) if i not in separators])


def score(tokens, clues, target):
    board(clues, blanks=True)
    if not grid_valid(target) or any(c and c != t for c, t in zip(clues, target)):
        raise ValueError('invalid reference/clue relation')
    try:
        cells = decode_board(tokens)
    except ValueError:
        return dict(structurally_valid=False, valid_grid=False, clues_preserved=False,
                    solved=False, target_agreement=False, matching_cells=0)
    valid = grid_valid(cells)
    kept = all(c == 0 or c == p for c, p in zip(clues, cells))
    return dict(structurally_valid=True, valid_grid=valid, clues_preserved=kept,
                solved=valid and kept, target_agreement=cells == target,
                matching_cells=sum(a == b for a, b in zip(cells, target)))


def inputs():
    """Four related formal smoke boards, not independent benchmark families."""
    rows = []
    for variant in range(4):
        target = [((r*3 + r//3 + c + variant) % 9) + 1 for r in range(9) for c in range(9)]
        clues = [0 if c == (r + variant) % 9 else target[r*9+c] for r in range(9) for c in range(9)]
        assert grid_valid(target)
        # Each row lacks exactly one digit; this independently certifies uniqueness.
        recovered = []
        for row in range(9):
            chunk = clues[row*9:(row+1)*9]
            missing = set(range(1, 10)) - set(chunk)
            assert len(missing) == 1 and chunk.count(0) == 1
            recovered.extend(v or next(iter(missing)) for v in chunk)
        assert recovered == target and grid_valid(recovered)
        rows.append(dict(id=f'formal-smoke-{variant}', family='cyclic-latin-3x3-v1',
                         clues=clues, target=target))
    return rows


def verify_upstream():
    pin = json.loads(read_bytes(ROOT/'upstream-pin.json', 65536))
    actual = {str(p.relative_to(ROOT/'vendor')) for p in (ROOT/'vendor').rglob('*') if p.is_file() and p.suffix != '.pyc'}
    if actual != set(pin['files']):
        raise ValueError('upstream source inventory mismatch')
    for name, expected in pin['files'].items():
        if digest(read_bytes(ROOT/'vendor'/name, 100000)) != expected:
            raise ValueError('upstream source mismatch: ' + name)
    for name, key in [('config.yaml', 'config_sha256'), ('checkpoint-card.md', 'checkpoint_card_sha256')]:
        if digest(read_bytes(ROOT/name, 65536)) != pin[key]:
            raise ValueError('upstream metadata mismatch')
    return pin


def source_bytes():
    repo = ROOT.parent.parent
    paths = sorted([p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
                   + [repo/'scripts/run_ncm_reference.py'])
    return {str(p.relative_to(repo)): read_bytes(p, 200000) for p in paths}


def request():
    verify_upstream()
    return dict(schema='ncm-reference-request-v1', source={k: digest(v) for k,v in source_bytes().items()},
                protocol=json.loads(read_bytes(ROOT/'protocol.json', 65536)), inputs=inputs())


def verify_weights(directory):
    pin = verify_upstream()
    verified = {}
    for name, entry in pin['weights'].items():
        path = Path(directory)/name
        data = read_bytes(path, entry['size'])
        if len(data) != entry['size'] or digest(data) != entry['sha256']:
            raise ValueError('checkpoint size/digest mismatch: ' + name)
        verified[name] = str(path.resolve())
    return verified
