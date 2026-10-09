"""Explicit, fixed-revision asset acquisition; never invoked by capture."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def acquire(directory):
    pin = json.loads((ROOT/'src/trinite/reference-model-pin.json').read_bytes())
    directory.mkdir(parents=True, exist_ok=True)
    for entry in pin['files']:
        target = directory/entry['path']
        if target.is_symlink(): raise ValueError('unsafe asset')
        if target.exists():
            if not target.is_file(): raise ValueError('unsafe asset')
            candidate = target
        else:
            url = f"https://huggingface.co/{pin['model']}/resolve/{pin['revision']}/{entry['path']}"
            partial = target.with_suffix(target.suffix+'.partial')
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise ValueError('unsafe partial asset')
            # Incomplete bytes are never admitted; restart a stale partial safely.
            if partial.exists(): partial.unlink()
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('xb') as stream:
                while chunk := response.read(1024*1024): stream.write(chunk)
            candidate = partial
        sha = hashlib.sha256(); blob = hashlib.sha1(f"blob {entry['bytes']}\0".encode())
        size = 0
        with candidate.open('rb') as stream:
            while chunk := stream.read(1024*1024):
                size += len(chunk); sha.update(chunk); blob.update(chunk)
        if size != entry['bytes'] or (sha.hexdigest() != entry['sha256'] if entry['sha256'] else blob.hexdigest() != entry['git_blob']):
            raise ValueError('asset differs from pinned revision: '+entry['path'])
        if candidate != target: candidate.rename(target)
        print(entry['path'], size, sha.hexdigest(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    acquire(parser.parse_args().directory)
