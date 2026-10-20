"""Explicit checkout-only NCM entrypoint, including isolated worker invocation."""
import sys
import tempfile
from pathlib import Path


def entry():
    # -I does not ignore -B, but -B alone still reads stale bytecode. An empty
    # private cache prefix prevents loading any prior package bytecode as source.
    with tempfile.TemporaryDirectory(prefix='ncm-empty-bytecode-') as empty:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = empty
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ncm_reference_v1.runner import main
        return main()


if __name__ == '__main__':
    raise SystemExit(entry())
