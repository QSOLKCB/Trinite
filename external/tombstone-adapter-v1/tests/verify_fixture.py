#!/usr/bin/env python3
"""Read-only source-bound fixture replay; never regenerate expected results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("adapter", ROOT / "scripts/tombstone_adapter_v1.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_hashes():
    paths = list((adapter.CRATE / "tests").glob("*.py"))
    paths += list((ROOT / "tests/tombstone_adapter_v1").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    record = json.loads((ROOT / "fixtures/tombstone-adapter-v1/conformance.json").read_bytes())
    adapter.verify_build()
    assert record["schema"] == "trinite.tombstone-source-bound-conformance.v1"
    assert record["runtime_sources"] == adapter.source_hashes(), "source closure changed; retain old fixture, version new semantics"
    assert record["test_sources"] == test_hashes(), "test closure changed"
    result = subprocess.run([sys.executable, str(adapter.CRATE / "tests/oracle.py"), str(adapter.BINARY)], capture_output=True, timeout=180, check=True)
    assert not result.stderr
    assert json.loads(result.stdout) == record["conformance"], "fresh actual-C99 replay differs"
    print("source-bound public conformance matches fresh actual-C99 execution")


if __name__ == "__main__":
    main()
