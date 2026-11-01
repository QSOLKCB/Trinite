#!/usr/bin/env python3
"""Fixed offline TOMBSTONE build/seal and bounded Linux process transport.

This is engineering infrastructure, not a learner/evaluation/training controller.
A trusted checkout and build record are prerequisites, not remote attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "external/tombstone-adapter-v1"
BINARY = CRATE / "target/release/trinite-tombstone-adapter-v1"
RECORD = CRATE / "target/build-record.json"
MAX_INPUT = 1024
MAX_OUTPUT = 4096  # combined stdout/stderr
TIMEOUT = 5.0
PIN = "5447558afdaa3e37f933015278b64dde6abef5d5"


class InfrastructureError(RuntimeError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hashes() -> dict[str, str]:
    """Bind the complete imported closure and this adapter/launcher, not targets."""
    paths = [ROOT / "scripts/tombstone_adapter_v1.py"]
    paths += [p for p in CRATE.rglob("*") if p.is_file()
              and "target" not in p.relative_to(CRATE).parts
              and "tests" not in p.relative_to(CRATE).parts]
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in sorted(paths)}


def verify_upstream() -> None:
    try:
        pin = json.loads((CRATE / "upstream-pin.json").read_bytes())
    except ValueError as error:
        raise InfrastructureError("invalid upstream manifest") from error
    if (type(pin) is not dict or set(pin) != {"schema", "repository", "commit", "license", "files"}
            or pin["schema"] != "trinite.tombstone-source.v1"
            or pin["repository"] != "https://github.com/QSOLKCB/TOMBSTONE"
            or type(pin["files"]) is not dict
            or pin["commit"] != PIN or pin["license"] != "Apache-2.0"):
        raise InfrastructureError("upstream identity mismatch")
    actual = {p.relative_to(CRATE / "upstream").as_posix(): digest(p.read_bytes())
              for p in (CRATE / "upstream").rglob("*") if p.is_file()}
    if actual != pin["files"]:
        raise InfrastructureError("upstream source mismatch")


def verify_build() -> None:
    verify_upstream()
    try:
        record = json.loads(RECORD.read_bytes())
    except ValueError as error:
        raise InfrastructureError("invalid build record") from error
    if record != {"schema": "trinite.tombstone-build.v1",
                  "sources": source_hashes(), "binary_sha256": digest(BINARY.read_bytes())}:
        raise InfrastructureError("build/source identity mismatch")


def build() -> None:
    verify_upstream()
    before = source_hashes()
    subprocess.run(["cargo", "build", "--release", "--offline", "--locked"],
                   cwd=CRATE, check=True, timeout=120)
    if source_hashes() != before:
        raise InfrastructureError("source changed during build")
    record = {"schema": "trinite.tombstone-build.v1", "sources": before,
              "binary_sha256": digest(BINARY.read_bytes())}
    # Build cache only, never a retained research receipt or historical fixture.
    temporary = RECORD.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    temporary.replace(RECORD)
    verify_build()


def bounded_stdin() -> bytes:
    deadline = time.monotonic() + TIMEOUT
    chunks = bytearray()
    # select() accepts regular files and pipes on the documented Linux lane.
    import select
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([sys.stdin.buffer], [], [], remaining)[0]:
            raise InfrastructureError("stdin deadline")
        chunk = os.read(sys.stdin.fileno(), MAX_INPUT + 1 - len(chunks))
        if not chunk:
            return bytes(chunks)
        chunks.extend(chunk)
        if len(chunks) > MAX_INPUT:
            raise ValueError("request too large")


def execute(operation: str, request: bytes) -> tuple[int, bytes, bytes]:
    if operation not in ("observe", "evaluate-privileged"):
        raise ValueError("unknown operation")
    if len(request) > MAX_INPUT:
        raise ValueError("request too large")
    verify_build()
    # No request-selected executable/path/arguments, shell, inherited preload,
    # Python path or runtime download. The build toolchain is a separate action.
    process = subprocess.Popen([str(BINARY), operation], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True,
                               env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    output = {"out": bytearray(), "err": bytearray()}
    deadline = time.monotonic() + TIMEOUT
    try:
        assert process.stdin is not None
        process.stdin.write(request)  # <=1024 bytes, below Linux pipe capacity
        process.stdin.close()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "out")
            selector.register(process.stderr, selectors.EVENT_READ, "err")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InfrastructureError("process deadline")
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fileobj.fileno(), MAX_OUTPUT + 1)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        output[key.data].extend(chunk)
                        if sum(map(len, output.values())) > MAX_OUTPUT:
                            raise InfrastructureError("combined output limit")
        returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if returncode not in (0, 2) or (returncode == 0 and output["err"]) or (returncode == 2 and output["out"]):
            raise InfrastructureError("native transport failure")
        return returncode, bytes(output["out"]), bytes(output["err"])
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "check", "observe", "evaluate-privileged"))
    args = parser.parse_args()
    try:
        if args.operation == "build":
            build()
        elif args.operation == "check":
            verify_build()
        else:
            status, out, err = execute(args.operation, bounded_stdin())
            sys.stdout.buffer.write(out)
            sys.stderr.buffer.write(err)
            return status
        return 0
    except ValueError:
        print("invalid request", file=sys.stderr)
        return 2
    except (OSError, KeyError, InfrastructureError, subprocess.SubprocessError):
        print("infrastructure failure", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
