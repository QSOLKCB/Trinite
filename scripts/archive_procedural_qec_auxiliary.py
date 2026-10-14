"""Archive engineering auxiliary runs, verify transport or restore historical source.

--archive performs fresh full native replay. --verify-archive is stdlib-only
byte transport verification, not numerical replay or a scientific result.
"""
import argparse
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trinite.procedural_qec_auxiliary.archive import (
    ArchiveError, build_archive, restore_historical_source, verify_archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", action="store_true")
    mode.add_argument("--verify-archive", action="store_true")
    mode.add_argument("--restore-historical-source", action="store_true")
    for name in ("output", "archive-file", "run", "request", "verification-request", "historical-source"):
        parser.add_argument("--" + name, type=Path)
    for name in ("request-identity", "verification-request-identity", "archive-identity", "inventory-identity"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    production = (args.run, args.request, args.request_identity, args.verification_request,
                  args.verification_request_identity, args.historical_source)
    transport = (args.archive_file, args.archive_identity, args.inventory_identity)
    if args.archive:
        if args.output is None or any(v is None for v in production) or any(v is not None for v in transport):
            parser.error("archive needs --output, --run, --historical-source and both external request paths/identities")
    elif any(v is None for v in transport) or any(v is not None for v in production):
        parser.error("transport verification/restoration needs --archive-file and both independent digests only")
    elif (args.output is None) == args.restore_historical_source:
        parser.error("only historical restoration needs a new --output directory")
    try:
        if args.archive:
            result = build_archive(args.output, args.run, args.request, args.request_identity,
                args.verification_request, args.verification_request_identity, args.historical_source)
        elif args.restore_historical_source:
            result = restore_historical_source(*transport, args.output)
        else:
            result = verify_archive(*transport)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["outcome"] == "passed" else 1
    except (ArchiveError, OSError, ValueError, RuntimeError, KeyError, TypeError,
            OverflowError, subprocess.TimeoutExpired) as exc:
        print("auxiliary archive: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
