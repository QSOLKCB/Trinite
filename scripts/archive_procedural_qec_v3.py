"""Archive fully replayed QEC procedural evidence or audit exact archived bytes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procedural_qec_v3_archive import ArchiveError, build_archive, verify_archive


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", action="store_true")
    mode.add_argument("--verify-archive", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--archive-file", type=Path)
    p.add_argument("--run", type=Path)
    p.add_argument("--request", type=Path)
    p.add_argument("--request-identity")
    p.add_argument("--verification-request", type=Path)
    p.add_argument("--verification-request-identity")
    p.add_argument("--archive-identity")
    p.add_argument("--inventory-identity")
    a = p.parse_args(argv)
    if a.archive:
        if (a.output is None or a.run is None or a.request is None
                or a.request_identity is None or a.verification_request is None
                or a.verification_request_identity is None or a.archive_file is not None
                or a.archive_identity is not None or a.inventory_identity is not None):
            p.error("archive needs --output, --run and both external committed request paths and identities")
    else:
        if (a.archive_file is None or a.archive_identity is None or a.inventory_identity is None
                or any(v is not None for v in
                       (a.output, a.run, a.request, a.request_identity,
                        a.verification_request, a.verification_request_identity))):
            p.error("read-only --verify-archive requires only --archive-file and both independent digests")
    try:
        result = (build_archive(a.output, a.run, a.request,
                  a.request_identity, a.verification_request,
                  a.verification_request_identity) if a.archive
                  else verify_archive(a.archive_file, a.archive_identity, a.inventory_identity))
        print(json.dumps(result, sort_keys=True))
        return 0 if result["outcome"] == "passed" else 1
    except (ArchiveError, OSError, ValueError, RuntimeError, KeyError,
            TypeError, OverflowError) as exc:
        print("procedural v3 archive: " + type(exc).__name__ + ": " + str(exc),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
