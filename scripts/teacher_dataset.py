#!/usr/bin/env python3
"""TT-002 offline formal teacher dataset; no training commands."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from trinite.contracts import ContractError, json_bytes  # noqa: E402
from teacher_dataset_v1 import core  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--packet", type=Path, required=True)
    verify.add_argument("--identity", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = {"manifest_identity": core.prepare(args.out), "packet": str(args.out),
                      "training_authorized": False}
        elif args.command == "verify":
            result = core.verify(args.packet, args.identity)
        else:
            _, manifest = core.make_packet()
            result = {"schema": manifest["schema"], "counts": manifest["split_counts"],
                      "independent_oracle_checks": manifest["independent_oracle_checks"],
                      "status": manifest["status"], "training_authorized": False}
    except (ContractError, ValueError, OSError, TypeError, KeyError) as error:
        print(f"TT-002 rejected: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
