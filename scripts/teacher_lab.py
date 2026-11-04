#!/usr/bin/env python3
"""TT-001 Teacher Lab CLI. Deliberately has no train/distill command."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trinite.contracts import ContractError, json_bytes  # noqa: E402
from teacher_lab_v1 import core  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TT-001: offline teacher research preflight")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "preflight", "preview-formal", "freeze"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--plan", default=core.DEFAULT_PLAN)
        if name == "freeze":
            cmd.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--identity", required=True,
                        help="sha256:... independently retained outside the packet")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = core.verify(args.run, args.identity)
        elif args.command == "preview-formal":
            result = core.preview_formal(args.plan)
        elif args.command == "freeze":
            digest = core.freeze(args.out, args.plan)
            result = {"request_identity": digest, "status": "PREFLIGHT_ONLY",
                      "path": str(args.out), "training_authorized": False}
        else:
            req, _, _ = core.request_and_rights(args.plan)
            result = ({"parameter_count": req["parameter_count"],
                       "candidate_source_count": len(core.inspect(args.plan)[2]["sources"]),
                       "status": req["status"], "training_authorized": False}
                      if args.command == "inspect" else req)
    except (ContractError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Teacher Lab rejected: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
