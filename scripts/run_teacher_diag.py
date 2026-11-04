#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""TT-003-DIAG-001 frozen-teacher read-only train behaviour audit."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trinite.contracts import ContractError, json_bytes  # noqa: E402
from teacher_diag_v1 import core  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--teacher-run", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    verification = commands.add_parser("verify-request")
    verification.add_argument("--request", type=Path, required=True)
    verification.add_argument("--identity", required=True)
    verification.add_argument("--dataset", type=Path, required=True)
    verification.add_argument("--teacher-run", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--request", type=Path, required=True)
    audit.add_argument("--identity", required=True)
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--teacher-run", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    for name in ("verify", "replay-forward"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--request", type=Path, required=True)
        cmd.add_argument("--identity", required=True)
        cmd.add_argument("--dataset", type=Path, required=True)
        cmd.add_argument("--teacher-run", type=Path, required=True)
        cmd.add_argument("--result", type=Path, required=True)
        cmd.add_argument("--result-identity", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "freeze":
            result = {"request_identity": core.freeze(args.out, args.dataset, args.teacher_run),
                      "training_authorized": False, "heldout_scored": False}
        elif args.operation == "verify-request":
            req = core.verify_request(args.request, args.identity,
                                      args.dataset, args.teacher_run)
            result = {"verified": True, "status": req["status"],
                      "training_authorized": False}
        else:
            from teacher_diag_v1 import native
            if args.operation == "audit":
                result = native.run(args.request, args.identity,
                                    args.dataset, args.teacher_run, args.out)
            elif args.operation == "verify":
                result = native.verify(args.request, args.identity,
                                       args.dataset, args.teacher_run,
                                       args.result, args.result_identity)
            else:
                result = native.replay_forward(args.request, args.identity,
                                               args.dataset, args.teacher_run,
                                               args.result, args.result_identity)
    except (ContractError, OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print(f"TT-003-DIAG-001 rejected: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
