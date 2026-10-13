"""Freeze, execute or independently replay the QEC-guided native learning matrix."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-request", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--run-profile", choices=("smoke", "research"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-identity")
    parser.add_argument("--verification-request", type=Path)
    parser.add_argument("--verification-request-identity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.freeze_request is not None:
        if (args.run_profile is None or args.request is not None or args.output is not None
                or args.request_identity or args.verification_request
                or args.verification_request_identity):
            parser.error("freeze requires only --run-profile and --freeze-request")
    else:
        if (args.run_profile is not None or args.request is None or args.request_identity is None
                or args.verification_request is None or args.output is None):
            parser.error("run/verify need --request, --request-identity, --output and --verification-request")
        if args.run and args.verification_request_identity is not None:
            parser.error("run must create a new external verification request")
        if args.verify_only and args.verification_request_identity is None:
            parser.error("verify-only requires the independently recorded verification identity")
    try:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        from trinite.procedural_qec import experiment
        if args.freeze_request is not None:
            value = experiment.freeze_request(args.freeze_request, args.run_profile)
        elif args.run:
            value = experiment.run_matrix(args.output, args.request,
                                         args.request_identity, args.verification_request)
        else:
            value = experiment.verify_only(args.output, args.request,
                args.request_identity, args.verification_request,
                args.verification_request_identity)
        print(json.dumps(value, sort_keys=True))
        return 0 if value.get("outcome", "completed") == "completed" else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
