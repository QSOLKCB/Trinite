#!/usr/bin/env python3
"""TT-003: explicit owner-local teacher training; frozen CPU research lane."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trinite.contracts import ContractError, json_bytes  # noqa: E402
from teacher_training_v1 import spec, cuda_preflight  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("inspect")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--dataset-identity", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    verify_request = sub.add_parser("verify-request")
    verify_request.add_argument("--request", type=Path, required=True)
    verify_request.add_argument("--identity", required=True)
    verify_request.add_argument("--dataset", type=Path, required=True)
    train = sub.add_parser("train-owner-local")
    train.add_argument("--request", type=Path, required=True)
    train.add_argument("--identity", required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--approval", type=Path, required=True)
    train.add_argument("--approval-identity", required=True)
    train.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify-run")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--manifest-identity", required=True)
    verify.add_argument("--request-identity", required=True)
    verify.add_argument("--dataset", type=Path, required=True)
    cuda = sub.add_parser("cuda-preflight")
    cuda.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            plan, config = spec.settings()
            result = {"role": "teacher", "parameters": config.parameter_count,
                      "plan": plan, "training_authorized": False,
                      "gpu_training_supported": False}
        elif args.action == "freeze":
            result = {"request_identity": spec.freeze(args.out, args.dataset,
                                                      args.dataset_identity),
                      "training_authorized": False, "path": str(args.out)}
        elif args.action == "verify-request":
            p = spec.verify_request(args.request, args.identity, args.dataset)
            result = {"verified": True, "status": p["status"],
                      "training_authorized": False}
        elif args.action == "cuda-preflight":
            result = cuda_preflight.live(args.device)
        else:
            # Torch not imported on inspect, freeze, approval check or CUDA
            # software-only contract checks. Native runtime only on these paths.
            from teacher_training_v1 import numerical
            if args.action == "train-owner-local":
                result = numerical.run(args.request, args.identity, args.dataset,
                                       args.approval, args.approval_identity, args.out)
            else:
                result = numerical.verify_run(args.run, args.manifest_identity,
                                              args.dataset, args.request_identity)
    except (ContractError, OSError, TypeError, KeyError, ValueError, ImportError) as error:
        print(f"TT-003 rejected: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
