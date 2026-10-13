"""Freeze, execute, and independently replay QEC procedural v3 research."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-request", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    p.add_argument("--run-profile", choices=("smoke", "research"))
    p.add_argument("--request", type=Path)
    p.add_argument("--request-identity")
    p.add_argument("--verification-request", type=Path)
    p.add_argument("--verification-request-identity")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.freeze_request is not None:
        if (a.run_profile is None or any(v is not None for v in
                (a.request, a.request_identity, a.verification_request,
                 a.verification_request_identity, a.output))):
            p.error("v3 freeze requires only profile and new request path")
    elif (a.run_profile is not None or any(v is None for v in
            (a.request, a.request_identity, a.verification_request, a.output))
          or (a.run and a.verification_request_identity is not None)
          or (a.verify_only and a.verification_request_identity is None)):
        p.error("v3 run/verify require independently supplied correct request identities")
    try:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        from trinite.procedural_qec_v3 import experiment
        if a.freeze_request is not None:
            value = experiment.freeze_request(a.freeze_request, a.run_profile)
        elif a.run:
            value = experiment.run_matrix(a.output, a.request, a.request_identity,
                                          a.verification_request)
        else:
            value = experiment.verify_only(a.output, a.request, a.request_identity,
                                           a.verification_request, a.verification_request_identity)
        print(json.dumps(value, sort_keys=True))
        return 0 if value.get("outcome", "completed") == "completed" else 1
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
