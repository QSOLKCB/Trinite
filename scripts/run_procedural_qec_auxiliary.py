"""Prepare an engineering request, run tiny conformance, or replay it read-only.

The prospective scientific profile is intentionally unavailable.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-request", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    p.add_argument("--historical-source", type=Path, required=True)
    p.add_argument("--run-profile", choices=("engineering", "research"))
    p.add_argument("--request", type=Path)
    p.add_argument("--request-identity")
    p.add_argument("--verification-request", type=Path)
    p.add_argument("--verification-request-identity")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.prepare_request is not None:
        if (a.run_profile is None or any(v is not None for v in
                (a.request, a.request_identity, a.verification_request,
                 a.verification_request_identity, a.output))):
            p.error("prepare requires only profile, historical source and new request path")
    elif (a.run_profile is not None or any(v is None for v in
            (a.request, a.request_identity, a.verification_request, a.output))
          or (a.run and a.verification_request_identity is not None)
          or (a.verify_only and a.verification_request_identity is None)):
        p.error("run/verify require independently supplied correct request identities")
    try:
        from trinite.procedural_qec_auxiliary.spec import settings
        if a.prepare_request is not None:
            settings(a.run_profile)  # Reject research before importing a backend.
        from trinite.procedural_qec_auxiliary.inputs import materialize_inputs
        inputs = materialize_inputs(a.historical_source)
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        from trinite.procedural_qec_auxiliary import experiment
        if a.prepare_request is not None:
            value = experiment.prepare_request(a.prepare_request, a.run_profile, inputs)
        elif a.run:
            value = experiment.run_matrix(a.output, a.request, a.request_identity,
                                          a.verification_request, inputs)
        else:
            value = experiment.verify_only(a.output, a.request, a.request_identity,
                a.verification_request, a.verification_request_identity, inputs)
        print(json.dumps(value, sort_keys=True))
        return 0 if value.get("outcome", "completed") == "completed" else 1
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
