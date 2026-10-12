"""Freeze, run and read-only verify CNR-001 against immutable v0.1.0 input."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.cnr001 import core


def expected_request():
    p, digest = core.protocol()
    data, _ = core.source_data()
    _, evidence = core.baseline_vectors()
    return {"schema": "trinite.cnr-001-request.v3",
            "protocol":p,"protocol_identity":digest,
            "baseline":evidence,"source":core.source_receipt(),
            "data_files":{name:identity(raw) for name,raw in data.items()},
            "release_tag":"v0.1.0","release_commit":core.BASELINE_SHA,
            "evidence_class":"retrospective-exploratory"}


def request_at(path, commitment):
    raw = core.read(path, 512*1024)
    if identity(raw) != commitment or raw != json_bytes(expected_request()):
        raise ContractError("external CNR request/source/data commitment differs")
    return parse_json(raw, canonical=True)


def freeze(path):
    raw = json_bytes(expected_request())
    with core.safe_output(path).open("xb") as stream:
        stream.write(raw)
    return {"request_identity":identity(raw),"status":"frozen",
            "baseline":core.BASELINE_SHA,"phase5_ready":False}


def run(request, commitment, output):
    request_at(request, commitment)
    result = core.analyze()
    result["request_identity"] = commitment
    raw = json_bytes(result)
    with core.safe_output(output).open("xb") as stream:
        stream.write(raw)
    return {"analysis_identity":identity(raw),"status":"completed",
            "research_class":"retrospective-exploratory","phase5_ready":False}


def verify(request, commitment, output, artifact):
    request_at(request, commitment)
    raw = core.read(output)
    if identity(raw) != artifact:
        raise ContractError("external CNR analysis commitment differs")
    expected = core.analyze()
    expected["request_identity"] = commitment
    if raw != json_bytes(expected):
        raise ContractError("CNR full baseline and biological replay mismatch")
    return {"status":"passed","read_only":True,
            "request_identity":commitment,"analysis_identity":artifact,
            "phase5_ready":False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-request",type=Path)
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--verify-only",action="store_true")
    ap.add_argument("--request",type=Path)
    ap.add_argument("--request-identity")
    ap.add_argument("--output",type=Path)
    ap.add_argument("--analysis-identity")
    a=ap.parse_args(argv)
    try:
        if a.freeze_request is not None:
            if any(x is not None for x in (a.request,a.request_identity,a.output,a.analysis_identity)):
                raise ContractError("freeze must not borrow run identities")
            outcome=freeze(a.freeze_request)
        else:
            if a.request is None or a.request_identity is None or a.output is None:
                raise ContractError("external request identity and output required")
            if a.run:
                if a.analysis_identity is not None: raise ContractError("run cannot borrow analysis identity")
                outcome=run(a.request,a.request_identity,a.output)
            else:
                if a.analysis_identity is None: raise ContractError("verification requires external analysis identity")
                outcome=verify(a.request,a.request_identity,a.output,a.analysis_identity)
        print(json_bytes(outcome).decode(),end="")
        return 0
    except (ContractError,OSError,ValueError,KeyError,TypeError) as error:
        print("CNR-001: "+str(error),file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
