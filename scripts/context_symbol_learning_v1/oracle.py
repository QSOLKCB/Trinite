"""Context-only fixed native transport; never dispatch other scientific tasks."""
import subprocess
import time

import queue_evaluation_v1 as gate
from procedural_dataset_v1 import cases
from .spec import ROOT

CRATE = ROOT/'external/context-symbol-relation-v1'
BINARY = CRATE/'target/release/trinite-context-symbol-relation-v1'


def binary_identity():
    return gate.digest(gate.read(BINARY,64<<20,allow_links=True))


def build():
    version = subprocess.check_output(['rustc','--version'],cwd=CRATE,timeout=10).decode('ascii')
    gate.require(version.startswith('rustc 1.90.0 '), 'Rust 1.90.0 required')
    subprocess.run(['cargo','build','--release','--offline','--locked'],cwd=CRATE,timeout=120,check=True)
    return binary_identity()


def call(request, operation, log=None):
    cases.validate_request('context',request)
    gate.require(operation in ('observe','evaluate-privileged'), 'fixed context operation required')
    before = binary_identity()
    start=time.monotonic()
    try:
        raw, evidence = gate.bounded_process([str(BINARY),operation],gate.encode(request),timeout=5,cap=8192)
    except gate.ProcessError as error:
        error.evidence.update(operation=operation,request_sha256=gate.digest(gate.encode(request)),
            wall_seconds_hex=(time.monotonic()-start).hex())
        raise
    if log is not None:
        log(dict(operation=operation,request_sha256=gate.digest(gate.encode(request)),
            wall_seconds_hex=(time.monotonic()-start).hex(),process=evidence))
    try:
        gate.require(not evidence['stderr_b64'] and binary_identity() == before, 'oracle identity/diagnostics changed')
        return gate.decode(raw)
    except Exception as error:
        raise gate.ProcessError(dict(evidence,failure='context_protocol')) from error


def observe(request, log=None):
    result = call(request,'observe',log)
    gate.require(result == cases.public_observation('context',request), 'native public observation mismatch')
    return result


def target(request, log=None):
    return cases.verify_outcome('context',request,call(request,'evaluate-privileged',log))
