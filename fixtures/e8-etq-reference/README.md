# ETQ reference conformance expectations

These small reference fixtures are mathematical/software expectations, not
training examples, neural measurements or PROVENANCE eligibility receipts.

`donor-commitments.json` records seven pre-existing published commitments from
QSOLKCB/SONIFICATION's canonical ETQ-303 fixture at
`27cfea99a3bab445c74b1aba5c38993fe931f568`, including its exact path/Git blob.
The donor's documentation is v3.0.1; its event runtime remains v3.0.0 and its
preserved base is ETQ-101 v2.0.0.

`donor-runtime-comparison.json` records execution of the actual three unmodified
pinned donor modules under Node v24.19.0 and comparison with the new Python
finite map: exact canonical event bytes, complete graph and site registry, and
all seven commitments. These are cross-language conformance checks, not an
independent empirical replication or proof of neural origin/benefit.

Expected fixtures are never refreshed by test execution. Source and fixture
material are MPL-2.0, Trent Slade / QSOL-IMC. No donor material is admitted as
training data. See [the reference contract](../../docs/E8-REFERENCE.md).

The original v1 record above is preserved unchanged. Its assertions are now
checked by `scripts/check_etq_donor.py`, which executes the exact files in
`donor/src/` after verifying their Git blob identities. `donor/examples/` retains
the original canonical fixture (including its absence of a trailing newline),
and `donor/LICENSE` retains the upstream licence. These copies were acquired at
the pinned commit and checked against its Git tree; no donor source was edited.

`donor-runtime-evidence-v2.json` is retained unchanged. It bound those files and
stdout but did not bind the live Node version, and its old verifier normalized
record bytes before comparison. It is superseded by the separately authored
`donor-runtime-evidence-v3.json`, which binds the current comparison source,
Node driver, all original records including v2, exact stdout, and live Node
version. `donor-runtime-output.json` remains the identical stdout under Node
v24.19.0. The driver invokes the upstream algorithms and serializer;
the verifier compares full payloads and all seven published hashes to Python.
Tests require Node and run in the separate ETQ donor CI job:

```sh
PYTHONPATH=src python scripts/check_etq_donor.py
PYTHONPATH=src python -m unittest discover -s tests/etq_upstream -v
```

Verification executes the donor anew and never refreshes these files. Missing
Node, a version mismatch, or inconsistent source, records or output is a failure.
The v3 record is compared as exact canonical bytes (one final LF), while v2 must
be canonical and match its v3-bound byte identity. Neither whitespace changes
nor a foreign runtime with identical output can verify against this evidence.
Failed version probes, donor exits and timeouts surface captured stdout/stderr
and exit/timeout details in the caller's error log, preserving arbitrary bytes
as escapes. The v3 record is new conformance evidence, not a reconstruction of
the original invocation or SONIFICATION's full implementation/custody receipt.
