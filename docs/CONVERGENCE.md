# Training-only convergence development

Status: implementation of the next post-foundations roadmap increment. This is
a bounded learning-rate diagnostic, not Phase 5 intervention. The original
comparison null result and foundations blocked learning gate are preserved.
The new [protocol](CONVERGENCE-PROTOCOL.md) was frozen at
`696b251f13d2e5dc82478f7b02d7e346605f5264` before its target outcomes;
byte identity `sha256:1c55d1834ad712f3a4ba14c00a486c221d540d093ed959fd1cbfc29ad21c1faf`.

## Implemented contract

`convergence.py` composes the admitted foundations generator, shared native
optimizer/update loop, prompt-only train scorer, safe checkpoint serializer and
closed upstream observer. It changes no native model, quantizer, optimizer,
foundation data, prior protocol or historical evidence bytes. Requests bind the
new protocol, source closure, fixed runner identity, admitted corpora and exact
CPU environment. No caller-supplied shell or executable is accepted.

Each of 27 dense candidate cells retains task counts and training predictions
at four fixed milestones, teacher-forced training losses, preclip gradient
norms/clipping counts, full update history and a final replayable checkpoint.
The inherited `validation` checkpoint field is explicitly a **training-only**
diagnostic in this experiment; the elected restore wrapper reinstates that data
view. Intermediate accuracy receipts are custody-checked but not independently
recomputed from intermediate model checkpoints. Final training loss is recomputed
from the final tensors. Source-bound replay is available as a separate numerical
check; seeds alone do not guarantee portability.

The summary verifies all evidence, checks candidate pairing before publishing
any groups, and retains 27 failed/completed records even on worker timeouts or
late pairing mismatches. Descriptive 90% training checks never unlock test data,
select a learning rate or certify a useful model. No held-out stage exists in
the runner. Existing learning/geometry gates remain conditional.

Affected contracts: TRI-I02/03 (native initialization and admission), TRI-I06/07
(detached observation and immutable bindings), TRI-I08/11 (scoped replay and
training-only diagnostics), TRI-I12/13/14 (modules, failure retention and a
separate frozen protocol).

## Execute and verify

Acquire the [hash-locked CPU environment](GETTING_STARTED.md), then use a new
request and output path. The entire matrix is explicit, not part of pull-request
CI. Export/download its complete directory before discarding the runtime.

```sh
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_convergence.py --freeze-request /tmp/convergence-request.json > /tmp/convergence-freeze.json
CONVERGENCE_REQUEST_ID=$(python -c 'import json; print(json.load(open("/tmp/convergence-freeze.json"))["request_identity"])')
python scripts/run_convergence.py --request /tmp/convergence-request.json --request-identity "$CONVERGENCE_REQUEST_ID" --output /tmp/convergence
python scripts/run_convergence.py --verify-only --request /tmp/convergence-request.json --request-identity "$CONVERGENCE_REQUEST_ID" --output /tmp/convergence
```

`--verify-only` reads and verifies all closed bundles/checkpoints/receipts and
compares the existing canonical summary without writing to the run. A different
source/data/environment rejects the old request; review historical results with
their retained producer. Failed summaries retain their reasons but are not
reported as verified complete runs. The manual workflow always attempts artifact
upload after worker or aggregation failure; cancellation/runner loss can still
interrupt retention.

Required checks include root/upstream/CPU/comparison/foundations suites and
`python -m unittest discover -s tests/convergence -v` across supported Python
versions. New tests cover exact short interrupted replay, training-only data,
request mutation, real observed evidence corruption, timeout retention,
transactional pairing and negative candidate diagnostics. Synthetic passing
scores in tests are software fixtures, not measured model results.

## Remaining gate

After reviewing training behavior, freeze a new learning protocol with a stated
candidate/budget decision before any held-out evaluation. Training memorization
cannot establish held-out accuracy, reasoning, a superior weight format or
Phase 5 readiness. More data, longer runs, parameter scaling and geometry loss
are separate conditional increments.
