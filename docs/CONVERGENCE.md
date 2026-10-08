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
The inner checkpoint’s inherited `validation` field is explicitly a **training-only**
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

The retained measured producer is `067aa254d1d30aaa32abfae75083ea4dbd36f2a2`.
Subsequent verifier/runner hardening checks actual runner bytes on every source
receipt and requires all named admitted/request inputs to be closed retained
members. It also documents the audited fixed subprocess boundary. Those changes
do not alter the numerical loop; the original producer/request/evidence remains
immutable. Future runs on the hardened implementation require a newly frozen
request. Review the historical run using its archived producer, rather than
claiming the hardened source produced the original results.

The hardened implementation also wraps shared checkpoint metadata in an explicit
`trinite.convergence-checkpoint.v1` envelope binding the train-only data view,
candidate, protocol and source. It rejects a native foundations checkpoint and
prevents interpreting this experiment’s diagnostic history as held-out validation.
The historical producer retains its original checkpoint encoding; its numeric
tensors and all results remain unchanged.

## Measured outcome

All 27 cells completed and their retained evidence verified. There were no
worker or aggregate errors. The execution summary is completed; **no candidate
passes every training task across all workloads/seeds**. Held-out scoring,
learning adequacy, Phase 5 readiness and release readiness remain false. No
candidate was selected automatically.

The table reports final training exact answers, including EOS. Passing an
aggregate count is insufficient: the gate is checked separately for every task.

| Workload | Rate | Seed 0 | Seed 1 | Seed 2 | Seeds passing every task |
| --- | --- | --- | --- | --- | --- |
| Arithmetic | 0.001 | 250/264 | 245/264 | 235/264 | 0/3 |
| Arithmetic | 0.003 | 160/264 | 159/264 | 161/264 | 0/3 |
| Arithmetic | 0.006 | 70/264 | 110/264 | 118/264 | 0/3 |
| Fold | 0.001 | 664/1,150 | 860/1,150 | 615/1,150 | 0/3 |
| Fold | 0.003 | 597/1,150 | 742/1,150 | 697/1,150 | 0/3 |
| Fold | 0.006 | 36/1,150 | 135/1,150 | 336/1,150 | 0/3 |
| Lattice | 0.001 | 209/210 | 210/210 | 205/210 | 2/3 |
| Lattice | 0.003 | 202/210 | 205/210 | 204/210 | 0/3 |
| Lattice | 0.006 | 142/210 | 158/210 | 140/210 | 0/3 |

The lower rate improves all arithmetic and lattice seeds relative to the
reference, but arithmetic addition still fails the per-task gate. Fold improves
in two seeds and regresses in the third. The higher rate is worse in every final
workload/seed count. These are training observations on exposed finite fixtures,
not held-out results or evidence for a superior weight format. All nine reference
cells reproduce the previous foundations dense tensors and update histories
byte for byte; the training-only diagnostics preserved numerical updates.

The next decision is a separately frozen training-development budget/schedule
or curriculum increment; rate adjustment alone did not close the gate. Any
later learning/evaluation protocol must name its decision and retain these
failures, without treating validation/test labels as development targets.

## Complete evidence retrieval

[Inventory](../fixtures/convergence-v1/inventory.json),
[request](../fixtures/convergence-v1/request.json) and
[summary](../fixtures/convergence-v1/summary.json) are retained alongside the
complete lossless archive in five binary parts under `fixtures/convergence-v1`.
This immutable review packet is a scoped exception to the small-fixture rule,
not a location for evolving runs. It contains original logs, final named
checkpoints/moments, all four milestone predictions and histories, closed
PROVENANCE bundles, and the exact producer plus CPU lock. It contains no
held-out predictions. Duplicate byte payloads are stored once for transport;
upstream evidence bytes and serialization are unchanged.

- Producer: `067aa254d1d30aaa32abfae75083ea4dbd36f2a2`.
- Request: `sha256:753b2e4419abe6f4de459aa72daeffcec5bda4be06db1cfd29a22607f72108cf`.
- Summary: `sha256:9ea412a474cdf004f474fe0df8982dfcbb2fa56c156420fa25e862e3cad3215e`.
- ZIP: `sha256:915ec795a27919512ac93779e47833416b1637b6a84842a90712a3f265efd439`, 36,619,124 bytes.
- Restored: 1,063 files, 198,242,574 bytes.

Use the current repository's bounded restoration helper, then the archived
producer for strict review. Its Python/package/CPU environment must match the
request; another host can audit transport bytes but cannot claim exact numerical
replay by dropping environment checks.

```sh
cat fixtures/convergence-v1/Trinite-convergence-v1.zip.part{01..05} > /tmp/Trinite-convergence-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-convergence-v1.zip /tmp/convergence-review --archive-identity sha256:915ec795a27919512ac93779e47833416b1637b6a84842a90712a3f265efd439
python /tmp/convergence-review/producer/scripts/run_convergence.py --verify-only --request /tmp/convergence-review/run/request.json --request-identity sha256:753b2e4419abe6f4de459aa72daeffcec5bda4be06db1cfd29a22607f72108cf --output /tmp/convergence-review/run
```

The original and restored complete run both passed read-only bundle/checkpoint/
receipt verification with the same summary identity. Named request and admitted
input membership was additionally checked for all 27 bundles. This establishes
scoped retained-evidence agreement; independent full numerical reproduction and
fresh confirmatory evaluation remain pending.
