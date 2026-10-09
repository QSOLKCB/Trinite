# Training-only budget development

Status: implementation of the next post-convergence increment. The [protocol](BUDGET-PROTOCOL.md)
was frozen at `0510c28d682b4d75824d9de25a17841e17e4fd79` before target outcomes;
byte identity `sha256:9e75f2f3642970e7ca56e81b65174d5f2794e07a008379cd0d621363c58d5087`.
This isolates four-times-longer exposure at constant 0.001 in nine dense cells.
It does not change the numerical model, native update loop, data or historical
protocols/results. Original plan construction limits stay unchanged.

`budget_training.py` owns frozen settings, source/request contracts and train-only state.
`budget_plan.py` explicitly validates the larger bounded resources and delegates
unchanged settings to the foundations plan validator. `budget.py` composes
admitted foundations inputs, shared train-only scoring, native AdamW updates,
checkpoint progress/tensor laws and closed PROVENANCE observation.
`budget_checkpoint.py` binds its distinct profile, train-only view, full plan,
request, source, data and environment. The shared named-tensor reader is factored
from the existing comparison restore into `restore_tensors`; comparison and
foundations restoration retain their existing schemas and size bounds.
No second model, optimizer, dataset generator or tensor inventory is introduced.

Every fixed milestone retains tensors/moments, safe metadata, consumed examples,
training predictions, losses and preclip/clipping diagnostics. Verification
restores **each** milestone, regenerates greedy train predictions and train loss,
and checks its history is the final history's prefix. This is stronger than
custody-only intermediate receipts; it is still not an independent full optimizer
trajectory replay. Short interrupted replay and prior-low-rate numerical prefix
checks are conformance tests. Descriptive training checks cannot unlock held-out
scoring, certify learning adequacy or select a candidate automatically.

## Execute and review

Use the [hash-locked CPU environment](GETTING_STARTED.md). These explicit full
runs are separate from PR CI and never evaluate held-out examples.

```sh
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_budget.py --freeze-request /tmp/budget-request.json > /tmp/budget-freeze.json
BUDGET_REQUEST_ID=$(python -c 'import json; print(json.load(open("/tmp/budget-freeze.json"))["request_identity"])')
python scripts/run_budget.py --request /tmp/budget-request.json --request-identity "$BUDGET_REQUEST_ID" --output /tmp/budget
python scripts/run_budget.py --verify-only --request /tmp/budget-request.json --request-identity "$BUDGET_REQUEST_ID" --output /tmp/budget
```

Requests reject changed source/environment/admission/protocol. Source receipts
check actual runner bytes, not just a constant. The runner fixes interpreter,
script and closed cell selectors; paths are separate argv entries, shell is
false and no caller-provided executable is accepted. Nine workers have bounded
720-second timeouts; all failures/logs remain and summaries retain nine records.
Aggregation is transactional. Read-only verification compares the retained
canonical summary without writing to the run. The manual workflow always
attempts retention; runner loss/cancellation may still prevent upload.

Required checks: root/upstream, existing CPU/comparison/foundations/convergence,
and `python -m unittest discover -s tests/budget -v` on Python 3.11–3.13. New
checks cover exact interrupted replay, old numerical prefixes, checkpoint
profile/data-view rejection, real closed evidence corruption, admission binding,
custody-valid false predictions, positive/failed/mismatched summaries, timeouts
and fixed subprocess/resource boundaries.

Affected requirements: TRI-I02/03/05/06/07/08/11/12/13/14. Native numerical
semantics, old frozen fixture identities and old evidence remain preserved.
The checkpoint reader refactor changes current source receipts; review historical
requests using their archived producer and freeze a new request for current runs.
Game theory and QEC/ETQ/UFT-ID research are separate curriculum/research additions;
none enters this frozen budget experiment.


## Measured outcome

All nine cells completed and the summary verified every retained milestone
checkpoint, regenerated training prediction set and training loss. There were
no worker/aggregate errors. **All nine final per-task training checks pass**.
This is training-only descriptive evidence on exposed finite fixtures; held-out
scoring, learning adequacy, Phase 5 readiness and release readiness stay false.

| Workload | Seed 0 | Seed 1 | Seed 2 | Seeds passing every training task |
| --- | --- | --- | --- | --- |
| Arithmetic | 264/264 | 264/264 | 264/264 | 3/3 |
| Fold | 1,087/1,150 | 1,114/1,150 | 1,060/1,150 | 3/3 |
| Lattice | 210/210 | 210/210 | 210/210 | 3/3 |

Counts require complete greedy answers and EOS. Every arithmetic task passes
separately; fold is above the frozen 90% boundary in all seeds. All nine first
1,024-update tensor payloads and update histories match the old low-rate cells
exactly; the longer budget preserved their numerical prefixes. A fresh full
4,096-step fold/seed-0 same-host trajectory reproduces final model/AdamW tensors,
metadata and history byte for byte. This scoped check is not independent-host
replication. [Prefix receipts](../fixtures/budget-v1/prefix-check.json) and
[replay receipt](../fixtures/budget-v1/replay.json) retain those checks.

The next increment is a separately frozen learning/evaluation protocol naming
this budget decision, matched lanes and exposure controls. No automatic selection
or held-out invocation occurs in this experiment. These positive training
observations do not overwrite the previous null/blocked results or show a
reasoning/geometry/game-theory benefit.

## Complete evidence retrieval

The [inventory](../fixtures/budget-v1/inventory.json),
[request](../fixtures/budget-v1/request.json) and
[summary](../fixtures/budget-v1/summary.json) accompany seven lossless binary
archive parts. They retain logs, every milestone's named safe checkpoint and
predictions/history, all closed PROVENANCE bundles, the exact measured producer,
CPU lock and additional prefix/replay receipts. This immutable evidence packet
is a scoped exception to the small-fixture convention, not an evolving-run store.

- Protocol: `0510c28d682b4d75824d9de25a17841e17e4fd79`.
- Measured producer: `917ebe82f05e3e37c1676f6ed7ad1684a9d76c96`.
- Request: `sha256:aec2fc5f307f79911606679286ac59a581afb07fa2adca8673b30e8be7fe8eae`.
- Summary: `sha256:95ad792069d993f5f24a98db3a353729e56c6df14a1bcbac2d18249e5550fb7e`.
- ZIP: `sha256:e202d25fdd8943d12aaf84d8d3c82a99adfd1dc577d9164f6e1e21e6110f927b`, 52,098,805 bytes.
- Restored: 606 files, 316,727,550 bytes; 401 distinct payload objects.

After measurement, frozen settings/state were separated into `budget_training.py`
to remove a lazy checkpoint-to-orchestration dependency. Explicit foreign-plan
rejection was added to budget/native checkpoint boundaries. Numerical updates
and the protocol are unchanged; current source identities differ. Freeze a new
request for current executions. Historical strict review uses the archived
producer, never a rewritten request pretending current code produced old results.

```sh
cat fixtures/budget-v1/Trinite-budget-v1.zip.part{01..07} > /tmp/Trinite-budget-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-budget-v1.zip /tmp/budget-review --archive-identity sha256:e202d25fdd8943d12aaf84d8d3c82a99adfd1dc577d9164f6e1e21e6110f927b
python /tmp/budget-review/producer/scripts/run_budget.py --verify-only --request /tmp/budget-review/run/request.json --request-identity sha256:aec2fc5f307f79911606679286ac59a581afb07fa2adca8673b30e8be7fe8eae --output /tmp/budget-review/run
```

The archive hash and reconstructed file inventories are checked before
publication. Strict numerical verification requires the archived locked CPU
environment; transport-byte verification alone does not certify numerical replay.

The published archive was restored and verified with its archived producer in the
locked CPU environment. Strict verification reproduced the original summary
identity. Restoration checked all 606 reconstructed file hashes; verification
added, removed or changed no run files. Python may rebuild producer bytecode
caches when loading the archived source; these are outside run custody.
