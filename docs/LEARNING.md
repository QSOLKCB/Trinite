# Matched learning and evaluation

This increment implements the [separately frozen protocol](LEARNING-PROTOCOL.md)
after the longer [training-only budget](BUDGET.md) cleared its exposed dense
training floor. It trains fresh dense, ternary and four-state models for 4,096
updates at constant 0.001 on the same arithmetic, fold and lattice facts. Geometry
loss remains zero. This is the learning gate before Phase 5, not a geometry
intervention or curriculum expansion.

`learning_data.py` completes per-item admission without importing new content.
Every row binds the generator revision, seed 31 and parent transformation lineage.
Its `ordering_identity` preserves the old example's exposure rank; texts, answers,
families and splits are unchanged. New content identities and admission manifests
live in `fixtures/learning-data-v1/`. Full regeneration and independent parent
oracle checks reject edited admissions even when their hashes are refreshed.
Historical fixtures and evidence retain their original producers.

The separate bounded `LearningPlan` reuses the native initializer, AdamW and update
loop. Diagnostic validation aliases training rows; it never selects a checkpoint.
The safe named-tensor checkpoint reader validates source, environment, request,
admission, model/lane/plan, exact exposure history and optimizer slots. Short
interrupted replays and parent-profile numerical equivalence cover every lane.

All 27 training workers terminate before one evaluation process freshly verifies
the entire matrix, regenerates each checkpoint's training predictions and loss,
checks every pair, and retains its decision. Every dense task must reach 90% exact
answer plus EOS across all seeds. A missing, failed or unmatched cell blocks all
held-out scoring. A persisted decision cannot authorize a later process.
Before each test call the selected checkpoint and training receipts are bound to
the freshly verified decision. Summary verification regenerates test predictions
and metrics from the elected models as well.

The held-out learning floor remains per-task 50% accuracy and ten percentage
points over the training-majority answer, for every dense seed. Summaries retain
all cells and failures, paired accuracy differences, wall-time ratios and latent
float32-byte ratios. Equal float32 storage does not imply packed savings.
Integrity completion can coexist with failed scientific learning gates. No
automatic format selection, Phase 5 objective or release unlock is implemented.

## Execute and verify

Use the hash-locked CPU environment from [GETTING_STARTED.md](GETTING_STARTED.md).
From the repository root:

```sh
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_learning.py --freeze-request /tmp/learning-request.json > /tmp/learning-freeze.json
LEARNING_REQUEST_ID=$(python -c 'import json; print(json.load(open("/tmp/learning-freeze.json"))["request_identity"])')
python scripts/run_learning.py --request /tmp/learning-request.json --request-identity "$LEARNING_REQUEST_ID" --output /tmp/learning
python scripts/run_learning.py --verify-only --request /tmp/learning-request.json --request-identity "$LEARNING_REQUEST_ID" --output /tmp/learning
```

Freeze requires the retained prior budget request and summary, whose exact
identities record the explicit budget choice. The runner also verifies its actual
source bytes. Changed sources require a new frozen request. Verification is
read-only and reruns current checks; it does not use retained verifier success.

Each training worker has a 600-second deadline, evaluation and summary have
1,200 seconds each. The manual workflow allows 360 minutes and always uploads
available artifacts. Worker timeouts continue through the full inventory. A
summary-worker failure retains its prior output and publishes a conservative
27-cell failed inventory marked unverified. Cancellation or runner loss can still
interrupt retention. This full experiment is manual, not part of routine PR CI.

## Original engineering failure

The first producer, `1c3792a842fe135d7f358669fd25762cdaba0abf`, completed all
27 training cells and cleared every dense training floor. Its evaluation guard
compared the upstream logical manifest identity with the hash of the complete
manifest file. Upstream excludes its self-identity field from the logical hash,
so these are different contracts. The guard rejected all 27 cells before any
held-out predictions were produced; the retained summary reports `failed`.

The corrected producer, `aad22ecce648a0daf9688b99faa79a07f5ddb664`, preserves
upstream logical identities and records the physical manifest file hash separately.
It requires unchanged manifest bytes across fresh upstream verification and binds
the selected training receipt to that physical hash before scoring. A regression
uses a real upstream bundle whose two identities differ. The scientific protocol,
runner and numerical update loop remain unchanged; changed source identities
require a fresh request and fresh full training matrix.

The [original failure inventory](../fixtures/learning-failure-v1/inventory.json),
[summary](../fixtures/learning-failure-v1/summary.json) and seven archive parts
retain that first request, all checkpoints, training evidence and logs, and its
exact producer. Its [read-only verification receipt](../fixtures/learning-failure-v1/verification.json) reproduces the original failed summary with that producer and changes no run files. No historical result or request was rewritten.

## Measured held-out outcome

The corrected matrix completed all 27 training and held-out evaluation cells.
Every dense training task passed its 90% floor before held-out scoring. Every
dense seed nevertheless fails at least one held-out task, so learning adequacy,
format selection and Phase 5 remain blocked. Geometry loss is still zero.

Training accuracy alone is insufficient here: dense fold reaches
1,087/1,150, 1,114/1,150 and 1,060/1,150. Ternary fold reaches 809/1,150,
904/1,150 and 918/1,150, failing its training floor in all three seeds;
four-state fold reaches 1,033/1,150, 952/1,150 and 1,036/1,150, passing only
seed 2. All lanes perfectly complete their arithmetic and lattice training rows.
The held-out unlock requires the dense training floor plus verified/pairable
records for every lane; it does not require quantized training adequacy.

Counts below require a complete greedy answer plus EOS. Means/ranges use the
frozen **family macro** metric; this differs from pooled accuracy for fold.

| Workload | Lane | Seed 0 | Seed 1 | Seed 2 | Family macro mean (range) |
| --- | --- | --- | --- | --- | --- |
| Arithmetic | dense | 0/64 | 1/64 | 1/64 | 1.04% (0.00–1.56%) |
| Arithmetic | ternary | 5/64 | 2/64 | 0/64 | 3.65% (0.00–7.81%) |
| Arithmetic | four-state | 0/64 | 0/64 | 0/64 | 0.00% (0.00–0.00%) |
| Fold | dense | 37/320 | 0/320 | 0/320 | 2.57% (0.00–7.71%) |
| Fold | ternary | 0/320 | 0/320 | 3/320 | 0.21% (0.00–0.62%) |
| Fold | four-state | 0/320 | 0/320 | 0/320 | 0.00% (0.00–0.00%) |
| Lattice | dense | 32/40 | 32/40 | 32/40 | 80.00% (80.00–80.00%) |
| Lattice | ternary | 32/40 | 32/40 | 32/40 | 80.00% (80.00–80.00%) |
| Lattice | four-state | 32/40 | 32/40 | 32/40 | 80.00% (80.00–80.00%) |

All four dense arithmetic tasks remain below the per-task held-out floor in
all seeds. Fold stays below 50%; seed 0 clears its baseline improvement margin
but still fails the accuracy floor. Lattice's 32/40 aggregate hides traversal
at 0/8 in every dense seed; address-to-role and role-to-address each score
8/8, and valid-address scores 16/16. This is why the
protocol requires every task rather than an aggregate alone.

The summary retains paired seed accuracy differences, training-wall ratios and
latent-byte margins. Ternary training-wall ratios range from 1.334 to 1.490 against dense; four-state
ratios range from 1.566 to 1.736. Both exceed the descriptive 1.25 margin in
every pair on this host. Fold seed 0 also misses the −0.05 quality margin for
both quantized lanes; the other paired quality margins pass. These timing and
quality margins do not supersede the failed dense learning gate. All lanes
still store the same float32 latent tensors;
byte ratio 1 is equal storage, not packed savings. Positive arithmetic results
in two ternary cells do not establish superiority: useful dense learning is a
prerequisite, there are only three seeds, and the visible fixtures are exploratory.
No lane passes the complete learning gate.

All nine corrected dense final model/optimizer payloads, full normalized
4,096-update histories and diagnostics match the retained prior budget exactly.
All 27 corrected payloads/histories also reproduce the original failed producer's
training run. These are scoped same-host checks. They demonstrate that the
manifest-receipt correction did not change this training trajectory; they do not
provide independent-host replication or fresh confirmatory evidence.

The next increment needs a separately frozen curriculum/generalization decision
before geometry intervention. More exposed training accuracy alone cannot close
this failed held-out gate. Preserve all previous null/blocked evidence and this
failure; do not tune this protocol against its held-out outcomes.

The finite corpora and their earlier outcomes are visible development material.
Family separation does not make this fresh confirmatory evidence. No claim of
general reasoning, quantum protection, packed storage advantage, independent-host
replication or release readiness follows from this increment. Game theory and
book candidates remain separate, untrained curriculum work.

## Complete evidence retrieval

The [inventory](../fixtures/learning-v1/inventory.json),
[request](../fixtures/learning-v1/request.json),
[summary](../fixtures/learning-v1/summary.json) and
[test decision](../fixtures/learning-v1/test-decision.json) accompany seven
lossless archive parts. The packet retains every final checkpoint, full update
history, prediction set, closed training/test PROVENANCE bundle and log, together
with the exact producer source/configuration/CPU lock and prior budget anchors.
This immutable review packet is a scoped exception to the small-fixture convention.

- Protocol commit: `c26d8535f97f70683c58200b58c3325eb5247d05`.
- Corrected producer: `aad22ecce648a0daf9688b99faa79a07f5ddb664`.
- Request: `sha256:4be53243cc1f561b37c5327cacd2f0d0976bdc066812713e76487432ba9af62e`.
- Summary: `sha256:e43e6670427aeb600677b90a7af281a2b8b88439f5671be385f62a784be99a59`.
- ZIP: `sha256:6faea186037dd540e92c54e4cf64cb9c407f0f9650f4033864f9f555cc98eb5d`, 54,095,183 bytes.
- Restored: 1,411 files, 437,080,232 bytes; 737 distinct payload objects.

[Parent parity](../fixtures/learning-v1/parent-parity.json) and
[corrected-run parity](../fixtures/learning-v1/corrected-parity.json) delimit the
same-host numerical checks. The [read-only verification receipt](../fixtures/learning-v1/verification.json)
records fresh archived-producer verification and an unchanged run-file inventory.
This verifies closed evidence, safe checkpoints and regenerated final predictions,
losses and summary; it does not independently replay every optimizer update.

```sh
cat fixtures/learning-v1/Trinite-learning-v1.zip.part{01..07} > /tmp/Trinite-learning-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-learning-v1.zip /tmp/learning-review --archive-identity sha256:6faea186037dd540e92c54e4cf64cb9c407f0f9650f4033864f9f555cc98eb5d
PYTHONPATH=/tmp/learning-review/producer/src python /tmp/learning-review/producer/scripts/run_learning.py --verify-only --request /tmp/learning-review/run/request.json --request-identity sha256:4be53243cc1f561b37c5327cacd2f0d0976bdc066812713e76487432ba9af62e --output /tmp/learning-review/run
```

Use the archived hash-locked CPU environment and one-thread settings from the
execution section. Transport verification checks the archive and every restored
file; numerical verification requires the exact environment. Review the original
failed packet similarly with its own inventory's archive/request identities and
its own archived producer. Changed sources require a new request; do not amend an
old request to attribute historical results to current code. Python may rebuild
producer bytecode caches outside run custody.
