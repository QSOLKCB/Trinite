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

## Evidence status

The protocol was published before target outcomes at
`c26d8535f97f70683c58200b58c3325eb5247d05`. The complete matched run and archived
producer verification are pending at the implementation commit. Conformance
fixtures and synthetic gate tests do not supply empirical learning evidence.

The finite corpora and their earlier outcomes are visible development material.
Family separation does not make this fresh confirmatory evidence. No claim of
general reasoning, quantum protection, packed storage advantage, independent-host
replication or release readiness follows from this increment. Game theory and
book candidates remain separate, untrained curriculum work.
