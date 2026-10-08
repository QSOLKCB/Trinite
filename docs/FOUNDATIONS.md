# Foundations learning-adequacy increment

This implements the first repository-informed roadmap increment before Phase 5:
short arithmetic/fractions, integer folds and symbolic lattice tasks. The
[protocol](FOUNDATIONS-PROTOCOL.md) was published at
`d85743aae0e732006c9a44d25c3455a4eee6cee0` before model outcomes;
its identity is `sha256:16063ebcb2f918b6c6bee9074f5949967c8fca4afd20a4245bf65e12789ca34d`.
Geometry loss remains zero. Phase 5 intervention remains pending.

## Data and admission

| Workload | Examples | Train / validation / test | Families | Answers |
| --- | --- | --- | --- | --- |
| Arithmetic | 392 | 264 / 64 / 64 | 10 | Add, subtract, multiply, reduced fractions, division-by-zero `ERR` |
| Fold | 1,550 | 1,150 / 80 / 320 | 15 | Count, sum, sum of squares |
| Lattice | 270 | 210 / 20 / 40 | 27 | Role/address conversion, validity, fixed traversal |

These are newly generated minimal formal facts. HERESY-API, CONSTRAINT-SHIFT and
LATTICE are pinned topic references in [RESEARCH.md](RESEARCH.md); no donor prose,
code or visible fixtures enter this corpus. Admission records identify the
commissioned author, exact generation/verification/rendering procedure, numeric
rights basis, source-hash supporting references, complete formal item evidence,
reviewer/date, scope and limitations. Software remains MPL-2.0. This is a scoped
automated admission, not legal certification or blanket permission to ingest
repositories. Standalone admission validation regenerates this exact policy.

`foundations_data.py` generates labels using explicit integer procedures;
`foundations_oracle.py` independently checks them using rational arithmetic,
frequency aggregation and closed-form traversal. A separate parser verifies both
carriers against the formal items. Retained files live in
`fixtures/foundations-data-v1`. Full audits recompute labels, prompt agreement,
source receipts, admission, tokenizer bounds, identities, deduplication and
semantic-family splits. Related carriers, signs, permutations, zero insertions
and lattice corruptions stay together. All tasks and finite fixtures are exposed;
family-held-out results are exploratory and do not establish fresh reasoning.

## Numerical and evidence contracts

The separate `FoundationsPlan` permits at most 1,024 updates, batch size 8,
131,072 scored targets, 100,000 parameters and 120 seconds per update invocation.
The frozen run uses 86,528 parameters, context 64 and constant AdamW 0.003.
`training.py` remains the authoritative numerical update loop. Native
`RunConfig` limits and existing dense/ternary/four-state definitions remain
unchanged. All lanes share initialization, inputs, schedule and actual target
budgets per workload/seed. Validation loss selects nothing.

Checkpoints retain every update/validation receipt, named float32 model tensors
and AdamW moments in bounded safetensors/JSON. Restoration rejects inconsistent
model/plan/lane/workload, source/data/environment/request changes, invalid
progress and missing/extra/nonfinite tensors. Checkpoints replay bit for bit.

Training scores use prompt-only greedy generation with a fixed workload cap and
require the complete answer plus EOS. The training-only majority-answer baseline
never examines test answers. Every retained prediction is checked against the
complete admitted scoring set and scores are recomputed during verification.
Closed training/test bundles use the existing pinned upstream PROVENANCE
serializer and verifier. Verification also binds names to consumed byte copies,
restores the elected checkpoint and checks report/history/resource agreement.
Observer execution occurs after numerical computation; there is no verifier
cache or alternative serializer.

All 27 fresh training workers finish before an immutable test decision is
retained. All must verify and pair correctly, and each of nine dense cells must
score at least 90% on every training task. Failure blocks held-out scoring for
all lanes. Eligible test workers reverify the decision and its training receipts
before loading or scoring; a forged eligibility flag cannot unlock evaluation.
Held-out adequacy additionally requires each seed/task to score at least 50% and
10 percentage points above its training-majority baseline. Paired quality, wall
and latent-float32-byte diagnostics are emitted only after every cell and metric
validates. Any late mismatch retains all 27 records with no partial groups.
Equal float32 storage is not packed savings. `phase5_ready` stays false.

## Run and verify

Acquire the CPU lock as described in [GETTING_STARTED.md](GETTING_STARTED.md).
Use a new request file and output directory for every execution:

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests/foundations -v
python -m trinite freeze-foundations /tmp/foundations-request.json
# Paste the exact printed request identity below.
python scripts/run_foundations.py --request /tmp/foundations-request.json --request-identity 'sha256:<64 hex digits>' --output /tmp/foundations-run
python -m trinite verify-observation /tmp/foundations-run/arithmetic-0-dense/train-provenance
```

The manual `foundations.yml` workflow provides a 240-minute envelope for at most
54 workers at 240 seconds each plus setup/aggregation/upload. Every timeout,
nonzero exit and blocked gate is retained. A blocked or inadequate run exits 1;
this is not silently converted into a successful experiment. Artifact upload
runs on failure and is kept 14 days. Cancellation/runner loss can still prevent
retention and needs explicit disclosure.

The implementation has separate standard-library data/policy tests and locked
CPU tests for all-lane replay, request binding, scorer/EOS rules, exact gates,
forged decisions, late mismatch retention, worker timeouts and actual retained
bundle tampering. CI runs these alongside the original required suites on
Python 3.11–3.13. Measurement and archive status are recorded below after the
first frozen execution; the old comparison remains unchanged.

## First frozen measurement

The complete 27-cell matrix was produced by commit
`2482868e039a483a5a26342de07f793875463597`, after the separate protocol freeze.
Request: `sha256:cec893221e4f07034e747a231e46fb0ee10ae12c3209fe6c303737fadf35dfc6`.
Summary: `sha256:7b8e35f061e3728f1bea4c011d41787bc637a65af9591bd4d89b27d069d2ef21`.
All 27 workers completed; all 27 upstream closed bundles and elected
checkpoints/prediction scores verified, with no worker or aggregate errors.
Every cell completed 1,024 updates. Actual target counts matched within every
seed/workload pair: arithmetic 21,099–21,101; fold 55,909–55,925; lattice
42,596–42,601. Each lane retained 346,112 latent float32 bytes. These are not
packed inference measurements.

The dense gate **failed**: all nine dense cells had at least one training task
below 90%. Outcome is `blocked`, with all 27 training records retained, no
held-out scoring or test workers, no paired quality groups, and false format
selection/Phase 5/release flags. The runner's exit code 1 is the required
scientific gate result, not an unreported worker failure.

Training complete-answer-plus-EOS counts below are descriptive. Aggregate
accuracy does not override per-task failure.

| Workload / lane | Seed 0 | Seed 1 | Seed 2 | Cells passing every training task |
| --- | --- | --- | --- | --- |
| arithmetic / dense | 160/264 | 159/264 | 161/264 | 0/3 |
| arithmetic / ternary | 254/264 | 250/264 | 215/264 | 2/3 |
| arithmetic / four-state | 263/264 | 222/264 | 244/264 | 1/3 |
| fold / dense | 597/1150 | 742/1150 | 697/1150 | 0/3 |
| fold / ternary | 595/1150 | 753/1150 | 635/1150 | 0/3 |
| fold / four-state | 592/1150 | 742/1150 | 708/1150 | 0/3 |
| lattice / dense | 202/210 | 205/210 | 204/210 | 0/3 |
| lattice / ternary | 203/210 | 210/210 | 203/210 | 1/3 |
| lattice / four-state | 202/210 | 206/210 | 199/210 | 1/3 |

The dense failures were all four arithmetic tasks at all three seeds; fold at
all three seeds; and lattice traversal at all three seeds (34/42, 37/42 and
36/42). Lattice address/role mappings and validity achieved 100% dense training
accuracy, illustrating why a global score cannot substitute for each-task
adequacy. Some quantized arithmetic/lattice cells learned their training tasks,
but this supplies no held-out superiority or capability conclusion. The original
zero-accuracy comparison remains intact.

No hyperparameter or budget was tuned after these outcomes. A subsequent
training-only development increment should investigate per-task convergence and
update stability, then freeze a new budget/protocol before another final
evaluation. Do not unlock test scoring selectively for the successful cells.
Independent replication and the hosted full matrix remain pending.

## Complete evidence retrieval and replay

The [complete archive parts](../fixtures/foundations-v1/)
assemble to `34,586,245` bytes with identity
`sha256:b494c4b39962892f205f378f572aac8f65e49b6f43f7d14c33f62c6a030c7aa2`. [inventory.json](../fixtures/foundations-v1/inventory.json)
records its request/summary/decision/member-inventory identities. All
504 physical ZIP members store repeated byte payloads once. The tested restore
reconstructs all 819 original/review files exactly; all original run and
producer source bytes were compared with the complete original archive.
This is a lossless transport convention, not a replacement for upstream
PROVENANCE canonical JSON, identity, events or verification.
`members.json` lists every other member's identity and size; the archive includes
every original run path, worker log, checkpoint/moment/history, prediction,
closed training bundle and elected producer source plus the CPU lock/licenses.
Transport limits require five binary-pinned parts; inventory records each part's size/hash and their exact assembly order. The assembled ZIP identity must
match before restoration. This is a public review archive, not a release or
independent replication.

Read-only replay reverified all 27 closed training bundles, restored checkpoints,
recomputed scores from retained predictions and reproduced the blocked summary
byte for byte. It does not generate new model outputs. To repeat with the strict
original environment, extract to a new directory and explicitly use the
archived producer sources:

```bash
cat fixtures/foundations-v1/Trinite-foundations-v1.zip.part{01,02,03,04,05} > /tmp/Trinite-foundations-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-foundations-v1.zip /tmp/foundations-review --archive-identity 'sha256:b494c4b39962892f205f378f572aac8f65e49b6f43f7d14c33f62c6a030c7aa2'
export PYTHONPATH=/tmp/foundations-review/producer/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python /tmp/foundations-review/producer/scripts/verify_foundations_run.py /tmp/foundations-review/run --request-identity 'sha256:cec893221e4f07034e747a231e46fb0ee10ae12c3209fe6c303737fadf35dfc6' --summary-identity 'sha256:7b8e35f061e3728f1bea4c011d41787bc637a65af9591bd4d89b27d069d2ef21'
```

Acquire the hash-locked dependencies first. Strict numerical replay additionally
binds Python, CPU/platform and backend settings; a different environment must
reject, not edit the original request. Byte-only upstream bundle verification
can still be performed separately. The read-only review helper was added after
measurement and is not claimed as a numerical producer.

A subsequent validation pass rejects a native plan masquerading as a foundations
checkpoint and contradictory aggregate/task score counts or nonfinite family
metrics. Admission comparison also rejects numeric/boolean substitutions using
canonical byte equality; the proposed fixtures were deliberately regenerated
and audited after that source change. Their numeric examples and family/order
identities are unchanged, while admission/source receipts change. These guards do not alter the frozen numerical protocol. Historical
measurement remains attributed to the exact archived producer above; current
source runs require a newly frozen request. A full matrix repeat on the current
head remains pending, alongside independent replication. The standard-library restore tool checks archive identity, complete object
inventory, payload hashes/sizes and safe paths before publishing a new directory.
Existing destinations, changed/missing/extra payloads and path/case collisions
reject. Restore never executes archived code.

Restore compares every file ancestor against the complete case-folded file
inventory, independent of index order. A file `run/A` and descendant `run/a/b`
reject before staging on every platform. Use the current repository restore
tool in the commands above; the immutable archive retains its historical review
helper. This guard does not change archived evidence or scientific settings.

CI exercises real
training/bundle verification plus explicitly synthetic test-stage conformance;
those test fixtures are not empirical held-out model results.
