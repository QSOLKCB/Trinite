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
