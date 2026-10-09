# Generalization diagnostics and scalar curriculum preparation

This increment follows the corrected 27-cell [learning matrix](LEARNING.md).
It implements the [frozen diagnostic protocol](GENERALIZATION-PROTOCOL.md), not
Phase 5 geometry intervention or a new training experiment. The preceding dense
models passed exposed training checks but failed held-out learning gates in every
seed. Their original negative results and both learning packets remain intact.

## What is implemented

`generalization.py` first executes the existing learning summarizer read-only.
Every closed training/test bundle, named checkpoint, final prediction/loss and
pairing must reproduce the exact pinned parent summary. Only then does it count
errors and coverage across every lane/seed/task and both textual carriers.
It checks the entire parent-file inventory before and after analysis.

Complete-answer/EOS results remain authoritative. Supplemental fold component
and traversal-coordinate counts describe which parts of complete canonical
outputs match; malformed or unfinished outputs receive zero component credit.
Vocabulary coverage is descriptive and does not establish the cause of failure.
The new report and full candidate outputs are bound by the pinned PROVENANCE
observer. Reverification regenerates the diagnostic and reruns the actual
upstream evidence verifier; retained success is not used as authorization.

`curriculum_data.py` prepares independent scalar count, sum and squares tasks,
and traversal value/x/y/z tasks. Arithmetic and non-traversal lattice prompts
and answers stay unchanged as controls. Child tasks and both carriers inherit
original family splits. No held-out fact moves into training. Every item records
its generator revision and parent example/semantic identities. An independent
histogram/modular oracle and prompt parser check procedural generator labels.

| Prepared workload | Train | Validation | Test | Change |
| --- | ---: | ---: | ---: | --- |
| Arithmetic | 264 | 64 | 64 | Unchanged numeric tasks/texts; new admission lineage |
| Fold | 3,450 | 240 | 960 | Three scalar projections for each original carrier |
| Lattice | 336 | 32 | 64 | Four scalar traversal projections; other tasks unchanged |

These are visible development examples. More rows are correlated projections,
not new independent samples. The [retained manifests](../fixtures/scalar-curriculum-v1/)
pin the exact generated dataset bytes without committing another evolving data
or model archive. No books, donor prose, game corpus, teacher outputs or new
numerical domains enter this increment.

## Execute and verify

The parent matrix requires its exact recorded CPU/source/environment fingerprint
for numerical verification. A different CPU, Python, kernel or package state
rejects. Routine CI exercises contracts and small local examples on each Python
lane; it does not report independent-host reproduction of this retained matrix.
Use the hash-locked CPU setup in [GETTING_STARTED.md](GETTING_STARTED.md).

Restore the lossless parent packet with the commands in
[LEARNING.md](LEARNING.md#complete-evidence-retrieval). Do not execute
arbitrary archive code; this increment calls the current unchanged authoritative
learning modules, whose bytes must match the parent request. Then, from the
repository root, with the restored parent's `run/` path:

```sh
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_generalization.py --freeze-request /tmp/generalization-request.json > /tmp/generalization-freeze.json
GENERALIZATION_REQUEST_ID=$(python -c 'import json; print(json.load(open("/tmp/generalization-freeze.json"))["request_identity"])')
python scripts/run_generalization.py --request /tmp/generalization-request.json --request-identity "$GENERALIZATION_REQUEST_ID" --run /tmp/learning-restored/run --output /tmp/generalization
python scripts/run_generalization.py --verify-only --request /tmp/generalization-request.json --request-identity "$GENERALIZATION_REQUEST_ID" --run /tmp/learning-restored/run --output /tmp/generalization
```

The new output directory contains the request, report, complete scalar datasets
and manifests, closed evidence, verification and worker stdout/stderr. One fresh
worker has a 1,200-second deadline. Failed or timed-out workers retain available
partial outputs and false eligibility gates. Read-only verification changes no
parent or diagnostic files. Stable local directories are required.

Scalar preparation itself needs only the standard library and can be regenerated
on another host without a model call. For example:

```sh
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from trinite.curriculum_data import dataset, audit_bytes
root = Path('/tmp/scalar-curriculum'); root.mkdir(exist_ok=False)
for workload in ('arithmetic', 'fold', 'lattice'):
    directory = root/workload; directory.mkdir()
    raw, manifest = dataset(workload); audit_bytes(raw, manifest)
    (directory/'dataset.json').write_bytes(raw)
    (directory/'manifest.json').write_bytes(manifest)
PY
```

## Retained diagnostic observations

The complete run passed fresh authoritative verification for all 27 parent cells.
All **1,306 parent files** retained the same content inventory through analysis.
The pinned upstream observer closed the new evidence, including full scalar data.
The following counts are descriptive breakdowns of the already visible parent
outcomes; they are not a new model experiment or causal attribution.

Every dense held-out output ended with EOS and had a strict ASCII body. Failed
answers were wrong-answer cases, not missing-EOS or misplaced-special-token cases.

| Dense fold seed | Complete answer/EOS | Count component | Sum component | Squares component |
| --- | ---: | ---: | ---: | ---: |
| 0 | 37/320 | 319/320 | 114/320 | 61/320 |
| 1 | 0/320 | 311/320 | 37/320 | 0/320 |
| 2 | 0/320 | 317/320 | 107/320 | 0/320 |

Dense traversal stayed **0/8** in every seed; its z coordinate was also **0/8**.
The other lattice tasks remained perfect, giving the historical 32/40 aggregate.
Arithmetic complete answers stayed 0/64, 1/64 and 1/64. None of these component
counts rescues the original per-task learning gate.

All held-out prompt bytes were present in the same task's training prompts.
Every held-out fold answer tuple was absent from training. Division and
multiplication held-out exact answers were absent from their task's training
answers; addition and subtraction answers were present. Lattice conversions
succeeded on novel complete answers, so exact-answer novelty alone cannot
explain all observed failures. These counts motivate examining scalar targets;
they do not establish that the preparation improves learning.

The [retained report and evidence inventory](../fixtures/generalization-v1/)
include the frozen request, upstream verification and a 1,345,178-byte lossless
packet containing all new outputs plus the exact diagnostic source closure.
The separately retained corrected learning packet is also required for numerical
replay. This is a bounded evidence-review exception to the small-fixture rule;
no evolving training directory enters git.

Protocol commit: `9924e82047b728e229094c37f5714bb268ebe6a3`.
Measured source producer: `5664b48a00df2cbf510828b7c866db076f51cffc`.
Report identity: `sha256:f0b802442f4fe5d81a3b7faaeb3086d6989ff1d08fcb918cdc14612274602ef5`.

Restore the new packet into a fresh destination:

```sh
python scripts/restore_foundations_archive.py fixtures/generalization-v1/Trinite-generalization-v1.zip /tmp/generalization-restored --archive-identity sha256:770d293d715c6b68f9ff1bf43ed81219722e2f0a74ec82a5726bcfeac7339c57
```

Use the exact source producer and recorded environment when calling read-only
verification; later source changes require their own new request. The archived
`producer/` is the diagnostic execution closure, not the entire repository or
the independently required old matrix. Its `COMMIT` label alone is not source
verification: check the actual files against the request's content identities
and the declared commit before execution.

## Completion and remaining gates

The implementation, retained diagnostic observations and validation are reported
with this PR. The scalar corpus is prepared and admitted within its exact scope;
**it has not been trained**. A separate frozen matched training budget, exposure
schedule, safe checkpoint profile and per-task learning protocol must precede
that experiment. Supplementary diagnostics cannot grant learning adequacy,
format selection, Phase 5 or release readiness. Geometry loss stays zero.

Affected contracts: TRI-I03, I07, I08, I11, I12, I13 and I14. The model, tokenizer,
quantizers, optimizer, old scientific protocols, learning thresholds, historical
fixtures and archived producers are unchanged. No larger-model or general
reasoning claim follows from these diagnostic counts.

A second fresh read-only reproduction regenerated the exact report and verified
both the parent matrix and new closed evidence again. All 37 new output files
still matched the restored packet byte for byte. The archived procedure also
passed request/source/environment conformance and fresh upstream verification.
[Reproduction receipt](../fixtures/generalization-v1/reproduction.json) and
[restored-packet checks](../fixtures/generalization-v1/restored-verification.json)
state their separate scopes. They do not replay every original optimizer update.

### Retained verification receipt correction

The measured producer reran the upstream verifier but did not compare the
published `verification.json` receipt. The current verifier requires that file
to match `json_bytes(verification)` from the fresh result exactly; missing,
modified or noncanonical receipts reject. Regression checks reproduce all three
failures in the earlier implementation and check successful read-only verification
after restoring the original receipt. This closes the retained-record check under
TRI-I07, I12, I13 and I14. Historical requests, reports and archived producer bytes
remain unchanged; execution with the changed source requires a newly frozen request.
The diagnostic counts and blocked gates are unchanged.

Local validation: 101 standard-library tests passed; all 122 CPU research cases
passed in each of six characterization runs with identical ordered identities;
upstream ran 51 checks with its unprivileged-permission case skipped locally.
All six hosted Python 3.11–3.13 jobs passed, covering all 274 current cases per
Python version including that permission case. Relative documentation links,
anchors, module boundaries and whitespace checks also passed.

The later [matched scalar experiment](SCALAR-LEARNING.md) trains this exact
preparation under a new [protocol](SCALAR-LEARNING-PROTOCOL.md). Its scalar-only
gates do not change the historical diagnostic counts or composed learning result.
