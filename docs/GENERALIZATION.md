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
