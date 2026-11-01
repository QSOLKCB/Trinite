# Context-symbol validation runtime repair

Engineering-only producer change after the [stopped evidence deposit](evidence/context-symbol-learning-v1/REPORT.md).
Affected requirements: TRI-I07/I08/I11/I13/I14. The native updater, tensor reader,
strict runtime guard, tokenizer, schedule and numerical semantics are unchanged.

`workers.validate_fit` initializes the existing deterministic CPU runtime before
restoring a checkpoint. The independent `_validate-fit` subprocess previously
reached `require_environment` without enabling process-local deterministic
algorithms. A configured training process cannot configure a fresh interpreter.
The initialization belongs to the validation boundary, so direct validation
callers also receive the same setup. It does not relax environment matching.

## Regression and limits

`tests/context_symbol_learning_v1/test_validation_process.py` creates temporary,
current-source-bound engineering inputs: random initial native weights, genuine
step-zero metadata, empty AdamW state, and a training-only packet with no heldout
observations. It runs the actual supervisor and fixed launcher. Correct behavior
is successful guarded native restoration followed by **rejection** for an
incomplete checkpoint/attempt budget; no fit certificate may be written.

A separately supervised fresh launcher invocation uses a Python call profiler
(not replaced runtime functions) to observe the initial disabled deterministic
setting, actual configuration, successful strict guard and native tensor restore.
It forbids optimizer/update-loop entry, checks zero calls, restored step/token
counts of zero, empty optimizer state and unchanged model identity. All input
hashes remain unchanged. Tamper cases retain anchor, source, environment, binding
and safe-tensor failures. Unsupervised worker access and an unconfigured direct
native guard still fail. No mocked configure, guard or restore certifies this path.

The focused test fails on the parent producer with the strict-runtime error,
then passes with this repair, in normal and optimized Python. The existing
[engineering CI workflow](../.github/workflows/context-symbol-learning-v1.yml)
discovers the regression in both modes:

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/context_symbol_learning_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/context_symbol_learning_v1 -v
```

These checks establish subprocess initialization and fail-closed restoration,
not a successful final-fit gate, learning result or optimizer replay. The stopped
run, evidence deposit, historical fixtures and frozen protocols remain unchanged.
No training, resume, scientific feasibility/freeze, heldout scoring, fresh attempt
or research/release gate advancement is authorized by this repair. A subsequent
scientific attempt requires a new freeze and explicit owner approval.
