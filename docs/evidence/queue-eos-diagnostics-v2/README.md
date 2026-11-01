# Queue EOS diagnostic preparation evidence

Additive v2 preparation; see [diagnosis](../../QUEUE-EOS-DIAGNOSIS-V2.md) and
[proposal](../../QUEUE-LEARNING-V2.md). No optimizer steps or fresh held-out
targets. `training-forward.json` retains all initial/final training outputs,
parse status, split supervision, both host fingerprints and exact diagnostic
source hashes. This is foreign-environment forward inspection, not v1 restore
or numerical/optimizer replay. `foreign-restore-rejection.json` retains the
unchanged v1 loader's rejection and both fingerprints; no historical receipt
was changed.

`baseline-ci.json` records 72 successful main push workflows for the baseline.
`proposal.json`, `requests.json`, `overlap.json` and `fit-selection.json` bind
the prospective recipe and observation-only audit, not an execution freeze or
approval. `engineering-checks.json` distinguishes passed checks from the 13
unchanged Phase 5 host-security test failures; that broader suite did not pass
locally. `SHA256SUMS` pins the files here. Private conversation metadata is
excluded. Original v1 evidence remains in its existing fixture unchanged.

Reproduce the forward report with the hash-locked CPU environment and a fresh
external extraction of the original archive (see its retained README):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON" -I -B scripts/run_queue_diagnostics_v2.py \
  --evidence /absolute/extracted/original-evidence --out /absolute/new/report.json
PYTHONPATH=src:scripts "$PYTHON" -B -m unittest discover -s tests/queue_learning_diagnostics -v
```

All 9,201 original file hashes are checked first. Current v1 source closure must
match. New host observations may differ numerically; environment and source
identities are part of the report. CI runs synthetic contracts and exposed
observation overlap checks only; it does not invoke this retained-model command,
fit, freeze, pilot training or held-out model evaluation.
