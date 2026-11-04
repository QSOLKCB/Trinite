# TT-003-DIAG-001 — Frozen teacher behaviour audit

**Status:** source-bound, train-only, read-only diagnostic implementation. No teacher/student training, optimizer updates, model modification, CUDA, hidden corpus, or scientific generalization claim. Original TT-003-EVIDENCE-001 checkpoints and historical evidence remain unchanged.

## Objective and immutable target

TT-003-EVIDENCE-001 completed 32/32 CPU updates on the original 90,624-parameter dense teacher. All eight retained train-only diagnostic outputs were token IDs [48,257] (zero plus EOS); two of eight were exact. Training loss fell from 0x1.62bb2745d1746p+2 to 0x1.fcf63b45d1746p+0. That is a successful training pipeline and a negative narrow generation result, not learned compositional reasoning. This diagnostic tests whether the same failure persists across **all 972 admitted TRAIN examples**, splitting **128 visited** from **844 unvisited** training examples, and comparing the initial seed-17 model against the frozen final checkpoint.

The exact identities in configs/teacher/tt003-diag-001.json are mandatory and separately enforced by source code. This protocol targets original Trinite commit 6f2ec05b407a3bafe7a5ed947fd9caadc4be65f7 and the owner-supplied verified handoff, with TT-002 manifest b90287c3549e2a877b4e23325b51f234f3e54eab3a48c0227f34c5f18d065687, original TT-003 request 72e9342f769c957c0d8c690d528bebb7fa3500cb96c7168e8d20aa32123fcd44, run manifest a74aec8d610a678e2e2fee0c32ece11dd1e1fcc8fa53d8e8da6db40d3ef73004, and checkpoint ec0e42658818fb9fb1712f11117eebc3c5adf101aa668e335ab788b5e2f6a551. All hashes in configuration are prefixed with sha256:. No model weights or personal handoff records are committed to this repository.

## Procedure

First **freeze** a new diagnostic request after cold source/input checks, complete TT-002 source-specific dataset regeneration, historical TT-003 member and rights identity comparisons and exact 32-step/128-visit history validation. Retain its SHA-256 independently outside the request folder. No Torch or optimizer is used during request freezing.

Second **audit** under the original pinned Linux x86_64 Torch 2.8.0+cpu environment, one intra-op thread, strict deterministic algorithms. Invoke TT-003's original read-only final-state verification; load the same closed safetensors through the unchanged native reader, recreate the seed-17 initial model without training, check both model identities, and reproduce the original eight training-only outputs. Perform uncached raw greedy generation for each of the 972 TRAIN rows with at most five generated tokens for each initial/final model. Preserve all raw token IDs, EOS presence, format validity and exactness, no answer-length hints or repaired predictions. Never score validation/test families or copy privileged targets into numerical inference.

Third compute exact count-only breakdowns by visited/unvisited cohort, scalar check versus vector apply/inverse, circuit, scalar label and crossed cohort/task. Distinguish exact answer plus EOS, valid-format wrong answer, premature EOS, missing EOS, excessive length and invalid symbols. Report most common raw sequences, number of distinct outputs and row-level initial/final improvements and regressions. This entire partition is **publicly exposed TRAIN data**, so an unvisited row is not a scientific holdout.

Fourth retain raw predictions.jsonl, summary.json, input request and independently anchored manifest.json in a **fresh directory outside the repository**. A failed/incomplete process leaves partial data/failure.json, not a valid success manifest. After completion, read-only **verify** recomputes source/row/schedule identity, classifications and every summary count, but does NOT replay native neural inference. An optional separately invoked **replay-forward** reruns the complete initial/final native forward pass and checks raw outputs byte-for-byte. Neither mode executes any optimizer step. Forward parity is not independent optimizer replay.

## Owner-local commands

Use a trusted checkout of the reviewed diagnostic PR on top of the original TT-003 code, and the existing pinned CPU environment. Extract the earlier handoff archive into a new private folder with careful path validation; do not run scripts stored in the archive or import its extracted source. Use only the reviewed checkout source.

~~~sh
python3 scripts/run_teacher_diag.py freeze \
  --dataset /path/to/handoff/dataset \
  --teacher-run /path/to/handoff/run \
  --out /path/to/new/diag-request

# Retain the emitted request_identity independently:
python3 scripts/run_teacher_diag.py verify-request \
  --request /path/to/new/diag-request \
  --identity sha256:RETAINED_REQUEST \
  --dataset /path/to/handoff/dataset \
  --teacher-run /path/to/handoff/run

python3 scripts/run_teacher_diag.py audit \
  --request /path/to/new/diag-request \
  --identity sha256:RETAINED_REQUEST \
  --dataset /path/to/handoff/dataset \
  --teacher-run /path/to/handoff/run \
  --out /path/to/new/diag-result

# Retain the emitted manifest_identity independently:
python3 scripts/run_teacher_diag.py verify \
  --request /path/to/new/diag-request \
  --identity sha256:RETAINED_REQUEST \
  --dataset /path/to/handoff/dataset \
  --teacher-run /path/to/handoff/run \
  --result /path/to/new/diag-result \
  --result-identity sha256:RETAINED_RESULT

# Optional complete native forward replay:
python3 scripts/run_teacher_diag.py replay-forward \
  --request /path/to/new/diag-request \
  --identity sha256:RETAINED_REQUEST \
  --dataset /path/to/handoff/dataset \
  --teacher-run /path/to/handoff/run \
  --result /path/to/new/diag-result \
  --result-identity sha256:RETAINED_RESULT
~~~

No new training approval is required for inference-only use of the previously authorised local checkpoint. The audit has a cooperative 1,800-second time limit, checked between completed rows, not a hard OS kill limit. A local OS supervisor may enforce a hard limit while preserving partial evidence. No retry or adaptive sampling; any incomplete result must be retained and reported accurately.

## Tests, rights and interpretation limits

CI exercises the strict source, classifier, schedule and frozen-request contracts under normal and optimized Python; a separate pinned CPU job performs a tiny forward-only canary on a freshly initialized model **without using the owner's actual checkpoint**. The full 972-row target inference is not performed in GitHub Actions and must not be claimed as completed based on a green CI run.

TT-003 native model/source/optimizer, TT-001/TT-002 datasets, QEC and original experiment evidence are unchanged. Source code remains MPL-2.0. Dataset distribution rights and produced weights licence remain separate and undecided; the original weights are not authorised for publication. The diagnostic does not verify neural comprehension, held-out generalisation, source authenticity, machine identity, cross-host determinism or optimizer replay.

**Next gate:** Interpret actual full-train observations before proposing a separately frozen fit-first / EOS-balanced training intervention. Do not enlarge the model, tune the frozen experiment, add seeds or begin TT-004 distillation automatically.

Affected TRI-I01/I02/I03/I05/I07/I08/I10/I11/I12/I13/I14. Preservation and correctness outrank speed.
