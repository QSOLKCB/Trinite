# Fixed-checkpoint forward comparison v1

This is an **exploratory, non-certifying, forward-only diagnostic**, separately
owner-authorized after the two context-symbol fit attempts. It does not resume,
repair, validate or replace either attempt. Affected requirements:
TRI-I01/I03/I05/I07/I08/I10/I11/I12/I13/I14.

The [launcher](../scripts/forward_checkpoint_comparison_v1.py) compares two fixed
final fit checkpoints on the same eight already-exposed public training rows.
It constructs the existing native dense `Decoder` for this diagnostic and loads
only its model tensors through safetensors. It does **not** call native training
checkpoint restoration, forge a matching runtime fingerprint, or weaken its
intentional foreign-environment rejection. Historical native guards, producers,
protocols, runs, negative results and evidence bytes remain authoritative and
unchanged. This adapter is not an alternate native certification route.

## Fixed inputs and execution boundary

Only the initial configured architecture is supported: byte vocabulary 259,
context 256, one block, width 32, four heads, feed-forward width 64, tied
embedding/output and 24,768 unique parameters. No quantization or architecture
search is part of this comparison.

The two input archives are local files, explicitly pinned by SHA-256:

| Input | Immutable deposit | Payload SHA-256 |
| --- | --- | --- |
| Prior | [PR #122](https://github.com/QSOLKCB/Trinite/pull/122), `docs/evidence/context-symbol-learning-v1/payload.zip` | `0e7e157d2238e7446760e71118df6b3bc715e979596e626eb01a31ae905cfa0e` |
| Current | [PR #124](https://github.com/QSOLKCB/Trinite/pull/124), commit `e44d11b6863f6dc920169b231c55f80ebb10b7d0`, `docs/evidence/context-symbol-learning-v1-fit-gate-578d4793/payload.zip` | `9148455a024825ec15a5a86add1b6190c52da207a2a1774511430d2773b7bc96` |

The second deposit is an external pinned input, not copied into this tool's
branch or silently fetched from a moving branch. Obtain the reviewed bytes
separately. The launcher does not download dependencies, inputs or model code.
Archive source files and native binaries are evidence data, never executable
inputs. No pickle loader or checkpoint-provided Python class is used.

Strict hash, path, bounded archive/JSON, key inventory, shape, dtype and finite
checks fail closed. Only the fixed training subset reaches model execution;
no heldout labels are generated and no heldout observations are evaluated.
Original metadata are retained as provenance, not rewritten to fit this host.
Observed host and explicit deterministic single-thread CPU settings are reported
separately, including original/observed environment mismatch. Deterministic
settings do not establish cross-host bitwise equivalence.

## Running the diagnostic

Use the repository's hash-locked CPU environment from
[GETTING_STARTED.md](GETTING_STARTED.md). Supply the two original archive files
and a fresh output path; do not place output inside either evidence deposit:

```sh
PYTHONPATH=src:scripts python -B scripts/forward_checkpoint_comparison_v1.py \
  --prior docs/evidence/context-symbol-learning-v1/payload.zip \
  --current /path/to/pinned-pr124/payload.zip \
  --output /path/to/fresh-forward-comparison.json
```

The command performs no optimizer construction/update, backward pass, training,
resume, feasibility experiment, fresh freeze, native fit-gate validation or
main-run execution. It does not execute an oracle or build a native binary.
Retain failures rather than changing pins, evidence or tolerances to obtain a
successful report.

## Reading the report

- Greedy output is the raw answer-token sequence, including EOS when emitted,
  with a fixed three-token cap. Missing EOS, extra tokens and malformed answers
  remain failures; no answer repair, constrained decoding or retry is applied.
- Teacher-forced correct/strongest-competitor logits and margins use the already
  exposed gold answer prefix. These are local prefix-conditioned diagnostics,
  not independent greedy predictions or a new label-blind evaluation.
- Tensor hashes, counts of float32 elements with different bit patterns, maximum absolute and symmetric relative
  differences describe the stored float32 operands. Relative difference is
  `abs(a-b)/max(abs(a),abs(b))`, zero when both are zero; near-zero operands can
  make relative differences large. These measurements are not formal floating
  point error bounds, and tensor differences do not identify the training step
  or cause of divergence.
- A change in greedy answer or margin on this host is an observed diagnostic,
  not proof of host nondeterminism. A match is not proof of optimizer replay.

The original reported 8/8 versus 7/8 fit outcomes remain attached to their own
runs and environments. This diagnostic cannot authorize the mandatory fit gate,
either main run, research advancement or release. It demonstrates neither unseen
learning/generalization nor operator blindness. It changes no training schedule,
checkpoint selection, data admission, model-size ceiling or other repository.

## Engineering checks

The [dedicated workflow](../.github/workflows/forward-checkpoint-comparison-v1.yml)
installs the existing hash-locked CPU requirements and exercises bounded
engineering tests in normal and optimized Python:

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/forward_checkpoint_comparison_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/forward_checkpoint_comparison_v1 -v
```

CI does not run the historical checkpoint comparison, retrieve the pending
external deposit, train a model, build an oracle or generate hidden labels.
Engineering conformance is not scientific gate success or native certification.
