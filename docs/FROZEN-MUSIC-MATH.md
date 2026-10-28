# Frozen music-to-arithmetic probe v1

Prospective checkout-only engineering, **not a target inference result**. This
small zero-shot barometer asks whether frozen music-overfit checkpoints produce
correct answers beyond their known music rows. It preserves all historical
sources, fixtures and negative results. No training, arithmetic bridge, model
selection, native numerical change or installed API extension is introduced.
Affected requirements: TRI-I01/02/03/05/07/08/09/10/11/12/13/14.

## Fixed question and checkpoint inventory

The new request freezes 32 addition questions before any target inference,
using only the plain ASCII prompt `a+b=`. Four eight-question strata cover
small one-digit sums, single-digit carry, two-digit addition without carry and
two-digit addition with carry, including longer answers. Targets vary to expose
constant-answer shortcuts. The protocol records exact and commutative overlap
with its explicitly audited historical arithmetic inputs; this bounded check is
not an exhaustive research-history audit or a claim of globally fresh families.
Known project-history overlaps are retained and disclosed, not silently replaced.
The questions are unseen to these music-only checkpoints, not a pristine
confirmatory benchmark across the project's research history.

All eight endpoints are required: the actual retained step-0 and step-64
checkpoints for subsets `one` and `four`, each at seeds 17 and 29. No best-checkpoint
selection, reconstructed initialization baseline, additional seed, replacement
run or outcome-dependent question choice is permitted. An initial model skeleton
constructed by the native safe loader is only a structural validation step: all
evaluated model tensors must come from the bound retained checkpoint.

## Historical custody and safe restoration

Use a complete original music-overfit run with its independently retained request
and matrix identities. Reports-only exports cannot substitute for the tensor
payloads and complete original inventory. The original producer is pinned to Git
commit `de28d4ec0a779bb91d6e464f12e6220aceb31fd9`, tree
`8282d8223bb22c155c15beff673ffecca77b22be`. Supply a clean local checkout of that
producer; no source or weight download occurs implicitly. Both runtime source
trees must be free of import caches and executable shadows, including ignored
`__pycache__`, `.pyc`, `.pyo` and extension files. `-B` prevents new cache writes
but does not prevent reading an existing unchecked-hash cache. Use a fresh
checkout and `-B`; rejection is not permission to erase historical evidence.
The original worker checks actual tracked runtime bytes against pinned Git blobs
before imports, not merely Git status or metadata.

The new request binds the **new probe producer** separately from the **original
checkpoint producer**, request, matrix and payload/metadata hashes. Do not compare
an old request to an evolved checkout, substitute current source identities, or
reseal old evidence. Original admission, environment and safe-reader checks must
pass without exceptions. The unchanged native `verify_checkpoint` validates
metadata/progress and loads bounded safetensors, not pickle; it performs no model
forward or update. Its optimizer construction/restoration is structural only.
No optimizer step, backward call, training replay or gold-prefix inference runs.

Original run files are read-only inputs. Before/after inventories detect observed
mutation; this is not an atomic snapshot against a hostile concurrent writer.
Use exclusively owned, quiescent local artifacts. Preserve all original bytes and
keep new requests, outputs, partial logs and reports in fresh external locations.

## Raw generation and scoring

Each decode uses the unchanged native greedy decoder on BOS plus the prompt only.
Gold answers are ordinary integer calculations kept outside inference inputs.
No solver, retrieval, teacher, answer-conditioned prefix, grammar mask, repair,
forced EOS, alternative prompt or adaptive retry is allowed. The global cap is
16 generated tokens, including EOS, within context 256. The fixed inventory is
256 decodes and at most 4,096 model forwards. Actual tokens/forwards may be lower;
equal caps do not imply equal actual work. The controller also enforces frozen
600-second per-worker wall-time, 1 MiB worker-log and 8 MiB attempt-artifact
bounds rather than relying only on token counts. Freeze admission is a separate
bounded worker operation with no forwards; these are not speed-comparison data.

Retain every raw token ID and EOS/cap termination. Primary success means exactly
canonical decimal answer bytes **followed by EOS**. Malformed bytes, extra text,
missing EOS and cap-ended outputs remain failures; display text is not a repair
or scoring channel. Model identities and absent gradients are checked around
inference. Failed, timed-out and incomplete runs remain retained, never silently
completed or excluded from the report.

Report per subset/seed/step/stratum exact counts, paired step-0/64 changes, output
diversity, malformed/EOS/cap rates and outcomes away from answers `6` and `14`.
Compare the fixed answers `6`, `14`, `5/4` and `7/4`, plus the maximum constant-answer
accuracy implied by the frozen target frequencies. Baselines are non-neural
scoring controls; no additional checkpoint generations are needed.

Strong varied correct responses beyond these shortcuts would be evidence against
*only reproducing the known memorized answers on this probe*. They would not
retroactively negate the training-overfit result, prove reasoning or establish a
music-specific causal transfer mechanism. Failure is also limited: plain addition
is a prompt-format distribution shift, and no arithmetic bridge was trained.
No original learning, Phase 5, scaling, format-selection or release gate advances.

## Owner-local workflow and authorization

Review and green CI for the exact published probe commit precede a separately
authorized owner-local target inference. Implementation and synthetic tests do
not authorize or imply that run. Use the hash-locked CPU environment and the
launcher `PYTHONPATH=src:scripts python -B scripts/run_frozen_music_math.py`.

- `preflight`: inspect the frozen protocol and question/overlap contract without
  target model execution.
- `freeze`: bind `--original-source`, `--original-run`,
  `--original-request-identity` and `--original-matrix-identity` into a fresh
  external `--request`. Retain the printed new request identity independently
  before inference. Do not publish machine-specific paths or private artifacts.
- `run --request ... --request-identity sha256:... --output /fresh/probe
  --authorize-inference`: one bounded attempt, requiring separate explicit
  authorization. Supply the same four `--original-*` arguments as at freeze;
  paths locate inputs, while the request binds their identities. Keep partial
  journals and failures; do not rerun adaptively.
- `audit --request ... --request-identity sha256:... --output /probe
  --expected-identity sha256:...`: read-only artifact/source/metric checking,
  not fresh inference, optimizer replay, hardware attestation or independent
  replication. Preserve the externally retained expected report identity.

Keep complete raw checkpoints and probe artifacts locally. A future compact
reports-only handover should include protocol and source/input identities,
prediction rows, summaries, failures and preservation receipts, but not weights
or private host details. Uploading private checkpoint or machine information is
not authorized by this implementation.

## Conformance, not performance evidence

The dedicated workflow runs standard-library contracts normally and under `-O`,
then separately locked native synthetic mechanics. Software doubles and any
synthetic gold-output positive are explicitly harness tests, not unseen-maths
results. No original target checkpoint is loaded by CI. Historical suites remain
separate; a local focused pass is not the full CPU aggregate.

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/frozen_music_math_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/frozen_music_math_v1 -v
# Requires requirements.lock; synthetic only, never the original run:
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/frozen_music_math_native -v
```
