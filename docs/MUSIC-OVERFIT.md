# Train-only music overfit diagnostic v1

Prospective implementation, **not a target training result**. The original
[negative-fit pilot](MUSIC-LEARNING-RESULT.md) was preserved first in merged
[PR83](https://github.com/QSOLKCB/Trinite/pull/83). This
new checkout-only lane neither edits its producer/fixtures nor changes native
model, optimizer, tokenizer, admission, scorer or numerical training semantics.
Affected requirements: TRI-I01/02/03/05/07/08/11/12/13/14. No installed API changes.

## Question and frozen exposure

Can concentrated repetition fit a tiny admitted training subset, and where do
first-answer versus EOS predictions fail? The original 64-update pilot is too
small to refute music learning. This diagnostic does not increase the curriculum
or per-cell update budget. There are two separately reported subsets:

- `one`: the shortest `attacks` answer among original train rows, tie-broken by
  example identity.
- `four`: first two distinct `attacks` answers and first two distinct `duration`
  answers in example-identity order, with tasks ordered as written.

Selection uses formal training data only, never model outcomes. Freeze actual
row identities, schedule and target totals in the request. Re-derive membership
from the original admitted corpus; trusting a submitted `split=train` is forbidden.
Corpus admission can regenerate the full formal corpus for identity validation,
but **no validation/test example enters any model call or update**. Original
admission is not extended to new examples or arbitrary generated text.

Both subsets use seeds17/29, unchanged CPU float32 dense native decoder
(context256, one block, width32, four heads, feed-forward64), AdamW LR0.001,
64 updates and batch8. There are512 scheduled visits per cell:512 per unique row
in `one`,128 per row in `four`. Unique diagnostic rows are not repeated-visit
accuracy. Actual target-token totals differ; equal updates/visits are not equal
compute. Ceiling32768 targets, cumulative120 update seconds, whole-worker900
seconds,1MiB worker log,16MiB per artifact and32MiB per-cell retained payload.
Four cells instead of the original two means aggregate ceilings480 update seconds,
3600 worker seconds and128MiB payload; this is explicit additional experiment
cost, not a claim of equal total budget. No adaptive reruns, budget extension,
new teacher/pretrained inputs, geometry/E8 or arithmetic bridge.

## Observation and fit

Fixed checkpoints0,8,16,32,64 retain native tensors/progress and selected-train
observations. Checkpoints are not candidates for election. Only final-step raw
whole-answer-plus-EOS exactness determines training fit, separately by subset,
requiring every selected row in both seeds. A failure says **not fit at this cap**,
not incapable of eventual fit. A success establishes exposed-row memorization,
not held-out learning, generalization or any learning/Phase5/release gate.

Keep teacher-forced answer accuracy/loss, first-answer and EOS predictions, full
259-way boundary logits, target rank/probability and raw greedy outputs distinct.
For P prompt bytes and A answer bytes, logits P predict the first answer token;
logits P+A predict EOS. Compare full teacher-forced and prefix-only forwards with
frozen float32 rtol=atol=1e-5. EOS after the gold answer is **not** free-generation
termination after a possibly wrong generated prefix. Raw generation delegates to
native greedy with64-token cap, no repair, forced EOS or answer leakage. Native
greedy re-forwards BOS+prefix, with no cache/carry state. Inspect carrier, causal
mask, positional alignment and shifted answer/EOS loss weights instead.

The unchanged native `validation` field aliases the selected unique train rows;
it is explicitly train loss, not held-out validation. Native updater invocations
reset their own deadlines, so the new lane separately accumulates actual update
elapsed time across chunks and enforces the original120-second total. The worker
controller additionally bounds initialization, diagnostics and writing.

## Workflow and authorization

Use a clean exact reviewed commit with green CI and the hash-locked CPU environment.
Commands use `PYTHONPATH=src:scripts python -B scripts/run_music_overfit.py`:

- `preflight`: standard-library admission, deterministic subsets and masks only;
  no model construction, inference or training.
- `freeze --request /outside/checkout/request.json`: new exact source, environment,
  admission, subset, schedule and protocol commitment. Retain printed identity
  independently before outcomes.
- `run --request ... --request-identity sha256:... --output /fresh/run --authorize-run`:
  **separate explicit owner-local execution authorization required** after review
  and green exact-commit CI. Implementation or CI is not that authorization.
- `audit --request ... --request-identity sha256:... --output /run --expected-identity sha256:...`:
  read-only source/request, hash-bound retained artifact/metadata consistency and
  unchanged native safe-reader tensor validation. It allocates a seeded model
  and optimizer for structural checks, but performs no forward pass or update.
  Independently retain the run's matrix identity. This is not prediction replay,
  optimizer replay, hardware attestation or closed PROVENANCE custody. Do not
  infer optimizer replay or learned fit from safe tensor structure alone.

Fresh exclusive paths outside source/request; no overwrite, symlink ancestors or
success-only filtering. Keep failures, timeout/interruption logs, partial tensors
and unattempted cells. Never reseal a changed original or bypass environment/source
mismatch. Read-only audit detects observed mutation, not an atomic hostile-writer
snapshot; exclusively own quiescent artifacts. Private machine environments,
checkpoints and logs are not automatically approved for public upload.

## Conformance versus research

Run standard-library contracts normally and with `-O`, then pinned CPU native
conformance. Synthetic scorer positives and small native update/parity tests
validate harness mechanics only; they are not the four target cells or evidence
that music fit succeeded. A gold-plus-EOS fake is labeled a fake, and missing EOS,
slash loops and malformed outputs remain failures. Tests must reject changed
carriers/answers, relabeled heldout rows, altered requests/artifacts and unsafe
paths. Historical music and native research suites remain separately required.

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_overfit_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_overfit_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_overfit_native -v
```
