# Verified teacher-origin formal inputs and CPU student pilot v1

Status: prospective implementation for review. No teacher inference, real model
asset verification or full learning result has been obtained by this PR. CI uses
fake byte artifacts, synthetic captures and bounded one-update software tests.
The existing negative [cyclic result](CYCLIC-LEARNING-CPU-EVIDENCE.md) is unchanged.

## Explicit policy and scope

On 2026-10-21 Trent Slade / QSOLKCB explicitly approved using teacher-origin data.
[DATA's new scoped exception](DATA.md#explicit-teacher-origin-formal-input-exception-2026-10-21)
is `trinite.teacher-origin-formal-policy.v1`. It admits only bounded numeric
operand proposals selected by a frozen gpt-oss-20b teacher, strictly parsed and
independently verified, with repository-generated prompts and oracle labels.
This is a project rights/admission decision, not a legal guarantee about all AI
outputs. Apache-2.0 licensing alone does not establish output rights.

Relabelling a proposal does not erase teacher lineage. Every item records that
origin explicitly. Raw responses, free-form prose, explanations, supplied
answers, reasoning traces and external tokenizer text are **quarantined** and
never passed to the student tokenizer. Correct numeric answers cannot certify
prose or reasoning. Teacher weights do not initialize the student. Historical
native admissions and fixtures are neither rewritten nor retroactively widened.
The unchanged authoritative native CPU updater is used under distinct model,
protocol and request identities, with geometry loss zero.

## Frozen proposal and intake contract

Exactly 32 indexed fresh independent conversations cycle moduli 5/7/9. Each slot
prescribes `n` and `a=(index//3)%n`; the teacher chooses `b` with `0<=b<n` and
returns only JSON with exactly keys `n,a,b` and integer values. Every prompt
includes its slot index and prescribed values. This avoids identical greedy
prompts but does not guarantee diversity. No feedback, examples, tools,
retrieval or answer-conditioned prompts. Temperature 0, top-p 1, seed 0, low
reasoning effort, maximum 512 generated tokens and 120-second timeout per call;
at most 16,384 generated tokens. No retry, refill, repair, quota chasing or
student-conditioned selection. The offline intake records the intended
protocol, not proof that an external runtime obeyed it.

Retain each complete raw response, including available bytes on timeout/error,
and exact `[final_start,final_end)` byte offsets for the final JSON channel.
Only that slice is parsed; the rest remains quarantined. The caller must bind
correct runtime channel extraction; arbitrary offset declarations are not
attestation. Retain at most 64 KiB per response; final JSON acceptance is at most
4 KiB. Larger captures cause a bounded failure, not silent truncation. Strict
UTF-8 JSON rejects duplicate keys, floats, bools, nonfinite constants, extra
keys/prose, wrong slot values and out-of-range operands. Commutatively identical
inputs share `(n,min(a,b),max(a,b))`; first valid occurrence wins. All 32 attempts
and every rejection remain in the sealed packet. At least 16 unique valid items
are required. This predeclared engineering minimum is not quality selection;
underfill blocks student execution, with no additional calls.

A bounded cyclic counter generates each label, independently checked by integer
modulo. Own minimal `addmod[NN,AA,BB]=` prompts and two-digit decimal answers
are rendered. Every item binds source/generator revision, seed/split procedure,
formal specification, family, rendered prompt/answer, verifier, request and raw
response identity, attempt, teacher origin, rights decision and reviewer/date.
Exact regeneration checks all fields rather than trusting claimed outcomes.
The packet's formal subset is `ADMITTED_FORMAL_ONLY`; complete raw capture is
`QUARANTINED_NOT_TRAINING_DATA`. Underfilled packets are `BLOCKED_UNDERFILLED`.

## Local teacher assets and provenance

Freeze a canonical runtime manifest before generation:

- `model`: `openai/gpt-oss-20b`;
- `revision`: actual immutable 40-hex model repository revision;
- `runtime`: exact local runtime/version/build;
- `quantization`: actual format, never an assumed compatibility claim;
- `artifacts`:4–128 entries with unique canonical relative `path`, `role`,
  positive `bytes` and SHA-256 `identity`. Required roles: `weights`, `tokenizer`,
  `runtime`, `chat_template`. Include all shards/configuration and dependencies.

The file bounds are 32 GiB each and 48 GiB total. `freeze` reads actual local regular
files and verifies declared hashes/sizes; traversal and symlink components are
rejected. It does not deserialize, execute, download, establish official origin
or prove inventory completeness. These are declared content-bound receipts,
not hardware/model-execution attestation. Review completeness before freezing
and keep assets unchanged during generation. Never upload weights or credentials.
Trusted checkout/local filesystem are assumed, not a hostile concurrent host.
Use the correct pinned Harmony/chat template. See the official
[model card](https://huggingface.co/openai/gpt-oss-20b) and
[source](https://github.com/openai/gpt-oss). Advertised memory fit is not verified
compatibility of Trent's GPU/runtime. No GPU runtime/inference adapter or download
is included here. The actual external command needs separate local-run approval
and exact runtime review after merge; this CLI never executes an arbitrary command.

## Intake CLI

Use a fresh working root and `reports/` folder. Python 3.11–3.13 needs no packages
for intake; run from the trusted checkout:

```sh
python -I -S scripts/run_teacher_intake.py freeze \
  --manifest /local/runtime-manifest.json --asset-root /local/assets \
  --output /local/reports/request.json
```

Retain the printed request identity independently **before** generation. Capture
canonical JSON has exactly `schema=trinite.teacher-capture.v1`,
`request_identity`, `attempts`. Every ordered attempt has exactly `index`,
`outcome` (`completed`, `timeout`, `runtime_error`), `raw_base64`, `final_start`,
`final_end`. Base 64 preserves complete raw bytes; offsets select final UTF-8 JSON.
No omitted or extra calls. Canonical JSON uses sorted keys, compact separators
and final LF via `trinite.contracts.json_bytes`.

```sh
python -I -S scripts/run_teacher_intake.py seal \
  --request /local/reports/request.json --request-identity sha256:RETAINED \
  --capture /local/reports/capture.json --output /local/reports/quarantine.json
python -I -S scripts/run_teacher_intake.py verify \
  --packet /local/reports/quarantine.json --packet-identity sha256:RETAINED
```

Bounded invalid inputs are retained in a unique full-filename attempt folder
before validation, with a failure receipt where storage permits. Valid packets
retain the request, complete capture, all decisions and admitted formal items.
Publication is complete same-directory atomic no-clobber linking. A preparation
receipt never claims final publication succeeded. Verification regenerates the
packet read-only using an externally retained digest, not a new teacher run.
Hashes provide integrity, not proof the transcript could not be fabricated.

## Source-bound CPU experiment

`trinite.verified-teacher-dense-64.v1` is dense, context 64, width 16, one block,
two heads, feed-forward 32. No ternary/performance claim. Random initialization,
byte tokenizer, native AdamW/updater and environment lock remain unchanged.
Freeze a **separate** CPU request binding the sealed admitted packet, complete
source map, controls, schedules and actual CPU environment before training.
No teacher call occurs during student training or replay.

First run a separate fixed train-only diagnostic: seed 0, four existing Phase 1
training items with four distinct answers, 256 updates, batch 4, 2,048 answer/EOS
targets. Native Phase 1 admission is actually regenerated/audited. Selection is
first distinct answers in the frozen generator order, never heldout or teacher
examples. Raw greedy complete-answer+EOS accuracy must be 4/4 after the fixed
budget. Failure is a retained `blocked_train_only_learnability` result and stops
the comparison. No adaptive budget, early stopping or tuning. Passing demonstrates
only memorization of these four visible training facts, not transfer or reasoning.

If it passes, train seeds 0/1 × arms `teacher_selected`/`oracle_only` from paired
identical initial weights: 256 updates × batch 4 × three fixed answer/EOS tokens
=3,072 targets per cell. All accepted teacher items are used. The control takes
the same number of unique commutative families per modulus from an independently
SHA-ranked full unordered-pair universe. Overlap is allowed and recorded, never
hidden or adaptively avoided. Both schedules group by modulus then independently
hash-rank within modulus and cycle: identical modulus exposure at every visit,
not necessarily identical operands/difficulty or unique-visit exposure per item.
The oracle control does not match the prescribed teacher slot-a constraints;
differences cannot be attributed solely to the teacher's choice of b.
The request records all 1,024 scheduled row identities per arm/seed. Unique train
diagnostics are separate from repeated visits. No teacher confidence, answer
correctness or student score selects the control.

All four training cells complete before heldout scoring. Validation is all 66
unordered pairs of modulus 11; test all 91 pairs of modulus 13. This is public
**modulus extrapolation**, not a secret benchmark, previous Z360 task, or proof
of independent underlying arithmetic knowledge. Entire moduli are isolated from
updates/diagnostics; the two arms share heldouts. Score raw unconstrained greedy
outputs against two decimal digits+EOS. Report every train/heldout prediction,
initial/final exact counts, model/optimizer identities, history, exposures and
negative outcomes. No model selection uses validation/test. Source/environment
must match at start and end; full replay reruns diagnostic and all updates.

With the hash-locked CPU environment from [GETTING_STARTED](GETTING_STARTED.md):

```sh
python -I scripts/run_teacher_student.py freeze \
  --packet /local/reports/quarantine.json --packet-identity sha256:RETAINED \
  --request /local/reports/student-request.json
python -I scripts/run_teacher_student.py run \
  --request /local/reports/student-request.json --request-identity sha256:RETAINED \
  --output /local/reports/student-report.json
python -I scripts/run_teacher_student.py replay \
  --request /local/reports/student-request.json --request-identity sha256:RETAINED \
  --expected-report /local/reports/student-report.json --expected-identity sha256:RETAINED \
  --output /local/reports/student-replay.json
```

Each training invocation has a 600-second update budget; the fresh controller
has a 3,600-second overall timeout covering initialization and scoring too. Retain the full
attempt folders, partial cells, both reports, failed replays and journals. Linux
cumulative reaped-child peak RSS is recorded with explicit scope, separately from
replay-compared numerical bytes. It is not a whole-machine memory cap or speedup.
No checkpoints are serialized; model/optimizer tensors are content-identified.
Atomic publication and durable publication observations distinguish worker,
comparison and publication failures. Trusted-checkout isolation is not a claim
against hostile interpreter caches/startup hooks.

Affected requirements: TRI-I02/I03 (explicit bounded exception/lineage), I05
(CPU resources), I07/I08 (source and replay scope), I11 (family isolation), I12
(no implicit downloads/backend substitution), I13 (all failures), I14 (new
versioned protocol, preserved history). No Phase 5, geometry, scaling, reasoning
superiority or release gate is unlocked by software conformance or this pilot.
