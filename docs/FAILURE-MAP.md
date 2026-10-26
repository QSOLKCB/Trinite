# Frozen addition failure map v1

Status: separately versioned **engineering implementation**, not a target result.
No target checkpoint is executed by this PR. A later owner-authorized local run
requires a reviewed exact green commit and a request frozen before its outcomes.
This is a bounded map of where each frozen model succeeds or fails, not a fix,
new training run, general-purpose LLM stress suite or proof of arithmetic ability.

## Fixed question and complete coverage

Run prompt-only greedy inference on **all 338 admitted inputs**: ordered operands
0–12, in each of the two existing symbolic carriers `A:a,b=` and `A(a,b):`.
The same [addition-development admission](ADDITION-DEVELOPMENT.md) supplies exact
labels, original row identities and rights evidence. No new natural-language,
external, teacher or model-generated input is admitted. All six final step-4000
checkpoints are mandatory: control/candidate × seeds 0, 1 and 2. There is no
best-seed selection, checkpoint election, retry, answer repair or training.

The complete 13 × 13 × 2 case map retains prompt, operands, exact answer, raw
byte/special-token output, category, family, checkpoint identity and status.
Counterexamples refer to these complete rows, not a cherry-picked display sample.
All 91 unordered families are represented; correlated carriers and operand swaps
are not independent trials. No confidence interval or whole-region guarantee is
inferred from these finite cases. Even a perfect map would not establish correct
unbounded arithmetic or instruction following. The two `0+0` renderings are
explicitly labelled trivial zero-answer controls, not proof of a learned algorithm.

### Exposure and controlled contrasts

- **Seen anchors:** 20 control and 40 candidate rows are seen training examples,
  explicitly **not generalisation**. The 20 expansion rows are training-exposed
  only in the candidate; control labels them as other-arm training families.
- **Historical development:** the 298 original shared development rows are already
  research-exposed. New requests or new reports do not make them fresh tests.
- **Answer familiarity:** sums seen in both training arms are flagged separately
  from family familiarity. This is descriptive conditioning, not an isolated
  causal intervention.
- **Range/digit-length shift:** operands 10–12 simultaneously leave the training
  range and become two digits. These are a **joint, confounded shift**, not two
  independently tested explanations. Answer familiarity is an additional flag.
- **Carrier contrast:** 169 pairs retain exactly the same ordered operands and
  answer while switching the existing carrier. This changes punctuation and
  prompt length together; it is not an isolated punctuation mechanism test.
- **Order contrast:** 156 pairs swap operands within the same carrier, holding
  sum/family fixed. Diagonal self-pairs are excluded rather than inflating agreement.

A fixed 48-cell cross-product of arm-specific seen status, answer familiarity,
joint range/length shift, carrier and order (ascending/descending/diagonal)
retains empty cells as **untested: no admitted case**, not passed. Nonempty cells
report exact counts and counterexample identities. Passing means all listed
finite cases passed; it does not certify unobserved inputs in a semantic region.

The native scorer's `test` label is used only as an ephemeral selection switch
for one row at a time. The original arm's unchanged training view supplies its
majority baseline. The source row's original split and admission identity remain
in the report with the separate scientific exposure label. The copied selection
view is not a newly admitted row, and its changed split is not claimed to hash to
the original identity. This is what permits bounded per-case progress without
rewriting the authoritative scorer or labelling seen anchors as held-out evidence.

## Correctness, termination and reproducibility are different

Exact correctness requires the canonical decimal answer **and final EOS**, with
the unchanged cap of five output tokens. Disjoint primary labels are:

| Label | Meaning |
| --- | --- |
| `exact_correct` | Exact answer bytes and EOS |
| `wrong_value` | Other well-formed canonical nonnegative integer followed by EOS |
| `immediate_eos` | EOS without answer bytes |
| `premature_eos` | A nonempty strict correct-answer prefix followed by EOS |
| `malformed_output` | EOS-terminated noncanonical numeric syntax or other bytes/tokens |
| `missing_eos` | Full five-token cap reached without EOS |

These labels are descriptive precedence, not identified causes: e.g. `2<EOS>`
for expected `13<EOS>` is wrong value, while `1<EOS>` is premature termination.
Missing-EOS takes precedence over syntax if the cap is exhausted. Impossible
retained greedy trajectories (EOS followed by more tokens, or a shorter
non-EOS sequence despite ample context) are rejected as invalid evidence.
Non-finite logits, invalid tensor shapes or context overflow fail the harness;
there is no silent truncation pass or alternative decoder.

Each carrier/order pair reports **consistency separately from correctness**:
both correct, both wrong with the same output, both wrong with different outputs,
or one correct. An identical wrong answer is consistent but never successful.
All pairs and their exact row identities are retained. Pair totals are explicitly
raw observations; replay-invalid pairs have a separate validity flag/count and
cannot support scientific interpretation.

Original final training and development predictions are retained byte-for-byte
and compared against the new output from the **same source-bound checkpoint,
input, environment and greedy cap**. An exact mismatch is a reproducibility or
plumbing failure, not automatically a learning regression. Its raw outputs and
comparison stay on disk, but completion fails closed. No accuracy threshold is a
completion gate: an entirely wrong yet reproducible model may complete its map.
Control's 20 expansion rows were not previously scored in that arm and are marked
`not_previously_scored`, not silently matched or reported as missing corruption.
The other 318 control and all 338 candidate rows have retained comparison outputs.

Malformed grammar, negative/out-of-range operands, leading zeros, Unicode digits,
extra answer text and long prompts are separate **parser-only boundary controls**.
Their outcome is `rejected_input`, with `model_executed: false`. Rejection says
something about harness admission, not a tiny symbolic model resisting prompts.
Timeouts, interruption and worker failures likewise are not wrong numerical answers.

## Source, custody and resources

The [frozen-token diagnostic](FROZEN-TOKEN-DIAGNOSTIC.md) model-only importer is
reused unchanged. It validates the externally committed original request, matrix
and all six model identities, original source/environment, full admission,
five milestones, history and safe tensor structure, closed provenance and named
score artifacts. It copies model parameters only, never constructs an optimizer
or executes uploaded producer code. Freeze performs safe structural tensor loading
but no model forward. It is not numerical or optimizer replay.

All 189 original producer commitments and the complete frozen-token diagnostic
source closure, including its bound contract, remain byte-identical. The new
request additionally binds this contract, every new runtime module and the fixed
launcher. Actual import origins are checked. Changed sources or an incompatible
CPU environment are rejected, never patched into historical evidence.

The fixed decoder is CPU float32/context 64 with deterministic single-thread
settings. The existing scorer and greedy implementation remain authoritative.
There are 338 row calls, each at most five forwards: at most **1,690 forwards per
cell, 10,140 across six cells**. This upper bound is not measured latency or equal
compute. Model, gradients, training mode and RNG are compared before/after;
finite-output hooks observe but do not modify logits. No GPU, teacher conditioning,
training, implicit download, optimizer step or reusable KV cache is introduced.

Each sequential fixed worker has a 900-second whole-worker deadline and a 1 MiB
log cap; six workers are at most 5,400 seconds plus controller/import/audit overhead.
Artifacts are capped at 16 MiB and retained payload writes at 32 MiB per cell.
Custody duplication/input imports are not a total-directory or hard RSS cap.
Sampled RSS includes imports. Progress is retained in 32-row chunks (last chunk18);
an interruption can lose the current unfinished chunk, not certify those rows.

Completed/failed/timed-out/interrupted/not-attempted outcomes retain all six
checkpoint identities. Noncompleted workers yield an explicit 338-row untested
map; partial files remain for diagnosis and are not promoted to audited results.
A replay mismatch retains its detailed measurement and originals before refusal
to create completion. Outer source/input/filesystem/controller failure may prevent
a final matrix or printed identity; retain all partial output and stderr too.

Use quiescent immutable input and checkout. Request/output/producer paths must
be disjoint and destinations fresh; observed links, unsafe members and mutations
are rejected. Inventories are not an atomic snapshot against a hostile concurrent
writer. Audit/summarize are read-only source/custody/structure and retained-metric
checks, **not independent inference replay** or hardware attestation. Completed run records state `target_inference_executed: true` and the 318/338
historical predictions compared; `audit_replays_inference: false` states that
subsequent audit does not independently rerun those model calls. All adequacy,
Phase 5, release, optimizer-replay and fresh-confirmation flags remain false.
Review local environment evidence privately before any public upload.

## Later authorized local workflow

The producer is the original `attempt` directory containing `matrix.json` and
six cell directories, not the enclosing handover directory. Retain external
identities independently, including failures. `models.json` maps exactly the six
cell names to their independently retained final-model `sha256:...` identities.
Never derive a supposedly independent expected ID from the artifact under audit.

```sh
PYTHONPATH=src:scripts python -B scripts/run_failure_map.py preflight
PYTHONPATH=src:scripts python -B scripts/run_failure_map.py freeze \
  --producer /ABSOLUTE/ORIGINAL/attempt \
  --producer-request-identity sha256:ORIGINAL_REQUEST \
  --producer-matrix-identity sha256:ORIGINAL_MATRIX \
  --model-identities /ABSOLUTE/INDEPENDENT/models.json \
  --request /ABSOLUTE/NEW/request.json
# Independently retain the printed request identity before target inference.
PYTHONPATH=src:scripts python -B scripts/run_failure_map.py run \
  --producer /ABSOLUTE/ORIGINAL/attempt \
  --request /ABSOLUTE/NEW/request.json --request-identity sha256:NEW_REQUEST \
  --output /ABSOLUTE/NEW/failure-map-attempt --authorize-run
# Independently retain the printed matrix identity, including failed matrices.
PYTHONPATH=src:scripts python -B scripts/run_failure_map.py audit \
  --producer /ABSOLUTE/ORIGINAL/attempt \
  --request /ABSOLUTE/NEW/request.json --request-identity sha256:NEW_REQUEST \
  --output /ABSOLUTE/NEW/failure-map-attempt --expected-identity sha256:MAP_MATRIX
# summarize accepts the same arguments and returns verified per-cell maps/summaries.
```

Never invoke the internal `_worker`, substitute a new expected hash after mismatch,
merge outcomes from different producers, or rerun automatically until success.
Preflight is data-only. Freeze is not authorization to execute the target run.

## Compatibility and engineering controls

Affected requirements: TRI-I01–I03, I05–I08, I10–I14. This additive checkout-only
lane changes no installed API, original admission/scorer/numerics, old fixtures,
source-bound documentation, negative result or learning gate. Tiny models,
short engineering schedules, synthetic prediction records and process fakes are
conformance controls only, never target research evidence. True-custody tests
exercise actual admission and the upstream observer/verifier.

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/failure_map_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/frozen_token_diagnostic_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/addition_development_v1 -v
PYTHONPATH=src python scripts/check_research_cpu.py
```

The historical CPU aggregate does not subsume these checkout-only topic suites.
No target six-model map is run by CI or claimed by this implementation.
