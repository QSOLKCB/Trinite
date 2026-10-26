# Coverage-balanced addition v1

Status: separately versioned **training and evaluation implementation**, with
engineering conformance, not a measured learning result. Review, green exact-commit
CI and explicit owner local-run authorization precede the target matrix. Preserve
[addition development](ADDITION-DEVELOPMENT.md), its frozen diagnostics and all
historical negative evidence unchanged. This is a new producer, not a continuation
or repair of those checkpoints.

## Question and limits, fixed before outcomes

Does a broader, position-supported addition training recipe transfer better to
shared unseen operand families at the same model capacity and token/visit budget?
The preceding development run's five answer strings were only 5, 7, 9, 11 and 13;
its observed failures motivated this experiment, not a proven root cause. Missing
output support cannot alone explain wrong answers whose values were trained.

**This is a data-recipe comparison, not an isolated causal output-digit ablation.**
Family count, per-example repetition, answer values, output-digit frequencies,
operand frequencies and co-occurrences change together. Coverage-balanced means
broad support and equal scheduled visits per unordered family within each arm;
it does not mean uniformly frequent answer digits. No architecture, tokenizer,
carrier, optimizer, decoding, digit-decomposition, geometry or KV-cache change is
combined with the recipe. Exact addition is still the appropriate ordinary
algorithm; this lane asks what the native neural model learns.

The public facts and previous diagnostic observations informed this design. All
heldouts are **development**, never newly uncontaminated confirmation. Heldout
selection was fixed by pre-run coverage constraints: all five control answer
values represented in shared development, all candidate sums supported, and the
24/16 answer-length family balance below. No new target outcomes select examples,
checkpoint, budget, seeds or reporting strata. Preserve all six outcomes even if
all are negative, failed or timed out.

## Exact corpus and family split

The new admission independently enumerates the same finite operand universe
0 through 12, with two unchanged carriers `A:a,b=` and `A(a,b):`, giving 338
unique rows and 91 unordered families. Labels use unit increments checked by
bounded integer addition and a separate strict prompt parser; the unchanged byte
tokenizer checks actual masks/lengths. Rights cover these locally generated
mathematical facts and minimal symbols only, not prose, external data, teacher
outputs or legal certification.

The control retains five families:

`(1,4), (2,5), (0,9), (3,8), (6,7)`.

The candidate uses every unordered pair `0 <= a <= b <= 9` **except** these 15:

```
(0,5), (1,6), (1,8), (2,4), (2,6), (3,3),
(2,9), (2,8), (3,9), (4,8), (5,8), (6,8),
(6,9), (7,7), (7,9)
```

Thus the candidate has 40 families/144 unique rows, including all five original
control families/20 rows. Both operand orders and carriers stay within a family.
The candidate trains every sum 0–18, all ten first-answer-position digits, and
second-position digits 0–8. Second-position 9 is structurally absent from this
training range: 19–24 are not trained answer values, and 19 requires an unseen
second-position digit. This is not universal position coverage. The source-bound
admission and request contain exact support tables and scheduled digit histograms.

Candidate training family counts for sums 0–18 are:

`1,1,2,2,3,2,2,3,4,4,4,3,2,2,1,1,1,1,1`.

Both arms have operands 0–9 in both roles. The candidate has multiple examples
for many, not all, answer values; sums 0, 1, 14–18 have only one training family.
Diagonal facts have two unique carrier renderings, not four independent facts.

## Shared evaluation and seen regressions

Every development family is excluded from **both** training arms. Both arms
score the exact same rows/order, only at the fixed final checkpoint:

| Shared stratum | Meaning | Families | Unique rows |
| --- | --- | ---: | ---: |
| `shared-answer` | New combinations with values 5,7,9,11,13, trained in both arms | 5 | 20 |
| `broadened-answer` | New combinations with other in-range values, trained only in candidate | 10 | 36 |
| `range-length` | At least one operand 10–12; range and digit length jointly shift | 36 | 138 |

The primary in-range development summary combines the first two strata: 56
renderings of 15 families. The 20-row common-answer subset covers all five old
answer values rather than relying only on the historical 5+8 survivor. It is
still small and historically observed, not fresh confirmation or 20 independent
trials. Of the range rows, 20/5 families have answer values trained in both arms,
76/19 have values trained only in candidate, and 42/12 have values trained in
neither. Per-arm support flags prevent labelling candidate-familiar values novel.

Training diagnostics use unique training rows, never repeated schedule visits.
The original 20 control rows are explicit **seen regression anchors** in both
arms, not generalisation evidence or replay of an old checkpoint. Candidate-only
training rows are not scored in the control; the report explicitly leaves those
unobserved and never compares pooled 338-row accuracy across unequal seen sets.

Every raw native prediction is retained. A new source-bound descriptive adapter
reuses the old pure failure classifier, not its fixed-admission producer importer.
It reports exact correctness including EOS, wrong value, immediate/premature EOS,
malformed output and missing EOS; per-answer and arm-specific support groups;
whole-family all-rendering correctness; and carrier/order pair agreement versus
both-correct counts. Consistently wrong is not correct. Diagonal order self-pairs
are omitted. Empty groups are untested, not successful. Training, anchors and
shared heldouts remain separate. No extra forward pass is needed for these maps.

## Exact schedule and fixed numerical budget

- Fresh dense CPU float32 models, paired initialization seeds 0,1,2; six cells.
  Context64, two blocks, width64, four heads, FF128: 86,528 parameters.
- Same native `TrainingState`, `run_steps`, AdamW settings, learning rate0.001,
  tokenizer, scorer and safe tensor writer/reader. Geometry loss0, KV unsupported.
- Batch8, 4,000 updates, milestones0/1000/2000/3000/4000. Only unique training
  observations at intermediate milestones; development never enters updates,
  training loss, checkpoint choice or optimizer diagnostics.
- The control160-slot cycle repeats its20 ordered/carrier rows eight times.
  The candidate contributes four slots per family: both orders × two carriers.
  For diagonal pairs the identical order is deliberately visited twice; the
  diagnostic set still contains only unique rows.
- The candidate queues family slots in lexicographic family/order/carrier order
  by `(answer length, carrier)`, then consumes each queue against the fixed
  control slot pattern. This matches every corresponding slot and batch's
  carrier, prompt length, input length and scored-target length, not answer values.
- Candidate24 one-digit-answer families and16 two-digit-answer families match
  control's60/40 length mix. Each cell consumes200 complete cycles: **32,000
  visits, 76,800 answer+EOS targets, 284,800 unpadded model-input tokens**.
  Repeated-family visits are32/control family or4/candidate family per cycle.
- Training diagnostic cost is larger for144 unique candidate rows than20 control
  rows. Equal scheduled token counts do not mean equal total experimental compute,
  wall time, information content or identical optimizer trajectories.
- Greedy prompt-only generation cap5, exact answer+EOS scoring. No solver fallback,
  answer repair, retries, early stopping, checkpoint election or selected seeds.
- Worker deadline900seconds/cell; cumulative update ceiling480seconds sampled at
  milestones. Six-worker bound5,400seconds plus setup/audits, not promised runtime.
  Payload16MiB/artifact and32MiB/cell; logs1MiB. Custody copies are extra; these
  are not a total-directory size cap, hard RSS quota or GPU authorization.

The two corresponding cells must have equal initialization identities and actual
scheduled target totals. Differences between seeds and arms are reported as
paired descriptive counts, not significance or proof of an internal mechanism.
No result automatically passes learning adequacy, Phase5, format, scaling or release.

## Frozen workflow and custody

The new launcher is `scripts/run_balanced_addition.py`; package, request, plan,
checkpoint, corpus, admission and reports use separate `balanced-addition.v1`
identities. Its source closure extends all206 historical bound files without
changing them. Existing token/failure-map importers intentionally cannot consume
this new producer. Reuse their pure classifier only; use this lane's adapter for
new predictions. No old request or checkpoint is relabelled.

Later authorized owner-run commands (not CI instructions):

```sh
PYTHONPATH=src:scripts python -B scripts/run_balanced_addition.py preflight
PYTHONPATH=src:scripts python -B scripts/run_balanced_addition.py freeze \
  --request /ABSOLUTE/NEW/REQUEST.json
# Keep printed request identity independently, before outcomes.
PYTHONPATH=src:scripts python -B scripts/run_balanced_addition.py run \
  --request /ABSOLUTE/NEW/REQUEST.json --request-identity sha256:REQUEST_ID \
  --output /ABSOLUTE/NEW/ATTEMPT --authorize-run
# Keep printed matrix_identity independently, including an incomplete matrix.
PYTHONPATH=src:scripts python -B scripts/run_balanced_addition.py audit \
  --request /ABSOLUTE/NEW/REQUEST.json --request-identity sha256:REQUEST_ID \
  --output /ABSOLUTE/NEW/ATTEMPT --expected-identity sha256:MATRIX_ID
PYTHONPATH=src:scripts python -B scripts/run_balanced_addition.py summarize \
  --request /ABSOLUTE/NEW/REQUEST.json --request-identity sha256:REQUEST_ID \
  --output /ABSOLUTE/NEW/ATTEMPT --expected-identity sha256:MATRIX_ID
```

Use a quiescent reviewed checkout and pinned CPU environment. Output parents must
exist; request/attempt destinations must be fresh and outside the checkout, and
attempt must not overlap request. Never invoke internal workers directly. Save
stdout/stderr, exit status, requests, complete attempts, all intermediate
checkpoints/raw results, logs and original independent identities. Controller
exceptions may leave partial files without a final matrix: retain those too,
without claiming a closed inventory. No silent rerun or repair is permitted.

Audit checks exact source/environment/admission/plan binding, upstream custody,
safe tensors, scheduled-history consistency, all retained metric calculations and
paired initialization/exposure. It does **not** replay optimizer updates or final
inference, prove numerical results on an independent host, or establish learning
adequacy. It reconstructs the initial native state for strict checkpoint checking;
this is not a model-only importer. Local environment details require private
review before public artifact upload; never redact/reseal an original packet.

## Compatibility and checks

Affected TRI-I01–I03, I05–I08, I10–I14. Native numerical semantics, original bound
contracts, historical sources/fixtures, negative results and learning gates stay
unchanged. This increment admits no external data or weights and runs no target
research experiment. Tiny injected numerical/custody controls are engineering
conformance, not evidence of this recipe's generalisation.

```sh
# Standard-library exact corpus/schedule controls:
PYTHONPATH=src:scripts python -B -m unittest discover \
  -s tests/balanced_addition_v1 -p test_data.py -v
# Pinned CPU dependencies; tiny plans only, never the target matrix:
PYTHONPATH=src:scripts python -B -m unittest discover \
  -s tests/balanced_addition_v1 -v
```
