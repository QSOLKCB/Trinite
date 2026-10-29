# Prospective train-only musical exposure diagnostic v1

**Implementation and conformance only; no target experiment has been run.**
This separate checkout-only lane follows the preserved
[negative musical-contrast training-fit result](MUSIC-CONTRAST-RESULT.md).
It preserves the original [contrast protocol](MUSIC-CONTRAST.md), every earlier
producer and fixture, the positive exposed-example fit, and the negative
arithmetic probe. It changes no installed API or native numerical operation.
Affected requirements: TRI-I01/02/03/05/06/07/08/11/12/13/14.

## One question, not a promised remedy

Does a fixed longer native training trajectory fit the same 40 musical contrast
examples, including correct note/rest and tie/reattack sensitivity, rather than
only learning EOS and the majority answer? The previous two seeds both emitted
`7` plus EOS on all final training rows: 20/40 exact, but no complete quartets
or both-correct contrast edges. That was a training-fit failure at 64 updates;
held-out evaluation remained unrun, not zero-scoring.

The new study keeps the 40 original canonical training rows byte-identical and
in their original order. It uses fresh seeds 17 and 29, not retained checkpoints
or an unaudited continuation of the old run. There is no larger model, changed
loss, new prompt, teacher, checkpoint pretraining, geometry intervention,
arithmetic bridge, curriculum extension, resampling or adaptive rescue.
Longer exposure also means more updates and a longer optimizer trajectory;
this is not a compute-matched causal isolation of repetition from optimization.

The finite task has eight events and admits the rule
`8 - rest_presence - tie_presence`. Training fit would not establish
arbitrary-length counting, general musical understanding, arithmetic transfer,
fresh confirmation or any learning/release milestone.

## Exact unchanged training data and controls

Two original training parents, five meters each and four note/rest ×
tie/reattack variants per group give 40 unique rows and ten quartets. Answers
within every quartet are 8, 7, 7, 6. Matched headers, prompt lengths and answer
positions prevent those fields alone from solving a quartet. Parent/group/variant
identifiers and split metadata never enter model prompts. The unchanged exact
rational and independent integer-tick checks, compact parser, tokenizer and
answer/EOS masks remain authoritative.

The new source-bound admission covers only these exact training rows and this
protocol; it does not extend or reseal the old admission. No held-out row,
pitch/tempo sibling or held-out inference stage is part of this lane. Native
`validation` aliases these training rows and is not a validation split.

Retain raw exact-answer-plus-EOS results, complete four-variant groups,
contrast edges with both answers correct, output sensitivity, output diversity,
and per-meter/per-parent results. Compare unchanged global, task/meter, header,
length and header-plus-length train-only majority controls. All these controls
score 50% raw accuracy here, but cannot solve a complete quartet. Output changes
without correct answers, lower loss, and successful EOS alone do not demonstrate
correct event sensitivity. These repeated observations over two seeds are
exploratory descriptions, not independent trials or statistical proof.

## Frozen trajectory and accounting

The unchanged native CPU float32 dense decoder has context 256, one block,
width 32, four heads and feed-forward width 64. AdamW uses the original constant
learning rate 0.001 and unchanged optimizer, clipping and answer/EOS target-mean
loss settings. The authoritative native updater cycles the original row order
with batch size eight. One fresh training state advances continuously to 640
updates; there is no restart, resume, checkpoint election or early stopping for
success. Native train-loss evaluation remains every 64 updates.

| Checkpoint | Scheduled visits per seed | Visits per unique row | Answer/EOS targets per seed |
| --- | ---: | --- | ---: |
| 0 | 0 | 0 | 0 |
| 64 | 512 | 32 rows receive 13; eight receive 12 | 1,024 |
| 640 | 5,120 | exactly 128 each | 10,240 |

The final schedule contains 1,280 sixes, 2,560 sevens and 1,280 eights per seed.
Across both seeds there are 1,280 updates, 10,240 visits and 20,480 answer/EOS
targets. The regenerated schedule contains 348,160 shifted live input positions
per seed, or 696,320 across both seeds; this is not ten times the original
64-update live-position count because that endpoint ends partway through a cycle.
A scheduled target is not a unique diagnostic example. The intermediate
step-64 observation supplies an in-run short-exposure reference; agreement with
the historical report is not assumed and cannot close historical run custody.

At exactly steps 0, 64 and 640, retain all 40 prompt-only raw greedy predictions,
their journals and safe checkpoints. The unchanged generation cap is 64 tokens,
including EOS: at most 240 training diagnostic decodes and 15,360 greedy
forwards across both seeds. Training and teacher-forced loss forwards are
separate from this bound. Journal copies are duplicates, not extra observations.
Diagnostic probes must preserve model/optimizer state, training progress,
scheduled order, RNG state and module modes; synthetic next-update parity tests
cover this observation boundary without executing the target curriculum.

### Time and retention limits

The **120-second per-seed cumulative update limit is unchanged**. It is elapsed
monotonic wall time summed around native update calls, including native
train-loss validation inside those calls. The accumulator is never reset at
step 64. It is not process CPU time or the whole-worker wall clock. Snapshot,
prediction and other orchestration time are outside the update counter but
inside the separate 900-second worker deadline. Two workers give aggregate
ceilings of 240 update seconds and 1,800 worker seconds. These are ceilings,
not predictions that all 640 updates will fit on a particular host.

Worker logs are bounded to 1 MiB, each artifact to 16 MiB, and the aggregate
retained payload to 128 MiB. The aggregate payload check is post-write/audit,
not a live disk quota or a whole-process RAM cap. Failed or oversized attempts
are retained rather than deleted to manufacture success.

## Outcomes, failures and explicit stopping rule

Run both prescribed fresh-seed cells and retain both outcomes. Only a completed,
verified step-640 result can report final training fit; fit requires all 40 raw
answers plus EOS in each seed. A successful structural audit or process exit is
not successful learning. A completed nonfitting run is a negative training-fit
result at this cap, not proof of an eventual capacity limit.

A timeout, interruption, source mismatch or incomplete observation is an
**incomplete run**, not failure at 128 visits per row. Preserve partial journals,
completed checkpoints, failure artifacts and explicit missing checkpoints;
never silently restart, extend the deadline or elect an earlier checkpoint.
A late update crossing the limit cannot be promoted into a completed result.
Even a complete positive result enables no held-out model call in this lane.
Any later evaluation, extra exposure or different intervention requires a
separately reviewed protocol and authorization.

## Commands, custody and execution boundary

Use the hash-locked CPU environment and a clean reviewed exact commit with green
CI. The fixed isolated launcher is `python -I -B scripts/run_music_exposure.py`:

- `preflight`: validate the finite train-only inventory, controls and accounting;
  no model execution.
- `freeze --request /fresh/outside/request.json`: create a new source,
  environment, protocol and admission-bound request. Retain its printed identity
  independently.
- `run --request ... --request-identity sha256:... --output /fresh/run --authorize-run`:
  perform the fixed two-cell matrix only after separate owner-local target-run
  authorization following review and green exact-commit CI.
- `audit --request ... --request-identity sha256:... --output /run --expected-identity sha256:...`:
  read-only verification of source-bound retained bytes, safe tensor structure,
  progress and score consistency. Retain the printed run identity independently.

The audit does not rerun inference or replay the optimizer, certify hardware or
prove generalization. New source/environment bytes need a new request, never a
substituted historical fingerprint. Use fresh exclusive paths outside the source
tree; no overwrites, linked ancestors or historical resealing. The isolated
launcher rejects import caches and package shadows before project imports.
Quiescent trusted source/runtime assumptions still apply. Private environment,
checkpoint and log bytes are not automatically approved for public publication.
Reports-only handoffs cannot establish complete original-run custody.

## Conformance, not target results

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_exposure_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_exposure_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_exposure_native -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_exposure_native -v
python -I -B scripts/run_music_exposure.py preflight
```

Run the affected historical suites, upstream checks, required research CPU
aggregate, dependency checks, documentation links and historical-byte comparison
as well. Synthetic models and process doubles test mechanics and rejection paths,
not target learning or original-run replay. CI never runs the two-seed target
study. No merge, tag, release or target execution is authorized by this document.
