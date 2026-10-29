# Musical exposure: positive exposed-training fit

The owner reports one completed run at [PR92 merge
`c239acb`](https://github.com/QSOLKCB/Trinite/commit/c239acb6e7b0b5c49798debbae92ca1e1264a64c).
The [prospective train-only protocol](MUSIC-EXPOSURE.md) remains unchanged.
This separate result note supersedes its implementation-time unrun statement
only; it changes no producer, budget, admission, source commitment or historical
result. No target model was executed to prepare this preservation increment.

## Both seeds fit all 40 exposed rows at step 640

All six supplied prediction files were independently rescored against the
unchanged source-pinned training corpus and scoring rules. Exact means the
complete raw answer followed immediately by EOS, without repair.

| Seed | Checkpoint | Exact + EOS | Complete quartets | Both-correct contrast edges | Changed-output edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0 | 0/40 | 0/10 | 0/40 | 0/40 |
| 17 | 64 | 20/40 | 0/10 | 0/40 | 0/40 |
| 17 | 640 | 40/40 | 10/10 | 40/40 | 40/40 |
| 29 | 0 | 0/40 | 0/10 | 0/40 | 0/40 |
| 29 | 64 | 20/40 | 0/10 | 0/40 | 0/40 |
| 29 | 640 | 40/40 | 10/10 | 40/40 | 40/40 |

At step 64 both seeds emit constant `7` plus EOS. At step 640 each seed emits
`6` plus EOS on ten rows, `7` plus EOS on twenty and `8` plus EOS on ten, matching
every answer. Both training parents score 20/20; every meter scores 8/8.
The final outputs therefore show correct note/rest and tie/reattack sensitivity
on these exposed examples, not merely successful EOS or majority-answer output.
All initial outputs reach the 64-token cap without EOS; all later outputs end
with EOS after one digit. The retained subset has 240 diagnostic observations
and 5,440 generated tokens. Its 240 journal entries duplicate those observations;
they are not additional independent tests or additional model calls.

The train-only global, meter/task, header, length and header-plus-length controls
remain at 20/40 exact, zero complete quartets and zero both-correct edges.
Each final seed exceeds these controls by 20 rows. The exact formal oracle is
a post-generation scoring ceiling, not a model input or an answer-repair step.

All eight step-0/64 prediction and journal files are byte-identical to the
corresponding files preserved with the [earlier negative contrast result](MUSIC-CONTRAST-RESULT.md).
This directly verifies agreement of the supplied early observations; it does
not prove that the historical and new optimizer trajectories or checkpoints
were identical, nor does it close either original run's custody.

## Static schedule versus reported execution

Source regeneration verifies the frozen 640-update, batch-eight schedule:
5,120 visits per seed, exactly 128 per unique training row, 10,240 answer/EOS
targets and 348,160 shifted live input positions. At step 64 the schedule gives
512 visits, with 32 rows receiving 13 visits and eight receiving 12.
These are independently verified **scheduled** counts, not a replay of actual
updates or proof of the omitted execution history.

The owner reports that both fresh seeds completed all 640 updates in one
continuous trajectory and one matrix invocation, with no failed cells,
missing observations, retry, resume, timer reset, checkpoint election, extra
seed or budget change. Reported cumulative update times were about 6.776 and
6.728 seconds, and cell wall times about 12.854 and 12.298 seconds for seeds 17
and 29 respectively. These timings and execution claims are owner-reported,
not independently measured from the public prediction subset. They are not a
hardware benchmark or evidence of universal throughput.

The original 120-second cumulative update and 900-second worker limits remain
unchanged. The owner also reports a successful read-only structural audit;
its original receipt and full run are not part of this subset.

## Meaning and limits

This is a **positive exposed-training fit result**, not a measured generalization
result. The later retained predictions distinguish all prescribed contrasts
correctly, whereas the step-64 predictions do not. More exposure also means more
optimizer updates, so this does not causally isolate repetition from optimization
or establish why the model improved. The same 40 rows and repeated observations
over two seeds are correlated, exploratory evidence.

The finite eight-event task admits the rule
`8 - rest_presence - tie_presence`. These outputs do not establish that the model
learned that rule rather than another fitting strategy, arbitrary-length counting,
general musical understanding, unseen-family performance or music-to-mathematics
transfer. The earlier negative 64-update result and negative arithmetic probe
remain intact; neither is erased by this later training fit.

**Held-out inference remains unrun and unmeasured, even after perfect fit.**
The frozen exposure protocol is training-only and contains no held-out route.
No automatic extra training, held-out assessment, numerical bridge, new curriculum,
geometry intervention or learning/release promotion follows. Any next experiment
requires a separately reviewed protocol and authorization.

## Preserved evidence and verification boundary

The [public fixture](../fixtures/music-exposure-result-20261029-v1/README.md)
retains the six supplied original prediction JSON files and six matching journal
JSONL files, together with newly derived public summary and provenance. All 240
prediction rows are retained and rescored, and each journal must exactly duplicate
its corresponding prediction records. The old fixtures remain byte-identical.

The new read-only standard-library checker authenticates exact fixture membership
and bytes, checks pinned source/import resolution before project imports, and
reuses unchanged corpus and scoring operations. It validates the retained subset,
not omitted execution artifacts. Before/after stability checks assume a trusted
interpreter and quiescent checkout, not a hostile-concurrent-writer guarantee.

The original request, matrix, safe checkpoint tensors and metadata, per-seed
reports, execution logs, command captures, environment fingerprints and audit
receipts were not supplied for independent full-run verification. The private
handoff/report and their private commitments are not published in this fixture.
Owner-reported original-run audit and no-retry claims remain attributed claims.
The public subset does **not** establish closed original-run custody, numerical
inference replay, optimizer replay, hardware attestation or fresh confirmation.
No model loading, prediction generation or update is performed by this checker.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -B scripts/check_music_exposure_result.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_exposure_result_v1 -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_exposure_result_v1 -v
```

Affected TRI-I07/08/11/13/14. All 2,104 earlier tracked files and frozen source
closures are preserved. This additive publication records the result; it does
not rerun the target experiment or authorize another one.
