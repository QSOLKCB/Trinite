# Musical contrast: negative training-fit result

The owner reports one completed run at [PR90 merge
`64031d5`](https://github.com/QSOLKCB/Trinite/commit/64031d5d230a8b3d3834a5c11d7cbe8a15c75e60).
The [prospective protocol](MUSIC-CONTRAST.md) is unchanged. This separate result
note supersedes its implementation-time unrun statement only; it changes no
curriculum, target budget, source commitment or historical evidence.

## Both seeds collapse to the majority answer

| Seed | Initial exact+EOS | Final exact+EOS | Final complete quartets | Final both-correct edges |
| --- | ---: | ---: | ---: | ---: |
| 17 | 0/40 | 20/40 | 0/10 | 0/40 |
| 29 | 0/40 | 20/40 | 0/10 | 0/40 |

Every final answer is ASCII `7` followed immediately by EOS. This exactly matches
the train-only majority, meter, header and length baselines: 50% raw accuracy but
zero complete contrast groups and zero both-correct edges. No final output
changes when a note/rest or tie/reattack edit changes the correct answer.
Each meter scores 4/8 in each seed; the two training parents each score 10/20.

All 80 initial outputs hit the 64-token cap without EOS; all 80 final outputs end
with EOS after the digit7. There are 160 training diagnostic observations and
5,280 generated tokens/greedy forwards. Journal copies duplicate observations;
they are not additional independent tests. Training and loss forwards are separate.

The controller outcome is **training_fit_failed**. Both seeds fail the 40/40
training gate. **All held-out inference, including pitch/tempo siblings, is
unrun and unmeasured**, not zero-scoring. Its successful process exit and reported
structural audit do not constitute learned fit. No training rescue or seed
selection is reported.

## Meaning and limits

The observed output pattern establishes majority-answer behavior on this finite
training set, not its underlying cause. Formatting and EOS succeeded, but
correct event-sensitive responses did not emerge at this budget. This is a
negative **training-fit** result, not a measured generalization failure.

The owner reports 64 updates × batch 8 per seed: 512 visits, 1,024 answer/EOS targets,
34,812 live input positions. The 40 rows receive 12 or 13 visits each, versus 128 visits
per row in the earlier four-example diagnostic. Different task mix, exposure and
target counts prevent a causal comparison. More training is a future hypothesis,
not an automatic remedy, a capacity conclusion or authorization to rerun.

This finite eight-event task admits the rule 8 minus rest-presence minus
tie-presence. It is not a test of arbitrary-length counting, general music
understanding or arithmetic transfer. The previous positive exposed-example fit
and [negative frozen arithmetic probe](FROZEN-MUSIC-MATH-RESULT.md) remain intact.
No numerical bridge, geometry, curriculum extension or research/release gate is
unlocked. Any next study requires a separately frozen protocol and authorization.

## Evidence retained and what is missing

The [public fixture](../fixtures/music-contrast-result-20261028-v1/README.md)
contains all 160 prediction observations and their 160 duplicate journal records,
plus newly derived summary/provenance. The private reports-only archive and inner
checksums passed; all fields were independently rescored. Canonical reconstruction
from the report reproduces all eight advertised prediction/journal SHA-256 values.
Those hashes bind these reconstructed bytes, not omitted checkpoints or execution.

Original raw requests, matrices, checkpoint tensors, host environment, command
captures and audit receipts remain local and were not independently audited here.
The owner reports audit success; that source/structure/score audit performs no
prediction or optimizer replay. The new checker authenticates the retained public
subset and reuses the unchanged source-pinned corpus/scorer. It does not execute
models, certify hardware, replay training, prove no leakage or close original-run
custody. Source imports require a trusted interpreter and quiescent checkout;
observed before/after equality is not protection against a hostile concurrent writer.

```sh
PYTHONPATH=src:scripts python -B scripts/check_music_contrast_result.py
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_contrast_result_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_contrast_result_v1 -v
```

Affected TRI-I07/08/11/13/14. All earlier tracked files and frozen source closures
are unchanged. This is preservation only, without a target model run.
