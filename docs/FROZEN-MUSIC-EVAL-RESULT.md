# Frozen musical evaluation: negative held-out development result

The owner reports a completed inference-only evaluation at [PR94 merge
`99a5896`](https://github.com/QSOLKCB/Trinite/commit/99a5896dd893f9d338e6c2c81c49dba29fbecbb1),
using both prescribed final step-640 checkpoints, seeds 17 and 29. The
[prospective frozen protocol](FROZEN-MUSIC-EVAL.md), original producer and
[positive exposed-training fit evidence](MUSIC-EXPOSURE-RESULT.md) remain
unchanged. This additive result note supersedes their earlier held-out-unrun
status only for this separately authorized evaluation; it does not add a
held-out route to the historical training-only producer.

**The retained outputs reproduce perfect training fit but do not reliably
transfer the contrasts to the two held-out structural parents.** Canonical
accuracy is below all five training-only majority controls, and neither seed
solves a complete held-out quartet in any view. We preserve this as a negative
held-out **development** result, not a failed execution. The frozen protocol
had no accuracy pass threshold: `completed` records execution/artifact
completion, not successful generalization.

No target model was executed to prepare this preservation increment. The new
checker authenticates supplied bytes and independently recomputes their scores;
it does not generate predictions or rerun the owner-local audit.

## Training replay and held-out scores

Both supplied training-replay sequences are 40/40 exact answer plus EOS and
match the corresponding final raw predictions already preserved with PR93.
Each has 10/10 complete quartets and 40/40 both-correct contrast edges. This
preserves the positive exposed-training result alongside the new negative
held-out result; the two statements are not contradictory.

The protocol required all 80 training replays to pass before any held-out call.
The journal order, replay-gate receipt and completed report are consistent with
that barrier. Their consistency does not independently attest the temporal
order of actual model execution.

| Seed | Held-out view | Exact + EOS | Complete quartets | Both-correct edges | Changed-output edges |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | Canonical primary | 16/40 | 0/10 | 6/40 | 12/40 |
| 17 | Pitch siblings | 10/40 | 0/10 | 0/40 | 0/40 |
| 17 | Tempo siblings | 16/40 | 0/10 | 6/40 | 12/40 |
| 29 | Canonical primary | 15/40 | 0/10 | 5/40 | 10/40 |
| 29 | Pitch siblings | 14/40 | 0/10 | 4/40 | 8/40 |
| 29 | Tempo siblings | 15/40 | 0/10 | 5/40 | 10/40 |

All 320 retained observations contain one answer digit followed by EOS: 640
generated tokens, no cap terminations. This independently verifies journal
token accounting, not measured model-forward execution or elapsed time.
Successful formatting does not establish correct counting. Some contrast edges
are both correct, but changed outputs alone are not correct contrast reasoning.

There are **two held-out structural parents**, `c` and `d`, each rendered in five
meters. Each view has 40 rows, 10 four-variant quartets and 40 contrast edges per
seed. Canonical is primary; pitch and tempo are correlated transformations.
Neither the 120 held-out rows per seed, meter renderings, siblings nor the two
seeds are independent subjects. These existing visible development families are
not a fresh confirmation sample. The task throughout is `attacks`.

## Pitch, rest and tempo patterns

The following descriptive patterns are derived from the complete retained
outputs, without selecting or repairing observations:

- **Pitch sensitivity:** for each seed, only 4/40 canonical-to-pitch pairs retain
  identical raw outputs; 36/40 change although the gold answer is invariant.
  Only 1/40 pairs are correct in both views. Seed 17 emits constant `6` plus EOS
  on all 40 pitch rows. Seed 29 emits `6` on 32 and `7` on eight.
- **Rest-only failures:** canonical variant `10` (rest present, no tie; gold
  answer `7`) scores 0/10 for both seeds. Canonical variants `00`, `01`, `10`,
  `11` score respectively 9, 6, 0, 1 out of ten for seed 17 and 9, 5, 0, 1 for
  seed 29. These row strata do not themselves contain complete quartets.
- **Tempo invariance, including errors:** all 40/40 canonical-to-tempo pairs
  retain identical raw outputs for each seed, but only 16/40 and 15/40 pairs
  are correct in both views. Consistent wrong answers are not successes.
- Across canonical/pitch/tempo triples, only 4/40 have identical raw outputs and
  only 1/40 is correct in all three views, for each seed.

Canonical parent scores are 8/20 on both `c` and `d` for seed 17, and 8/20 on
`c`, 7/20 on `d` for seed 29. The preserved report retains the complete
parent/meter/task/variant and transformation summaries; no observed stratum is
excluded from the scoring. These patterns describe the failures, not their
causal mechanism or a verified repair strategy.

## Controls and interpretation

The unchanged global, task/meter, header, length and header-plus-length majority
controls fit only the 40 unique canonical training rows. Each scores 20/40 exact
in each held-out view, with zero complete quartets and zero both-correct edges.
Canonical model accuracy is therefore four and five rows lower, respectively.
Unknown control keys use the fixed training-majority answer `7`; tempo header
and header-plus-length controls use that fallback on all 40 rows. No held-out
row fits a control. The exact formal oracle remains an offline scoring ceiling,
not an inference input or answer-repair channel.

This result does not demonstrate reliable unseen-family contrast transfer,
general musical understanding, arbitrary-length counting, arithmetic or
music-to-mathematics transfer, self-correction, or a causal learning mechanism.
The finite eight-event gold task still admits
`8 - rest_presence - tie_presence`; the retained outputs do not establish that
either model learned that rule. No fresh-confirmation, learning-adequacy,
Phase 5 or release-readiness gate is cleared. The earlier negative results and
positive training fit all remain in the evidence record. No curriculum change,
additional training, intervention or further target run is authorized here.

## Public evidence and verification boundary

The [public fixture](../fixtures/frozen-music-eval-result-v1/README.md) preserves
six supplied files byte-for-byte: the original attempt record, prediction
journal, replay gate, report and worker-completion record, plus the owner-local
audit receipt. It also retains the original handoff checksum manifest; this
manifest names omitted files and is not a claim that the public subset is the
complete handoff. New provenance and this interpretation are explicitly derived.

The read-only standard-library checker pins the supplied artifact identities
and the unchanged scoring-source closure before project imports, regenerates
the fixed row inventory, rescores all 320 journal records, checks retained
completion/report/gate consistency and compares all 80 training-replay outputs
with PR93's retained final predictions. It checks before/after byte stability
under a trusted interpreter and quiescent filesystem; this is not protection
against a hostile concurrent writer or hardware attestation.

The complete original exposure run, evaluation request, checkpoint tensors and
metadata, environment commitments and raw execution/admission logs remain
owner-local. The public subset cannot rerun the full source-bound owner-local
`audit`. The supplied audit receipt says that read-only audit passed; it is a
retained owner artifact, not an independently reproduced full audit. Original
request/report identities link the supplied files but do not recover omitted
artifacts or close full-run custody.

The owner reports one bounded inference-only attempt, without training,
optimizer construction, checkpoint modification, election, repair or retry.
Reports-only consistency does **not** independently prove actual inference,
its timing, the absence of other attempts, or those runtime claims. Neither the
owner's audit nor this public checker is optimizer replay. No target tensors
are loaded or model/optimizer operations executed by this preservation checker.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -I -B scripts/check_frozen_music_eval_result.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -B -m unittest discover -s tests/frozen_music_eval_result_v1 -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/frozen_music_eval_result_v1 -v
```

Affected TRI-I07/08/11/13/14. All historical files, source closures, fixtures and
numerical semantics remain unchanged. This publication preserves the result;
it does not authorize a merge, tag, release or another experiment.
