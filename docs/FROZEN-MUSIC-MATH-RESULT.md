# Frozen music-checkpoint arithmetic: negative exploratory result

The owner reports one completed frozen probe at [PR #88 merge
`def96eb`](https://github.com/QSOLKCB/Trinite/commit/def96eb86ffbd257e71bef15a69bd6370643d28a).
The [prospective protocol](FROZEN-MUSIC-MATH.md) remains byte-for-byte unchanged;
this later evidence updates only its implementation-time unrun status. No new
training, inference, bridge, checkpoint selection, prompt tuning or repair occurs.

## Results from all report-transcribed predictions

| Music training subset | Seed | Actual step 0 | Actual step 64 |
| --- | ---: | ---: | ---: |
| One example | 17 | 0/32 | 0/32 |
| One example | 29 | 0/32 | 2/32 |
| Four examples | 17 | 0/32 | 0/32 |
| Four examples | 29 | 0/32 | 1/32 |

Primary accuracy requires exact decimal bytes followed immediately by EOS.
All 256 records are retained in the [derivative public fixture](../fixtures/frozen-music-math-result-20261028-v1/README.md).
The same 32 questions recur across eight endpoints: these are correlated
observations, not 256 independent problems. Initial endpoints total 0/128;
final endpoints total 3/128. All three successes are one-digit, no-carry sums.
Every carry and two-digit stratum scored zero.

The one-example seed-29 endpoint emits `6` plus EOS on every question, scoring
2/32 by constant output. The four-example seed-29 endpoint emits `4` plus EOS
on eight questions, catching `1+3=` once. **No endpoint exceeds the best constant
baseline (2/32).** There is no demonstrated input-sensitive arithmetic rule.
Overall: 2,897 generated tokens/forwards, 88 EOS endings, 168 cap endings and
216 malformed outputs. Cap endings are also malformed; these counts overlap.

## What this does and does not mean

The earlier owner-reported [one/four-example diagnostic](MUSIC-OVERFIT.md) fit
its selected exposed music rows at final step 64 in both seeds. That is a
positive exposed-training-fit observation, not held-out music competence.
This arithmetic probe does not erase that observation or establish cheating.
It does supply negative evidence for direct arithmetic transfer from those
particular checkpoints, under plain `a+b=` notation, without a trained bridge.

The format shift, tiny one/few-example training scope and lack of a numerical
bridge limit conclusions. This does not establish a capacity limit, the cause
of prior failures, absence of all reasoning, or a refutation of music-first
learning. Seventeen questions overlap the protocol's bounded historical-project
arithmetic inventory: unseen to the music checkpoints is not globally fresh
confirmation. No music-specific causal transfer is established.

A separately versioned musical-contrast study may test event-sensitive answers
while holding meter fixed, with parent-family separation and trivial controls.
It must freeze its own data, budgets and stopping criteria before outcomes.
This result does not authorize that run, automatic extra training, an arithmetic
bridge, new geometry or promotion of any research/release gate.

## Evidence boundary and verification

Only a checksummed reports-only handover was supplied for this preservation.
Its archive and inner checksums were verified privately, and all 256 token rows
were independently rescored against the frozen question inventory. The public
JSON records are **new derivative transcriptions**, not original journal bytes.
Their hashes authenticate this retained derivative, not omitted original files.

The owner reports the original controller audit passed; that audit checks
report/source/score consistency and explicitly performs no tensor/inference
replay. Original raw journals, checkpoints, requests, host environment, logs and
receipts remain local and were not independently audited here. No private
handover bytes, local paths or environment records are published. Public
producer Git pins are provenance context, not a substitute for closed custody.

The new standard-library verifier pins derivative bytes and historical scoring
sources, reuses the unchanged authoritative scorer, and rejects altered,
missing, extra, linked or non-regular fixture files and scorer import caches.
It constructs no model and runs no forward, update or checkpoint replay.
Use quiescent files; observed before/after equality is not hostile-writer proof.

```sh
PYTHONPATH=scripts python -B scripts/check_frozen_music_math_result.py
PYTHONPATH=scripts python -B -m unittest discover -s tests/frozen_music_math_result_v1 -v
PYTHONPATH=scripts python -O -B -m unittest discover -s tests/frozen_music_math_result_v1 -v
```

Affected TRI-I07/08/11/13/14. All historical files and source closures are
unchanged. Learning adequacy, fresh confirmation, Phase 5 and release readiness
remain false. This is additive preservation, not a capability certification.
