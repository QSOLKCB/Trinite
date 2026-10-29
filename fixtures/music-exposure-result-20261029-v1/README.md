# Reports-only musical exposure result

This additive public subset preserves **positive exposed-training fit**, not
held-out musical performance or arithmetic transfer. Both prescribed seeds
score 0/40, 20/40 and 40/40 exact answers plus EOS at steps 0, 64 and 640.
At step 640 both also achieve 10/10 complete quartets and 40/40 both-correct
contrast edges. These are six observations of the same 40 training rows,
not 240 independent examples. Held-out evaluation remains unrun.

## What is retained

- Six supplied original predictions JSON files and six supplied original
  journal JSONL files, copied without changing any byte. The flattened names
  preserve seed and checkpoint; each journal duplicates its corresponding
  40-row prediction observation and is not additional evidence.
- A newly computed `summary.json`, using the unchanged, source-pinned
  authoritative musical-exposure scorer. It includes all checkpoint scores,
  contrast metrics, train-only controls, output diversity and strata.
- A newly authored `provenance.json` separating independently checked subset
  integrity and scores from owner-reported execution, timings and audit.

All eight step-0/64 prediction and journal files are byte-identical to the
previously preserved [negative contrast result](../music-contrast-result-20261028-v1/README.md).
The checker pins the historical files' known hashes as well as the new subset.
This establishes prediction-byte agreement, not identical hidden states,
optimizer replay, or independent original-run custody.

## Evidence boundary

The owner reports one completed, fixed-budget trajectory per seed at producer
commit `c239acb6e7b0b5c49798debbae92ca1e1264a64c`, with no restart, retry,
checkpoint election, extra seed or held-out model call, and a successful local
structural audit. These execution and audit claims were **not independently
replayed or closed here**. Timings in provenance are owner-reported, not a
benchmark reproduced by this checker.

The original request, matrix, checkpoint tensors and metadata, completion and
attempt receipts, worker/command logs, source/environment commitments and audit
receipt were not supplied in this reports-only subset. Private handoff reports,
local paths, environment fingerprints, archive checksums and original private
run commitments are intentionally not published. Subset hashes cannot certify
those missing originals, training dynamics, hardware provenance or no leakage.

This is a small finite training-fit result. Longer exposure also changes the
number of updates and optimizer trajectory; it does not isolate repetition
causally. No generalization, arbitrary-length counting, music-to-maths transfer,
fresh confirmation, learning adequacy, Phase 5 or release gate is established.
Historical fixtures and protocols remain unchanged. No target run, inference,
replay, bridge or next experiment is authorized by this preservation.

## Read-only check

```sh
PYTHONPATH=src:scripts python -B scripts/check_music_exposure_result.py
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_exposure_result_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_exposure_result_v1 -v
```

The checker uses only the standard library. It verifies exact membership and
bytes, source/import resolution, canonical schemas and record context, duplicate
journal agreement, historical byte parity and recomputed scores. It does not
construct a model, load tensors or execute predictions or updates. The trusted
interpreter and a quiescent checkout remain prerequisites; observed before/after
checks do not defeat arbitrary concurrent hostile filesystem modification.
