# Reports-only arithmetic probe preservation

This is a **derivative public transcription**, not a byte-exact original run
subset or a replacement request/report. The owner supplied a checksummed archive
containing REPORT.md, HANDOVER.md and SHA256SUMS.txt on 28 October 2026.
Its two reports and archive checksums were checked privately. All 256 reported
prediction rows were independently rescored against the frozen PR #88 questions.

`predictions.json` retains every report-transcribed generated token ID in frozen
checkpoint/question order. No output is repaired or excluded. `summary.json` is
recomputed by the original standard-library scorer. `provenance.json` records
public producer pins, evidence class, explicit exclusions and derivative hashes.
Derivation: the report's complete prediction table was parsed without executing
any embedded commands. Each checkpoint label was split into subset, integer seed
and integer step; its question ID and numeric generated-token list were copied.
Rows were reordered to the frozen protocol's checkpoint/question order. Prompt,
gold, display escapes and reported correctness were independently checked but
are not fed into the scorer as predictions. No token normalization or answer
repair occurred. The original scorer computes summary.json from these records.
These new identities bind only these public derivative bytes, not original
checkpoint execution or omitted source artifacts.

The private handover, source archive identifiers, environment, local paths,
checkpoint tensors, raw journals, requests, logs and audit receipts are omitted.
The owner retains the full original run locally. The reported completed run and
local audit passed are owner evidence, not independently replayed findings here.

See [result and limits](../../docs/FROZEN-MUSIC-MATH-RESULT.md).
Read-only verification: `PYTHONPATH=scripts python -B scripts/check_frozen_music_math_result.py`.
No Torch, model loading, training, inference or original-run audit is performed.
