# Reports-only negative musical training-fit evidence

This new public subset reconstructs eight prediction/journal files from the
owner's checksummed REPORT.md. Each reconstructed file's SHA-256 equals the
corresponding original-file hash printed in that report. This is stronger than
an unbound transcription, but **the original raw files, tensors and execution
receipts were not supplied**. It is not independent numerical execution or
closed original-run custody. The private reports and environment remain private.

## Exact derivation

For each of the eight 40-row JSONL report blocks, retain the `raw` object in row
order, without its row-number/display wrapper. Serialize JSON with sorted keys,
UTF-8, compact separators and a trailing newline. Prediction files contain
`{"predictions": [...]}`; journals contain one such raw record per line. Replace
only the relative directory separator in public filenames with `--`; no record
field changes. Compare each resulting SHA-256 to the corresponding report anchor.
All eight match. Each prediction/journal pair is duplicate evidence of the same
40 observations, not separate trials. There are 160 observations, not 320.

`summary.json` is newly derived with the unchanged source-pinned musical scorer.
`provenance.json` distinguishes observed token metrics from owner-reported run,
audit and exposure statements. No original request, matrix, checkpoint, private
archive hash, environment fingerprint or local pathname is included.

Both final seeds always emit `7` then EOS: 20/40, the majority baseline, with no
complete quartet or correct contrast edge. Both initial endpoints score 0/40.
The training-fit gate failed; held-out model evaluation is **unrun**, not zero.
See [the result note](../../docs/MUSIC-CONTRAST-RESULT.md) for interpretation.

Run the standard-library checker with:

```sh
PYTHONPATH=src:scripts python -B scripts/check_music_contrast_result.py
```

No model, training, replay, bridge or later experiment is invoked by this check.
