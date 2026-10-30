# Reports-only controlled pitch replacement result

This is an additive, byte-preserved public subset of the owner's PR96 result,
not a complete run or new experiment. See the [derived interpretation](../../docs/MUSIC-PITCH-RESULT.md).

`original-run/` contains 23 unchanged supplied originals: the matrix,
development gate and audit stdout; attempt, completion, report and diagnostic
journal for each of four cells; and the development attempt, completion, report
and prediction journal. The checker authenticates their fixed byte identities,
regenerates the declared inventory and scores all 800 retained observations.

Only **21 of 43 matrix-bound files** are present. The matrix and audit stdout
are the other two supplied originals; “23 originals” does not mean full custody.
Absent matrix-bound files are each cell's `diagnostic-predictions.json`,
`final-checkpoint.json`, `final.safetensors`, `request.json`, `worker.log` (20),
plus `development/worker.log` and root `request.json` (2). Full requests,
tensors, metadata, environment commitments and execution logs remain owner-local.

`source-SHA256SUMS` is the unchanged source handoff manifest, including names of
four omitted owner-derived operational files; it is not a manifest claiming
that the whole original handoff is present here. `provenance-DERIVED.json` and
this README are new derived descriptions, not original producer receipts.

The source archive SHA-256 is
`d7ea7dd7c223cdfe8332a5a1c037c9a6bd9656413fe992206292c085722f38df`.
The original matrix SHA-256 is
`59c92c9217a35b2b5a5594fa9ba839a903807398b30788a1c286095f4b69b6db`.
The original request identity is an owner-retained receipt, not independently
verified request custody in this subset. Available links and score consistency
do not attest actual model execution, optimizer replay, timing, temporal gate
execution, omitted tensor state or absence of other attempts. The supplied audit
receipt records an owner-local audit, which cannot be rerun from this subset.
The verifier uses only the standard library and performs no model operations.

```sh
python -I -B scripts/check_music_pitch_result.py
```

Run the command from the repository root. Preserve historical evidence and all
source closures unchanged; no target run, curriculum change or promotion follows
from this preservation increment.
