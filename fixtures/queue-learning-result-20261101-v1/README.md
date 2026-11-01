# Complete queue-learning v1 pilot evidence

Negative two-seed result; see [report](../../docs/QUEUE-LEARNING-RESULT.md).
`evidence.zip` contains 9,201 regular-file entries, 12,269,082 uncompressed bytes
(5,304,893 compressed bytes). `ARCHIVE.sha256` pins its bytes; `SHA256SUMS` pins
every extracted file. `summary.json` and `verification.json` are unchanged copies
of the retained summary and read-only verification output.

Included: all frozen preparations and observations; all run files, attempts,
checkpoints, raw inference and resource journals; independently retained freeze,
preparation, checkpoint, prediction and run anchors; the owner-approval operator
declaration; feasibility and engineering/CI receipts; all 185 source-closure files
and the two original Rust executables under `producer/`. Full Git history, the
virtual environment, compiler distribution and private conversation provenance
are not included. The public producer commit is
`7a2ea15de2fd732ce88a2d1532911391ecdf663d`. Both seeds were executed once; no retry
or failed pilot attempt is omitted. Engineering tests include intentional negative
cases, not additional pilot attempts.

## Read-only reproduction

Validate `ARCHIVE.sha256` in this directory, extract with a standard ZIP utility
into a **fresh external directory**, then validate `SHA256SUMS` against every
extracted member. Do not overwrite historical files. The included binary files
are preserved for exact byte-identity reproduction, not an invitation to trust
unknown binaries. Use the original hash-locked CPU environment (including the
recorded Python/platform/CPU fingerprint); a foreign environment rejection is
intentional and must not be patched away. These commands do not retrain.

```sh
sha256sum -c ARCHIVE.sha256
# EVIDENCE is a fresh absolute destination outside any source checkout.
unzip evidence.zip -d "$EVIDENCE"
(cd "$EVIDENCE" && sha256sum -c /absolute/path/to/SHA256SUMS)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# PYTHON is the matching pinned CPU virtual environment's absolute interpreter.
"$PYTHON" -I -B "$EVIDENCE/producer/scripts/run_queue_learning_v1.py" verify \
  --frozen "$EVIDENCE/frozen" \
  --freeze-sha 71e56232314dc32bb0b55dff9b15b3587a44d35b7f1703962466aa3e0f7b90c2 \
  --anchors "$EVIDENCE/anchors/preparation-anchors.json" \
  --anchors-sha 99ab87c9dd4f8fd4cd4f6088dc0143476b851afbd10e37e3dee9a91724d3be93 \
  --run "$EVIDENCE/run" --run-anchors "$EVIDENCE/anchors/run-anchors" \
  --checkpoint-sha 5003f199abeaa3205d66338c72421aa6344d60ca0e8cb6158181737bddfc131b \
  --barrier-sha 161e917035a5d6432847f4bea7c05472c6b4ad49cbeb81faa469c9e514f94435 \
  --run-sha c4a7a84b34475bced8c7b89c1e3e0c4a8dbcf891b246e31df41e8830163dcef9
```

Expected output: `verified: true`, `optimizer_replay: false`,
`operator_blindness_proved: false`. This authenticates custody and reruns queue
scoring/safe checkpoint verification, not optimization or scientific adequacy.
The approval JSON is an operator declaration, not a cryptographic owner signature.
No newly authored extractor, verifier or training workflow is introduced here.
