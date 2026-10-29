# Frozen musical-family evaluation: reports-only preservation

This new public subset preserves the owner-supplied completed evaluation at
[PR94 merge `99a5896`](https://github.com/QSOLKCB/Trinite/commit/99a5896dd893f9d338e6c2c81c49dba29fbecbb1).
See [the result and limits](../../docs/FROZEN-MUSIC-EVAL-RESULT.md).
No target model was loaded or run to prepare this preservation increment.

## Original and derived bytes

The five files under `original-attempt/` and `audit-receipt.json` are supplied
originals, copied byte-for-byte. `original-SHA256SUMS.txt` is the original handoff
manifest, also unchanged; it lists both included and intentionally omitted
handoff members. It is not a manifest asserting that all those files are present
here. The complete private reports-only archive's SHA-256 and the exact public
subset hashes are recorded in `provenance-DERIVED.json`.

This README and the provenance record are newly derived publication material.
The original report already contains the full summary; the checker recomputes
it from all 320 retained journal records using the unchanged, source-pinned
scorer. Its first 80 train-replay records are compared with the earlier preserved
PR93 final predictions. The original positive training-fit evidence is unchanged.

## Interpretation

Both final checkpoints retain 40/40 training answers. Held-out canonical scores
are 16/40 (seed 17) and 15/40 (seed 29), below the 20/40 training-only majority
control. No held-out quartet is wholly correct. This is a negative held-out
**development** result, not an accuracy gate retroactively added to the protocol.
`completed` identifies execution/report completion, not successful transfer.
Pitch/tempo siblings and the two seeds are correlated; only two structural
held-out parents are present. Full counts and pitch/rest patterns are in the
linked result note. No general music, arbitrary counting or arithmetic-transfer
claim follows.

## Verification boundary

The checker authenticates this subset, the scorer closure, retained report links,
raw-token scores, and earlier final-prediction equality, with before/after checks.
It does not run inference, construct an optimizer, train, repair outputs or
write evidence. The original audit receipt is retained evidence of the owner's
reported audit, not an independently rerun full source-bound audit here.

The original evaluation request and worker log, original producer request/matrix,
checkpoint tensors/metadata, and full environment remain owner-local and are
not supplied in this public subset. Omitted derived handoff notes are listed
in provenance; they are not needed to recompute the retained scores. No full-run
custody, actual inference timing, absence of other attempts, actual weight
immutability, hardware attestation, optimizer replay or fresh confirmation is
established by these public bytes. The trusted interpreter and quiescent checkout
assumption remains; this is not a hostile-concurrent-writer guarantee.
