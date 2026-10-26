# Coverage-balanced addition: mixed/negative development result

Recorded **26 October 2026**, after the owner-authorized local run of the frozen
[PR #74 recipe](BALANCED-ADDITION.md), on merged PR #75 commit
[`37ff4434fb51b176540c1a1e4764f7fe64f5953b`](https://github.com/QSOLKCB/Trinite/commit/37ff4434fb51b176540c1a1e4764f7fe64f5953b).
All six predeclared cells completed. No new training, checkpoint selection,
retries, answer repair or model inference was performed to preserve this result.

## What happened

All counts below are raw complete answers **including EOS** at the fixed final
4,000-update checkpoint. Unique training is not repeated scheduled visits.

| Arm / seed | Unique training | Seen anchors | Primary in-range | All-correct in-range families | Shared-answer | Broadened-answer | Range/length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control 0 | 20/20 | 20/20 | 4/56 | 1/15 | 4/20 | 0/36 | 0/138 |
| Candidate 0 | 144/144 | 20/20 | 8/56 | 2/15 | 4/20 | 4/36 | 0/138 |
| Control 1 | 20/20 | 20/20 | 4/56 | 1/15 | 4/20 | 0/36 | 0/138 |
| Candidate 1 | 144/144 | 20/20 | 0/56 | 0/15 | 0/20 | 0/36 | 0/138 |
| Control 2 | 20/20 | 20/20 | 4/56 | 1/15 | 4/20 | 0/36 | 0/138 |
| Candidate 2 | 144/144 | 20/20 | 0/56 | 0/15 | 0/20 | 0/36 | 0/138 |

The candidate broadened output support and fit every one of its 144 unique
training rows without losing the original 20 seen anchors. **It did not produce
a stable transfer improvement:** seed 0 improved by four in-range renderings,
while seeds 1 and 2 lost the control's four. The control solves family {5,8} in
all three seeds; candidate 0 solves {5,8} and {2,6}. No other development family
was completely correct. The range/length slice failed in all six cells.

The primary failure is not merely formatting: all wrong in-range predictions
are classified as wrong values except one missing-EOS control-0 prediction.
Range/length includes wrong values and immediate/premature EOS. The original
per-row outcomes and full failure/pair maps are retained, not just this table.

Each cell used the same 86,528-parameter dense architecture, 4,000 updates,
32,000 scheduled visits, 76,800 answer-plus-EOS targets and 284,800 unpadded input
tokens. Candidate training diagnostics cover more unique examples and cost more;
equal scheduled tokens do not mean equal total wall time or experiment compute.
The retained reports' exposure fields agree with the frozen schedule. This
public subset does not independently replay those updates or training history.

## Interpretation and next decision

**Broader coverage alone was not a reliable generalisation fix under this exact
recipe and budget.** Fitting more training rows is useful evidence of training
capacity, not proof of a learned addition algorithm. This result does not prove
universal architectural incapacity, establish memorisation as a causal mechanism,
or show that more compute, a different representation, ternary weights or KV
caching would fix transfer.

All 56 candidate in-range gold answer-plus-EOS sequences occur in its training
answer support, yet 48, 56 and 56 rows fail respectively. Absent full-answer
output support therefore cannot explain those particular failures.

This is a data-recipe comparison: family coverage, output/operand frequencies
and repetition change together. It cannot isolate output-digit support as the
cause of either improvement or regression. All families are development-exposed;
56 rows are correlated renderings of 15 families, not 56 independent trials.
Seeds are retained paired descriptive observations, not a selected winning seed
or a significance claim. Range and operand digit length shift together.

The existing [position preparation](POSITION-COUPLING.md) is conformance only.
A next learning PR may freeze a separate format-matched, CPU-first representation
comparison with explicit carry coverage, offsets, held-out family inventories,
paired initialization, budgets and stop criteria. It is not implemented or
executed by preserving this result. Do not retune this matrix or silently treat
its exposed families as fresh confirmation. KV-cache and efficiency work remain
separate from generalisation claims. All historical negative evidence, immutable
checkpoints and scientific gates stay unchanged.

## Evidence, privacy and verification limits

The [public subset](../fixtures/balanced-addition-result-20261026-v1/README.md)
retains 43 original files byte-for-byte: original matrix plus seven records per
cell. It covers **1,656 final predictions** (492 unique-training observations
across cells plus 1,164 development observations); repeated examples across
seeds/arms are not independent cases. No seed or final prediction is dropped.

- Original request: `sha256:897d9dae78eed8c50e6c899610810c49867ffec5067a941cc60ea7e4828f76d7`.
- Original matrix: `sha256:930040ebe307083052bf8f1c004af20f65ca697bce94b4e5a2c4331b1294d412`.
- Original private archive: SHA-256 `b44e926edbe7bf387a5e50ea6ef9f7b9feabc648c437ebd9499078eb3c2784c4`.

Private read-only review checked archive/manifest hashes, the original 216-file
balanced producer closure against published sources, admitted corpus identities,
all retained histories and prediction metrics, provenance links and static tensor
model identities. Independently recomputed arithmetic labels, all 1,656 final
predictions, failure categories and family counts agreed. These checks are not
fresh numerical prediction or optimizer replay, or hardware attestation.

The newly authored `source-bindings.json` is explicitly a **derivative private-
audit receipt**, separate from original bytes. The omitted original request and
its environment cannot be reconstituted from this source-only extract. Original
matrix/completion references to absent private/tensor artifacts are unchanged;
this public subset is **not closed custody**, complete checkpoint verification
or independent-host replication. Intermediate training summaries are preserved
in original reports but not recomputed without their omitted raw artifacts.

The checker reuses authoritative admission, score validation and descriptive
functions. Those import Torch, so use the pinned CPU environment; no decoder is
constructed, no tensors loaded and no numerical run is launched. It rejects
changed hashes, missing/extra files, symlinks, foreign source roots and changed
producer sources, verifies all six original completion linkages, and recomputes
final counts/maps read-only. Existing producer sources and bound contracts are
not modified to accommodate this later result.

```sh
PYTHONPATH=src:scripts python -B scripts/check_balanced_result.py
PYTHONPATH=src:scripts python -B -m unittest discover \
  -s tests/balanced_result_v1 -v
```

Affected TRI-I03, I07, I08, I11, I13 and I14: evidence preservation and bounded
verification only. No numerical semantics or source closure changes. Original
fold/composed adequacy, fresh confirmation, independent replication, format
selection, geometry-on Phase 5, scaling and release remain separately gated;
`learning_adequate`, `phase5_ready` and `release_ready` remain false.
