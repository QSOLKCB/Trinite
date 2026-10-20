# Frozen native batch-composition conformance v2

Status: all three pinned CPU matrix runs and exact hosted inference replays passed
on the published v2 producer, followed by independent offline evidence review.

The original published v1 producer completed its first numerical run, but strict
replay rejected canonical JSON before inference: dictionary key sorting changed
the order of recomputed comparison edges. v2 explicitly uses the frozen layout
order. Model operations, tolerances, workload, budgets and the exact replay gate
are unchanged. New request/report/protocol schemas are v2; existing module and
launcher paths remain for one implementation. Original results and the failed
replay remain separate. Original source recovery uses the public commit/tree and
all 149 exact content/Git-blob identities, not a published self-contained ZIP.

This separate inference-only lane tests the four actual final-step-60 endpoints
from [the fixed-code-refit training pilot](SCALE-REFIT-TRAINING.md), not an untrained
toy and not the external NCM model. All original model operations, checkpoint
readers, numerical producers, fixtures and source closures remain unchanged.
It is engineering conformance on visible development inputs, not a new learning,
fresh-generalization, quantizer-selection, performance or release experiment.

## Frozen inputs and interpretation

The [protocol](../scripts/batch_composition_v1/protocol.json) binds the exact PR #49
report, all absmean/refit × seeds 0/1 final endpoints, the admitted formal dataset
and manifest, and the previous [frozen-decoding result](FROZEN-DECODING.md).
Every one of the 36 training and six validation prompts is retained in fixed
train-then-validation original fixture order. Six test items are excluded from
numerical execution. No checkpoint selection, updates, optimizer construction,
weight downloads, GPU execution, NCM change or paid compute is involved.

Two distinct lanes prevent an important interpretive mistake:

1. Fixed-prefix comparison uses BOS plus the original prompt, then BOS plus that
   prompt and the first digit from this exact endpoint/item's previously recorded
   syntax trajectory. It compares all 259 next-token logits at each stage, along
   with raw-greedy and syntax decisions. These reference tokens are previous native
   model outputs, not gold answers or external teacher data. This is forced
   reference-trajectory conformance, not independently generated quality.
2. Greedy and syntax generation also run independently under every layout. Each
   receives a label-free prompt projection only, with the unchanged two-token and
   two-active-forward-per-item limits. Syntax's forced EOS still costs a real
   forward. No repair, extra tokens, ignored EOS or completion beyond context is
   allowed. Raw greedy BOS/PAD outputs are retained rather than silently stripped.

A fixed-prefix match does not prove independently evolving trajectories match.
Independent numerical comparisons only use steps whose prefixes really match;
divergent continuations are counted as unaligned, not compared as if identical.
Exact token sequences and stop reasons are separate requirements.

These checkpoints previously generated immediate EOS in raw greedy and constant
syntax answers by seed. Passing batch conformance would not cure those limitations
or demonstrate input-sensitive arithmetic. Seeds name historical initialization;
there is no sampling or assertion of sampled RNG-invariant trajectories.

## Layouts and numerical gates

The eight preregistered layouts are unpadded singleton; singleton padded to 32;
contiguous B8 padded to 32; contiguous B4 partitions; reversed rows within B8;
even-index-then-odd-index B8 recomposition; B8 padded to its current longest live
prefix; and B8 padded to 32 with alternate masked payloads `[0,257,255]`.
Final partial batches are retained. Right-padding preserves absolute live token
positions; left-padding is not supported and would change semantics.

Every layout is compared against the unpadded singleton. Additional directed
edges compare padded singleton against padded B8, then padded B8 against B4,
reordering, recomposition, dynamic padding and alternate masked payload. This is
not an all-pairs claim: numerical tolerances are not transitive.

The historical native oracle's `atol=2e-6`, `rtol=1e-5` are frozen before outcomes.
Every coordinate must satisfy `abs(candidate-reference) <= atol + rtol*abs(reference)`.
Full final-live logit vectors are retained as exact finite float32 hex values.
Maximum absolute error, normalized error, descriptive relative error, violation
coordinates and exact-equality counts are separate. All returned model logits,
including masked-out positions, must be finite; masked output positions must be
zero. Selection gathers each row's last live index, never its padded last column.

Both numeric and exact-decision gates are required. Exact ties use lowest token
ID; top-two margins and near-tie diagnostics never waive a decision mismatch.
Within-group finished rows remain unchanged carriers and receive no further
emitted tokens. Their actual computed rows still count against the resource bound.
Model weights, tied storage, eval state, absent gradients and Python/NumPy/Torch
RNG state are checked unchanged. Required trainability flags are preserved.

## Resources, evidence and reproduction

The unchanged hash-locked Linux x86_64 CPU environment uses float32, one thread,
strict deterministic algorithms, eval and no-grad, with no autocast. The bound is
3,000 physical model calls and 8,064 computed rows per report, maximum batch eight
and context 32. The fixed fresh worker has a 300-second hard deadline including
imports/loading; reports and partial streams are bounded at 64 MiB. The workflow
has a 20-minute limit. These are conservative limits, not fit/speed guarantees.
Whole-worker wall time and peak RSS remain separate observations.

Publish the complete producer/protocol before target execution. A request binds
all relevant source bytes, launchers, tests, existing publication helper, protocol,
input and checkpoint identities, explicit layouts and current CPU environment.
The existing exact checkpoint importer and atomic no-clobber helper are reused,
not rewritten. No historical fixture is regenerated or resealed.

```sh
export PYTHONPATH=src:scripts
python -B -m unittest discover -s tests/batch_composition_v1 -v
python -B -m unittest discover -s tests/batch_composition_review -v
request_id=$(python -B scripts/run_batch_composition.py freeze --request /new/request.json)
python -B scripts/run_batch_composition.py run --request /new/request.json \
  --request-identity "$request_id" --output /new/report.json
python -B scripts/run_batch_composition.py verify --request /new/request.json \
  --request-identity "$request_id" --output /new/report.json \
  --expected-identity sha256:INDEPENDENTLY_RETAINED_REPORT_DIGEST
```

Use the public launcher for actual experiments. Direct Python helpers and the
hidden worker entry are internal APIs, not alternatives to the parent deadline.
Each run/replay gets a unique attempt directory containing its invocation,
completed-layout/cell JSONL, worker log, status, measurements and computed report
when available. A nonconformant complete result is retained and exits 2; failures,
timeouts and exact replay mismatches remain evidence. Replay stores both computed
bytes and an explicit false-match receipt before rejecting a mismatch. Existing
outputs, including symlinks, are never replaced. Atomic visibility is not a claim
of directory crash durability. A hard-killed parent may leave only its attempt.

Exact replay recomputes the entire deterministic report in the same bound
source/input/environment. Batch-comparison tolerance does not relax exact replay.
Observed timing, RSS, random attempt names and parent status are excluded from
scientific report equality. This is inference replay, not optimizer replay or
PROVENANCE closed custody/hardware attestation.

## Retained outcomes and limits

The [v2 commitment](../fixtures/batch-composition-v2/commitment.json) retains all
three matrix requests, complete raw-vector reports, all original run/replay
members through a lossless storage map, and independent offline audit receipts.
Python 3.12 was designated the primary retained environment before matrix outcomes;
all other matrix outcomes are also retained rather than selected for performance.

| Python | Actual hosted CPU/kernel | Maximum paired absolute logit difference | Outcome |
| --- | --- | --- | --- |
| 3.11.17 | AMD EPYC 7763 / AVX2 | 1.1920928955078125e-6 | Both gates and exact replay pass |
| 3.12.15 (primary) | AMD EPYC 7763 / AVX2 | 1.1920928955078125e-6 | Both gates and exact replay pass |
| 3.13.16 | Intel Xeon Platinum 8573C / AVX512 | 1.430511474609375e-6 | Both gates and exact replay pass |

Every report covers four endpoints and 10,920 same-prefix full-vector comparisons,
with zero violating coordinates and no exact token/stop differences. Observed
compute was 2,500 physical forwards and 6,720 computed rows per report, inside the
unchanged bounds. Maximum normalized error across the matrix was about 0.0424 of
the declared coordinate-wise allowance. Floating-point values were **not bitwise
invariant across batch layouts**. The environments also differ in CPU/kernel and
Python, so this is not an attribution to Python or a cross-environment bitwise
claim. Exact replay is asserted only within each bound environment.

Producer publication preceded these runs:
[`f05f8396`](https://github.com/QSOLKCB/Trinite/commit/f05f8396a98fb77134e26486fa9482fcca0913b0).
The [hosted run](https://github.com/QSOLKCB/Trinite/actions/runs/39463163844) completed
all three CPU jobs; each fresh run and fresh replay process stayed within 300
seconds. The reviewer independently checked source, raw vectors, call accounting,
all partial streams and exact report/receipt bindings without executing a model.
This is hosted inference replay plus independent offline review, **not** a claim
of a second local inference replay or optimizer replay. The temporary local CPU
environment disappeared during the task; its unfinished historical aggregate is
not claimed passed, and it was not replaced by another GPU or dependency variant.

The primary uncompressed report is
`sha256:4723415336b3a6d398b17fba9c18345f951b0723cd61a0b51f635ad1bf4d3d00`.
All reports/large partial streams use lossless XZ with externally pinned compressed
and uncompressed identities and a 64 MiB expanded-file bound. Each matrix's
`members` map reconstructs all 19 original hosted artifact members; identical
bytes are stored once. The corrected [offline auditor v3](../scripts/audit_batch_composition_v3.py)
operates on that original extracted member layout, never on a model. Use it only
on the hash-checked retained packet, with the repository as `--source-root` and
its [prepublication review](../fixtures/batch-composition-v2/prepublication-review.json)
as `--approved-review`. Its output must be a new path outside the artifact tree,
including after symlink resolution. Root evidence tests perform actual data
admission and strict recorded-report validation without Torch; they do not
substitute for the retained hosted inference replay.

```sh
python -B scripts/audit_batch_composition_v3.py \
  --artifact-dir /path/to/extracted-original-hosted-packet \
  --source-root . \
  --approved-review fixtures/batch-composition-v2/prepublication-review.json \
  --output /separate/new-offline-audit.json
```

### Offline auditor correction

The [historical v2 audit script](../fixtures/batch-composition-v2/audit_ci_artifact.py)
and its original receipts remain byte-preserved for provenance, **not as the
current verification entry point**. Review reproduced three transport defects:
Python optimization removed its assertion checks, an output inside the artifact
tree could be created after its unchanged-input check, and a contradictory
`timeout` outcome with exit code zero could be certified as an exact replay.
Do not use that historical script to issue a new verification receipt.

The additive v3 auditor uses explicit validation exceptions under normal Python,
`-O`, and `PYTHONOPTIMIZE`; rejects resolved output aliases inside the audited
tree before reading it; and validates exact integer status codes against the
runner's outcome pairings. Only `completed/0` or `completed_nonconformant/2`,
consistent with the report, can support a completed attempt. A failure or timeout
is never certified as replay. The v3 receipt binds the exact auditor bytes,
approved-review bytes, original producer/source map and all input artifact bytes.
The auditor, source, approved review and artifact inventory are checked unchanged
before exclusive output publication. This fixes offline evidence semantics only;
it does not change or reseal any numerical protocol, request, report or old
receipt, and does not rerun inference.

The [v3 audit commitment](../fixtures/batch-composition-audit-v3/commitment.json)
retains corrected offline receipts for all three unchanged hosted packets. The
3.11 packet was checked normally, primary 3.12 with `-O`, and 3.13 with
`PYTHONOPTIMIZE=1`, all using the local standard-library Python 3.12.14 auditor.
These are optimization-mode checks of the auditor, not new runs of those model
Python environments. All three strict full-vector audits passed, and independent
review confirmed unchanged scientific values and complete artifact inventories.

The [original v1 commitment](../fixtures/batch-composition-v1/commitment.json)
retains the first local result, failed replay, partial streams and failure audit.
Its source-recovery manifest names
[`7be8c442`](https://github.com/QSOLKCB/Trinite/commit/7be8c44288ea645ed8ac04e00abebbe452e2e567),
tree `07c93f7ad42f5d1dc13cd2c4f433cc748bf50fff`, and every original source identity.
The optional source ZIP is not published; no check silently skips a declared
archive. The public Git commit is the explicit original-source recovery basis.
The original replay failure is not converted into a successful replay receipt.

## Attribution and compatibility

The testing idea is informed by DeepInfer's pinned
[batch-invariant test](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/tests/models/test_batch_invariant.py).
It samples a batch and forces singleton `golden_token_ids` to the batch trajectory
before comparing results. It does not establish independent sampling invariance.
This implementation is original MPL-2.0 code; no Apache-2.0 donor source, kernels
or runtime are copied. No new third-party package is introduced.

Affected requirements: TRI-I01/I02/I03/I05/I06/I07/I08/I10/I11/I12/I13/I14.
Native model/tokenizer/quantizer/scorer behavior, prior negative findings, external
NCM's batch-1 protocol and all scientific/release gates remain unchanged.
