# Frozen-model decoding comparison v1

Status: completed visible-fixture comparison; syntax fixes format, but outputs are constant answers.

This additive checkout-only comparison follows [PR #49's training pilot](SCALE-REFIT-TRAINING.md).
It imports all four existing final-step-60 endpoints (absmean/refit, seeds 0/1)
from the exact retained report, without training, optimizer construction, checkpoint
selection, retrieval, tools or across-item memory. Original sources, fixtures,
model IDs, scorer semantics, and training replay evidence remain unchanged.

## Frozen comparison

The [protocol](../scripts/frozen_decoding_v1/protocol.json) compares ordinary
lowest-ID-tie greedy generation against syntax-masked greedy on identical checkpoint
weights and prompt bytes. Both modes have two generated-token and two model-forward
call caps, including EOS. Even syntax's forced EOS consumes a real model forward.
No repair, completion or extra tokens are allowed after either cap or context exhaustion.
All four endpoints and all 36 training plus six validation examples are retained.
Test items are admission-audited as part of the historical input but never generated
or scored. This is an already visible development fixture, not new held-out evidence.

The grammar is exactly ASCII `[0-9]` followed by EOS. It includes 9 even though
this fixture's labels do not reach 9. It does not inspect a modulus, solve arithmetic,
look up answers or receive labels. Generation accepts only model, prompt and mode;
labels are used afterward in scoring. This is an explicit task-format prior, not
learned syntax or a model-only capability gain. The primary criterion remains the
historical exact complete answer plus EOS. Syntax validity, malformed sequences,
valid-but-wrong semantic errors, actual tokens and actual forward calls are separate.
No teacher-forced accuracy or partial credit substitutes for the primary outcome.

## Import and replay boundary

The original report's externally fixed SHA-256 is checked before any tensor is used.
Compressed and decompressed reads are bounded at 16 MiB; canonical JSON and exact
historical source identities are required. Only the four exact final endpoint records
are accepted. The loader checks config, input, protocol, environment consistency,
model inventory, tensor names and cardinalities, and uses the existing finite
canonical hex parser with exact float32 roundtripping. It preserves tied embeddings.
No pickle, downloaded code, optimizer, or arbitrary checkpoint input is accepted.

The historical model validates trainability flags, so flags remain unchanged;
`.eval()` and `torch.no_grad()` govern all inference. Model identities and absent
gradients are checked after each mode. Every ordinary prediction must also equal
its retained PR #49 prediction. Historical training replay remains separately
retained evidence, not reexecuted or relabeled as inference verification.

A pre-run request binds producer source, protocol, original report, every checkpoint,
current pinned CPU environment and original inputs. Exact read-only replay compares
all deterministic report bytes. Observed per-item wall latency is kept in a separate
sidecar, not falsely compared for exact replay. It includes model invocation and
masking; serial greedy-then-syntax ordering is fixed, not randomized or a speed study.
Seeds identify original initialization, not stochastic decoding. There are no sampling
RNG draws. The grammar starts at an empty prefix for each item and exposes no
prefix-resume/dead-end repair interface. Foreign inference environments have their own bound requests; original
training environment metadata stays intact, and no cross-host equality is claimed.

## Current execution: transport v2

Use `scripts/run_frozen_decoding_v2.py` for all new requests and runs. The original
`run_frozen_decoding.py` remains byte-frozen solely for historical v1 reproduction;
it retains its original non-atomic publication and same-stem sidecar limitations.
Do not use that legacy launcher for new artifacts. Its original workflow remains
historical conformance. The separate v2 workflow exercises the current transport.
Changing the historical launcher would invalidate its retained source-bound request
and evidence, so the fix adds a transport wrapper, not another numerical producer.

V2 atomically publishes complete request, timing and final report bytes with a
same-directory temporary file, complete-write loop, flush/fsync, then hard-link
creation with no-clobber semantics. Existing destinations, including symlinks and
racing files, are never replaced. Catchable exceptions and KeyboardInterrupt clean
the temporary file; abrupt SIGKILL may leave a hidden temporary file, but cannot
expose an incomplete final. This is atomic visibility, not a promise of directory
crash durability. If cleanup fails after successful linking, an error may accompany
a complete final artifact; inspect the destination before retrying. Filesystems must support same-directory hard links; unsupported
operations fail closed without falling back to overwriting rename/replace.

Each attempt retains sidecars named from the **entire output filename** plus a
fresh attempt nonce: `report.json.attempt-<nonce>.cells.jsonl` and
`report.json.attempt-<nonce>.timings.json`. `report.txt` has a different prefix.
Paths are logged before generation. A failed producer or final publication can
leave completed-cell/timing sidecars, deliberately preserved as failure evidence;
a retry at the same final output path gets new sidecars and does not erase them.
No nonce or observed timing enters deterministic scientific replay.

Canonical `json_bytes` already appends one LF; original retained JSONL was valid.
V2 explicitly tests multi-record line parsing rather than adding a second newline.
The v2 request binds transport source plus the unchanged v1 inference request;
the v2 report wraps a freshly computed unchanged v1 inference report. This does
not rewrite or reseal original v1 artifacts. Use the legacy launcher only to
verify the original retained v1 report, and v2 to verify new v2 transport reports.

## Execution and assurance

```sh
export PYTHONPATH=src:scripts
python -m unittest discover -s tests/frozen_decoding_v1 -v
python -m unittest discover -s tests/frozen_decoding_transport_v2 -v
request_id=$(python scripts/run_frozen_decoding_v2.py freeze --request /new/request.json)
python scripts/run_frozen_decoding_v2.py run --request /new/request.json \
  --request-identity "$request_id" --output /new/report.json
python scripts/run_frozen_decoding_v2.py verify --request /new/request.json \
  --request-identity "$request_id" --output /new/report.json \
  --expected-identity sha256:YOUR_RETAINED_DIGEST
```

The producer/protocol must be published before target outcomes. Output and sidecars
are exclusively created; partial cells and failure logs must be preserved. The
maximum report performs 336 generations and 672 forward calls, single-threaded on
hash-locked CPU Torch; replay repeats inference only. Dedicated CI has a 20-minute
job limit and retains completed or failed records. Unit tests cover historical greedy
parity, EOS cost, all ten digits, tie-breaking, context exhaustion, invalid logits,
metric partitions, tamper rejection, and exact tensor import without training.

Affected requirements: TRI-I01/I02/I03/I05/I06/I07/I08/I10/I11/I12/I13/I14.
Numerical model, quantizer and original scorer compatibility is preserved. This
changes decoding policy only. No learning adequacy, original folded/composed-answer
gate, fresh confirmation, format selection, geometry, scaling or release is unlocked.


## Retained result: constant answers, not input-sensitive reasoning

The producer was published before outcomes at
[baff6270](https://github.com/QSOLKCB/Trinite/commit/baff6270ce868a31bf74f7da41abbed8b3d6b3dc).
The [commitment](../fixtures/frozen-decoding-v1/commitment.json),
[request](../fixtures/frozen-decoding-v1/request.json),
[full report](../fixtures/frozen-decoding-v1/report.json.gz) and
[timings](../fixtures/frozen-decoding-v1/report.timings.json) retain every cell.
Read-only inference recomputation exactly matched report identity
`sha256:7f5b8c6819ab5162546f881a8105148e409b17c37689bd72e52e9799fc06a688`.

Ordinary greedy emits EOS immediately for every example: 0/36 training and 0/6
validation complete answers in all four endpoints, identical to PR #49. Syntax
masking produces valid digit-plus-EOS output for all examples, but **every seed-0
output is the constant digit 2, and every seed-1 output is the constant digit 0**,
for both absmean and refit. Neither arm shows input-sensitive arithmetic here.

The resulting exact-answer counts are 12/36 training in every cell, and 4/6
validation for seed 0 versus 2/6 for seed 1. Those counts equal the respective
constant answers' prevalence: training labels contain twelve zeros and twelve twos;
validation contains two zeros and four twos. This is descriptive inspection of
retained predictions, not an additional tuned baseline or selected checkpoint.
All 24/36 training semantic errors and the 2/6 or 4/6 validation semantic errors
remain visible. Refit and absmean are identical on these generated outputs.

**Equal caps did not mean equal actual compute:** ordinary greedy uses one token
and one forward per item, while syntax uses two of each. Across the four endpoints,
ordinary totals 168 tokens/forwards and syntax totals 336. Observed whole report
runtime was about 5.50 seconds and replay about 4.37 seconds; the full per-item
latency sidecar is retained, but fixed order, small workload, masking overhead and
unequal actual calls preclude a speed/compute-efficiency claim. Peak process RSS
was approximately 287 MiB. These are observations of this host, not model guarantees.

The format prior improves the harness's exact-answer score by preventing empty
answers. It does not establish improved learned reasoning, new learning, fresh
held-out generalization or a quantizer advantage. All prior gates remain blocked.


## Transport v2 verification

The [v2 producer](https://github.com/QSOLKCB/Trinite/commit/c42bba23ba54832be2d703f15a87d5a8eb885864)
was published before its actual transport smoke. All twelve fault-injection,
no-clobber, race, retry and JSONL tests passed, as did the nine original decoding
tests and original evidence audit. The new [v2 commitment](../fixtures/frozen-decoding-transport-v2/commitment.json)
retains the source-bound request, complete report, unique-attempt sidecars and
run/replay logs. Exact v2 replay matched
`sha256:24d6a2f1574cd64d05b5354222dea18c62909c76035f18c5c1bdffa7a3efa3de`.
Its embedded v1 inference bytes exactly reproduce the original report identity
`sha256:7f5b8c6819ab5162546f881a8105148e409b17c37689bd72e52e9799fc06a688`.
This is artifact-transport conformance, not a new scientific outcome or a reseal
of old evidence. The constant-answer finding and all original limitations remain.
