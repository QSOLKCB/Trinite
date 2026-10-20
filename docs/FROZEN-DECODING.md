# Frozen-model decoding comparison v1

Status: implementation and prospective protocol; target results not yet measured.

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

## Execution and assurance

```sh
export PYTHONPATH=src:scripts
python -m unittest discover -s tests/frozen_decoding_v1 -v
request_id=$(python scripts/run_frozen_decoding.py freeze --request /new/request.json)
python scripts/run_frozen_decoding.py run --request /new/request.json \
  --request-identity "$request_id" --output /new/report.json
python scripts/run_frozen_decoding.py verify --request /new/request.json \
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
