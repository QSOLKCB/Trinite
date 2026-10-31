# Memory traffic accounting v1

Status: additive engineering instrumentation, **not a SPARTA implementation,
cache backend, hardware-traffic measurement or performance result**. Native
model, quantizer, training, generation, CPU ledger v1 and historical evidence
remain unchanged. No target checkpoint run is included or authorized by the CLI.

## Why separate the quantities?

[SPARTA, 26 October 2026 preprint](https://arxiv.org/abs/2610.34824)
combines ternary compute with selective attention. Its accelerator evaluation
uses RTL synthesis and cycle-level simulation; reported benefits are not Trinite
measurements or fabricated-silicon results. Here the useful first step is to
separate weights from attention storage and logical data movement. Selective
page/token access changes outputs and needs a separately frozen quality/cost
experiment; it is not added to the canonical model by this increment.

Trinite's current native `Decoder` **has no persistent KV cache**. Each generation
forward recomputes every prefix position. Dense, ternary and four-state lanes
all retain float32 parameters and activations. Ternary/four-state linear calls
also materialize effective float32 weights. Tied output embeddings consume one
resident parameter allocation, but the output operation still uses those weights.
The existing [CPU cost ledger](CPU-COST-LEDGER.md) remains the separate
source/checkpoint-bound timing and observation workflow.

## Analytic report: no Torch, checkpoint, inference or training

Supported CLI platform: Linux, Python 3.11+. Native observation uses the existing
hash-locked CPU environment.

```sh
python -I -B scripts/report_memory_traffic.py \
  --plan configs/memory-traffic-v1-example.json
```

The example is a **synthetic shape plan**, not a training or target result.
To inspect another configuration, prepare a separate JSON plan with exactly:

- `config`: a complete native `ModelConfig.to_dict()` object;
- `lane`: `dense`, `ternary` or `four-state`;
- `batch`: 1–8 allocated rows, including padding;
- `prompt_tokens`: 1–context length, including BOS;
- `forward_calls`: 1–16 successive full-prefix calls.

The first row uses `prompt_tokens`, then one more prefix token per row. These
are planned calls, not observed outputs: EOS may shorten real generation.
All rows must fit native context, parameter, quantizer and attention bounds.
A nonblocking, regular-file-only 64-KiB reader rejects final-component symlinks and special files,
duplicate/unknown keys, floats,
non-finite numbers and Boolean integers. The command writes only stdout.

The deterministic integer-only JSON binds the plan, relevant runtime sources,
this contract and the CPU dependency lock with
SHA-256. Formula-bearing native files are pinned and fail closed if changed.
Hashes identify bytes, not independent execution, hardware or custody. This is
not an imported historical ledger receipt or its audit/replay replacement.

### Fields and counting conventions

- **Parameters:** analytic native float32 tensor payload with tied weights counted
  once. This excludes the model object, allocator overhead, buffers and checkpoint
  container metadata. For ternary only, a separate hypothetical fixed-2-bit
  payload rounds each linear matrix up independently to whole bytes, adds one
  float32 scale per matrix, and leaves embeddings/norms float32. This is not an
  export file size or a resident packed backend. No compression ratio is claimed.
- **Selected activations:** Q/K/V, attention-score/probability, FFN and logit tensor
  payloads. Scores are the full square `[batch, heads, length, length]` allocated
  by the current implementation, even though a causal mask hides future entries.
  Padding does not reduce the allocated tensor dimensions. Values aggregated over
  layers/calls are **not simultaneously live memory**.
- **Logical operand payload:** each listed weight or K/V operand counted once per
  operator invocation. These are useful shape/cost counters, **not physical DRAM
  reads**. Hardware caches, tiling, internal reuse, validation scans and other
  operators are not represented. Categories overlap; do not sum them into RSS,
  bandwidth, a complete traffic total, or an activation peak.
- **Quantizer materializations:** selected effective FP32 weight, int8 code and
  scalar-scale product sizes across calls. Float64 reductions, unit/mask and
  conversion intermediates and their lifetimes are excluded. Complete temporary
  peak remains null, not zero.
- **Native KV cache:** unsupported with null payload/read/write fields. Recomputed
  K/V activations are reported separately; they are not relabelled as cache reads.
- **Hypothetical dense FP32 cache:** for ordinary multi-head attention,
  `2 × blocks × batch × cached_tokens × width × 4` bytes. Width already includes
  all heads. Prefill appends every prompt token; each later row appends one and
  reads the occupied K/V operands once, including that token. This convention
  describes operand payload, not per-query accesses or a measured memory bus.
  Metadata, capacity slack, reallocation/copies and maintenance are excluded.
  No selective cache, eviction, quantized KV or reserved-capacity claim is made.

`physical_dram_traffic_bytes` and `peak_live_activation_bytes` remain null. A
prospective cache may avoid recomputation while increasing retained memory;
these formulas alone do not establish the actual bottleneck or a speedup.

## Optional in-process native observation

`scripts/memory_traffic_v1/native.py::observe_forward(model, input_ids, mask)`
returns the unchanged native `ModelOutput` and a scalar-metadata report around
**one** forward of an already supplied `Decoder`. It requires eval and explicit
no-grad/inference mode; it never selects a checkpoint, changes caller modes,
loads data or invokes another generation pass. The caller owns run authorization
and must follow its existing source/input/evidence protocol. This API is not
wired into frozen historical producers.

It observes actual unique parameter storage bytes, named linear-output shapes,
dtypes/logical bytes and logits bytes. It does not copy or retain activation
payloads in its hooks. Hooks are removed on success and failure; existing hooks
are preserved. No token IDs, text, tensor contents or paths are emitted. The
shape-derived counterpart is labelled separately, and physical traffic/peak/cache
measurements remain unavailable. It is diagnostic metadata, not a complete run
receipt or checkpoint-custody assertion.

## Compatibility and tests

Affected requirements: TRI-I01, I04–I08, I10, I12–I14. Additive checkout-only API;
no installed model behavior, historical hashes/fixtures, numerical tolerance,
training schedule, budget policy or scientific gate changes.

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/memory_traffic_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/memory_traffic_v1 -v
# Existing hash-locked CPU environment, no silent skips:
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/memory_traffic_native -v
```

Tests independently enumerate tensor counts, per-matrix packing rounding,
prefill/append/read formulas, head-vs-width behavior, context/admission boundaries
and strict CLI inputs. Tiny randomly initialized CPU examples check exact
observer-on/off logits, all three lanes, masked padding, parameter/RNG/mode
preservation and hook cleanup after exceptions. They are engineering checks,
not target inference benchmarks, learned capability or paper reproduction.
