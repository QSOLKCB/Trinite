# Proposed computational and KV-cache harness

Status: prospective design, 2026-10-26. No harness or cache implementation, benchmark,
model run or resource-limit change is introduced by this document. The canonical
CPU float32 numerical lane and historical evidence remain unchanged.

## What exists today

Inspected baseline: `be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a` (v0.4.0).

- The native [attention block and decoder](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/model.py#L78-L205)
  recompute Q, K and V for every input position. `Decoder.forward` returns logits
  and requested hidden-state captures, not reusable KV state. Its causal/right-padding
  masks and learned absolute position embeddings are part of the numerical contract.
- [Frozen generation](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/scripts/frozen_decoding_v1/experiment.py#L72-L96)
  feeds the entire growing prefix into the decoder for each generated token.
  [v2 transport](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/scripts/frozen_decoding_transport_v2/runner.py#L10)
  still uses that numerical producer. Neither is an existing cached-decode backend.
- [Comparison instrumentation](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/comparison.py#L192-L243)
  records full-prompt forward timings after a warm-up, excluding generation, and
  training-worker lifetime RSS including imports. This is not a KV-cache benchmark
  or incremental inference-memory measurement.
- External [SSE metadata parsing](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/scripts/tiny_sse_v2.py#L59-L104)
  recognizes cache-token/timing metadata. Such server reports are not measurements
  of native Trinite cache allocation. The [tiny compatibility probe](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/scripts/tiny_compatibility_v2.py#L368)
  explicitly leaves unmeasured memory unavailable.

A measurement wrapper is feasible without changing the model. Native KV reuse,
however, needs a separately versioned inference API, not a wrapper switch.

## Stage A: uncached measurement and resource contracts

Freeze a new inference-only request before observations. Bind checkpoint/tensor
identity, model configuration, lane, tokenizer, prompt bytes, decoding/stopping
policy, source closure, environment and worker/thread settings. Do not train,
change a checkpoint or silently substitute an external model.

Record separately:

1. Model load/startup, prompt forward (prefill-equivalent), each subsequent decode
   step and end-to-end wall time. Include process CPU time and actual input/output
   token counts, forward calls and total prefix tokens processed. A token cap is
   not a compute cap. Name whether tokenization, imports, transfers and verification
   are included. Distinguish cold and warmed runs; retain repetitions and failures.
2. Artifact bytes, unique resident parameter tensor storage (do not double-count
   tied embeddings), runtime/process overhead, retained cache payload if supported,
   and measured transient peak. Report units and scopes rather than adding
   overlapping RSS/allocator measurements into a fictitious total.
3. Fresh-worker lifetime peak RSS versus sampled process-tree RSS, explicitly
   distinguishing each from a guaranteed peak or enforced memory ceiling. External
   server memory must not be attributed to the client process. GPU instrumentation,
   if separately authorized, requires synchronized timings and separately labelled
   allocator allocated/reserved and device observations, not a whole-device cap claim.

Use null plus an explicit reason for unsupported/unavailable metrics. Cache mode
is `unsupported` on the unchanged native decoder, not an invented measured zero.
Keep timings/resource observations separate from replay-compared numerical bytes.

Freeze bounded input/output/context/batch sizes, maximum forward calls/processed
positions, worker deadline and output-artifact limits. Validate predicted tensor
sizes before allocation. Distinguish predictive admission checks, sampled stop
conditions and genuinely enforced process limits; none substitutes for the others.
Report a limit failure without returning a truncated run as success.

## Stage B: a versioned inference-only cache backend

Keep the uncached canonical path as the reference. Introduce a new, separately
reviewed producer/API only after Stage A instrumentation and resource-contract tests.
No cache is retained in training/autograd, and no historical producer is rewritten.

For each layer, represent K/V tensors plus logical token length, absolute position
and batch-mask state. Bind cache ownership to immutable checkpoint/configuration,
lane, dtype/device, tokenizer and the exact request/prefix. Validate tensor shapes,
finite contents, position ranges and append bounds before use; reject stale,
cross-model, cross-request or altered-prefix state. Start fresh for every request
by default. Shared-prefix reuse, persistence, cache quantization, paging and eviction
are later separately specified experiments, not initial features.

For this decoder's ordinary multi-head attention, a prospective dense payload estimate is:

`KV payload bytes = 2 × blocks × batch × cached tokens × width × element bytes`.

This excludes metadata, allocation rounding, temporary concatenation/workspace and
other runtime memory. Current canonical float32 uses four bytes per element; ternary
weights do not imply ternary KV tensors. Derive estimates from the admitted model,
not from a headline model-file size. Allocate only within a frozen capacity bound,
and report reserved capacity separately from occupied logical length.

Appending must preserve the existing learned absolute-position convention and
causal/right-padding semantics. Do not silently extend context, roll positions,
truncate a prefix or evict tokens to fit: those change the task. Any bounded-storage
strategy must have its own semantics, cost accounting and tests.

## Acceptance gates

- Compare cached and uncached outputs at every prefix, not just final answers,
  across admitted lanes, seeds, batch/mask shapes and context boundaries. Freeze
  numerical comparison rules before results. Changed float32 operation shapes can
  change rounding; do not promise bit identity in advance or loosen historical
  tolerances to admit the new path. If exact parity is required and absent, the
  optimization fails that gate and remains outside the canonical lane.
- Require exact emitted-token/EOS/stop-reason and lowest-ID tie behavior under the
  declared deterministic decoder on the frozen test matrix. Numerical closeness
  alone does not establish discrete-output equality or general equivalence.
- Cover one-token prompts, varying prefix lengths, right-padded mixed batches,
  maximum context and one-past-limit rejection, early EOS and output caps. Reject
  mismatched checkpoints, masks, lane/dtype, changed prefix and cache corruption.
- Verify independent requests cannot observe each other's cache, cache reset is
  effective, and cancellation/timeout/failure/repeated runs release owned storage
  and preserve partial failure receipts. Test budget rejection before large
  allocation, no hidden retry/fallback and no training-time cache reuse.
- Only after correctness gates, compare paired workloads on the same hardware,
  environment, thread count and decoding policy. Report prefill, decode, end-to-end
  time, RSS/cache payload and exact-answer accuracy together, including cold-start
  and short-context overhead. Speed is an observation, not a correctness criterion.

KV reuse may reduce repeated projections and prefix computation while increasing
retained memory. It does not eliminate attention work, establish better reasoning,
or guarantee a speedup for Trinite's small contexts. This harness measures and
bounds costs; measured improvements require a separately approved execution plan.
