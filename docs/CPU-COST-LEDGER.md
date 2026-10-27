# Stage A: CPU inference cost ledger

Status: engineering implementation, **no target measurement result**. This is the
separate inference-only Stage A of the [compute/KV proposal](COMPUTE-KV-HARNESS-PROPOSAL.md),
not a cache backend, optimizer change, training run, speed claim or learning gate.
The native model, authoritative `trinite.comparison.greedy`, historical producers,
source closures and fixtures are unchanged. Review and green exact-commit CI, a
reviewed checkpoint/prompt plan and **separate owner execution authorization** are
required before any target run. The CLI flag alone is not authorization.

## Input and freeze

Use the hash-locked CPU environment (`requirements.lock`), Linux x86-64,
CPU float32, deterministic algorithms, one intra-op and one inter-op thread,
one worker, batch one. No implicit download, GPU, external model code, data
admission, training/optimizer construction or checkpoint modification is supported.

This first increment accepts a **new model-only safetensors** input, containing
exactly `model.<name>` for each native `Decoder.named_parameters()` entry.
Tied embeddings occur once; optimizer-bearing historical checkpoints are rejected,
not silently stripped. Any conversion/export and its original lineage must be
reviewed separately: this tool does not export, rewrite or certify historical
checkpoints. Byte identity plus finite named tensors proves input binding, **not
training lineage, native data admission or producer custody**. Select a checkpoint
whose lineage/rights are already authorized. No checkpoint is selected by results.

Prepare one JSON plan with exactly these fields:

- `config`: the complete native `ModelConfig.to_dict()` object, no defaults inferred;
- `lane`: `dense`, `ternary` or `four-state`;
- `checkpoint_identity`: independently retained `sha256:` of the model-only file;
- `model_identity`: existing native named-tensor identity, independently retained;
- `prompts`: ordered distinct exact UTF-8 strings (1–4, empty prompt is allowed);
- `repetitions`: 1–3 fixed repetitions per prompt;
- `output_tokens`: common 1–16 cap, no per-answer selection;
- `worker_seconds`: 1–600 whole-worker wall deadline.

Freeze binds the exact plan, UTF-8 bytes, initial BOS/prefix identities, full source
closure, tokenizer, decoding policy, environment and worker/thread settings.
Freeze checks checkpoint bytes/size without constructing a model or running inference.
Keep its printed request identity outside the output directory before execution.

```sh
export PYTHONPATH=src
python -B scripts/run_cpu_cost_ledger.py preflight --plan /outside/plan.json
python -B scripts/run_cpu_cost_ledger.py freeze --plan /outside/plan.json \
  --checkpoint /outside/model.safetensors --request /outside/request.json
# ONLY after separate reviewed execution authorization:
python -B scripts/run_cpu_cost_ledger.py run --authorize-run \
  --request /outside/request.json --request-identity sha256:EXTERNAL_REQUEST_HASH \
  --checkpoint /outside/model.safetensors --output /outside/fresh-attempt
python -B scripts/run_cpu_cost_ledger.py audit --request /outside/request.json \
  --request-identity sha256:EXTERNAL_REQUEST_HASH --output /outside/fresh-attempt \
  --expected-identity sha256:EXTERNAL_COMPLETION_HASH
```

`audit` verifies externally committed bytes, closed observation-bundle integrity,
source/environment/input bindings, row order, logical work and recorded parity.
It does **not** replay inference, authenticate timings/hardware or establish
checkpoint producer custody. It does not train or perform optimizer replay.
Keep the completion identity independently. Original checkpoint bytes are not
copied into the observation bundle and must be preserved separately.

## What is measured

Every prompt/repetition runs **uninstrumented before → instrumented → uninstrumented
after**, calling the same authoritative greedy function. Exact emitted IDs/counts,
EOS/output-cap stop and before/after output parity are mandatory; mismatch retains
the failed journal and cannot complete. Model identity and absent gradients are
checked before/after. The transparent proxy changes no model operations, tensors
or returned logits; nonfinite logits fail after the forward timer.

The measured middle pass is **warmed by the preceding parity pass**. The first
baseline is separately labelled the first generation in a fresh worker, not a
cold OS page-cache/CPU-cache guarantee. No cache flushing or silent warmup is done.

| Observation | Scope |
| --- | --- |
| Startup | Launcher clock through imports/configuration before worker request validation; includes importing measurement/workflow modules |
| Loading | Bounded file read/hash, safe decode, temporary native construction, parameter copies, finite/state checks and model hashing; not pure disk IO |
| Prompt forward | First full-prefix native call; includes native input/state validation and math, excludes tokenization/input tensor creation and proxy audit |
| Decode steps | Each later full growing-prefix call, **not incremental KV decode**; exact checkpoint/prefix/mask/absolute-position binding |
| Instrumented end-to-end | Greedy tokenization, input tensor construction, all forwards, argmax/stopping and proxy audit; overlaps component timers |
| Parity/audit | Separate baseline generation times, identity checks, prefix/finite instrumentation and provenance creation/verification; not estimated by subtracting noisy runs |
| Worker through measurement | Launcher through inference and journal writes, before provenance; whole controller wall is also retained separately |
| Process CPU | `process_time_ns`, current process only, not machine or child-tree CPU |
| Worker lifetime RSS | Linux `ru_maxrss × 1024`, high-water bytes through measurement, including imports/load/baselines/audit; later provenance allocations not included |
| Parameter storage | Unique resident tensor storage, tied embedding counted once; not serialized file size |
| Artifact bytes | Exact input checkpoint length, separate from resident storage/RSS |

No transient forward peak, process-tree sampling or isolated runtime-overhead
memory is measured; each is null with an explicit reason. Native KV mode is
`unsupported`, payload **null, not zero**. RSS, tensor storage and file sizes overlap
or measure different things: do not add them into a fictitious memory total.
Observations/timings stay separate from `numerical.json` parity bytes. There is no
speed threshold, automatic optimization selection or accuracy/learning claim.

## Bounds and failures

The plan admits at most the native 5M parameters and context 256 before model
allocation; it checks predicted parameter bytes and attention cells. The entire
worst-case output fits context; no truncation/eviction/rolling positions is allowed.
Safe input bytes are capped at 32 MiB before safetensors decoding, with strict
names, shapes, CPU float32 and finite checks after decode. This is not unrestricted
pickle, and no foreign installed Trinite runtime is accepted.

For prompt input length `L` including BOS and output cap `C`, reserve `C` calls
and `C*L + C*(C-1)/2` processed positions **for each of all three passes**, times
fixed repetitions. Early EOS reduces observed work; it never changes the frozen
upper bound. A complete trio is admitted before its first call; the proxy also
checks its call/position allowance before each native call. Output tokens across
all passes, workers, threads and batch are explicit in the frozen bounds.

The parent enforces the fresh worker's wall deadline and a 64-KiB log cap, kills
and waits on timeout/cancellation, and records non-success with partial artifacts.
Each ledger artifact write is capped at 4 MiB; the reused exclusive `Store` bounds
its own writes to 32 MiB, and the provenance collector bounds retained payloads
to 64 MiB/64 items. These are **application write/collector limits**, not an OS
whole-directory quota (provenance metadata and separate controller writes exist).
There is **no enforced memory ceiling**; predictive allocation admission and
lifetime RSS observation are not a hard RAM cap. Worker launch is sequential and
there are no hidden retries or alternative backends. Output/request paths are
fresh, outside source and separate from immutable inputs; writes do not clobber.

## Evidence-first decision

Only a separately authorized fixed-checkpoint run can identify whether startup,
prompt forward, repeated full-prefix decode, parity/audit or another component
actually dominates. Retain all prompts/repetitions/failures, not a favorable subset.
**Stop KV optimization if the suspected bottleneck is elsewhere.** A future cache
backend requires its own versioned checkpoint/prefix/position-bound state and
cached-versus-uncached numerical/token/EOS parity tests. This increment does not
implement or authorize it. Any efficiency improvement is not improved learning.

Affected requirements: TRI-I01, I05–I08, I10, I12–I14. Compatibility is additive,
checkout-only; installed native APIs and immutable evidence are unchanged.
Tests use tiny synthetic randomly initialized checkpoints and bounded forwards,
not target checkpoint benchmarks or learned capability evidence:

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/cpu_cost_ledger_v1 -v
# Pinned CPU environment required; no skipped numerical dependencies:
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/cpu_cost_ledger_native -v
```
