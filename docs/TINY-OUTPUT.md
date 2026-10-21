# Tiny model-plus-harness output comparison v1

Prospective, output-only, public exploratory conformance lane. No target model
has been loaded or executed by this PR. Synthetic CI is not model compatibility,
performance, quality, GPU fit, native Trinite learning or release evidence.
No student training or data admission is performed.

## Frozen comparison

32 fixed public rows per model, 16 literal numeric field extractions and 16 exact
modular additions (four each at moduli 5, 7, 11, 13). Each row binds generator
version, deterministic enumeration, split, exact answer and verifier outcome.
Report families separately, including all failures in the denominator of 16.
Constant zero and best-constant oracle diagnostics distinguish valid formatting
from input-sensitive accuracy. The best constant is calculated from frozen gold,
not selected from a model's outputs. This tiny reused formal suite is not held out.

One Needle3 checkpoint at one frozen depth and one text-only Bonsai2-2B artifact.
Same semantic prompts, generic unrestricted integer answer object, no gold enums,
per-row bounds, examples, retrieval or executing generated function calls.
Model-specific wrappers and tokenizer/runtime/artifact hashes must be frozen.
Needle uses an inert function named `answer`, parameters exactly the shared answer
schema, through `complete()`, never `run()`. Prism receives the row prompt followed
by `Return only a JSON object with the integer field answer.` through its pinned
chat template with thinking disabled. No image/projector input.

The profile freezes experimental greedy decoding, 256 output tokens, 120 seconds
per call, 64 KiB retained and 4 KiB accepted response bytes, fresh independent state.
A supervisor must enforce the deadline externally; an API's startup timeout is
not a per-call deadline. Permit at most five seconds cleanup. A killed/failed call
remains failed even if its retained output contains the correct answer. Never
retry, refill, rescue suppressed calls or apply additional Trinite repair.
Different tokenizers and runtime constraints mean equal caps are not equal compute.
Unsupported settings or incompatible artifacts must stop preflight, not be ignored.

## Official source pins, not execution results

- [Needle source](https://github.com/cactus-compute/needle/tree/b4fc11b04f1852888d375c8c6dfbaca44572d014), Apache-2.0;
  [Needle3 model](https://huggingface.co/Cactus-Compute/needle3/tree/d6c7d6a2e9655505e0f8e23339356dd1705900eb).
  Current repository engine 3.3.3 versus model configuration 3.1.0 is an unresolved
  compatibility question. A supported exact pair must be established before target
  calls. Explicit local weights/library, offline operation, telemetry off,
  `auto_date=False`, `stateless=True`, and fixed depth are mandatory.
- [Prism model revision and Apache-2.0 notices](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/commit/683f7be2629342a50987627b40881e54761a2430):
  PQ2_0 artifact is 517,312,992 bytes, SHA-256
  `d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928`.
  [Prism MIT runtime release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715)
  source `e3de715222b2667c54a267441159a5729f11155e`.
  Ternary learned matrices use a rotated basis, FP16 group scales and F32 auxiliary
  tensors. This is not Trinite's representation or an all-tensor ternary claim.

Needle is an engine-assisted comparison, not raw model tokens: its engine applies
built-in deterministic argument repair before returning the envelope. Per-call
repair activity is not observable here. Capture a faithful SDK-return JSON
serialization including all metadata, labeled `sdk_serialization`, not original
native bytes. Prism captures the complete final text bytes, labeled `final_text`.
Do not invent raw-token claims, remove flags, or select answer substrings.
The conservative Needle acceptance subset requires type `call`, one active
`answer` function and exactly one integer argument named `answer`. Reject reported
errors, false success, suppression, ungrounded arguments, negation and reported
confidence below 0.1. Missing confidence is not invented; this subset does not
claim a confidence value when the SDK omits it. Optional
success/error fields vary upstream; these are our preregistered conservative
rules, not a complete verified upstream schema. Unknown envelope or validation fields fail closed and require prospective
contract review, never relaxation after target outputs.

## Offline interfaces and local preflight

`scripts/run_tiny_output.py` only supports `freeze`, `seal`, `verify`; it never
starts runtimes, installs software, downloads assets or connects to the GPU.
A live adapter is deliberately **not implemented or certified** in this PR.
Before a later explicitly authorized local run, review and freeze an adapter and
non-target compatibility receipt, including exact wrapper/template bytes, runtime
settings, capture serialization and actual support for every required setting.
Do not relax these rules after seeing any target response. A manifest declaration
or successful local hash check does not attest compatibility or execution.

The manifest has exactly `needle3` and `bonsai2-2b` entries. See
`validate_manifest` in `scripts/tiny_output_v1.py` for the bounded exact schema.
Each entry binds the pinned profile, full runtime commit, hardware, settings,
compatibility receipt identity, launch configuration identity and local artifacts for weights, runtime, tokenizer,
chat template, compatibility receipt and complete launch configuration. The latter
includes every resolved sampling, confidence/gating, penalty, token, timeout and
environment option, with defaults explicitly attributed to the exact immutable
runtime revision. It is not sufficient to record temperature alone. Freeze the
actual invocation and adapter configuration bytes; preflight must reject any
unrecorded override. These declarations bind configuration but this offline tool
cannot prove a runtime actually honored them. Include actual executable/library bytes,
not merely a source revision. All paths are relative to one asset root, regular,
non-symlink files; every byte count/hash is recomputed at freeze. Include adapter
source in the runtime artifacts. Both models and all rows freeze together before
any target response. Keep the printed request identity independently.

Commands (canonical JSON inputs, fresh output paths):

```sh
python -I -S scripts/run_tiny_output.py freeze --manifest manifest.json --asset-root assets --output request.json
python -I -S scripts/run_tiny_output.py seal --request request.json --request-identity sha256:... --capture capture.json --output packet.json
python -I -S scripts/run_tiny_output.py verify --packet packet.json --packet-identity sha256:...
```

Capture must contain all 64 ordered model/slot records. Missing records are an
incomplete experiment, not fabricated errors. Preserve incomplete captures and
logs. Each attempted call stores status, complete retained bytes as canonical
base64, channel, elapsed time as finite hexadecimal float, diagnostics, generated
token count and scoped memory. Unavailable measurements use null plus a reason,
never invented zero; state whether memory is peak RSS, device peak or snapshot.
Time covers the call, not loading. Retain load-time/hardware inventory logs with
the local bundle separately. No raw response or teacher answer is admitted.

Seal retains bounded request/capture inputs before validation, plus prepared
packet/failure receipts in a unique full-filename attempt directory. Atomic
no-clobber publication preserves old outputs. The packet itself is the published
outcome. Verify is read-only recomputation from the retained capture, **not a
fresh inference replay or hardware attestation**. Keep originals and SHA-256
independently; checksums bind bytes, not the truth of a local execution claim.

Historical gpt-oss evidence measured operand proposals, not matched answers;
it is context only. Colibri is an inference engine, not a tiny model.
Qwen/LFM comparisons remain the following planned step. No causal claims about
parameter count, ternary precision or student benefit are earned by this lane.

Affected TRI-I02/I03/I04/I05/I07/I08/I10/I11/I12/I13/I14. All historical producers,
source closures and fixtures remain untouched. Run the synthetic workflow/tests
normally and under `-O`, plus repository root/upstream suites. Target runtime,
asset compatibility, inference, measurements and learning remain unrun.
