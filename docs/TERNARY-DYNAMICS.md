# Ternary training dynamics — STE saturation diagnostic v1 (PR #31)

**Status: implemented read-only instrument and synthetic conformance, not an empirical proof of a learning defect.** This instrument does not change `src/trinite/quantizer.py`, `model.py`, the optimizer, scheduling, admission, numerical checkpoints or historical evidence. It adds no geometry loss and does not unlock held-out evaluation, Phase 5, format selection or release readiness.

## Question and interpretation

Trinite's native ternary matrices use detached float32 absmean scale `alpha`, forward codes `clip(round(W/alpha),-1,+1)` and a masked straight-through gradient `g_incoming * (abs(W/alpha) < 1)`. Unlike a straight-through estimator with an unmasked backward, **the outside region has zero gradient through the quantizer**. The zero-code region is generally **inside** the gradient support: `code=0` is not automatically a trapped parameter. `alpha` remains detached and latent `W` remains floating point.

The measured **STE-disabled fraction** describes how many weight entries are outside the current surrogate gradient support, *not* how many beneficial updates were lost. An observed AdamW first moment reflects history; it does not prove that a current incoming gradient was suppressed. Observed latent changes can arise through decay/optimizer momentum even with a locally masked quantizer gradient. Equally, a code transition can reflect changes in other weights via the shared absmean scale. No causal claim or per-task capability claim follows from these measurements.

## Frozen protocol and subject

The prospective instrument's canonical procedure is [`scripts/ternary_dynamics/protocol.json`](../scripts/ternary_dynamics/protocol.json), SHA-256 **`14261ecc9c6bf5f9487a71a1def5ce9115298de6cefd9fd9b702eb91a06bbebf`**. The protocol was first committed separately as [38d5d007f364](https://github.com/QSOLKCB/Trinite/commit/38d5d007f36457314eab2055c452ec58d5819cee), before adding this measurement implementation. It fixes the quantizer, metric definitions, maximum input and output bounds, and the absence of optimizer or learning intervention. Later selection of checkpoint steps requires a separate, canonical selection request. Using already-inspected historical snapshots is retrospective description, **not** independent preregistration.

A selection declares one `ternary` seed and 2–16 **distinct increasing** checkpoint steps, with exact safe basename pairs of existing safetensors payloads and canonical JSON metadata. The freeze pass hashes the selected bytes **without inspecting tensors or selecting a winner**. The analysis pass requires an externally supplied `sha256:` identity for those frozen request bytes. A later `--verify-only` pass regenerates the report and compares it with the retained bytes without rewriting either. Requests and reports use `trinite.ternary-dynamics-request.v2` and `trinite.ternary-dynamics-report.v2` canonical Trinite JSON. Older v1 requests/reports are not silently reinterpreted.

Supported metadata schemas for descriptive reading are **only** `trinite.training-checkpoint.v1` and `trinite.comparison-checkpoint.v1` (the authentic `trinite.maths-task-balance-checkpoint.v1` producer is **dense-only** and cannot be interpreted as ternary). The diagnostic verifies snapshot SHA-256 commitments, retained model tensor identities, complete declared model parameter names/shapes, fixed lane and seed, source/environment/plan and data identities across steps; it requires the producer's **exact `source["quantizer.py"]` SHA-256 to match** the current observer quantizer before using the existing `quantize()` and `tensor_identity()` implementations. A different/missing quantizer digest blocks analysis, even when every snapshot agrees with each other. It does **not** perform full optimizer replay or certify upstream custody against a separately held ledger. For scientifically admissible comparison, the originating producer and its *original authoritative verifier* must be run independently and their custody evidence retained.

## Metrics and controls

For **every** quantized attention/FFN matrix at each selected step, retain exact element counts for code `-1/0/+1`, the actual quantizer scale (`float.hex`), latent/code content identities, normalized magnitude, strict inside/outside STE-mask counts and populations within `1/64` of the normalized code and STE thresholds (`0.5` and `1`). Record optimizer first-moment nonzero counts inside/outside only when the retained snapshot actually includes those tensors; otherwise record `null`, **never zero by assumption**.

For each adjacent selected checkpoint pair, retain the exact observed code-change and code-stability counts, latent changes with stable code, unchanged latent with changing code, zero↔nonzero counts, and direct -1↔+1 transitions. These are **between chosen steps**, not every intermediate optimizer update, unless every step was retained. No gradient values arriving at the quantized forward weight are stored in these ordinary checkpoints; the report declares that empirical upstream gradient evidence is unavailable. To study truly suppressed *incoming* gradients, a new separately scoped producer would need to capture them through a validated, noninterfering observer and bind every measurement to the same native training step.

The request binds the complete current tool source closure (including `quantizer.py`, `model.py`, `inspection.py`, `training.py` for the runtime guard, the lock, launcher and protocol) without adding any file beneath historically sealed `src/trinite/`. The **CLI** sets the two canonical native-runner PyTorch flags (`torch.set_num_threads(1)` and `torch.use_deterministic_algorithms(True)`) **before** invoking the existing `trinite.training.require_environment()` CPU backend guard during selection freeze or report calculation, and the guard rechecks the runtime at the end of inspection. A freshly launched interpreter therefore works with the pinned dependencies without a manual Python setup step; direct calls to the underlying Python library remain strict and require callers to configure their own process first. The CLI does **not** rewrite a foreign dependency lock, dtype, device, or backend to bypass the guard. The exact **observed environment** (pinned Torch, NumPy, safetensors and dependency versions; CPU/kernel capabilities, determinism, threads, dtype and device) is bound in **both request and report**, distinct from the checkpoint producer's `context.environment_identity`. A foreign observer runtime or changed runtime is rejected, including in `--verify-only`. Never modify or reseal the older v1/v2 experiment packets. Keep all negative/failed samples and report them as such. A model result remains ineligible if its upstream learning gate fails.

## Example offline workflow

Use the hash-locked CPU environment in [`GETTING_STARTED.md`](GETTING_STARTED.md) (Torch 2.8.0+cpu on Linux x86_64, Python 3.11–3.13) and a copy of **already existing** authenticated checkpoint payload/metadata pairs in a stable, read-only directory such as `/evidence/stable/`. The example is illustrative; it does not generate experimental checkpoints.

Create `selection.json` with Trinite canonical JSON bytes (sorted keys, compact encoding, final LF):

```json
{"lane":"ternary","schema":"trinite.ternary-dynamics-selection.v1","seed":0,"snapshots":[{"metadata":"0.json","payload":"0.safetensors","step":0},{"metadata":"1.json","payload":"1.safetensors","step":1}]}
```

Then run:

```sh
PYTHONPATH=src:scripts python scripts/analyze_ternary_dynamics.py \
  --freeze-selection /outside/evidence/selection.json \
  --root /evidence/stable --output /outside/evidence/dynamics-request.json

# Independently retain the request identity printed above before analyzing.
PYTHONPATH=src:scripts python scripts/analyze_ternary_dynamics.py \
  --request /outside/evidence/dynamics-request.json \
  --request-identity sha256:EXTERNALLY_RECORDED_IDENTITY \
  --root /evidence/stable --output /outside/evidence/dynamics-report.json

# Recompute the same result without modifying its retained record.
PYTHONPATH=src:scripts python scripts/analyze_ternary_dynamics.py \
  --request /outside/evidence/dynamics-request.json \
  --request-identity sha256:EXTERNALLY_RECORDED_IDENTITY \
  --root /evidence/stable --output /outside/evidence/dynamics-report.json --verify-only
```

New outputs are exclusive; output paths beneath the selected evidence root, symlink traversal and unknown metadata/tensor representations fail. The same-UID concurrent-filesystem threat model is not solved by path validation alone; operate over stable evidence snapshots. The report explicitly keeps `phase5_ready=false`, `release_ready=false`, `full_optimizer_replay=false`, evidence class `DESCRIPTIVE` and `incoming_training_gradients_available=false`.

## Acceptance and next step

The dedicated `Ternary dynamics diagnostic conformance` workflow runs real Torch quantizer/safetensors tests under the existing hash-locked CPU environment; the **required combined** `scripts/check_research_cpu.py` suite now includes `ternary_dynamics`, and root standard-library discovery validates the frozen protocol byte identity. Synthetic examples establish software conformance only. No live research matrix, paired dense-versus-ternary learning experiment, new gradients or held-out result is produced by this PR.

A separately frozen next increment may compare a versioned alternative STE/scale against the **unchanged native baseline**, matched by seed, data, exposure and budget. Such a study must record per-task exact-answer/EOS metrics and every negative seed; altered surrogate semantics require new protocol/source/checkpoint identities. PR #32 may reuse independently verified #31 diagnostic receipts to compare quantization behavior with an Aizawa geometry **reference**; that still is not evidence of a geometry learning benefit.

Affected contracts: TRI-I02/I03/I04/I05/I07/I08/I09/I11/I12/I13/I14.
