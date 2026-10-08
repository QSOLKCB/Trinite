# Phase 2 reference model

Status: implemented and locally checked; remote CI/review/merge remain PR gates. These are software and numerical conformance results from untrained models.

## Interfaces

| Interface | Implemented behavior |
| --- | --- |
| model.Decoder(config, lane, seed, parameter_budget) | Explicit CPU float32 dense/ternary decoder with a private initialization generator; identical latent tensors for matching seeds/configs |
| model.Decoder(ids, attention_mask, capture=...) | CPU int64 IDs and bool nonempty right-padded masks; explicit causal attention, exact GELU, RMSNorm, tied output; ModelOutput(logits, captures) |
| model.CaptureSpec(layers, positions, max_bytes) | Explicit embedding/block.N/final layer selection and absolute token positions; detached independent copies, no pooling or attention dump |
| quantizer.quantize(weight) | Bounded finite float32 matrix to int8 ternary codes, detached float32 scalar scale, and effective float32 weight with the declared surrogate gradient |
| inspection.inventory(model, max_bytes=...) | Actual unique tensors, names/aliases, shapes/dtypes/trainability, modules, ternary scope, code counts/scales/hashes, source/config/state identities and environment |
| inspection.quantizer_snapshots(model, names, max_bytes=...) | Detached independent copies of only selected ternary codes/scales after aggregate byte preflight |
| inspection.capture_inventory(captures) | Shape/dtype/device and raw-byte identity for detached states |
| CLI inspect-model | Optional model inventory, forward on supplied text, and explicitly selected capture metadata; no training, loading or text generation |

The default architecture has 1,247,232 unique parameters: 1,179,648 attention/feed-forward linear elements, 67,584 floating forward elements, and 4,988,928 bytes of latent float32 parameter payload. There are 36 ternary linear matrices; token/output sharing is an explicit registered alias and counted once. Dense mode has the same architecture and latent tensors but no quantized forward matrices. Bias/dropout and geometric loss remain absent/zero.

The quantizer reduces detached absmean in float64, floors it at 1e-8, casts the scalar scale to float32, and performs float32 division, round-to-even, clipping, and scale multiplication. This freezes v0's overflow behavior before native training. Backward passes the upstream gradient only where abs(W/alpha) < 1, with no scale gradient. An explicit autograd function avoids subtract/add cancellation in the forward. It is a surrogate, not the derivative of rounding; finite-difference gradcheck would test a different contract.

## Bounds and integrity

Construction rejects configs above a caller's parameter budget before allocating tensors: default 2,000,000, hard maximum 5,000,000. Quantization rejects empty/non-matrix inputs and matrices above 2,000,000 elements. Forward accepts batch 1–8 and sequence length 1–config context (at most 256), with at most 4,194,304 attention cells per block. It rejects invalid IDs, mask gaps/all-padding rows, device/dtype/layout mismatches, nonfinite parameters/logits, altered tensor shapes/sharing/lane, and CPU autocast.

Captures default to a 65,536-byte aggregate selected-state budget, with a 4 MiB hard limit. Inventory defaults to an 8 MiB unique-parameter payload budget, with a 32 MiB hard inspection limit. Selected code snapshots default to 65,536 bytes and include four bytes per scalar scale. These bounds describe tensor payloads and attention cells; they do not assert a process RSS ceiling. The CPU runtime, temporary tensors, autograd state and Python objects add memory.

Inspection hashes contiguous little-endian raw tensor bytes in bounded chunks. Metadata binds shapes, dtypes, aliases, config and source bytes as well as tensor/code identities. Numeric scales use exact float hex strings in JSON, preserving the existing no-float artifact contract. environment is reported separately from the inventory identity; its version/platform/thread/determinism fields must accompany reproduction claims. A constructor seed records the requested initialization, while tensor identities describe the current state; the seed alone cannot establish lineage after callers edit tensors.

Capture arrays are returned only through the explicit Python API. The CLI reports identities and metadata, not a durable capture bundle. Capture persistence, span/source binding for research, pooling and geometry compatibility are later gates. The API does not change process RNG, thread counts or determinism settings; the explicit CLI selects one CPU thread and strict deterministic algorithms for its process.

## Conformance evidence

The foundation suite has 31 checks and remains runnable without site packages. The separate model suite has 17 checks and requires the installed CPU backend rather than silently skipping. It covers actual reference shape/count/sharing, matched initialization without global RNG changes, finite forwards/backwards, causal/padding/batch isolation and prefix agreement, exact quantizer codes/scales and surrogate boundaries, zero/subnormal/extreme/nonfinite inputs, capture output/gradient/RNG parity, independent copies, budget rejection, state validation, CLI success/failure, and backend identity.

fixtures/model-v0/forward.json retains dense and ternary logits from tests/model/oracle.py, an independent scalar implementation using math/struct rather than Torch or the model. Its one-block width-four configuration uses explicit hand-authored tensor values. Tests compare against the frozen oracle and then against the actual forward with absolute tolerance 2e-6 and relative tolerance 1e-5; these tolerances are compatibility bounds, not accuracy scores. Oracle logits are retained as hex strings with the oracle source identity. Tests/CI do not regenerate this expectation. The Phase 1 data/source-bound fixture remains unchanged.

The local scope is Linux x86_64, Python 3.12.14, torch 2.8.0+cpu, NumPy 2.3.5, one Torch thread and strict deterministic algorithms. requirements.lock includes exact versions and official-index wheel hashes for CPython 3.11/3.12/3.13; CI has separate dependency acquisition and offline conformance steps. Other operating systems, architectures, backends and numerical scopes need their own acquisition/parity evidence. CPU dependency provisioning needs network access or a verified wheel cache; execution after provisioning is offline.

This increment supplies model-conformance evidence toward TRI-I01, TRI-I02, TRI-I04, TRI-I05, TRI-I07, TRI-I08, TRI-I12 and TRI-I14. It does not prove learning, reasoning, semantic interpretation, full-open release readiness, packed-memory execution, device speed or universal edge support. Phase 3 owns tiny training, safe checkpoints/resume and the separately pinned observer integration.
