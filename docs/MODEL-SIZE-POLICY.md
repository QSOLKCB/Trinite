# Prospective native model-size ceiling v1

**Decision:** native Trinite models must not exceed **8,000,000,000 (decimal 8B)
unique model parameters**. This is a ceiling, not a target, preferred size,
training authorization or promise of CPU feasibility. Smaller justified models
remain preferred. Existing learning, data, geometry, resource and release gates
remain in force. This ceiling does not change an external reference model's
identity or retroactively redefine the separately authorized teacher lanes.

## Counting rule

Count the total unique physical model's scalar parameter slots, not just
trainable parameters, nonzero weights, compressed storage words, or parameters
active for one token. Include embeddings, output heads, biases, normalization,
learned scales, adapters, routers and **all** experts of any future MoE model.
Frozen learned tensors count even if registered as buffers in a future format.
A tied/shared parameter is counted once only when it is genuinely the same
underlying parameter; equal-valued independent copies count separately. Parameter
views must have their overlap resolved by a reviewed architecture-specific
inventory, not by name, value equality or an unverified claimed alias.

Non-learned constants, activation/KV caches, gradients and optimizer-state copies
are not extra model parameters; they still consume separately budgeted resources.
Quantization changes representation/precision and may reduce stored bytes, but
**does not reduce the number of parameter slots**. Learned quantization scales
are additional parameters; deterministically derived export scales are metadata,
not additional learned parameters. Distillation into a genuinely smaller student
can reduce parameter count; it is separate from quantization.

## Implemented enforcement and its limits

[`scripts/model_size_policy_v1.py`](../scripts/model_size_policy_v1.py) is a small,
checkout-only prospective boundary, not a new framework or installed API:

- `preflight_parameter_count(count)` accepts only exact built-in positive integers
  up to the inclusive ceiling. It rejects booleans, floats, strings, subclasses,
  negative/zero and over-limit counts, including integers beyond machine width.
  It neither allocates tensors nor certifies a caller's inventory is complete.
- `require_matching_count(expected, actual)` fails closed on either invalid
  count or any disagreement. It does not estimate a count from memory size.
- `build_native_decoder(config, ...)` revalidates an exact native `ModelConfig`,
  checks its authoritative shape-derived count **before importing the decoder
  or allocating tensors**, then delegates to the existing `Decoder`. It checks
  actual `parameters()` against the preflight count before returning. The native
  constructor already checks config versus actual inventory and ties its output
  to the token embedding. These checks include frozen parameters; they do not
  filter `requires_grad`. This adapter supports only the current native layout,
  not arbitrary shared-storage/view models, MoE, learned-buffer representations
  or checkpoint loading. Each future architecture needs its own reviewed exact
  pre-allocation formula and complete actual-inventory/alias check.

**Current native limits are unchanged:** `Decoder` permits at most 5,000,000
parameters, with a default per-run budget of 2,000,000 and additional config,
attention and inspection bounds. The new 8B ceiling does **not** raise those
limits. The reference configuration is still 1,247,232 unique parameters. A
policy-valid 8B scalar count is not a constructible current model.

No historical constructor, launcher, config, fixture or producer source is
retrofitted. Existing call sites do not automatically invoke this new adapter.
**This PR is not universal runtime enforcement across all present/future models.**
For new native workflows, explicit adoption must bind this policy source and the
architecture/config/inventory in a new source-bound request, invoke preflight
before allocation/loading, and check actual inventory before training/inference.
Loaders must validate shape/count metadata before tensor materialization and
reject mismatch; this increment adds no loader. Unsupported inventory semantics
must block adoption rather than silently fall back to trainable/active counts.
Historical replay continues with its original source closure and limits.

Example for a new, separately authorized checkout workflow (not an execution
request):

```python
from trinite.contracts import ModelConfig
from model_size_policy_v1 import build_native_decoder

model = build_native_decoder(ModelConfig(), lane="dense", seed=0)
```

A request still needs a smaller measured resource budget: parameter count alone
says nothing sufficient about host RAM, peak activation/optimizer/dequantization
memory, context/KV usage, CPU latency, training time, storage or energy. Retain
existing stricter limits; justify changes separately with measured evidence.

## Roadmap: our own teacher → smaller student → quantized candidate

Planned only; no teacher/student training, distillation, quantization pipeline,
model acquisition or target run is implemented or authorized here.

1. First establish reproducible useful learning and fresh held-out behavior in
   a native Trinite teacher **at or below the ceiling**. It need not reach 8B.
   Scaling needs its own scientific and resource review; current failed gates
   are not waived.
2. Propose a separate versioned own-teacher/smaller-student study. Preserve
   random-initialization ancestry, teacher/student/checkpoint/source identities,
   data rights, explicit distillation disclosure (TRI-I02), generated-output
   lineage and admission. Ownership of a model does not automatically confer
   rights to all its training data or outputs. Follow [DATA](DATA.md),
   [PROVENANCE](PROVENANCE.md) and the existing [teacher intake boundary](TEACHER-INTAKE.md);
   do not silently broaden that numeric-only exception to unverified prose.
3. Freeze training-only distillation inputs, family-level exposure isolation,
   matched non-distilled student controls, budgets, seeds and success/failure
   criteria before target outcomes. Keep validation/test labels out of teacher
   generation, student updates and tuning. Retain negative results and complete
   replay; evaluate teacher and student independently on fresh held-out families.
4. Separately evaluate a quantized teacher or student candidate under the same
   parameter ceiling. Follow [INFERENCE](INFERENCE.md) for safe artifacts,
   declared tensors/scales, output parity/tolerances, quality regressions and
   measured CPU costs. Quantization is not assumed lossless or faster, and the
   existing ternary/four-state paths are not a chosen deployment format.

Affected requirements: TRI-I01/I02/I04/I05/I07/I08/I10/I11/I12/I13/I14.
Compatibility is additive and prospective: no numerical, optimizer, scoring,
admission or historical evidence semantics change; no learning/release gate is
advanced by policy conformance.

## Verification

Boundary tests use scalar counts at 8B and enormous integers without large
allocations. Native tests construct only a tiny model, verify deterministic
parameter equality to the original constructor, tied/frozen counting, reject
actual/config mismatch and prove stricter budget rejection precedes allocation.
They perform no target training, checkpoint loading or inference.

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/model_size_policy_v1 -v
# Pinned requirements.lock CPU dependencies:
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/model_size_policy_native -v
```
