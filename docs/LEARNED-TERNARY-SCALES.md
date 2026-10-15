# Learned positive ternary scales: engineering v1

Status: isolated, opt-in checkout-only conformance. **No scientific training,
learning improvement, task-gate passage, format selection or release is claimed.**
The native model, quantizers, updater, PCGrad lane and historical evidence remain
byte-identical. There is no PCGrad or auxiliary-objective integration.

## Academic source and version boundary

Steven K. Esser et al., [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153),
ICLR 2020, motivates learning the quantizer step size and normalizing its
surrogate gradient by `1/sqrt(N*Qp)`. This implementation independently derives
and implements a **symmetric ternary adaptation** with Qp=1. Signed two-bit LSQ
has four levels; this three-level lane is not an LSQ paper replication or a
reproduction of its accuracy results. No external implementation is imported.

The engineering [specification](../scripts/learned_scales_v1/protocol.json) was
published before conformance outcomes in
[430b988a013ec1fa9523c75a293b5c698ebc7b2c](https://github.com/QSOLKCB/Trinite/commit/430b988a013ec1fa9523c75a293b5c698ebc7b2c),
SHA-256 `08554101a99dde509336f213d2a0277a4700e4d77e4d7086a920f317e821b4be`.
It preserves main `00eddce`, including absmean, fixed-code refit, declared-state
and PCGrad work. The new numerical identifiers are
`trinite.learned-positive-ternary-ste.v1` and
`trinite.learned-ternary-decoder.engineering.v1`.

Files under `scripts/learned_scales_v1/` are not installed-wheel APIs and are not
new native checkpoint lanes. The wrapper reuses native initialization and the
native decoder forward, normalization, GELU, attention and tied floating token
embedding/output. Only the six attention/feed-forward matrices per block gain
one explicitly registered scalar scale. The inherited architectural config is
unchanged; its historical quantizer field is not the new numerical identifier.
The new inventory records separate model/arm identities. The inherited internal
`dense` structural label allows reuse of native structural validation, while
all actual linear modules are checked to be the versioned ScaleLinear type.
Do not pass this wrapper to native training/export/checkpoint interfaces.

## Exact forward, gradient and positive-scale policy

Inputs, scales and effective weights are finite strided CPU float32 tensors.
Matrices contain 1–65,536 entries. Each scale is a scalar at least the float32
representation of 1e-8. Its initialization is the detached float64 reduction
of mean absolute latent weights, floored at 1e-8 and cast to float32.

Forward computes float32 U=W/s, then ties-to-even round, clamp to [-1,+1], int8
codes, and actual float32 codes*s. Both ±0.5 ties map to zero. The actual product
is returned by custom autograd directly, without a subtract/add STE expression
that can lose small values through cancellation. Activations remain float32;
this is not packed integer inference.

For incoming g, the frozen chosen surrogate is:

    dW = g where abs(U) < 1, otherwise zero
    h = -1 where U <= -1
        +1 where U >= +1
        code - U otherwise
    ds = float32(sum_float64(g * h) / sqrt(N))

The residual h is formed in float32, then promoted for the product/reduction.
The strict weight mask and h(±1)=±1 are explicit endpoint conventions. At an
endpoint the scale surrogate jumps from the interior residual limit zero;
this is intentional, not a calculus derivative of the discontinuous forward.
Ordinary finite differences of rounding are not a valid STE-gradient oracle.

Finite W divided by a tiny valid s can overflow U. Forward clamps its rounded
codes; the backward residual clamps U before subtraction and explicitly selects
the saturated signs, avoiding infinity subtraction and zero-times-infinity.
Incoming nonfinite gradients and nonfinite reduced/cast scale gradients reject.
No claim of finite output for arbitrary overflowed surrounding neural operations
is made; the native decoder also rejects nonfinite logits.

Scales are **direct parameters**, not log/softplus parameters. After each toy
AdamW step, reject nonfinite scales and project them to at least float32(1e-8).
Optimizer moments remain unchanged by this projection. Zero/negative/below-floor
scales reject on forward; projection is the explicit post-update operation.
There is no upper clamp, hidden chain factor, adaptive threshold or scale refit.

The hand fixture W=[-2,-1,-.5,0,.5,1,2], s=1 and incoming g=[1,2,3,4,5,6,7]
has codes [-1,-1,0,0,0,1,1], weight gradient [0,0,3,4,5,0,0], and scale gradient
9/sqrt(7), rounded float32. Tests include a separate Python/Fraction scalar
oracle, adjacent threshold floats, zeros, subnormals, finite extremes, invalid
dtypes/scales, gradient overflow and positive projection with unchanged moments.

## Actual tiny matched control

The fixed conformance decoder has two blocks, width 4, two heads, FF width 8,
context 4 and byte vocabulary 259. Its native unique parameter count is 1,328;
actual inventory checks count **12 additional scales**, for 1,340 parameters.
The learned arm trains all 1,340. The frozen arm registers the same 12 scalar
values as nontrainable parameters and trains 1,328. Frozen scales have no AdamW
moments or weight decay because they receive no gradient. General block counts
are tested to add exactly six scales per block; twelve is not a universal count.

Both arms/seeds 0 and 1 have byte-identical initial latent tensors and scales,
identical private row-order RNG, fixed inputs/targets, four updates, six synthetic
targets and one backward/AdamW call per update. Inputs are explicit numeric toy
arrays, not admitted scientific examples, final-answer/EOS tasks or validation
labels. Mean cross entropy is over all six arbitrary target positions. There
is no accuracy scorer, learning threshold, outcome selection or early stopping.
The report records every cell, including every scale and actual rounded
forward matrix, code changes, saturation/zero counts, gradients and losses.

The existing native AdamW constructor and validator are reused (LR .003,
betas .9/.999, epsilon 1e-8, weight decay .01; single tensor). Norm clipping is
one, followed by the stated scale projection. This isolated engineering loop
never replaces native run_steps. Matching initialization/batches/updates does
not imply equal wall time or parameters, and post-step trajectories can diverge.
Global clipping includes scale gradients in the learned arm: identical clipping
policy does not imply identical clipping factors for the shared latent weights.
Changes in codes arise from both latent-weight and scale changes. A lower toy
loss or changed scale does not imply better exact-answer performance.

## Full replay and evidence boundary

New bounded canonical JSON snapshots record all unique model parameters,
including frozen scales, complete named AdamW moments/counters, private RNG,
full history, inventory, source, locked environment and pre-outcome spec.
Resume reconstructs the full bounded prefix from its seed and compares every
byte against the externally supplied checkpoint identity before continuing.
It never loads arbitrary tensor arrays, invokes pickle, restores global RNG or
introduces another native safe-tensor reader. This is full-prefix replay-to-resume,
not constant-cost restart, native historical replay or cross-host portability.

The report checks stops 0/1/3/4 in both arms and both seeds. Changed model,
optimizer, scale, source, RNG, history, identity or arm rejects. Generation is
exclusive; verify-only reruns actual numerical work and leaves input bytes
unchanged. Source/environment binding and exact numerical replay are distinct
from a closed PROVENANCE custody bundle; closed_custody remains false.

Use the [hash-locked CPU environment](GETTING_STARTED.md), one Torch thread and
strict deterministic algorithms. The launcher configures those runtime flags.

```sh
export PYTHONPATH=src:scripts
python -m unittest discover -s tests/learned_scales_v1 -v
python scripts/check_learned_scales.py --output /new/external/scales-report.json
python scripts/check_learned_scales.py --output /new/external/scales-report.json \
  --verify-only --expected-identity sha256:YOUR_RETAINED_DIGEST
python scripts/check_learned_scales_cpu.py
```

The aggregate preserves the existing research CPU, scale-refit and PCGrad suites
and adds this suite. Dedicated CI uses Python 3.11/3.12/3.13 and fresh source-bound
reports. The retained [receipt](../fixtures/learned-scales-v1/report.json) and
[commitment](../fixtures/learned-scales-v1/commitment.json) bind their actual local
environment. Other environments verify its structure/source and produce fresh
receipts; they do not pretend to reproduce the local environment identity.

All non-engineering profiles reject before state allocation. Four steps is the
hardcoded horizon; no config path or CLI override activates research. Later
scientific training needs a separately frozen protocol, native integration and
checkpoint review, named host/time/memory/storage/replay authorization, dense
and absmean references plus identically initialized frozen-scale controls,
matched inputs/budgets and exact original answer+EOS per-task gates. No PCGrad
objective change may be bundled into that scale comparison. Held-out confirmation,
geometry loss, scaling, weight-format selection and release remain separately gated.

Affected requirements: TRI-I01/I02/I04/I05/I06/I07/I08/I12/I13/I14.
TRI-I03/I09/I10/I11 remain explicit scientific-data/claim boundaries. Existing
negative evidence, native source, updater, scorer and all historical fixtures
remain unchanged. The earned positive claim is trainable-scale engineering
correctness and tiny optimizer replay, not a learning win.
