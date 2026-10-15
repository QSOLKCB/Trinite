# Matched PCGrad engineering lane v1

Status: opt-in checkout-only engineering conformance. **No scientific protocol is
frozen or enabled, no native learning experiment is run, and no learning gain is
claimed.** This increment implements the gradient-interference recommendation
without changing the historical updater, protocols, data order, task gates,
checkpoints, CNR source closure, or retained negative evidence.

## Source and exact boundary

Tianhe Yu et al., [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), NeurIPS 2020, describes PCGrad.
This is an independent implementation of its two-objective projection equations;
no upstream source or TensorFlow dependency is imported. The original
[author repository](https://github.com/tianheyu927/PCGrad) is MIT licensed, but its
implementation is not the native runtime here.

The engineering-only [specification](../scripts/pcgrad_v1/protocol.json) was
published before measurements in
[56b253af6f6630ade09b001cff7023d42c652d01](https://github.com/QSOLKCB/Trinite/commit/56b253af6f6630ade09b001cff7023d42c652d01).
Its byte hash is pinned by the launcher. Its baseline reference is PR #45 main
`5d7e4ba`; this implementation also preserves the subsequently merged PR #46
checkout-only declared-state helpers. The spec is a toy conformance contract,
not approval or a freeze of a research experiment.

Files live under `scripts/pcgrad_v1/`, outside the installed native package.
`kernel.combine` accepts exactly two finite CPU float32 flat vectors, each with
1–65,536 entries. Inputs are detached and cloned, never mutated. There is no
model hook, implicit optimizer replacement, dataset ingestion, geometry loss,
new quantizer or native checkpoint lane. Future integration needs its own
reviewed source/protocol change.

## Algebra and floating behavior

For each original vector g_i, use the immutable other original g_j:

    if dot(g_i, g_j) < 0 and dot(g_j, g_j) > 0:
        projected_i = g_i - dot(g_i, g_j) / dot(g_j, g_j) * g_j
    else:
        projected_i = g_i
    combined = projected_0 + projected_1

The control performs the same gradient collection but combines the unprojected
originals by sum. Neither arm averages the two objectives. Each objective loss
is independently target-normalized; both have coefficient one. Applying a
projection to an already modified target would be a different algorithm.

Dot products, squared norms and projection arithmetic use float64 over the
actual float32 gradients. Each projected vector is rounded to float32 before
the float32 sum. Zero dot is nonconflicting; zero norm never divides. There is
no epsilon, hidden tolerance, or cosine threshold in the conflict decision.
Subnormal float32 norms do not underflow in the float64 calculation; extreme
float32 squares do not overflow it. Nonfinite projected or summed outputs fail
before an optimizer update, including finite inputs whose sum overflows.

The report includes actual rounded residual dot products. Real-arithmetic
orthogonality need not survive the final float32 cast. For the hand fixture
[1, 2] and [-3, 1], exact projected vectors are [7/10, 21/10] and
[-14/5, 7/5]; their actual float32 residuals are measured, not asserted zero.
The tests also cover aligned, orthogonal, opposed and zero pairs, and an
independent exact Fraction oracle. Float64 intermediates themselves also round:
for [-27, -7] against [49, 0], cancellation leaves a tiny nonzero component
where exact arithmetic gives zero. A separate regression test retains this
residual instead of clamping it away; no universal exact-rational equality is
claimed. These local algebra checks prove neither
monotonic aggregate loss nor improvement of every objective.

PCGrad acts in Euclidean gradient coordinates. AdamW momentum, adaptive
preconditioning, weight decay, clipping and a finite step change the actual
parameter displacement. No descent or learning guarantee follows from the
projection alone. Gradient conflict is a hypothesis for future investigation;
it is not an established cause of historical procedural-QEC failures.

## Actual matched toy execution

`toy.py` executes eight updates of a two-parameter linear model on two fixed,
public mathematical input arrays. Labels `final` and `auxiliary` identify toy
objectives only; these are not Trinite answer/EOS tasks or procedural-QEC rows.
Both arms and seeds zero/one use identical initialization, inputs, objective
weights, target means, two separate reverse-mode passes and one optimizer call
per step. Both gradients are computed before any parameter update within that
arm. Arm trajectories may then diverge; same-snapshot does not mean treatment
and control remain at the same parameters after different updates.

The harness reuses the native `create_optimizer` and `require_optimizer` with
pinned single-tensor AdamW settings (LR .003, betas .9/.999, epsilon 1e-8,
weight decay .01) and clips the combined gradient to norm one. It does not edit
or replace native `run_steps`. Full histories retain before/after parameters,
separate losses and gradients, projected/summed/clipped vectors, norms, order,
backward counts, optimizer-call counts and target counts. Eight steps mean
sixteen backwards, eight optimizer calls and sixteen targets per objective.
There is no test/validation input, accuracy scorer or early stopping.

A private CPU generator initializes parameters and produces `randperm(2)` for
the outer objective order each step. Both arms consume the same draws. Each
objective has only one other target, so this order cannot change the
mathematical two-objective result. It is not a general multi-task permutation
implementation. No Python, NumPy or global Torch RNG is read or changed by the
harness. Caller execution must use the hash-locked CPU environment with strict
determinism and one Torch thread; the CLI establishes those execution settings.

## Checkpoint and replay meaning

The toy checkpoint is bounded canonical JSON, with actual float hex parameter
and AdamW first/second moments and step, complete history, private RNG bytes,
config, engineering spec, complete native Python/runtime source hashes, locks,
launcher/tests/workflow, and CPU environment. The complete checkpoint digest
must be supplied independently by the caller.

Resume is deliberately **full-prefix replay-to-resume**: reconstruct the fixed
zero-to-eight-step prefix from its seed, serialize its complete state, compare
every byte against the retained expected checkpoint, and continue that verified
state. It does not decode arbitrary tensor lists into a model and introduces
no new general tensor reader. No pickle, native checkpoint loading, historical
model replay, cross-host replay or constant-cost restart is claimed. Malformed,
noncanonical, over-bound and altered source/config/model/moment/RNG/history
snapshots fail. Verification leaves supplied bytes unchanged.

Tests cover every interruption point, both seeds and both arms. Reports also
retain checkpoint identities at steps 0/3/7/8 and complete final state for all
four cells. A fresh report independently reruns the toy and compares all
numerical state. This is source/environment/input binding and numerical replay,
not a closed PROVENANCE custody bundle or native research archival closure.
Existing CNR-001 fixtures and recursive native source inventory remain intact.

## Running and evidence

Use the existing [hash-locked CPU setup](GETTING_STARTED.md), not a fallback
backend. From a checkout:

```sh
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests/pcgrad_v1 -v
python scripts/check_pcgrad.py --output /new/external/pcgrad-report.json
# Supply the printed digest from the preceding generation step:
python scripts/check_pcgrad.py --output /new/external/pcgrad-report.json \
  --verify-only --expected-identity sha256:YOUR_RETAINED_DIGEST
# Unchanged historical aggregate, scale-refit suite and new PCGrad suite:
python scripts/check_pcgrad_cpu.py
```

Generation is exclusive and never refreshes a fixture. Verification recomputes
current results; it cannot bless a foreign environment or changed source. CI
runs fresh reports for Python 3.11/3.12/3.13. The separately retained engineering
[receipt](../fixtures/pcgrad-v1/report.json) and
[commitment](../fixtures/pcgrad-v1/commitment.json) describe their actual producer
environment; cross-environment CI checks source/structure and then creates its
own fresh numerical receipt. CI or launcher failures remain in logs and must
not be described as passes; this small harness is not the full research
failure-archive controller. Runtime/RSS observations, when retained, are
separate observations rather than deterministic replay fields or scientific
resource authorization.

`--profile research` and all non-engineering profile requests reject before
state creation. No flag, config path or numeric horizon override activates
research. The harness never performs more than eight toy updates per state.

## Prospective scientific comparison, still gated

A later protocol must specify final-versus-auxiliary grouping, frozen paired
batches and initialization, original historical final exposure/order, target
normalization, objective weights, clipping and all optimizer settings. Compare
same-snapshot sum against same-snapshot PCGrad with identical backward and
optimizer-call counts. Preserve final-only and extra-final rehearsal controls
for a claim that auxiliary supervision itself helps. These controls are part
of the prospective design, not executed arms of this toy receipt.

A sequential-update reference may connect to the existing auxiliary recipe,
but its intervening parameter update and additional optimizer/momentum step
create a different trajectory. It is not the matched summed-gradient control.
Keep geometry loss zero. Do not retrofit PCGrad into historical runs or relabel
old results as projected training.

Before any target outcomes, freeze the scientific protocol, same-host paired
seeds, unique original complete-answer-plus-EOS scoring, all original per-task
gates, final-checkpoint selection and failure policy. A proposed positive fold
contrast requires both treatment seeds to beat their matched control counts
while retaining every original absolute task gate (including arithmetic
60/66 per task, folds 1,035/1,150, GF(3) 49/54); these thresholds are unchanged,
not activated here. An absolute gate failure cannot be hidden by an aggregate
loss or paired gain. Keep held-out confirmation separate and untouched.

Explicit authorization must name source/protocol, CPU host, total wall time,
memory, storage, full replay and diagnostics. Measure wall time, peak RSS,
tokens, backwards, optimizer calls and per-example exposure for every arm;
the existing auxiliary 960-second cap is not this experiment's resource grant.
No research training, protocol freeze, Phase 5 geometry activation, format
selection, scaling, release, tag or later programme phase follows from this PR.

Affected requirements: TRI-I01/I02/I03/I05/I06/I07/I08/I11/I12/I13/I14.
TRI-I09/I10 remain explicit scientific-claim boundaries. Native source,
quantizers, scorer, admission, original evidence and thresholds are unchanged.
