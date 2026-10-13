# Aizawa attractor — bounded geometry reference (proposed PR #32)

**Status:** Pure-CPU, standard-library synthetic reference and numerical conformance checks. This is *not* a trained model, observed neural dynamics, a physical attractor experiment, a proof of chaos, a verified geometry-training objective, or evidence of learning improvement. **`phase5_ready` and `release_ready` remain false.** No original data, weights, optimizer, quantizer, archived protocol, geometry kernel or retained evidence changes.

## Frozen source and procedure

The independently reviewable [canonical protocol](../scripts/aizawa_reference/protocol.json), schema `trinite.aizawa-reference-protocol.v1`, has SHA-256 identity:

`sha256:cb904c512d45d8abba55a258c6b649e320c2ca37a7c04c9464d4c5623ca4857f`

It was committed **first** at [`c7fa03d25d21`](https://github.com/QSOLKCB/Trinite/commit/c7fa03d25d21d1f3526d15cb76d5c725a798724c), before this reference producer and its output tests. This is Git-revision ordering, **not** an independently timestamped preregistration. A prospective experiment must separately retain its frozen request identity *before* executing a research run; reproducing known synthetic outcomes is a software check, not fresh confirmation.

The autonomous Aizawa system is exactly

\[
\dot{x}=(z-b)x-dy,\qquad
\dot{y}=dx+(z-b)y,\qquad
\dot{z}=c+az-\frac{z^3}{3}-(x^2+y^2)(1+ez)+fz x^3.
\]

The frozen parameters are `a=0.95`, `b=0.7`, `c=0.6`, `d=3.5`, `e=0.25`, `f=0.1`, with initial state `(0.1,0,0)` and integration interval `[0,100]`.

The standalone reference applies an **explicit classical fixed-step RK4** integrator using Python binary64 floats. It integrates 20,000 update steps (`h=0.005`) and independently repeats the same interval at 40,000 half-size steps (`h=0.0025`). It samples **201** inclusive equally spaced time positions from each integration, corresponding to `t=0,0.5,...,100`; it does not treat sampled output positions as integrator steps. The original idea's SciPy `solve_ivp(RK45)` code is a separate *potential* external numerical comparison, not a runtime dependency or an executed validation in this PR.

A frozen **early-horizon** pointwise convergence check compares the first 21 sampled positions (`t=0..10`) using a maximum absolute coordinate tolerance of `0.0009765625`. If it fails, the reference fails. The complete 201-point coarse/fine discrepancy is **reported but is not a gating tolerance**: chaotic sensitivity can amplify numerical perturbations over long horizons, and convergence of two discretizations is not a proof of an invariant chaotic attractor. Each integration records the SHA-256 of the complete little-endian packed binary64 *state stream*, including the initial state and every intermediate state. The report exposes all 201 coarse/fine sample coordinates as exact `float.hex` strings, but does not store every intermediate state.

No GPU, Torch, NumPy, SciPy, internet, pretrained model, or implicit backend fallback is needed. The declared exact-replay scope is a matching Python build, host/libm and pinned source closure; a different runtime requires a **different source/environment-bound request** rather than editing an old report. Exact binary64 equality across all Python versions, platforms or CPUs is **not** promised.

## Geometry metrics and controls

All seven 201×3 trajectories enter the existing **pinned** `trinite.geometry.points()` / `geometry.metrics()` / `geometry.alignment()` functions; no replacement implementation of path length, finite differences, arclength or Menger curvature is shipped. The upstream numerical kernel and pin files remain unchanged. The metrics retain complete path length, order-one/two differences, 199 represented-binary64 Menger curvatures, their mean, and the order-one same-index cosine alignment relative to the unmodified Aizawa path. The output is canonical Trinite JSON with hex-encoded measured floats.

| Condition | Purpose | Interpretive restriction |
| --- | --- | --- |
| `native` | Actual sampled Aizawa trajectory | Synthetic reference, not neural activations |
| `shuffle-seed-11` | SHA-256-ranked deterministic temporal shuffle | Preserves sampled points, destroys native time order |
| `shuffle-seed-37` | Independent deterministic temporal shuffle | Separate seed, not an inferred null distribution |
| `rademacher-seed-17` | Pinned Trinite independent-sign diagnostic | Different value distribution; not a matched attractor |
| `smooth-periodic` | Analytic nonchaotic periodic curve | Different system, not fitted to Aizawa |
| `affine-line` | Collinear deterministic reference | Exactly zero represented Menger curvature |
| `reverse-diagnostic` | Reversed native point order | **Not** an independent temporal-shuffle null; checks reversal invariances |

Each condition's full ordered coordinates, hashed coordinates, metrics and, for the two shuffles, complete permutation and hash, appear in the report. Shuffles derive every index order deterministically from Trinite canonical JSON and SHA-256; the controls never draw from model, global NumPy, Torch or Python PRNG states. Reversal must preserve path length and reverse the Menger-curvature sequence; this tests software orientation symmetry and does **not** demonstrate sequence understanding. Controls are fixed rather than selected after reading results.

The integer-only stencil `a−2b+c` annihilates affine index series; a conformance test checks this simple fact and points to [COSMO's source-pinned Lean theorem](https://github.com/QSOLKCB/COSMO/blob/51bdc9221553b06668168d1d8c21ec245b9b6707/cosmovirus.lean). The PR **does not** compile COSMO Lean inside Trinite or claim that COSMO proves floating-point RK4, Menger curvature, an attractor, or neural learning.

## Read-only evidence workflow

From the repository root in a stable offline environment, use an existing writable directory **outside the checkout** for new requests/reports. All output files are exclusively created, never overwritten; a report can be independently regenerated without writing to it. Capture and externally retain the actual request SHA-256 before the analysis step.

```sh
export PYTHONPATH=src:scripts
python -S scripts/run_aizawa_reference.py \
  --freeze-request --output /tmp/aizawa-reference-request.json

# Copy the exact request identity printed by the freeze step.
python -S scripts/run_aizawa_reference.py \
  --request /tmp/aizawa-reference-request.json \
  --request-identity sha256:THE_RECORDED_REQUEST_DIGEST \
  --output /tmp/aizawa-reference-report.json

python -S scripts/run_aizawa_reference.py \
  --request /tmp/aizawa-reference-request.json \
  --request-identity sha256:THE_RECORDED_REQUEST_DIGEST \
  --output /tmp/aizawa-reference-report.json --verify-only
```

`trinite.aizawa-reference-request.v1` binds the frozen protocol, exact source closure, pinned geometry receipt, actual Python/platform/libm environment and the complete planned controls. `trinite.aizawa-reference-report.v1` binds the original request identity, full source/environment, integration stream identities, sampled hex vectors, convergence diagnostics, per-control metrics, and explicit **simulation-only** claims. Invalid request hashes, changed versions/pins/source/runtime, missing controls, an early convergence failure, invalid numerical values, symlink paths, stale output, and inconsistent retained reports fail closed. The input reader and output writer do not claim secure snapshots against an actively hostile same-UID filesystem; use independently retained input artifacts for archival custody.

Use:

```sh
PYTHONPATH=src:scripts python -m unittest discover -s tests/aizawa_reference -v
PYTHONPATH=src:scripts python -m unittest discover -s tests -p test_aizawa_reference_contract.py -v
PYTHONPATH=src python scripts/check_research_cpu.py   # pinned CPU dependencies required
```

The dedicated [`Aizawa geometry reference conformance`](../.github/workflows/aizawa-reference.yml) workflow runs actual pinned geometry source under Python 3.11/3.12/3.13 without external numerical dependencies; the combined CPU runner includes the new suite as well. These are software conformance checks, **not** independent experimental replication.

## Scientific boundary and handoff

This benchmark **does not read any model weights, outputs or hidden states**; its references cannot indicate whether Trinite's ternary STE loses representation capacity. [PR #31's separately source-bound diagnostic](TERNARY-DYNAMICS.md) can later supply observed quantizer occupancy and transitions, but a *new preregistered* mapping would be required to compare those measures with any model activation trajectory and Aizawa-style geometric controls. That future work must define sample units, non-posthoc projections, seed/task exposure, controls, independent outcome custody and per-task quality metrics before looking at model results. A deterministic chaotic trajectory is not an explanation of reasoning.

After review, merged GitHub CI and explicit release approval, v0.2.0 may freeze the complete PR #31/#32 diagnostic baseline. Do **not** tag or imply positive Phase 5 gates here. COSMO's E₈/Weyl, triadic lattice, Lean implementation correspondence and later ECC control investigations are reserved in [the roadmap](ROADMAP.md) for **after** that immutable checkpoint.

**Affected invariants:** TRI-I01/I04/I05/I06/I07/I08/I09/I11/I12/I13/I14; all existing source closures, empirical negative results, checkpoint formats, optimizer/quantizer behaviour, and historical protocols remain unchanged.
