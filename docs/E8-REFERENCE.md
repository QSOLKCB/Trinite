# E8 triality reference and second-difference proposal

Requested by Trent on 2026-10-12, from the idea headed “E8 Triality + theta=pi/2
+ [1,-2,1]” dated 23 July 2026. Status: exact reference calculator implemented;
neural mapping, target activation study and training intervention pending.

## What runs now

```sh
PYTHONPATH=src python scripts/run_e8_reference.py
PYTHONPATH=src python -m unittest discover -s tests -p test_e8_reference.py -v
```

Both commands work offline with the standard library. The calculator emits
`trinite.e8-reference.v1` / `TRI-E8-REF-001`, labelled `ANALYTIC_REFERENCE`. It
prints exact root/orbit counts, the declared triality numerator, an explicit
3D receiver, the quarter-turn example and a synthetic second difference using
the existing pinned geometry kernel. It does not load a checkpoint or inspect
neural activations. This output is mathematical conformance, not a source-bound
neural experiment, PROVENANCE assurance receipt or learning result.

`src/trinite/e8_reference.py` owns the new exact rational algebra. The thin
launcher delegates order-two differences to `geometry.metrics`; no alternative
geometry kernel, neural updater or scorer is introduced. The model, optimizer,
old geometry protocol, producer snapshots and retained outcomes are unchanged.

## Corrected interpretation

E8 has **240 roots** in eight-dimensional root space. Its Lie algebra has
dimension **248**, including eight Cartan dimensions. The exceptional outer
triality symmetry belongs to **D4 / Spin(8)**. Here we use an explicitly embedded
`D4 × D4` action that preserves the E8 root set; we do not call it an E8 outer
automorphism. See Baez's primary mathematical exposition:
[Spinors and Trialities](https://math.ucr.edu/home/baez/octonions/node7.html) and
[Kostant on E8](https://math.ucr.edu/home/baez/kostant/summary.html).

The calculator freezes these algebra conventions:

- Integer-family doubled roots have two coordinates in `{−2,+2}` and six zeros:
  112 roots. Half-integer-family doubled roots have eight coordinates in
  `{−1,+1}` with even negative parity: 128 roots. Every `2r` has squared norm 8;
  the conventional root `r` has squared norm 2.
- On each four-coordinate block, `A = N/2`, with numerator rows
  `(1,1,1,1)`, `(1,1,−1,−1)`, `(1,−1,1,−1)`, `(−1,1,1,−1)`.
  The full map is `T = A direct-sum A`, with `T^3 = I` and `T^T T = I`.
- Lexicographic root order and triality action order give 12 fixed roots and
  76 complete three-cycles. Exact Fraction arithmetic avoids rounded root matching.
- The demonstration receiver is `P x = (x_0,x_1,x_2)`. It is a declared lossy
  display choice. The composition is `R_z(pi/2) P T x`.

For example, in doubled coordinates:

```text
(2,0,0,0,2,0,0,0)
  -> (1,1,1,−1,1,1,1,−1)
  -> (1,1,1, 1,1,1,1, 1)
  -> (2,0,0,0,2,0,0,0).
```

A single vector `[1,−2,1]` does not specify a 3D subspace of 8D space or an E8
root. General rational 8D inputs are accepted explicitly; floats/bools, wrong
widths and inputs exceeding 256-bit numerator/denominator bounds are rejected.
The exact action APIs preserve the caller's coordinate scaling.

The separate spatial example chooses an active right-handed +90° z rotation:

```text
R_z(pi/2) (x,y,z) = (−y,x,z)
(1,−2,1) -> (2,1,1)
squared norm = 6 on both sides
dot product = 1
cos(angle between vectors) = 1/6, so angle is about 80.406°.
```

A quarter-turn about z leaves the z component unchanged, so it does not make
these 3D vectors perpendicular. Triality has order three; this spatial rotation
has order four. They are separate operations. Orthogonal display rotations
preserve Euclidean norms, distances and dot products and cannot by themselves
improve an invariant geometry metric.

For this P, no universal 3×3 triality matrix can satisfy `P T = B P`: `P e_3 = 0`
but `P T e_3 = (1/2,−1/2,−1/2)`. Projected roots can collide and projected lengths
and angles need not equal their 8D values. The tests retain this counterexample.

## Useful connection to Trinite

As a stencil, `[1,−2,1]` means `x_i − 2 x_(i+1) + x_(i+2)`, the second difference
at the middle sample. It annihilates constant and affine sequences at equally
spaced indices. The synthetic quadratic samples `0,1,4` give 2. Trinite already
computes this as `geometry.metrics(trajectory)['order_2_hex']`; reuse it.

Raw token-index differences have activation units and depend on sampling and
scale. They are not the Menger curvature metric. Interpreting a second difference
as a derivative requires declaring a common spacing h and division by h²;
arclength sampling has different semantics from integer token spacing. Fewer
than three samples yield an empty difference sequence, not measured zero.

A later neural observation must freeze its own protocol and publish the producer
before inspecting target outcomes. Specify checkpoint/capture identities,
layers/tokens, normalization, step units and spacing, an explicit hidden-width-to-8D
map, whether it is fixed or fitted, and training-only fitting/exposure. Keep
native-space metrics alongside projected metrics and all failures. Exact 8D
symmetry does not imply that a neural hidden state carries the corresponding
representation. Applying T to arbitrary mapped activations preserves norms by
construction; a small norm-preservation residual is therefore a software check,
not evidence of learned triality.

A useful hypothesis is whether native second-difference patterns distinguish
same-item/different-carrier pairs from carrier-matched/different-item pairs,
and whether an explicitly declared E8 anchor mapping retains information beyond
matched alternatives. Predeclare independently shuffled sequences, shuffled
anchor/orbit labels, matched random/non-E8 anchors, and random orthogonal receivers
as appropriate to the metric. Preserve capacity, counts and fitting budget;
record correctness and costs separately. A three-cycle built into a receiver
cannot serve as evidence that the model discovered that cycle. Coordinate
rotation is an invariance check, not an independent reasoning null.

Training remains geometry-off. Learning adequacy and the separately reviewed
[Phase 5](ROADMAP.md) objective, quality/cost margins and matched ablations still
precede a geometry intervention. This reference increment bypasses no gate.

## Source, compatibility and checks

The mathematical construction matches
[QSOLKCB/E8 at 292cc9a](https://github.com/QSOLKCB/E8/tree/292cc9a41dd08bb8bc48b060c9e36157dbe3d1f0),
specifically its `e8-core.js` and `docs/MATHEMATICAL_RECEIVERS.md`. This is a new
Python implementation of the declared formulas, not a vendored JavaScript
runtime. Donor and Trinite are MPL-2.0; Trent Slade / QSOL-IMC attribution and
SPDX notices are retained in the new source. No donor material is admitted as
training data, and no ETQ selector, sonification, physics or quantum protection
claim is imported.

Affected requirements: TRI-I01/I05/I09/I12/I14. Numerical model, native training,
existing `TRI-GEO-OBS-001` metrics/controls and their byte-bound producers remain
compatible and unchanged. This reference has a separate identity; it does not
revise that historical protocol. The new module imports only contracts and
standard-library exact arithmetic; existing model/training modules acquire no
new imports. Root discovery and the existing CPU CI include the reference tests.

The ten new checks cover root families/parity/norms, all root images and all
57,600 pairwise inner products, exact full-space order/orthogonality, a known
vector/spinor cycle, complete disjoint orbits, quarter-turn closure, lossy
projection, input bounds/immutability, analytic stencil cases and the actual
launcher without optional site packages. These establish software conformance,
not an improvement in AI reasoning.
