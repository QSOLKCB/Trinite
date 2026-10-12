# E8 triality reference and second-difference proposal

Requested by Trent on 2026-10-12, from the idea headed “E8 Triality + theta=pi/2
+ [1,-2,1]” dated 23 July 2026. Status: exact reference calculator and ETQ-101/303 map implemented;
neural mapping, target activation study and training intervention pending.

## What runs now

```sh
PYTHONPATH=src python scripts/run_e8_reference.py
PYTHONPATH=src python -m unittest discover -s tests -p test_e8_reference.py -v
```

Print the complete ETQ-compatible finite map as canonical JSON:

```sh
PYTHONPATH=src python scripts/run_e8_reference.py --etq-map > etq-map.json
```

All commands work offline with the standard library. The calculator emits
`trinite.e8-reference.v1` / `TRI-E8-REF-001`, labelled `ANALYTIC_REFERENCE`. It
prints exact root/orbit counts, the declared triality numerator, an explicit
3D receiver, the quarter-turn example the ETQ reference commitments and a synthetic second difference using
the existing pinned geometry kernel. It does not load a checkpoint or inspect
neural activations. This output is mathematical conformance, not a source-bound
neural experiment, PROVENANCE assurance receipt or learning result.

`src/trinite/e8_reference/__init__.py` owns the new exact rational algebra. The thin
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
training data. The compatible finite ETQ selector/events below are reference
software; no rendered audio, physics or quantum protection claim is imported.

Affected requirements: TRI-I01/I05/I09/I12/I14. Numerical model, native training,
existing `TRI-GEO-OBS-001` metrics/controls and their byte-bound producers remain
compatible and unchanged. This reference has a separate identity; it does not
revise that historical protocol. The isolated nested package preserves the frozen flat v1 source inventory and
imports only contracts and
standard-library exact arithmetic; existing model/training modules acquire no
new imports. Root discovery and the existing CPU CI include the reference tests.

The eleven E8 checks cover root families/parity/norms, all root images and all
57,600 pairwise inner products, exact full-space order/orthogonality, a known
vector/spinor cycle, complete disjoint orbits, quarter-turn closure, lossy
projection, input bounds/immutability, analytic stencil cases and the actual
launcher without optional site packages. These establish software conformance,
not an improvement in AI reasoning.

## ETQ-compatible state map

Trent requested SONIFICATION compatibility on 2026-10-12. The additional
`e8_reference/etq.py` implements the finite ETQ map from
[SONIFICATION at 27cfea9](https://github.com/QSOLKCB/SONIFICATION/tree/27cfea99a3bab445c74b1aba5c38993fe931f568).
Its documentation is v3.0.1; the event runtime contract remains **v3.0.0** and
its preserved base is **ETQ-101 v2.0.0**. This is a new Python implementation
using the already implemented exact root/triality functions. Runtime remains
standard-library only and imports no donor JavaScript, model or RNG.

| Layer | Exact map now implemented | Meaning |
| --- | --- | --- |
| Selected roots | First two fixed roots, then first 33 complete triality triples | 101 distinct root labels in the donor's canonical order |
| Internal sectors | `j=0,1` are singlets; `j=2+3m+q`, `m=0..32`, `q=0..2` | `C² direct-sum (C³³ tensor C³)`; 33 direct-sum blocks |
| SCL operator | Singlets zero, each internal block `diag(1,−2,1)` | A diagonal observable, distinct from a row stencil on trajectory samples |
| Phase and cycle | `F_3=X_3 exp(−i pi D_3/2)` | Phase first, then cyclic label shift; multipliers `[-i,−1,−i]` |
| External fibre | `tensorIndex=3j+a`, `a=0,1,2` for every site | Independent fibre, including on the two internal singlets; 303 states |
| Support traversal | `(j,a) -> (j+1 mod101,a+1 mod3)` | Selector-order shift; exact order 303, with phase closure |
| Root graph | Selected adjacency from doubled inner product 4; Cartesian lift with `C_3` | 1,687 base edges, 5,364 lifted edges, lifted degrees 24–57 |
| Symbolic note code | Singlets 13/113; internal `(m,q)` gives `14+33q+m` | Preserved integer labels; no acoustic tuning or audio renderer |

This supplies the precise ETQ meaning of the proposed combination:

```text
D_3 = diag(1,−2,1)
U_D(pi/2) = diag(−i,−1,−i)
F_3 |q> = exp(−i pi d_q/2) |q+1 mod3>
F_3³ = I, since (−i)(−1)(−i) = 1.
```

Here theta is a **phase angle**, distinct from the spatial z quarter-turn in
the earlier demonstration. Phase labels are exact Gaussian-unit exponents
`[3,2,3]` modulo four; no rounded complex arithmetic is needed. The internal
`F_Q` fixes both singlets. ETQ-303 instead uses the separate selector-order
ring shift `R_101 tensor F_3`; it is not the internal root triality permutation.
The CRT inverse `n=j+101*(2*(a−j mod3) mod3)` reconstructs the event index.

The compatible event document contains all 101 labelled sites and all 303
events, with internal q and external a kept distinct. It does not assert that
the 303-event ordering is a graph walk. The lifted graph is available through
`etq.lifted_graph()`. The canonical JSON serializer delegates to Trinite's
strict integer-only serializer and removes its final LF for ETQ compatibility;
the existing serializer/protocol bytes are unchanged.

All **seven** published SONIFICATION commitments match: selected basis,
adjacency, labelled degree-potential numerators, site registry, event sequence,
lifted edge list and the complete canonical event document. The latter has the
published domain-separated SHA-256
`d0ab60df11084a230e641d737ec22b2b2ca7d3a3ba5b8759ba669ac4d5254dc0`.
The [donor commitments](../fixtures/e8-etq-reference/donor-commitments.json) bind
the exact upstream canonical fixture path and Git blob. The actual unmodified
pinned donor Node runtime was also executed: canonical event bytes, every
lifted graph field/edge and every site registry entry agree with Python. Its
[comparison record](../fixtures/e8-etq-reference/donor-runtime-comparison.json)
identifies the source blobs and Node version. This is cross-language finite
software conformance, not an independent neural or empirical replication.

The original v1 comparison record is preserved as historical evidence. The
review fix adds [v2 execution evidence](../fixtures/e8-etq-reference/donor-runtime-evidence-v2.json)
and [complete donor stdout](../fixtures/e8-etq-reference/donor-runtime-output.json).
The exact three donor modules, published canonical fixture and MPL-2.0 licence
are retained under `fixtures/e8-etq-reference/donor/`. The offline verifier checks
their Git blob identities before executing those snapshot bytes in a temporary
directory. Its small JavaScript driver calls the upstream selector, graph,
registry, event builder, serializer and hash functions directly. No npm package
or network access is used. The seven expectations come from the byte-verified
upstream fixture, rather than trusting the separately transcribed commitments.

```sh
PYTHONPATH=src python scripts/check_etq_donor.py
PYTHONPATH=src python -m unittest discover -s tests/etq_upstream -v
```

These **additional evidence checks require Node** (the retained execution and
CI use v24.19.0); the Python calculator/exporter and original root suite remain
standard-library only. A separate `ETQ donor runtime conformance` job runs on
every PR. Missing Node, execution failure, altered source, differing complete
payloads, incorrect historical flags, changed producer identities or altered
retained output fail the check; there is no skip or fixture-refresh fallback.
Every verification runs Node anew, compares full event bytes, every graph field
and registry entry with Python, validates all seven hashes, and checks the v1
record against those results. The v2 record binds donor and comparison-producer
files, the original records and complete stdout with byte lengths and SHA-256.
It is a read-only conformance check, not a signed custody receipt or a replay of
the unretained original invocation. The current Node version is printed separately
from the historical v1 version. This strengthens TRI-I07/I08/I14 evidence;
the finite map and all historical training/geometry producers stay compatible.

Ten additional ETQ checks preserve the published commitments, internal/fibre
separation, complete phase/order closure, exact address inverses, graph
product/triality invariance, symbolic note bijection, input domains, immutability,
large-power composition and the actual offline canonical exporter. Fixtures
are retained expectations; tests never regenerate them.

The separate nine-test donor suite includes real upstream execution and
corruption cases for every donor blob, copied commitments, historical flags and
metadata, producer/output identities, retained stdout, complete payloads with
unchanged hash labels, and missing Node. Neither suite rewrites expected evidence.

This port covers the finite map, not SONIFICATION's full implementation-identity
bundle, receipt/export pipeline, D4-TIA covariant engine, continuous generator
simulation or physical dynamics. Its SONIFICATION **event payload** matches;
a Trinite producer would have a separate source identity and custody receipt.

### Later neural coupling

An AI hidden vector does not automatically have 101 ETQ amplitudes or 303 tensor
coordinates. A future adapter must explicitly define a read-only encoder from
selected hidden states to site scores/amplitudes, its dimension/capacity and
normalization, and whether/how it is fitted on training-only inputs. Keep the
101 root labels, internal q and independent external a typed throughout; a
303-output encoder is a separate capacity choice, not a free reinterpretation
of a 101-output encoder. Preserve native hidden-state diagnostics and compare
matched shuffled-label and non-E8 receivers.

Applying a phase kick alone leaves diagonal populations unchanged. A phase
experiment therefore needs a declared phase-sensitive observable or subsequent
noncommuting operation; population differences cannot be attributed to the
kick itself. Any learned head, auxiliary loss or changed forward operation
requires a separately frozen intervention protocol with quality/cost controls.
No such encoder, fitting, training or activation observation is enabled here.
