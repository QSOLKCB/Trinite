# Phase 4 evidence and limits

Phase 4 implements detached prompt capture, a pinned QSOL-GEO-REASON numerical adapter and bounded observation orchestration. Dense/ternary architecture, quantization, numerical training and the admitted dataset remain unchanged. Geometry loss stays zero; Phase 5 intervention is pending.

## Protocol and numerical conformance

The immutable [protocol commit](https://github.com/QSOLKCB/Trinite/commit/7ebe475f1e62f4675e2525dbed7ddba441db1998) preceded the first target geometry extraction. Its canonical configuration identity is sha256:d6f35530f9b0dc04a1836e4b8e2026ca16e2a690eb0825802405cef9e4795931. The packaged protocol has the same bytes. Per-run requests then bind the exact source, dependency/environment, data and checkpoint bytes before extraction. Local clock assurance is not independent registration.

QSOL-GEO-REASON commit e770f585bf3136b47a657772156116d5f02b1e77 supplies the unmodified kernel, Apache-2.0 license, mathematical/scientific contracts and original 19 tests, all bound in geometry-pin.json. Analytic tests add known path length, finite differences, cosine/zero conventions, arclength, straight/curved Menger cases, degenerate rejection, bounds and Decimal-context isolation. Every analysis uses the actual pinned kernel and lossless native float32 coordinates promoted to binary64.

The root standard-library suite has 60 checks, including the original 19 geometry tests. Required CPU suites separately exercise 17 model checks, 16 training/replay checks and 9 capture/observation checks. The unchanged PROVENANCE suites have 51 tests; one permissions test skips when run locally as root, and hosted CI runs it without that skip. Fixture audit, exact source pins, wheel packaging and all required CI lanes are review gates.

Capture tests cover Unicode byte spans, detached native states, cumulative causal-prefix parity with independent truncated forwards, source/token/shape/dtype/hex/tensor identity rejection and preservation of gradients/parameters/RNG. Integration tests train both elected checkpoints, retain all six families and all pairs/layers/controls, exclude validation/test prompts, verify the actual closed PROVENANCE bundle, compare repeated observer-on/off/failing results byte-for-byte, reject changed request/input/source/environment/pins and run freeze/measure in separate CLI processes.

## First descriptive observation

The local smoke uses the exact seed-0 tiny plan: 6,752 unique parameters, 60 steps, both CPU float32 lanes, geometry loss zero, the 36 exposed training prompts, and six equally weighted modulus families. Primary trajectories select every prompt byte at final; block.0 is a declared secondary diagnostic. Alignment uses arclength resampling and mean order-1 cosine. Full native path/difference/curvature metrics, all nine pairs per family and every diagnostic remain in the generated lane artifacts.

The contrast is mean same-item/different-carrier alignment minus mean same-carrier/different-item alignment. Here item means exact operands/modulus; logical operation type stays modular addition. Values below are rounded for reading; artifacts retain exact float.hex values.

| Layer / condition | Dense contrast | Ternary contrast |
| --- | ---: | ---: |
| final / native (primary) | -0.371087 | -0.335223 |
| block.0 / native (secondary) | -0.422628 | -0.327063 |
| final / deterministic Rademacher vectors | -0.012180 | -0.012180 |
| block.0 / deterministic Rademacher vectors | -0.027798 | -0.027798 |

Both primary contrasts are negative: these trajectories align more strongly for matching carriers with different operands than for matching items across carriers. This does not support the anticipated item-over-carrier direction in this exposed fixture. It does not falsify the broader reasoning hypothesis, because logical structure is not independently varied and no fresh confirmatory task is evaluated. result_status is inconclusive, scientific evidence class OBSERVATION, replication_status not_attempted; the random-vector diagnostic is separately SIMULATION. Six-family values/min/max are descriptive, with no inferential interval or p-value.

The frozen reversed-token-order condition reverses both trajectories. Its primary scores agree with native, as expected from orientation invariance of paired arclength/order-1 cosine; secondary floating-point differences are only rounding scale. Although the frozen protocol calls it an order-sensitivity control, it cannot serve as an independent order-scrambling null. This limitation is recorded in results and the contract. A revised shuffle requires a new protocol before its own target outcomes; this PR preserves the original control unchanged.

The smoke's completed bundle verified with the actual pinned upstream verifier: closed scope, zero known missing artifacts, no unresolved errors and integrity verified. Observer evidence eligibility is narrower than release readiness, which remains false. Hash verification demonstrates integrity/reference consistency, not reasoning, scientific validity or custody. The hosted manual workflow retains complete review artifacts for 14 days; archive them explicitly for longer retention.

## Reproduction and remaining gates

[GETTING_STARTED.md](GETTING_STARTED.md) gives current-tree train/freeze/measure/verify commands. Source and environment fingerprints remain strict; prior-tree checkpoints require their original implementation for exact replay. Wheel and repository execution use the same shipped source/pin/protocol bytes. Automated CPU CI checks each supported Python version; the manual workflow trains and observes both lanes on one worker to avoid cross-worker checkpoint fingerprints.

No geometry loss, packed export, new corpus, teacher, external comparator, independent replication, fresh evaluation, broader reasoning superiority, edge-device timing or release archive is implemented here. Phase 5 must freeze its differentiable objective, quality/cost margins and matched auxiliary-loss controls before intervention outcomes. The negative result and reversal limitation remain available when deciding that next experiment.
