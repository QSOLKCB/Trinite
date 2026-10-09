# Documentation

Trinite investigates whether a small, fully inspectable model can combine ternary linear weights with controlled geometric reasoning experiments and practical CPU inference.

Phases 1–4 implement the offline foundation, a frozen symbolic fixture, dense/ternary CPU models, inspection, native training, safe replay, detached prompt capture, pinned geometry analysis, and a pinned PROVENANCE observer. Tiny visible-fixture learning and geometry observations are conformance/descriptive evidence. The three-lane/QEC detour adds bounded greedy generation and a complete initial comparison; all lanes scored zero held-out complete answers, leaving format selection inconclusive. General generation/export, geometry intervention, and fresh capability evaluation remain planned.

## Reading map

| Document | Purpose |
| --- | --- |
| [IDEA.md](IDEA.md) | Original brainstorming, preserved verbatim; questions are not decisions |
| [INVARIANTS.md](INVARIANTS.md) | Openness, data, scientific, and runtime requirements |
| [ARCHITECTURE.md](ARCHITECTURE.md) | First model specification and system boundaries |
| [MODULES.md](MODULES.md) | Responsibilities, interfaces, and permitted dependencies |
| [TECH-STACK.md](TECH-STACK.md) | Language, dependencies, alternatives, and source references |
| [DATA.md](DATA.md) | Admissible data, lineage, curation, and split policy |
| [TRAINING.md](TRAINING.md) | Quantizer, training procedure, reproducibility, and workflow plan |
| [RESEARCH.md](RESEARCH.md) | Geometric reasoning, controls, comparisons, and evidence limits |
| [INFERENCE.md](INFERENCE.md) | CPU execution, export, inspection, and resource measurements |
| [PROVENANCE.md](PROVENANCE.md) | Observation boundary and evidence requirements |
| [ROADMAP.md](ROADMAP.md) | Ordered implementation phases and completion gates |
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | Contributor and implementing-agent workflow |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Runnable foundation/model commands, dependency acquisition, and artifacts |
| [PHASE-1.md](PHASE-1.md) | Foundation conformance evidence and limits |
| [PHASE-2.md](PHASE-2.md) | Reference-model interfaces, numerical checks, and limits |
| [PHASE-3.md](PHASE-3.md) | Native-training protocol, replay/observer evidence, and limits |
| [GEOMETRY-PROTOCOL.md](GEOMETRY-PROTOCOL.md) | Protocol frozen before the first target geometry outcomes |
| [GEOMETRY.md](GEOMETRY.md) | Capture, numerical adapter, request binding, and artifact contracts |
| [PHASE-4.md](PHASE-4.md) | Geometry conformance, observed contrasts, and control limitations |
| [OPTIMIZATION.md](OPTIMIZATION.md) | Exact reuse, CI coverage, local timing evidence and rollback contracts |
| [COMPARISON-PROTOCOL.md](COMPARISON-PROTOCOL.md) | Frozen three-lane / exact QEC task experiment |
| [COMPARISON.md](COMPARISON.md) | Matrix commands, task admission, results and scientific limits |
| [FOUNDATIONS-PROTOCOL.md](FOUNDATIONS-PROTOCOL.md) | Frozen basics curriculum and learning gates before Phase 5 |
| [FOUNDATIONS.md](FOUNDATIONS.md) | Foundations admission, replay, scoring, evidence and measured gate |

## Source and contract precedence

The current user instruction controls task scope. Within this plan, INVARIANTS defines requirements; the topic contracts define their specific interfaces; ROADMAP defines implementation order. IDEA is historical motivation. A contradiction must be resolved explicitly before implementing the affected feature, with all affected documents updated together.

The phrase “fully open” is a release goal covering source, configuration, tokenizer, data lineage and admissible data, training procedure, weights, inspection, and evaluation evidence. Publication of weights alone cannot satisfy it. Inspecting activations does not guarantee a complete explanation of the learned computation.

Keep the root limited to the short project README, LICENSE, and eventually necessary build/configuration entrypoints. All narrative documentation belongs in docs/; the logo is a documentation asset in docs/assets/. Planned source layout is described in MODULES; no empty module scaffolding is introduced now.

Tail-end reproduction, formal verification and distribution plans are in
[ROADMAP.md](ROADMAP.md#tail-end-reproduction-verification-and-distribution).
The initial pre-launch agent handoff is [QBRAID.md](QBRAID.md).

The next training-only diagnostic is [CONVERGENCE.md](CONVERGENCE.md), with
its frozen [protocol](CONVERGENCE-PROTOCOL.md).

- [Training budget development](BUDGET.md) and [frozen protocol](BUDGET-PROTOCOL.md).
- [Exact game-theory foundations](GAME-THEORY.md); numeric corpus prepared, training pending.
- [Ethics, reality and narrative curriculum](ETHICS.md); Harvard source intake pending.
- [Curated public-domain sources](PUBLIC-DOMAIN.md); editions/rights/exposure admission pending.

- [Matched learning/evaluation](LEARNING.md) and [frozen protocol](LEARNING-PROTOCOL.md); complete three-lane evidence and scoped learning gates.

- [Generalization diagnostics and scalar preparation](GENERALIZATION.md) — read-only verified failure/coverage analysis, inherited family splits and untrained scalar targets.
- [Generalization protocol](GENERALIZATION-PROTOCOL.md) — frozen parent identities, diagnostic definitions and conservative gates.

- [Scalar curriculum learning](SCALAR-LEARNING.md) and [frozen protocol](SCALAR-LEARNING-PROTOCOL.md) — matched scalar training, fresh gates and separate composed-output limits.
