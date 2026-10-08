# Documentation

Trinite investigates whether a small, fully inspectable model can combine ternary linear weights with controlled geometric reasoning experiments and practical CPU inference.

Phases 1–4 implement the offline foundation, a frozen symbolic fixture, dense/ternary CPU models, inspection, native training, safe replay, detached prompt capture, pinned geometry analysis, and a pinned PROVENANCE observer. Tiny visible-fixture learning and geometry observations are conformance/descriptive evidence. Generation/export, geometry intervention, and fresh capability evaluation remain planned.

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

## Source and contract precedence

The current user instruction controls task scope. Within this plan, INVARIANTS defines requirements; the topic contracts define their specific interfaces; ROADMAP defines implementation order. IDEA is historical motivation. A contradiction must be resolved explicitly before implementing the affected feature, with all affected documents updated together.

The phrase “fully open” is a release goal covering source, configuration, tokenizer, data lineage and admissible data, training procedure, weights, inspection, and evaluation evidence. Publication of weights alone cannot satisfy it. Inspecting activations does not guarantee a complete explanation of the learned computation.

Keep the root limited to the short project README, LICENSE, and eventually necessary build/configuration entrypoints. All narrative documentation belongs in docs/; the logo is a documentation asset in docs/assets/. Planned source layout is described in MODULES; no empty module scaffolding is introduced now.
