# Documentation

Trinite investigates whether a small, fully inspectable model can combine ternary linear weights with controlled geometric reasoning experiments and practical CPU inference.

Phase 1 implements the contracts/configuration, tokenizer, data, and CLI foundation modules, a frozen 48-example symbolic conformance fixture, and bounded CPU CI. Model, training, inference, geometry, and PROVENANCE integration remain planned. The fixture validates software; it is not a held-out capability benchmark or a training result.

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
| [GETTING_STARTED.md](GETTING_STARTED.md) | Runnable offline Phase 1 commands and artifacts |
| [PHASE-1.md](PHASE-1.md) | Implemented contracts, conformance evidence, and remaining gates |

## Source and contract precedence

The current user instruction controls task scope. Within this plan, INVARIANTS defines requirements; the topic contracts define their specific interfaces; ROADMAP defines implementation order. IDEA is historical motivation. A contradiction must be resolved explicitly before implementing the affected feature, with all affected documents updated together.

The phrase “fully open” is a release goal covering source, configuration, tokenizer, data lineage and admissible data, training procedure, weights, inspection, and evaluation evidence. Publication of weights alone cannot satisfy it. Inspecting activations does not guarantee a complete explanation of the learned computation.

Keep the root limited to the short project README, LICENSE, and eventually necessary build/configuration entrypoints. All narrative documentation belongs in docs/; the logo is a documentation asset in docs/assets/. Planned source layout is described in MODULES; no empty module scaffolding is introduced now.
