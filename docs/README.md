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
| [ROADMAP.md](ROADMAP.md) | Current development-order entrypoint and preserved phase gates/history |
| [ROADMAP-V1.md](ROADMAP-V1.md) | Diagnostic-first milestones, RSH/integer/capacity tracks and conditional v1 scope |
| [Compute/KV harness proposal](COMPUTE-KV-HARNESS-PROPOSAL.md) | Uncached resource instrumentation followed by separately gated cached inference |
| [Research update, 26 October](research/UPDATE-2026-10-26.md) | Later research candidates and source-pinned shrek_GPT lessons |
| [Research assessment, 22 October](research/ASSESSMENT-2026-10-22.md) | Dated supporting rationale and original source/rights limitations |
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | Contributor and implementing-agent workflow |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Runnable foundation/model commands, dependency acquisition, and artifacts |
| [PHASE-1.md](PHASE-1.md) | Foundation conformance evidence and limits |
| [PHASE-2.md](PHASE-2.md) | Reference-model interfaces, numerical checks, and limits |
| [PHASE-3.md](PHASE-3.md) | Native-training protocol, replay/observer evidence, and limits |
| [GEOMETRY-PROTOCOL.md](GEOMETRY-PROTOCOL.md) | Protocol frozen before the first target geometry outcomes |
| [E8-REFERENCE.md](E8-REFERENCE.md) | Exact E8/ETQ-101/303 map, phase cycles and second-difference proposal; no neural intervention |
| [GEOMETRY.md](GEOMETRY.md) | Capture, numerical adapter, request binding, and artifact contracts |
| [PHASE-4.md](PHASE-4.md) | Geometry conformance, observed contrasts, and control limitations |
| [OPTIMIZATION.md](OPTIMIZATION.md) | Exact reuse, CI coverage, local timing evidence and rollback contracts |
| [COMPARISON-PROTOCOL.md](COMPARISON-PROTOCOL.md) | Frozen three-lane / exact QEC task experiment |
| [COMPARISON.md](COMPARISON.md) | Matrix commands, task admission, results and scientific limits |
| [PROCEDURAL-QEC-V3.md](PROCEDURAL-QEC-V3.md) | Frozen cursor implementation, historical-order limitation and pinned reproduction boundary |
| [PROCEDURAL-QEC-V3-NEGATIVE-RESULT.md](PROCEDURAL-QEC-V3-NEGATIVE-RESULT.md) | Complete local four-cell negative result, original-byte evidence and archive identities |
| [PROCEDURAL-QEC-NONDISPLACING-PROPOSAL.md](PROCEDURAL-QEC-NONDISPLACING-PROPOSAL.md) | Review-only extra-update hypothesis, historical-order prerequisite and explicit scientific/resource gate |
| [FOUNDATIONS-PROTOCOL.md](FOUNDATIONS-PROTOCOL.md) | Frozen basics curriculum and learning gates before Phase 5 |
| [FOUNDATIONS.md](FOUNDATIONS.md) | Foundations admission, replay, scoring, evidence and measured gate |

## Source and contract precedence

The current user instruction controls task scope. Within this plan, INVARIANTS defines requirements; the topic contracts define their specific interfaces; ROADMAP defines implementation order. IDEA is historical motivation. A contradiction must be resolved explicitly before implementing the affected feature, with all affected documents updated together.

The phrase “fully open” is a release goal covering source, configuration, tokenizer, data lineage and admissible data, training procedure, weights, inspection, and evaluation evidence. Publication of weights alone cannot satisfy it. Inspecting activations does not guarantee a complete explanation of the learned computation.

Keep the root limited to the short project README, LICENSE, and eventually necessary build/configuration entrypoints. All narrative documentation belongs in docs/; the logo is a documentation asset in docs/assets/. Planned source layout is described in MODULES; no empty module scaffolding is introduced now.

Tail-end reproduction, formal verification and distribution plans are in
[ROADMAP.md](ROADMAP.md#tail-end-reproduction-verification-and-distribution).
The initial pre-launch agent handoff is [QBRAID.md](QBRAID.md).

The historical training-only diagnostic is [CONVERGENCE.md](CONVERGENCE.md),
with its frozen [protocol](CONVERGENCE-PROTOCOL.md). For the current proposed
next workflow, see [ROADMAP-V1.md](ROADMAP-V1.md).

- [Training budget development](BUDGET.md) and [frozen protocol](BUDGET-PROTOCOL.md).
- [Exact game-theory foundations](GAME-THEORY.md); numeric corpus prepared, training pending.
- [Ethics, reality and narrative curriculum](ETHICS.md); Harvard source intake pending.
- [Curated public-domain sources](PUBLIC-DOMAIN.md); editions/rights/exposure admission pending.

- [Matched learning/evaluation](LEARNING.md) and [frozen protocol](LEARNING-PROTOCOL.md); complete three-lane evidence and scoped learning gates.

- [Generalization diagnostics and scalar preparation](GENERALIZATION.md) — read-only verified failure/coverage analysis, inherited family splits and untrained scalar targets.
- [Generalization protocol](GENERALIZATION-PROTOCOL.md) — frozen parent identities, diagnostic definitions and conservative gates.

- [Scalar curriculum learning](SCALAR-LEARNING.md) and [frozen protocol](SCALAR-LEARNING-PROTOCOL.md) — matched scalar training, fresh gates and separate composed-output limits.

- [Fold-sum exposure development](FOLD-EXPOSURE.md) and [frozen protocol](FOLD-EXPOSURE-PROTOCOL.md).

- [Fold-order development](FOLD-ORDER.md) and its [frozen protocol](FOLD-ORDER-PROTOCOL.md): equal-exposure training-only ordering control.


## Finite maths curriculum

[MATHS.md](MATHS.md): admitted exact finite maths/algebra, independently checked oracles, family isolation, paired native-training control, full prediction/checkpoint/PROVENANCE retention and explicit exploratory limits.

[MATHS-RETENTION.md](MATHS-RETENTION.md): fixed balanced arithmetic rehearsal versus the unchanged pooled maths curriculum, explicit exposure accounting and fresh safe-state verification.

[MATHS-RETENTION-V2.md](MATHS-RETENTION-V2.md): separate verification tiers, external artifact commitments, complete optimizer replay, strict admitted task counts and real integration coverage.

- [Balanced rehearsal budget development](MATHS-RETENTION-BUDGET.md): training-only 4,096/8,192-update comparison and lossless replay archive.
- [Task-balanced maths scheduling](MATHS-TASK-BALANCE.md): separately frozen training-only task exposure comparison at 8,192 updates in both arms.

## Recursive 3×3×3 scaffold

[LATTICE-SCAFFOLD.md](LATTICE-SCAFFOLD.md): frozen integer Menger-style
parent/child geometry and exact face adjacency, source-bound reproducible
artifact verification, independent tests and blocked neural-mapping gates.

## Trained activation-to-lattice observation

[ACTIVATION-LATTICE.md](ACTIVATION-LATTICE.md): source-bound retrospective
64→3 projection of retained real native trained captures, four edge/cardinality
matched lattice controls, split-restricted normalization and no new training.

## Post-v0.1.0 CNR-001 structural study

[CNR-001.md](CNR-001.md): CC BY MaleCNS source, v0.1.0 native activation
replay, ETQ-101/303 reduced graph comparisons, exact full 303-state cycle and
actual lifted-graph walk, degree-preserving controls and separate read-only
research evidence. No trained model changes or Phase 5 claim.

- [Non-displacing auxiliary engineering implementation](PROCEDURAL-QEC-AUXILIARY-IMPLEMENTATION.md): actual historical mapping and bounded native conformance; scientific execution remains disabled.

- [Fixed-code scale-refit engineering candidate](SCALE-REFIT.md): separately
  versioned numerical approximation tests; no native training-lane activation.

- [Declared-state observation and legacy migration](DECLARED-STATE.md): strict optional label codec, raw-text checks, seven-file audit; no training or epistemic capability claim.

- [Matched PCGrad engineering lane](PCGRAD-ENGINEERING.md): exact two-vector
  projection and same-snapshot toy optimizer replay; scientific execution gated.

- [Learned ternary scale engineering](LEARNED-TERNARY-SCALES.md): isolated learned/frozen controls and exact tiny replay; no learning claim.

- [Fixed-code refit training pilot](SCALE-REFIT-TRAINING.md): paired versioned model integration, exploratory formal-fixture quality and complete bounded replay.

- [External NCM reference feasibility pilot](NCM-REFERENCE-PILOT.md): isolated
  pretrained Sudoku adapter and CPU contracts; [RTX smoke result and archive](NCM-RTX-SMOKE-RESULTS.md)
  retain the initial failure and successful four-board replay, not generalization evidence.
