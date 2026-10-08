# Roadmap

Phases 0–4 are merged. The merged OPT detour applies exact reuse and CI orchestration improvements, with independent equivalence checks and scoped timing evidence in [OPTIMIZATION.md](OPTIMIZATION.md). The following three-lane/QEC detour is implemented and measured in [COMPARISON.md](COMPARISON.md). Phase 5 remains the next research phase; fresh confirmatory evaluation and independent replication remain empirical gates. Phases 5–9 are pending. Phase numbers describe work order, not published version tags. See [PHASE-1.md](PHASE-1.md), [PHASE-2.md](PHASE-2.md), [PHASE-3.md](PHASE-3.md), and [PHASE-4.md](PHASE-4.md) for evidence and limits.

| Phase | Deliverable | Completion gate |
| --- | --- | --- |
| 0 — Contracts | Clean docs layout, stack, model/module plan, research/data/evidence boundaries | Reviewed docs-only diff; original idea/artwork retained; internal links resolve |
| 1 — Foundation | Python package, configuration validation, byte tokenizer, rights-admissible formal generator, family splits, CPU CI | Offline tokenizer/span fixtures; rights/split audits; dependency lock and bounded CI |
| 2 — Reference model | ~1.25M dense and ternary models, quantizer, bounded inspection | Shape/count assertions; masks; quantizer/gradient conformance; dense/ternary forward fixtures |
| 3 — Native training | Tiny controlled supervised runs, explicit checkpoints/resume, first PROVENANCE adapter | Frozen run settings; learning smoke; interrupted replay; pinned upstream acquisition/conformance and Trinite mapping checks in PROVENANCE; observer isolation and integrity checks |
| 4 — Geometry measurement | Trinite capture instrument and pinned QSOL-GEO-REASON cross-check | Synthetic conformance; capture identity/span/dtype checks; preregistered observation and controls |
| 5 — Controlled intervention | Frozen geometry objective and matched ablations | All lanes/controls in RESEARCH evaluated; uncertainty, extra compute, null/negative outcomes retained |
| 6 — CPU export | Packed format and offline importer/inference/inspection | Invalid-input rejection; code/logit/token/capture parity; measured CPU resources |
| 7 — Reproduction and edge evidence | Repeated runs, at least one measured laptop and one capable SBC lane | Scoped device reports and replay outcomes; no universal hardware claim |
| 8 — Conditional expansion | Larger tier and optional Unsloth/native-backend experiments | Earlier results justify resource spend; compatibility, exposure, budgets, and parity frozen |
| 9 — Release and optional formal layer | Full source/data/weights/protocol/evidence inventory, archive; focused exact proofs if useful | Full-open checklist satisfied; artifact verification; stationary release target before formalization |

The Phase 3 observer gate uses the exact revision and acceptance checks in [PROVENANCE.md](PROVENANCE.md). An unavailable or incompatible upstream contract blocks that integration and Phase 3 completion; it must not be replaced with an unverified local schema. Native training performed while the observer gate is blocked remains explicitly unintegrated.

## Three-lane / QEC detour

Authorised by Trent Slade on 2026-10-09 after OPT: defer Phase 5 again to implement the dense/ternary/four-state comparison matrix and independently verified QEC task lane. The protocol is frozen at 3bc49f478ce0997b0ab46d36c79b9eb7ba7319cd, with 27 paired cells, native geometry loss zero, all failures retained and exact answer generation as primary evidence. See [COMPARISON.md](COMPARISON.md) and [COMPARISON-PROTOCOL.md](COMPARISON-PROTOCOL.md). This introduces classical quantization and generated numeric tasks, without quantum hardware or packed inference. The complete local 27-cell matrix and all six required CI jobs passed their software/integrity gates. All lanes scored 0% held-out complete answers, so weight-format selection remains inconclusive; a separately frozen learning-adequacy increment is needed before drawing comparative capability conclusions. Complete evidence is retained; independent replication and the manual hosted full matrix remain pending. Original Phase 4 negative results remain unchanged.

## Next roadmap phase

After this comparison detour merges and a separately reviewed learning-adequacy gate is established, scope Phase 5's controlled intervention: freeze the differentiable objective and acceptable quality/cost margins, then evaluate matched dense/ternary geometry-off/on lanes and auxiliary-loss controls. The first Phase 4 contrasts are negative, and reversal of both paths is an orientation-invariance diagnostic; neither supplies evidence of a reasoning benefit. Any revised order-scrambling or confirmatory observation needs a new protocol frozen before its target outcomes. Geometric loss remains zero until that reviewed increment. Curriculum intake still requires a separate source-admission and evaluation/exposure increment; no wholesale repository import.

Each phase should be a reviewable increment with stated requirements, completed checks, and unresolved gates. A software fixture does not close an empirical milestone. “Implemented” and “measured” are separate statuses.

Scaling is conditional: first prove the data, semantics, and measurement procedure, then decide whether a larger parameter tier is worth training. The initial 0.5B target is not a prerequisite or an automatic promise. Optional formal verification comes after a stable implementation target; it cannot prove language-model reasoning or substitute for empirical evaluation.

## Planned training corpus and curriculum

Status: curriculum approved by Trent Slade on 2026-10-09; corpus preparation, admission, and training remain pending. Teach short, verified basics first, then introduce curated first-party material and scale only after measured results. This section does not import data or extend the current Phase 3 training-conformance task.

| Tier | Source | Training target | Admission / evaluation gate |
| --- | --- | --- | --- |
| 1 — Foundations | Independently verified generated arithmetic, logic, short sequences, and basic vector/matrix operations | Correct answers, structured completion, and simple transformations | Formal verification, content identities, and family-held-out evaluation; visible software fixtures remain conformance data |
| 2 — YAML | A new generated YAML stress corpus | Structure, indentation, types, validation, and minimal repairs; harder cases introduced gradually | Pin YAML version/parser and resource limits; verify answers with that oracle; keep valid/invalid variants and repair counterparts in the same family |
| 3 — Epistemic discipline | Selected public [QSOL-SUBSTRATE](https://github.com/QSOLKCB/QSOL-SUBSTRATE) contracts and source-bound records | Distinguish known, retrieved, inferred, unknown, conflicting, and fictional claims; preserve claim maturity | Pin source revision/record hashes and metadata; select stable interpretation rules for training; deliver changing project facts through retrieval |
| 4 — Contextual language | Curated [AUSTRALIAN-FOR-AIS](https://github.com/QSOLKCB/AUSTRALIAN-FOR-AIS) training examples | Banter, understatement, ambiguity, and context-dependent meaning | Preserve benchmark/pilot evaluation sets; create separate training examples and group related context contrasts; preserve annotation uncertainty |

AUSTRALIAN-FOR-AIS currently documents 60 synthetic pilot prompts awaiting human annotation. They are not human-labelled gold training targets. Referenced comedy programmes and other third-party works remain research references rather than automatically admitted dialogue.

The authorised data-policy addition is **Trent's own material explicitly approved for training**, alongside the existing eligible formal/public-domain inputs. Before importing it, reflect this category consistently in DATA and the affected invariants and record the author/rights holder, permission scope, supporting evidence, immutable source identities, transformations, and split/exposure audit. Repository ownership is not a blanket admission of bundled third-party material or private records.

Each source must be selected at file/record level rather than dumped wholesale. Maintain separate training, validation, and fresh confirmatory evaluation families; benchmarks used for training or tuning cannot also support an uncontaminated generalisation claim. QSOL-SUBSTRATE mutable facts should retain source/date context rather than becoming timeless memorised assertions.

At the initial roughly 1.25M parameter tier, target narrow completion, classification, and transformation tasks. Establish reproducible learning and held-out behavior before broader language training, larger parameter tiers, or the conditional 0.5B goal. Curriculum acquisition requires a separately reviewed training/data increment after this Phase 3 conformance gate; expansion remains conditional in Phase 8.

## Foundations reference intake

Suggested by Trent on 2026-10-09: [Codecademy linear-algebra cheatsheet](https://www.codecademy.com/learn/dsml-math-for-machine-learning/modules/math-ds-linear-algebra/cheatsheet) and [quanghuy0497/Cheatsheet-collection](https://github.com/quanghuy0497/Cheatsheet-collection/tree/19a000df986efad76cefabdd7544b00dee21baad). These are curriculum-planning references, not admitted training data or changes to the completed comparison protocol. The collection is an index of material from multiple authors; inspect each original source and its rights separately.

Use the topic map to design our own short numeric examples: vector addition/scaling, dot products, matrix shapes/products, identity/permutation operations, and small exact linear systems. Build labels with independent bounded integer/rational oracles, retain generation/admission evidence, and group related examples and transformations into the same split family. Keep ordinary integer/rational algebra distinct from GF(3) and packed-ququart algebra. Probability and optimisation topics can follow after the basic exact-answer/EOS gates succeed. Third-party explanations, sheets and bundled examples require separate file-level admission before any training use.
