# Roadmap

Phase 0 is merged. Phase 1 foundation implementation and local conformance are complete in the current PR; merge and the PR's remote CPU CI results remain review gates. Phases 2–9 are pending. Phase numbers describe work order, not published version tags. See [PHASE-1.md](PHASE-1.md) for evidence and limits.

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

## Next PR

Implement Phase 2 only: the dense/ternary reference decoder, quantizer and surrogate-gradient checks, causal/padding masks, configuration-to-tensor inventory, and bounded detached inspection. Introduce and lock the CPU PyTorch dependencies at that point. Verify the declared 1,247,232-parameter reference architecture against actual tensors. Do not begin long training, geometric losses, or PROVENANCE integration during model-conformance work.

Each phase should be a reviewable increment with stated requirements, completed checks, and unresolved gates. A software fixture does not close an empirical milestone. “Implemented” and “measured” are separate statuses.

Scaling is conditional: first prove the data, semantics, and measurement procedure, then decide whether a larger parameter tier is worth training. The initial 0.5B target is not a prerequisite or an automatic promise. Optional formal verification comes after a stable implementation target; it cannot prove language-model reasoning or substitute for empirical evaluation.

## Planned training corpus and curriculum

Status: curriculum approved by Trent Slade on 2026-10-09; corpus preparation, admission, and training remain pending. Teach short, verified basics first, then introduce curated first-party material and scale only after measured results. This section does not import data or extend the current Phase 1 implementation task.

| Tier | Source | Training target | Admission / evaluation gate |
| --- | --- | --- | --- |
| 1 — Foundations | Independently verified generated arithmetic, logic, and short sequences | Correct answers, structured completion, and simple transformations | Formal verification, content identities, and family-held-out evaluation; visible software fixtures remain conformance data |
| 2 — YAML | A new generated YAML stress corpus | Structure, indentation, types, validation, and minimal repairs; harder cases introduced gradually | Pin YAML version/parser and resource limits; verify answers with that oracle; keep valid/invalid variants and repair counterparts in the same family |
| 3 — Epistemic discipline | Selected public [QSOL-SUBSTRATE](https://github.com/QSOLKCB/QSOL-SUBSTRATE) contracts and source-bound records | Distinguish known, retrieved, inferred, unknown, conflicting, and fictional claims; preserve claim maturity | Pin source revision/record hashes and metadata; select stable interpretation rules for training; deliver changing project facts through retrieval |
| 4 — Contextual language | Curated [AUSTRALIAN-FOR-AIS](https://github.com/QSOLKCB/AUSTRALIAN-FOR-AIS) training examples | Banter, understatement, ambiguity, and context-dependent meaning | Preserve benchmark/pilot evaluation sets; create separate training examples and group related context contrasts; preserve annotation uncertainty |

AUSTRALIAN-FOR-AIS currently documents 60 synthetic pilot prompts awaiting human annotation. They are not human-labelled gold training targets. Referenced comedy programmes and other third-party works remain research references rather than automatically admitted dialogue.

The authorised data-policy addition is **Trent's own material explicitly approved for training**, alongside the existing eligible formal/public-domain inputs. Before importing it, reflect this category consistently in DATA and the affected invariants and record the author/rights holder, permission scope, supporting evidence, immutable source identities, transformations, and split/exposure audit. Repository ownership is not a blanket admission of bundled third-party material or private records.

Each source must be selected at file/record level rather than dumped wholesale. Maintain separate training, validation, and fresh confirmatory evaluation families; benchmarks used for training or tuning cannot also support an uncontaminated generalisation claim. QSOL-SUBSTRATE mutable facts should retain source/date context rather than becoming timeless memorised assertions.

At the initial roughly 1.25M parameter tier, target narrow completion, classification, and transformation tasks. Establish reproducible learning and held-out behavior before broader language training, larger parameter tiers, or the conditional 0.5B goal. Curriculum acquisition begins within reviewed Phase 3 training/data work after the model-conformance gate; expansion remains conditional in Phase 8.
