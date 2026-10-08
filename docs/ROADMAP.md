# Roadmap

All executable phases are pending. Phase 0 is delivered by the documentation foundation PR and completes only after review/merge. Phase numbers describe work order, not published version tags.

| Phase | Deliverable | Completion gate |
| --- | --- | --- |
| 0 — Contracts | Clean docs layout, stack, model/module plan, research/data/evidence boundaries | Reviewed docs-only diff; original idea/artwork retained; internal links resolve |
| 1 — Foundation | Python package, configuration validation, byte tokenizer, rights-admissible formal generator, family splits, CPU CI | Offline tokenizer/span fixtures; rights/split audits; dependency lock and bounded CI |
| 2 — Reference model | ~1.25M dense and ternary models, quantizer, bounded inspection | Shape/count assertions; masks; quantizer/gradient conformance; dense/ternary forward fixtures |
| 3 — Native training | Tiny controlled supervised runs, explicit checkpoints/resume, first PROVENANCE adapter | Frozen run settings; learning smoke; interrupted replay; observer isolation and integrity checks |
| 4 — Geometry measurement | Trinite capture instrument and pinned QSOL-GEO-REASON cross-check | Synthetic conformance; capture identity/span/dtype checks; preregistered observation and controls |
| 5 — Controlled intervention | Frozen geometry objective and matched ablations | All lanes/controls in RESEARCH evaluated; uncertainty, extra compute, null/negative outcomes retained |
| 6 — CPU export | Packed format and offline importer/inference/inspection | Invalid-input rejection; code/logit/token/capture parity; measured CPU resources |
| 7 — Reproduction and edge evidence | Repeated runs, at least one measured laptop and one capable SBC lane | Scoped device reports and replay outcomes; no universal hardware claim |
| 8 — Conditional expansion | Larger tier and optional Unsloth/native-backend experiments | Earlier results justify resource spend; compatibility, exposure, budgets, and parity frozen |
| 9 — Release and optional formal layer | Full source/data/weights/protocol/evidence inventory, archive; focused exact proofs if useful | Full-open checklist satisfied; artifact verification; stationary release target before formalization |

## Next PR

Implement Phase 1 only: package/configuration contracts, byte tokenizer with special tokens/spans, a tiny independently verified formal generator, deterministic family-level splits, and CPU conformance CI. Keep all fixtures within DATA admission and all modules within MODULES boundaries. Do not begin long training or a geometry loss during foundation work.

Each phase should be a reviewable increment with stated requirements, completed checks, and unresolved gates. A software fixture does not close an empirical milestone. “Implemented” and “measured” are separate statuses.

Scaling is conditional: first prove the data, semantics, and measurement procedure, then decide whether a larger parameter tier is worth training. The initial 0.5B target is not a prerequisite or an automatic promise. Optional formal verification comes after a stable implementation target; it cannot prove language-model reasoning or substitute for empirical evaluation.
