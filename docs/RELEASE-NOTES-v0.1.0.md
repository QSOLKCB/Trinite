# Trinite v0.1.0 — Initial Research Milestone

**Release type:** immutable GitHub source/research checkpoint, following merged PRs #1–#26 (October 2026). **Status:** experimental; **Phases 0–4 implemented, Phases 5–9 pending**. This milestone does **not** assert a production-capable AI, a passing learning-adequacy gate, portable performance superiority, or an approved model-weights release. The Python package metadata still reports its separately managed pre-release version.

## What succeeded

- **Inspectable CPU foundation:** contract-first architecture, strict UTF-8 byte tokenizer, deterministic family-split synthetic data, bounded native dense/ternary/four-state models, explicit quantizers and numerical/gradient/capture checks. CPU-oriented conformance covers Python 3.11, 3.12 and 3.13. [#1](https://github.com/QSOLKCB/Trinite/pull/1)–[#4](https://github.com/QSOLKCB/Trinite/pull/4)
- **Training and evidence engineering:** hash-locked environment, controlled AdamW training, lossless safe-tensor checkpoint/restoration, independent scorer/admission, reproducible prompt capture, source- and environment-bound experiment requests, pinned [PROVENANCE](PROVENANCE.md) custody and exact replay. The later v2 assurance distinguishes final-inference checks from full optimizer/model/AdamW/history replay. [#4](https://github.com/QSOLKCB/Trinite/pull/4), [#8](https://github.com/QSOLKCB/Trinite/pull/8)–[#10](https://github.com/QSOLKCB/Trinite/pull/10), [#21](https://github.com/QSOLKCB/Trinite/pull/21)
- **Repeatable experimental infrastructure:** frozen protocols before outcomes, conservative handling of partial/failed workers, retained negative results, historical producer preservation, executable verification and lossless archival. OPT-guided reuse reduced declared geometric-null constructions from 432 to 144 with exact output parity in the scoped comparison; no portable speedup is claimed. [#6](https://github.com/QSOLKCB/Trinite/pull/6)
- **Verified synthetic curricula and finite reference maps:** formal arithmetic, folds, exact game-theory/QEC probes, 3,154 finite-maths/algebra examples across twelve tasks, and the canonical E8/ETQ-101/303 reference. The latter matches published donor events, graph and registry via the actual pinned Node runtime—not a claim of learned E8 structure. [#7](https://github.com/QSOLKCB/Trinite/pull/7), [#12](https://github.com/QSOLKCB/Trinite/pull/12), [#19](https://github.com/QSOLKCB/Trinite/pull/19), [#23](https://github.com/QSOLKCB/Trinite/pull/23)
- **Measured geometry, with controls:** detached trained/native captures, pinned QSOL-GEO-REASON analysis, a separately pinned Qwen3 stock/SYSTEM-only comparison, an integer Menger-style 3×3×3 lattice (up to 8,000 cells and 14,976 leaf face edges), and retrospective native-activation/20-anchor matched controls. The lattice mapping remains observational and training-exposed. [#5](https://github.com/QSOLKCB/Trinite/pull/5), [#16](https://github.com/QSOLKCB/Trinite/pull/16), [#25](https://github.com/QSOLKCB/Trinite/pull/25)–[#26](https://github.com/QSOLKCB/Trinite/pull/26)

## What failed, regressed or remains unproven

| Experiment / PR | Retained observation | Consequence |
| --- | --- | --- |
| Initial geometry [#5](https://github.com/QSOLKCB/Trinite/pull/5) | Final item-minus-carrier contrasts were **−0.371087 dense / −0.335223 ternary** on exposed prompts. Reversing both paths is orientation-invariant, not an independent shuffle control. | No evidence of useful geometric reasoning or causal benefit. |
| Three-lane QEC comparison [#7](https://github.com/QSOLKCB/Trinite/pull/7) | 27 training/comparison cells completed; all dense/ternary/four-state lanes scored **0% held-out complete answers** under this tiny budget. | Format selection inconclusive; loss reduction alone is insufficient. |
| Foundations and convergence [#8–#10](https://github.com/QSOLKCB/Trinite/pull/10) | All nine dense foundations cells failed at least one 90% task gate; the 27-cell learning-rate sweep had no universally passing candidate. | Held-out gates stayed blocked, with failures retained rather than hidden. |
| Longer-budget curriculum [#12–#15](https://github.com/QSOLKCB/Trinite/pull/15) | Nine dense 4,096-step cells passed exposed training gates in #12. The later full 27-cell held-out matrix failed in every dense seed: arithmetic **0–1/64**, folds **0–37/320**, lattice **32/40** with traversal **0/8**. Scalar follow-up again failed fold-sum 90% training adequacy. | More training and successful custody do **not** establish generalization. |
| Fold exposure/order [#17–#18](https://github.com/QSOLKCB/Trinite/pull/18) | Repeated 1×/2×/4× fold-sum exposure never passed all tasks; in equal-exposure grouped/mixed order, count and squares passed but sum failed every seed. | No schedule selected or causal ordering claim. |
| Maths pilot / rehearsal [#19–#20](https://github.com/QSOLKCB/Trinite/pull/20) | Expanded maths improved some tasks but regressed arithmetic training to **153/264 and 157/264**. Balanced rehearsal lifted arithmetic to **256/264 and 263/264** while maths declined and all 12 maths training tasks remained below their gates. | A measured retention/training tradeoff, not jointly adequate learning. |
| 8,192-update rehearsal [#22](https://github.com/QSOLKCB/Trinite/pull/22) | Maths improved **975→1,206/1,998** and **1,074→1,269/1,998**, arithmetic passed, and full optimizer replay succeeded—but **every maths task** still failed its 90% gate in both candidate seeds. | Aggregate improvement is a genuine result; the preregistered joint criterion remains false. |
| Task-balanced scheduler [#24](https://github.com/QSOLKCB/Trinite/pull/24) | Frozen 8,192-step control/candidate implementation and conformance merged; the complete four-cell **research** matrix, full replay and archive were not documented as finished outcomes in the merged PR. | **Do not infer a positive or negative research result** from green CI. |
| External / lattice observation [#16](https://github.com/QSOLKCB/Trinite/pull/16), [#26](https://github.com/QSOLKCB/Trinite/pull/26) | The 16-probe trained-native comparison reported 16/16 visible answers versus 0/16 Qwen stock and 4/16 SYSTEM-only; exposure/context differ. Lattice analysis reuses those **training-exposed** captures. | Neither model ranking, fresh generalization, nor a neural geometry mechanism is established. |

**Scientific gate status:** original fold/composed-output adequacy, fresh independent confirmation, model-format selection, controlled geometry intervention (Phase 5), packed CPU export (Phase 6), device reproduction (Phase 7), scaling (Phase 8) and release-readiness certification (Phase 9) are **not satisfied**. Native geometry loss remains `0`. A green CI badge signifies software conformance only.

## PR-by-PR milestone ledger

All listed merged PRs contribute to the source milestone. **PR #11 was an unmerged revert proposal and contributed no merged changes**—the #1–#26 range contains 25 merged pull requests, not 26.

| PR | Contribution / outcome |
| --- | --- |
| [#1](https://github.com/QSOLKCB/Trinite/pull/1) | Architecture, invariants, source/data/research contracts and original idea preservation |
| [#2](https://github.com/QSOLKCB/Trinite/pull/2) | Offline foundation, strict tokenizer, admission, splits and CPU conformance |
| [#3](https://github.com/QSOLKCB/Trinite/pull/3) | Dense/ternary CPU reference decoder, quantizer and bounded inspection |
| [#4](https://github.com/QSOLKCB/Trinite/pull/4) | Native training, resumable safe checkpoints and pinned PROVENANCE observer |
| [#5](https://github.com/QSOLKCB/Trinite/pull/5) | Phase 4 geometric capture and initial **negative** contrasts |
| [#6](https://github.com/QSOLKCB/Trinite/pull/6) | OPT-guided exact reuse, controls and CI orchestration |
| [#7](https://github.com/QSOLKCB/Trinite/pull/7) | Three-lane QEC comparison; **0% held-out** complete answers |
| [#8](https://github.com/QSOLKCB/Trinite/pull/8) | Foundations admission and learning gate; **blocked** |
| [#9](https://github.com/QSOLKCB/Trinite/pull/9) | Frozen training-only convergence sweep |
| [#10](https://github.com/QSOLKCB/Trinite/pull/10) | Checkpoint binding, retained sweep; **no passing candidate** |
| [#11](https://github.com/QSOLKCB/Trinite/pull/11) | **Not merged:** revert proposal, excluded from this release |
| [#12](https://github.com/QSOLKCB/Trinite/pull/12) | 4,096-step exposed-training success; held-out still untested |
| [#13](https://github.com/QSOLKCB/Trinite/pull/13) | Longer 27-cell held-out evaluation; **learning gate failed** |
| [#14](https://github.com/QSOLKCB/Trinite/pull/14) | Fresh generalization diagnoses and exact scalar task preparation |
| [#15](https://github.com/QSOLKCB/Trinite/pull/15) | Scalar comparison; **fold-sum training gate failed** |
| [#16](https://github.com/QSOLKCB/Trinite/pull/16) | Retained native/Qwen prompt-state comparison; descriptive only |
| [#17](https://github.com/QSOLKCB/Trinite/pull/17) | Fold-sum exposure controls; **no joint task pass** |
| [#18](https://github.com/QSOLKCB/Trinite/pull/18) | Equal-exposure ordering experiment; **sum failed all cells** |
| [#19](https://github.com/QSOLKCB/Trinite/pull/19) | Finite maths corpus and corrected pilots; arithmetic regression |
| [#20](https://github.com/QSOLKCB/Trinite/pull/20) | Balanced rehearsal restored arithmetic at a maths cost |
| [#21](https://github.com/QSOLKCB/Trinite/pull/21) | AGENTS.md and complete native optimizer/evidence replay |
| [#22](https://github.com/QSOLKCB/Trinite/pull/22) | 4,096/8,192 budget study; aggregate maths gain, failed per-task gates |
| [#23](https://github.com/QSOLKCB/Trinite/pull/23) | Exact E8/ETQ-101/303 donor-runtime finite parity |
| [#24](https://github.com/QSOLKCB/Trinite/pull/24) | New task-balanced schedule; full scientific matrix pending |
| [#25](https://github.com/QSOLKCB/Trinite/pull/25) | Bounded recursive 3×3×3 scaffold and graph/source verification |
| [#26](https://github.com/QSOLKCB/Trinite/pull/26) | Retrospective trained-activation lattice controls and evidence reader |

## Verification and reproducibility

Three independently linked GitHub workflows report current software status: [CPU conformance](https://github.com/QSOLKCB/Trinite/actions/workflows/cpu.yml), [reference backend conformance](https://github.com/QSOLKCB/Trinite/actions/workflows/reference-conformance.yml) and [ETQ donor runtime conformance](https://github.com/QSOLKCB/Trinite/actions/workflows/etq-donor.yml). Other historical manual/research matrices are **not** represented by these badge checks. Sources, frozen protocols, original producer hashes, audits and lossless evidence packets are retained in this repository and its documented research records; replication depends on the specified inputs, environments and run profiles.

Use [GETTING_STARTED.md](GETTING_STARTED.md), [ROADMAP.md](ROADMAP.md), [PHASE-4.md](PHASE-4.md), [MATHS-RETENTION-BUDGET.md](MATHS-RETENTION-BUDGET.md) and [ACTIVATION-LATTICE.md](ACTIVATION-LATTICE.md) for authoritative commands and experiment-specific limits.

## Immutable publication note

Tag **`v0.1.0` only after the final documentation PR has merged and the exact intended `main` SHA is checked**. Release title: **Trinite v0.1.0 — Inspectable CPU Research Baseline, Verified Evidence and Negative Results**. Publish this narrative as an initial source milestone, not a production release. GitHub release immutability must remain enabled; never move the tag, rewrite the published archive or backfill failed evidence. Corrections require a subsequent version with explicit provenance. No weights, hosted service, PyPI wheel or DOI are represented as released here.
