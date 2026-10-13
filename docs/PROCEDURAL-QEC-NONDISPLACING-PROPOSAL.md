# Trinite non-displacing procedural supervision proposal

**Review-only design, 13 October 2026. Not implemented, frozen, or authorized for execution.**

Recommend one prospective comparison: retain the exact historical eight-row final-answer batch at the start of each of 4,096 macro-rounds, then perform one additional eight-row native update. Compare extra final-answer rehearsal with extra procedural supervision of the same task kind. Both fresh arms would perform 8,192 AdamW updates on seeds 0 and 1. Historical schedule reconstruction is a separate prerequisite, not a second scientific hypothesis.

This draft confers no new compute budget or scientific authority. A focused PR must remain reviewable at the user's authorization gate. Start subsequent implementation from the then-latest verified `main`; this draft starts at verified main `7c3a2c8c07f97fe56bfc54eb4425698ce0a2c2fb`. The relevant native numerical files inspected at `d01ed20a11213761836ae9f022c9f1ca38c3780b` remain unchanged there.

## Evidence motivating the question

The completed current-v3 study passed all six assurance checks and full native optimizer replay but failed its training gate. Procedural fold-sum counts were 614 versus 808 in seed 0 and 762 versus 778 in seed 1. No procedural primary-task count exceeded its paired current-source control. Intermediate learning was partial: only 14 of 1,024 visited fold traces were correct in either procedural seed. These results justify considering a bounded alternative; they do not predict success.

V3 displaced 2,048 fold-final and 2,048 GF(3)-final visits per cell. Its historical final-only order also changed when source-dependent example identities changed. Cross-version differences cannot isolate the cursor correction. The proposed question is:

**With the same historical final-answer rehearsal subsequence and matched native update shapes, does spending an additional update on exact procedural targets improve fold-sum final-answer learning relative to spending that update on further final answers of the same kind, while meeting the existing per-task training gates?**

This tests the specified additional-update recipe. It is not an isolated cursor comparison, an unchanged historical eight-row control trajectory, or evidence about reasoning as a general mechanism.

## Separate historical schedule and custody prerequisite

Complete and review this boundary before freezing the prospective experiment. It needs no model calls or learning outcomes.

1. Preserve the existing executable historical audit and original comparison receipt unchanged. A passing audit reproduces the ordering failure; it never establishes historical parity.
2. Reconstruct the original admitted data and all 4,096 eight-row final-only batches for each seed from producer `93522bf8f18fe38970ceafdd2ace792f8922c382`. Validate its complete pinned source and original data/manifest commitments. Do not substitute two scheduling functions fed the same current rows.
3. Establish a bijection to prospective admitted rows by comparing the full schema and every field except the top-level `generator_revision` and `example_identity`. Preserve both actual identities and record the mapping separately. Reject missing, extra, duplicate, ambiguous, or otherwise changed rows. Prompt/answer equality alone is insufficient.
4. Bind the mapping and full prospective schedule in a new request. For every seed and macro-round, removing the additional update must recover the original eight-row batch in exact slot order, including prompt/answer bytes and all normalized row content. Check historical sequence commitments using the retained audit's exact encoding; differently encoded checkpoint schedule digests are not interchangeable.
5. Separately bind prospective data/manifest identities and the complete new source closure. Source changes must fail closed or receive newly reviewed identities, never silently rerank primary rows or rewrite old producers.

This establishes a prospective ordered-input guarantee. It does not repair the v3 run, reproduce historical parameters or logits, or close the separate earlier task-balance archival obligation. All historical protocols, receipts, negative results, numerical artifacts and immutable baselines remain unchanged.

## One proposed four-cell recipe

For each seed 0 and 1, create fresh paired `extra-final` and `extra-procedural` arms under one new producer and the same declared host/software/thread environment. Initialization identities must agree within each pair. Retain the existing dense float32 CPU model: 86,528 parameters, two blocks, width 64, four heads, feed-forward width 128, context 64, random initialization, constant learning rate 0.001, existing AdamW settings and gradient clipping, and geometry loss zero.

Each macro-round `m`, from 0 through 4,095, contains two consecutive native updates:

- Update `2m + 1`: the reconstructed historical eight-row final-only batch, identical in both arms.
- Update `2m + 2`: eight additional rows of the selected kind. The treatment uses the selected procedural mode's own cursor; the control uses a separate final-answer cursor for that kind.

The additional-batch cadence is fold prefix, GF(3) prefix, fold trace, GF(3) trace, repeating every four macro-rounds. Each treatment `(kind, mode)` cursor advances by eight only when selected. Each control `(kind, final)` cursor likewise advances by eight only on that kind's extra update. Within each stream, use original historical seed-hash ranking, translated through the verified row mapping. Do not rank regenerated current identities and assume order stability.

This control matches kind frequency and additional update count. Because final and procedural streams have independent rankings/cursors, it does not match each additional parent, prompt length or target-token exposure. Those differences belong to the recipe and must be reported.

Each treatment auxiliary stream receives 1,024 batches and 8,192 row visits, covering all 1,150 fold or 54 GF(3) rows before repeating. This exceeds v3 auxiliary exposure as part of the fixed additional-update recipe. No ratio, mode, budget or ordering is tuned from new outcomes.

| Visits per cell | Shared primary subsequence | Extra-final total | Extra-procedural total |
| --- | ---: | ---: | ---: |
| Arithmetic final | 16,384 | 16,384 | 16,384 |
| Fold-sum final | 10,240 | 26,624 | 10,240 |
| GF(3) final | 6,144 | 22,528 | 6,144 |
| Fold prefix | 0 | 0 | 8,192 |
| Fold trace | 0 | 0 | 8,192 |
| GF(3) prefix | 0 | 0 | 8,192 |
| GF(3) trace | 0 | 0 | 8,192 |
| All row visits | 32,768 | 65,536 | 65,536 |

The treatment displaces none of the historical primary visits. Total final-answer exposure is deliberately different because extra final rehearsal is the comparator. Intervening updates, extra weight-decay applications, AdamW evolution and padding prevent claiming preservation of the historical numerical trajectory. No third baseline arm is proposed.

## Supported native composition and normalization

Source inspection and a nontraining standard-library plan/tokenizer probe establish these existing capabilities:

- `BudgetRetentionPlan` accepts 8,192 steps, batch eight and diagnostics every 2,048 steps without modifying its implementation. The configured model remains 86,528 parameters.
- `ByteTokenizer.encode(..., pad_to=64)` preserves unpadded tokens and answer-plus-EOS target counts, appending PAD with zero attention/loss masks. Existing padding/causality and shifted-loss tests were inspected, not rerun for this draft.
- Encoding every training row to length 64 makes the unchanged native `target_loss` call the Decoder with `[8, 63]` inputs at every update in both arms. There are 127,008 attention-score cells per layer per update, below the existing 4,194,304 guard. Tensor shapes and dense operation structure match; actual time, energy and memory must still be measured.
- The sole `run_steps` loop performs each complete `zero_grad`, loss, backward, gradient-clipping and AdamW update. New composition can provide the interleaved admitted rows and existing numerical plan. It must not introduce another updater, gradient accumulation, a hidden optimizer or monkey-patched bounds.

Keep `target_loss` unchanged: each update sums scored answer-plus-EOS cross-entropies and divides by that update's scored-token count. Primary and additional batches are separate optimizer updates with this same normalization. Do not replace them with a combined loss or add a final/auxiliary coefficient. Real target/input counts differ despite identical padded shapes.

A nine-row update was considered and rejected: both `model.py`'s `MAX_BATCH=8` and the plan validators reject it. Raising these limits adds avoidable compatibility scope. Alternating supported eight-row updates needs no model architecture, tokenizer, numerical loss, optimizer or quantizer change.

This is a feasible composition route, not an implemented producer. New request/checkpoint/report schemas, source closure, restored-state validation and archive integration still require implementation and review. Ordered-input equality, padding mask/target equivalence and appropriate native forward/loss/gradient conformance checks remain prerequisites; no historical bitwise trajectory parity is promised.

## Outcomes and existing gates

Use the final checkpoint after all 8,192 updates. Do not select a checkpoint, winning seed or larger follow-on budget from results. Keep diagnostics on unique admitted training/development rows, separate from scheduled repetitions. Proposed diagnostic points are native steps 2,048, 4,096, 6,144 and 8,192, following 1,024, 2,048, 3,072 and 4,096 macro-rounds.

The existing 90% complete canonical final-answer-plus-EOS gates remain unchanged for **every task in both treatment seeds**:

- Add, subtract, multiply and divide: each at least 60/66.
- Fold-sum: at least 1,035/1,150.
- GF(3): at least 49/54.

Report the same criteria for controls, all six paired count differences, final-only and mixed losses, per-task/mode losses, exposure, and unique auxiliary correctness. Repeated visits never increase diagnostic denominators. An empty unvisited set has no accuracy estimate or held-out interpretation.

For review, keep two distinct judgments: the training gate passes only under the thresholds above; a positive fold contrast additionally requires the treatment's fold count to exceed its paired control in both seeds. Mixed, zero and negative contrasts remain visible even if an absolute gate passes. This proposed contrast rule is prospective, not an amended historical criterion. Two training-exposed seeds support only a bounded exploratory result.

All four cells must complete and pass all six assurance checks before reporting a complete verified matrix. Retain failures, timeouts and inconclusive results. No validation/test-family model calls are permitted. Original composed-answer adequacy, fresh family confirmation, independent-host replication, format selection, geometry-on training, scaling and release remain separate gates.

## Data-only planning counts and added computation

The [read-only planning script](evidence/procedural-qec-nondisplacing-proposal/planning_counts.py) reconstructs and admits the exact historical dataset through the existing isolated audit worker, checks source before and after, and enumerates the recipe without importing a numerical backend or calling a model. Its [planning receipt](evidence/procedural-qec-nondisplacing-proposal/planning-counts.json) binds the script, this proposal, the original audit and historical source/data/manifest identities. It is **pre-outcome mathematical planning**, not a frozen experiment or learning result. Prospective-current-source mapping is still unimplemented and must reproduce these mathematical counts.

| Seed | Arm | Primary targets | Additional targets | Total targets | Unpadded input tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Extra-final | 79,550 | 72,879 | 152,429 | 870,915 |
| 0 | Extra-procedural | 79,550 | 192,960 | 272,510 | 1,023,719 |
| 1 | Extra-final | 79,552 | 72,878 | 152,430 | 870,954 |
| 1 | Extra-procedural | 79,552 | 192,941 | 272,493 | 1,023,800 |

Targets count answer bytes plus EOS; unpadded inputs include BOS and EOS but exclude PAD. All four cells fit the existing 1,048,576 scored-training-target ceiling. Each cell has 4,194,304 padded encoded positions and 4,128,768 shifted model-input positions. These counts exclude diagnostic/generation and replay work.

The training matrix contains **32,768 native updates, 262,144 row visits and 849,862 scored targets**. Updates and row visits double relative to the four-cell v3 matrix. Fixed context padding adds work relative to historical dynamic padding, so this is not a measured two-times FLOP or wall-time estimate.

Preserve all three full replay passes used for the retained v3 study: matrix-time native replay, separate read-only native replay, and archive-time native replay. Each complete four-cell replay adds 32,768 native updates and its diagnostics. Training plus those three passes therefore entails **131,072 native updates** overall, before diagnostics and transport/source verification. Declare diagnostic and archive overhead, approved host, total time and storage allowance, and failure policy before authorization. Record actual targets, input/padding counts, per-stage wall time, peak RSS and failures. Any reduced-assurance proposal would require its own explicit review rather than silently omitting a pass.

The existing plan allows at most 960 seconds per invocation. That is an implementation bound, not granted expenditure or evidence the proposal fits. If supported time/resource bounds prove insufficient, stop for a separate reviewed decision; never silently raise limits or restart failed cells until one succeeds.

## Approval gate and prerequisites

The next focused PR should preserve this prospective design for review. Keep historical schedule/custody prerequisites distinct from the single learning hypothesis. Before any freeze or target training, require:

1. Review of the hypothesis, extra-final control, historical mapping, fixed cadence/padding, native normalization and proposed contrast rule.
2. Passing historical-order, mapping/admission, independent-cursor, exposure, padding, held-out exclusion, interruption/replay, malformed-input and immutable-source tests. Preserve frozen fixtures and nine-row rejection behavior.
3. New protocol/request/report/checkpoint identities; full runtime source and launcher closure; source-bound CPU environment; external verification commitments; lossless retention and availability plan.
4. Applicable CI and exact-final-head review, followed by explicit scientific and resource approval. Publication, reviewer approval or green CI for this draft does not authorize dispatch, merging the scientific change, protocol freeze, release or an unspecified resource budget. The user must explicitly approve the applicable scientific actions and bounded resources after prerequisites are demonstrated.

If a prerequisite fails, retain it and leave execution blocked. Current v3 is not a substitute comparison. This document does not change or freeze any protocol, source, request or research result.

A portable reconstruction command and the independently retained planning checksum are in the [planning evidence README](evidence/procedural-qec-nondisplacing-proposal/README.md). That check performs no training.

## Historical commitments and inspected sources

The following values were checked against the repository's original report, pinned audit and retained receipt. The comparison receipt's actual local SHA-256 was recomputed and matched. The planning pass separately regenerated and admitted the historical dataset, but did not rerun the complete old/current audit or any numerical replay.

| Commitment | Historical v2 | Compared current v3 |
| --- | --- | --- |
| Producer | `93522bf8f18fe38970ceafdd2ace792f8922c382` | `cc996e272800a5c02667001f7c96c8d11cb513cc` |
| Git tree | `c8cc28e560729722539a2a6ee47408f50a7a802e` | `061173b9f71e7aa1f5c1c3838e3d1b5e4f07a8ca` |
| Dataset SHA-256 | `120b95ac68825b50b92dc0210d1f142a30980bfe5acedcfafe6d8f3cdadf83c4` | `9ea3223ebd099933f9c05a3ccb9db43e4a7d19e3c1edf5922c326c2eec16d944` |
| Manifest SHA-256 | `315168dd6be4909cead81a45a4c3b5b1abb125cc69f3ca773b5145f18b0ac412` | `528cb27db98781db8148dd4f029e9a392d0d2a786d393f2ef399dcbdad95ff49` |

Original comparison receipt SHA-256: `7e0160ac9ca5c6241dd8001bbc8775c6dd1c51266755533705debd819d9a087b`. It records 3,876 normalized equal rows, all 3,876 identities changed, and 32,576/32,768 and 32,521/32,768 differing historical final-only visits in seeds 0 and 1.

Historical final-only sequence identities, in the retained receipt's encoding:

- Seed 0: `sha256:263b1878b2ee6eef523cd9dcc2cc616874e746ca2e398bd6ec4c43a4990f3527`.
- Seed 1: `sha256:b5680dd1f7f3e4275613882443108759c0ef4e01702ccba4c7685423effb870d`.

Inspected references:

- [V3 negative result and exposure](https://github.com/QSOLKCB/Trinite/blob/d01ed20a11213761836ae9f022c9f1ca38c3780b/docs/PROCEDURAL-QEC-V3-NEGATIVE-RESULT.md).
- [Historical audit](https://github.com/QSOLKCB/Trinite/blob/d01ed20a11213761836ae9f022c9f1ca38c3780b/docs/evidence/procedural-qec-control-order/README.md) and [original receipt](https://github.com/QSOLKCB/Trinite/blob/d01ed20a11213761836ae9f022c9f1ca38c3780b/docs/evidence/procedural-qec-control-order/comparison.json).
- [Existing 8,192-step plan](https://github.com/QSOLKCB/Trinite/blob/d01ed20a11213761836ae9f022c9f1ca38c3780b/src/trinite/retention_v2/budget/plan.py), [native loss/updater](https://github.com/QSOLKCB/Trinite/blob/d01ed20a11213761836ae9f022c9f1ca38c3780b/src/trinite/training.py), [padding](https://github.com/QSOLKCB/Trinite/blob/d01ed20a11213761836ae9f022c9f1ca38c3780b/src/trinite/tokenizer.py#L35-L64), and [model bounds](https://github.com/QSOLKCB/Trinite/blob/d01ed20a11213761836ae9f022c9f1ca38c3780b/src/trinite/model.py#L156-L175).

Affected requirements: TRI-I02/I03/I05/I06/I07/I08/I11/I12/I13/I14. TRI-I09/I10 remain claim boundaries. Historical compatibility means preserving original producers and evidence, not promising that extra updates reproduce old trajectories.
