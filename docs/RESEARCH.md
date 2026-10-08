# Research and evaluation

Status: hypotheses and planned protocols. Trinite has no model observations or capability results.

## Questions and falsification

| Hypothesis | Test | Outcome against it |
| --- | --- | --- |
| A small ternary model learns the selected formal tasks with acceptable quality/cost. | Dense versus ternary lanes on family-held-out tasks, matched data/token budget | Quality loss or cost outweighs the preregistered acceptable margin |
| Representation geometry distinguishes logical structure from surface carrier. | Same logic/different carrier and same carrier/different logic controls | Native-space differences vanish or follow carrier alone |
| A geometric training intervention improves held-out reasoning. | Controlled loss intervention and ablations | Geometry changes without task improvement, or gains disappear after controls |
| The small model can outperform a selected larger comparator in a scoped task. | Frozen comparator, prompts, scoring, exposure, and resource protocol | Gain disappears under contamination, prompt, budget, or uncertainty controls |

“Outperform larger models” is a research goal, not an established property. Report task, metric, comparator revision, uncertainty, and cost. A narrow task advantage does not imply general intelligence or broad reasoning superiority.

## Geometry reference boundary

Reference [QSOL-GEO-REASON's scientific contract](https://github.com/QSOLKCB/QSOL-GEO-REASON/blob/e770f585bf3136b47a657772156116d5f02b1e77/SCIENTIFIC-CONTRACT.md) and [mathematical specification](https://github.com/QSOLKCB/QSOL-GEO-REASON/blob/e770f585bf3136b47a657772156116d5f02b1e77/MATH-SPEC.md). The reviewed revision contains a numerical kernel, a separate exact-mathematics proof layer, and a capture instrument/preregistered request; it reports no executed production model observation. None of these artifacts establishes a geometric training benefit for Trinite.

A pinned reference adapter must pass synthetic conformance fixtures before use. Trinite's custom model capture is a new instrument and does not inherit another model's capture validation. Cross-check finite differences, path length, cosine convention, Menger curvature, alignment/resampling, degenerate cases, and numeric tolerances against the pinned definitions. A differentiable approximation for a later loss is separately specified and validated; it cannot silently replace the analysis metric.

## Capture and protocol freeze

A representation state records model/checkpoint identity, layer, token IDs and spans, cumulative/isolated context mode, pooling, transform, dtype, device/backend, and extraction implementation. A trajectory step has a declared unit; mixing token steps and problem steps requires an explicit alignment rule.

Freeze train/validation/test families, primary metric, selected layers, pooling, alignment, inclusion/exclusion, uncertainty method, seed set, multiple-comparison policy, and acceptable quality/cost margins before confirmatory outcomes are inspected. Use native-space statistics as the primary evidence. Projections remain exploratory and record fit scope and label influence.

## Intervention matrix

The first geometry experiment compares dense/no geometry, ternary/no geometry, dense/geometry, and ternary/geometry under matched data and token budgets. Geometry-on lanes also need a matched shuffled-target or non-geometric auxiliary-loss control before attributing gains specifically to geometry. Account for added trainable parameters, auxiliary labels, extra compute, and teacher information. Report the no-geometry result even if the intervention fails.

For output correctness, use deterministic formal scorers, answer-normalization rules frozen before scoring, and family-level held-out aggregates. State the independent sampling unit for uncertainty; do not treat paraphrases of one underlying problem as independent evidence.

Larger pretrained comparators are a separate external lane: record full model/tokenizer revision, training exposure as exposed/unexposed/unknown with basis/confidence, quantization, prompts, generation policy, and backend. Unknown exposure remains unknown and limits parameter-efficiency claims.

## Evidence records

Use SIMULATION, OBSERVATION, ASSOCIATION, PERTURBATION, INTERVENTION, or MECHANISM according to the claim actually supported, following the reference contract. Record replication status separately as not_attempted, replicated, failed, or mixed, with scope and source-result identities when attempted. Exact mathematical proof is a separate evidence object, not an empirical capability class.

Every result records protocol/revision, complete metrics and controls, uncertainty, artifacts, result status (supports/null/contradicts/inconclusive), limitations, and provenance. No training run, geometry plot, successful hash check, or proof automatically grants a reasoning claim.
