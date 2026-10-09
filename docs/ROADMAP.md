# Roadmap

Phases 0–4 are merged. The merged OPT detour applies exact reuse and CI orchestration improvements, with independent equivalence checks and scoped timing evidence in [OPTIMIZATION.md](OPTIMIZATION.md). The following three-lane/QEC detour is implemented and measured in [COMPARISON.md](COMPARISON.md). Phase 5 remains the next geometry phase; the foundations learning-adequacy gate precedes it; fresh confirmatory evaluation and independent replication remain empirical gates. Phases 5–9 are pending. Phase numbers describe work order, not published version tags. See [PHASE-1.md](PHASE-1.md), [PHASE-2.md](PHASE-2.md), [PHASE-3.md](PHASE-3.md), and [PHASE-4.md](PHASE-4.md) for evidence and limits.

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
| 9 — Release readiness and formal verification | Full source/data/weights/protocol/evidence inventory; multi-prover verification of a stable candidate; distribution and archive | Full-open checklist; scoped Lean and Isabelle/HOL proofs with implementation correspondence; pre-launch stress reports; verified publication inventory |

The Phase 3 observer gate uses the exact revision and acceptance checks in [PROVENANCE.md](PROVENANCE.md). An unavailable or incompatible upstream contract blocks that integration and Phase 3 completion; it must not be replaced with an unverified local schema. Native training performed while the observer gate is blocked remains explicitly unintegrated.

## Three-lane / QEC detour

Authorised by Trent Slade on 2026-10-09 after OPT: defer Phase 5 again to implement the dense/ternary/four-state comparison matrix and independently verified QEC task lane. The protocol is frozen at 3bc49f478ce0997b0ab46d36c79b9eb7ba7319cd, with 27 paired cells, native geometry loss zero, all failures retained and exact answer generation as primary evidence. See [COMPARISON.md](COMPARISON.md) and [COMPARISON-PROTOCOL.md](COMPARISON-PROTOCOL.md). This introduces classical quantization and generated numeric tasks, without quantum hardware or packed inference. The complete local 27-cell matrix and all six required CI jobs passed their software/integrity gates. All lanes scored 0% held-out complete answers, so weight-format selection remains inconclusive; a separately frozen learning-adequacy increment is needed before drawing comparative capability conclusions. Complete evidence is retained; independent replication and the manual hosted full matrix remain pending. Original Phase 4 negative results remain unchanged.

## Next roadmap phase

After the foundations learning-adequacy gate is measured and passes under a separately frozen protocol, scope Phase 5's controlled intervention: freeze the differentiable objective and acceptable quality/cost margins, then evaluate matched dense/ternary geometry-off/on lanes and auxiliary-loss controls. The first Phase 4 contrasts are negative, and reversal of both paths is an orientation-invariance diagnostic; neither supplies evidence of a reasoning benefit. Any revised order-scrambling or confirmatory observation needs a new protocol frozen before its target outcomes. Geometric loss remains zero until that reviewed increment. Curriculum intake still requires a separate source-admission and evaluation/exposure increment; no wholesale repository import.

Each phase should be a reviewable increment with stated requirements, completed checks, and unresolved gates. A software fixture does not close an empirical milestone. “Implemented” and “measured” are separate statuses.

Scaling is conditional: first prove the data, semantics, and measurement procedure, then decide whether a larger parameter tier is worth training. The initial 0.5B target is not a prerequisite or an automatic promise. Formal verification comes after a stable implementation target and before launch; it cannot prove language-model reasoning or substitute for empirical evaluation.

## Planned training corpus and curriculum

Status: curriculum approved by Trent Slade on 2026-10-09. The first short generated pack is implemented and measured in FOUNDATIONS; its dense-learning gate failed. Broader corpus preparation, admission and training remain pending. Teach short, verified basics first, then introduce curated first-party material and scale only after measured results. This section does not import data or extend the current Phase 3 training-conformance task.

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

## Repository-informed learning-adequacy increment

Trent requested intake of the QSOLKCB scan on 2026-10-09. Candidate revisions,
concrete tasks, source boundaries and evaluation limits are recorded in
[RESEARCH.md](RESEARCH.md#qsolkcb-repository-intake-research). This is a planned
increment before Phase 5, implemented in [FOUNDATIONS.md](FOUNDATIONS.md), with scoped numeric admission
and a separately frozen protocol. Its empirical gate is reported there. It does not change the frozen comparison experiment or admit a
repository wholesale.

| Order | Proposed addition | Completion gate |
| --- | --- | --- |
| 1 — Short foundations | HERESY-API arithmetic/fraction normalization; CONSTRAINT-SHIFT count/sum/squares; LATTICE address parsing, role mapping and traversal | File-level admission; new bounded examples; independently checked labels; semantic-family splits; frozen train and held-out exact-answer/EOS learning gates |
| 2 — Structure and errors | QSOL-MACH typed-contract classification and the generated YAML stress corpus; RUNE rounding/overflow tasks | Frozen error precedence, parser/numeric rules and limits; invalid-input coverage; related valid/invalid/repair examples grouped |
| 3 — Exact linear algebra | Small vector/matrix operations, E8 finite integer transformations and a narrow SONIFICATION rational/polynomial subset; TFT as a topic reference | Independent exact oracles; orbit/transformation grouping; measured learning before increasing task length or model size; approximate float diagnostics kept separate |
| 4 — Epistemic/context assessment | Fresh synthetic QSOL-ARK and QSOL-SEMANTIC-RELAY task families alongside QSOL-SUBSTRATE; later WHOAMI Gauntlet-style assessment | Evaluation answer keys excluded from training; real/absent/corrupt-context controls; unknown/contradictory evidence scored explicitly; visible donor fixtures remain conformance references |
| Separate engineering gate | Pinned PROVENANCE research-manifest integration; RUNE CPU export/resource patterns in Phase 6 | Upstream canonical/schema conformance and retained source/adaptation evidence; exported numerical parity and measured resources; no inferred truth, rights certification or speedup |

First establish useful learning on the short pack with an explicit preregistered
budget and thresholds. Then compare dense, ternary and four-state lanes using
matched inputs, initialization and training budgets, retaining all failures.
The thresholds and budget are frozen in [FOUNDATIONS-PROTOCOL.md](FOUNDATIONS-PROTOCOL.md). The measured 27-cell foundations run retained a blocked dense-learning gate; no held-out scoring was unlocked.
Preserve the initial null/inconclusive comparison and freeze a new request for
changed sources; changed tasks or scientific settings require a separately frozen
protocol. Keep Phase 5 intervention and larger-tier expansion conditional on
these measured gates.

The foundations implementation shares the native update loop, independently
checks new finite labels and prompt carriers, retains a source-bound training
decision and prevents held-out evaluation if dense learning fails. Software
conformance does not close the empirical gate; see [FOUNDATIONS.md](FOUNDATIONS.md).
YAML, typed contracts, linear algebra and geometry intervention remain later work.

The first foundations run is measured and integrity-verified, but **not
learning-adequate**. All nine dense cells fail at least one training task.
Held-out evaluation and format selection remain blocked. The next increment
is training-only convergence/stability development followed by a separately
frozen learning protocol; Phase 5, structure/algebra expansion and scaling
remain conditional. Preserve this blocked result alongside the earlier null
comparison. [FOUNDATIONS.md](FOUNDATIONS.md) retains the counts and evidence.

## Tail-end reproduction, verification and distribution

Requested by Trent after merging PR #8. These are planned deliverables, not
completed integrations or evidence that the failed learning gate passed.

| Placement | Deliverable | Completion gate |
| --- | --- | --- |
| Phase 7 — Accessible reproduction | Google Colab Free and Kaggle CPU notebooks; no GPU requirement | Commit-pinned source and hash-locked CPU dependencies; small inspect/audit/replay examples; clean-session execution on both providers; explicit environment differences, resource limits and downloadable evidence |
| Phase 7 / pre-launch — Independent stress | qBraid agent handoff in [QBRAID.md](QBRAID.md), updated for the stable candidate | Bounded CPU jobs, pinned agent/runtime/source, negative cases, complete failure retention and independent reproduction; human review of findings |
| Phase 9 — Formal software verification | Lean plus Isabelle/HOL for selected exact software contracts | Fixed release-candidate revision, explicit specifications/assumptions/trusted bases, checked proofs without unresolved proof holes, and tests or translation connecting specifications to implementation |
| Phase 9 — Hugging Face | Stable weights and model card, with optional permitted promotion | Owner chooses namespace/visibility; rights and current platform terms reviewed; safe tensors, model/config/tokenizer/source identities, reproduction instructions and honest limitations; verified download parity before public announcement |
| Phase 9 — README access | Centered notebook links and real version, CI and DOI badges directly below the logo | Targets actually exist and resolve; CI describes the named workflow, version names a real release, DOI names its deposited immutable archive; no placeholder success or DOI badges |

The notebooks use existing modules and bounded commands, rather than a second
training implementation. Keep interactive CPU demos small enough to recover
from interrupted sessions; full experimental matrices are explicit separate
runs. Colab's free resources are variable and not guaranteed. Check current
Kaggle settings/quotas at execution; record what actually ran. Do not use free
notebook hosts as persistent model-serving or background agent infrastructure.

Formal scope starts with tokenizer round trips, family split separation,
quantizer code/threshold laws, checkpoint consistency, archive path collision
rejection and evidence-gate state transitions. Lean and Isabelle/HOL provide two
proof environments; agreement on a specification alone does not verify Python,
PyTorch, filesystem behavior or floating-point kernels. Require a concrete
implementation correspondence argument and disclose those remaining trusted
components. TLA+ may supplement the gate/orchestration state machine if useful;
Rocq, F* and Why3 remain alternatives if a specific implementation boundary
justifies them. Avoid multiplying tools without a proof obligation.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
[qBraid Lab](https://qbraid.com/lab),
[Hugging Face model cards](https://huggingface.co/docs/hub/model-cards),
[Hub terms](https://huggingface.co/terms-of-service),
[Lean theorem proving](https://lean-lang.org/theorem_proving_in_lean4/) and
[Isabelle overview](https://isabelle.in.tum.de/overview.html).

## Training convergence development

The next training-only diagnostic is implemented in [CONVERGENCE.md](CONVERGENCE.md),
with a separately frozen [protocol](CONVERGENCE-PROTOCOL.md). It retains all
dense learning-rate candidates and never selects a new learning protocol or
unlocks held-out scoring automatically. Phase 5 remains conditional.

The complete convergence development matrix is measured and verified: 27 dense
cells, no worker/aggregate errors, and no candidate passes every training task
across all seeds. Lower-rate arithmetic/lattice improvements do not close the
fold gate. The next increment needs a separately frozen training budget/schedule
or curriculum decision before another learning/evaluation protocol. Complete
counts, failures and retrieval instructions are in [CONVERGENCE.md](CONVERGENCE.md).
Held-out scoring and Phase 5 remain blocked.


## Training budget development and game-theory curriculum

The next training-only budget increment is implemented in [BUDGET.md](BUDGET.md)
under its separately frozen [protocol](BUDGET-PROTOCOL.md): nine dense cells,
constant 0.001 and four times the prior updates, with replayable checkpoints at
every milestone. Held-out evaluation, format selection and Phase 5 remain gated.

Trent also requested game theory. The introductory [exact numeric pack](GAME-THEORY.md)
and independent oracle are implemented; teaching/training is a separate pending
curriculum protocol. Progress from payoff comparison and best responses through
strict dominance and pure Nash equilibria, then independently checked rational
mixed-strategy/minimax tasks and bounded repeated games. Keep symmetry-related
games in one split family and retain ties/no-pure-equilibrium cases. A claim of
strategic advantage requires scoped, fresh evaluation and comparator evidence.

QEC geometry, ETQ and UFT-ID 3.0 are recorded in [RESEARCH.md](RESEARCH.md#qec-geometry-etq-and-uft-id-intake).
Exact algebra/invariant tasks can precede a geometry-feature experiment after
learning succeeds. E8 features, harmonic observation, neural representations,
finite decoder games and physical quantum behavior require separate contracts
and evidence. Lean proofs of an observation specification do not verify a neural
model or the Python implementation automatically.


## Ethics and public-domain narrative intake

Trent requested an [ethics crash course](ETHICS.md) and reality/narrative
separation, using Asimov's laws as a fiction discussion reference and Harvard's
public-domain collection as a potential source. Curriculum preparation is
planned alongside game theory: evidence/fiction classification, consent/privacy,
conflicting duties, uncertainty, safe alternatives and escalation, followed by
fresh human-reviewed scenarios. It is not an admitted or trained ethics corpus.

The [Harvard intake review](ETHICS.md#harvard-public-domain-intake) records access
and rights boundaries. Select bounded exact editions with source, rights,
acquisition, OCR and exposure receipts before admission. Keep
real and fictional contexts explicitly labelled, and preserve disagreement in
ethical rubrics. Pre-launch stress and formal software verification cannot
substitute for this behavioural evaluation.

The [curated public-domain shortlist](PUBLIC-DOMAIN.md) proposes Mill, Kant (Abbott translation), Boole and Shelley for later small, source-bound ethics/logic/narrative intake. Exact editions, file identities, Australian rights, transformations and exposure gates precede admission; none enters the current training experiment.

The [Australian shortlist](PUBLIC-DOMAIN.md#australian-source-candidates) adds
Lawson, Banjo Paterson, Miles Franklin and Catherine Helen Spence for historical
Australian language, figurative interpretation and narrative perspectives.
Begin with bounded Lawson stories alongside separately admitted modern material.
Exact editions, contributor rights, transformations and distribution/exposure
review remain prerequisites; these are source candidates, not downloaded,
admitted or trained books.


The complete budget-development matrix is measured: all nine cells passed every
final per-task **training-only** check, with all milestone predictions/losses
verified against retained models. Prefix and one full same-host replay checks
are retained in [BUDGET.md](BUDGET.md). This resolves the exposed training floor
for the chosen longer dense budget, not held-out learning adequacy. The next
reviewable increment is implemented in [LEARNING.md](LEARNING.md), with a
separately frozen learning/evaluation protocol, matched lanes and this explicit
budget decision. Its corrected complete 27-cell run scored held-out examples, but every dense seed failed at least one per-task learning gate. Arithmetic/fold generalization stays low; lattice’s 80% aggregate hides traversal failure. Format selection and Phase 5 remain blocked. The next increment requires a separately frozen curriculum/generalization decision; curriculum training remains pending. Earlier negative and
blocked evidence stays preserved.

## Generalization diagnosis and scalar curriculum decision

The next separately frozen increment is implemented in
[GENERALIZATION.md](GENERALIZATION.md) under
[GENERALIZATION-PROTOCOL.md](GENERALIZATION-PROTOCOL.md): fresh read-only
verification of the complete corrected learning matrix, per-task/carrier
coverage and EOS/component diagnostics, and independently checked scalar
fold/traversal preparation over unchanged family splits. Arithmetic remains an
unchanged-text control. Diagnostic observations are descriptive; no new model
is trained in this increment.

The next training experiment must freeze its matched budget, exposure schedule,
checkpoint profile and per-task gates before outcomes. It should measure the
scalar curriculum and subsequently test original composed outputs under a
separately declared protocol. Correlated projections cannot count as independent
samples, component credit cannot replace exact-answer/EOS gates, and unseen
answers/vocabulary observations cannot establish a causal explanation of failure.
Phase 5, format selection, structure/algebra expansion and scaling stay blocked.

The diagnostic is now measured and retained: all 27 parent cells freshly verified,
with the 1,306-file parent inventory unchanged. Dense fold count components
reached 311–319/320 while complete answers remained 0–37/320; traversal stayed
0/8 and all dense held-out errors were wrong-answer rather than EOS failures.
All 27 lanes/cells remain in the report. This does not close learning adequacy;
the subsequently frozen scalar training experiment is reported below.

## Matched scalar curriculum learning

The next separately frozen training increment is implemented in
[SCALAR-LEARNING.md](SCALAR-LEARNING.md), under
[SCALAR-LEARNING-PROTOCOL.md](SCALAR-LEARNING-PROTOCOL.md). It trains the prepared
scalar corpus in all 27 matched fresh CPU cells with 4,096 updates, fixed parent/
projection exposure, safe checkpoints and fresh verified per-task gates. Arithmetic
texts remain the control. Original composed outputs are outside this experiment;
their assessment requires a separately declared protocol. Scalar success cannot
close composed learning adequacy, select a format or unlock Phase 5.

All 27 training workers completed. Dense arithmetic and lattice passed their
training tasks in every seed, but fold sum reached 934/1,150, 868/1,150 and
882/1,150, below the unchanged 90% training gate. The verified matrix decision
therefore blocked every held-out model call. The complete negative result,
checkpoints, update histories, receipts and exact producer closure are retained
with this increment. Nine arithmetic control payloads and normalized histories
match the earlier learning run exactly on the same host.

The next learning increment needs a separately frozen training-only decision
about fold-sum exposure or budget before further outcomes. Original fold-tuple
and traversal-address assessment still needs its own protocol. Geometry
intervention, format selection, broader corpus admission and release remain
blocked; this result does not justify proceeding directly to Phase 5.


## Separate external reference observation

The [reference geometry increment](REFERENCE-GEOMETRY.md) compares internal
activations from retained Trinite arithmetic training with pinned open-weight
Qwen stock and a SYSTEM-only Modelfile-derived profile. It adds byte audits, an
independent exact arithmetic oracle, fresh worker replay and separate closed
PROVENANCE bundles. Routine CI uses tiny simulations; pretrained model pilot
evidence is reported separately. These visible training probes do not establish
learning adequacy, matched compute or generalization. The fold-sum scalar gate,
held-out scoring, geometry intervention and Phase 5 remain blocked.

The retained pretrained pilot completed all three conditions with exact paired
fresh-worker capture/output replay, independent arithmetic checks and four
verified closed PROVENANCE bundles. The lossless packet includes exact producer
source and all captures; stock/profile final-state CKA was 0.996344 while output
format changed. This small descriptive result does not unlock later phases.

## Training-only fold-sum exposure development

The next reviewable increment is implemented in [FOLD-EXPOSURE.md](FOLD-EXPOSURE.md)
under a separately frozen [protocol](FOLD-EXPOSURE-PROTOCOL.md). Nine dense cells
compare fixed sum multiplicities 1, 2 and 4 over unchanged scalar examples and
4,096 updates. All count/sum/squares training gates and candidates are retained.
All nine cells completed and freshly verified; no candidate passes every
training task in every seed. Extra consecutive sum exposure reduces sum accuracy
in every seed and the 4× schedule also fails squares. All three 1× controls match
prior model/AdamW, history and prediction bytes exactly. There are no held-out
model calls or automatic schedule selection. Any subsequent learning or composed assessment needs a separate
protocol. Phase 5, format selection and scaling remain blocked.


## Training-only fold-order control

Following the negative fold-exposure result, the next separately frozen
training-only increment is implemented in [FOLD-ORDER.md](FOLD-ORDER.md), under
[FOLD-ORDER-PROTOCOL.md](FOLD-ORDER-PROTOCOL.md). Six dense cells compare parent
grouping with deterministic mixed-task order, preserving exactly eight visits
per example and equal total scored targets. The shared 3,450-update budget is an
explicit new decision, not a relabelling of the preceding 4,096-update evidence.
All six cells completed and freshly verified without worker or pairing errors;
both orderings fail the sum training gate in every seed while passing count and
squares. Mixed order improves one seed and worsens two. No schedule is selected.
A new separately frozen training-only decision must precede further outcomes. Held-out and composed assessment, format
selection, scaling and Phase 5 remain blocked.


## User-authorised finite maths pilot

Requested by Trent Slade after PR #18 on 2026-10-09. [MATHS.md](MATHS.md) implements a separately frozen finite maths/linear algebra corpus and paired arithmetic-control training pilot. This explicit new scope permits exploratory held-out diagnostics for its own protocol while historical fold-sum and composed adequacy remain blocked. It selects no format and unlocks no geometry phase or scaling; corpus admission and training results are reported separately.
