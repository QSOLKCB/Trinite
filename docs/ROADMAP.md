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

A separate [Phase 5 authorization preflight](PHASE-5-PREFLIGHT.md) now validates prospective source/data/protocol bindings, a complete paired geometry-off/on/null-control matrix, external trust-policy digest and distinct Ed25519 steward/verifier signatures. It produces only an immutable **blocked** receipt: neither a positive learning gate, full optimizer replay, independent confirmation, objective conformance, nor a geometry runner exists in this increment. No `phase5_ready` state is unlocked; measured prerequisites and explicit future authorization remain necessary.

After the foundations learning-adequacy gate is measured and passes under a separately frozen protocol, scope Phase 5's controlled intervention: freeze the differentiable objective and acceptable quality/cost margins, then evaluate matched dense/ternary geometry-off/on lanes and auxiliary-loss controls. The first Phase 4 contrasts are negative, and reversal of both paths is an orientation-invariance diagnostic; neither supplies evidence of a reasoning benefit. Any revised order-scrambling or confirmatory observation needs a new protocol frozen before its target outcomes. Geometric loss remains zero until that reviewed increment. Curriculum intake still requires a separate source-admission and evaluation/exposure increment; no wholesale repository import.

Each phase should be a reviewable increment with stated requirements, completed checks, and unresolved gates. A software fixture does not close an empirical milestone. “Implemented” and “measured” are separate statuses.

Scaling is conditional: first prove the data, semantics, and measurement procedure, then decide whether a larger parameter tier is worth training. The initial 0.5B target is not a prerequisite or an automatic promise. Formal verification comes after a stable implementation target and before launch; it cannot prove language-model reasoning or substitute for empirical evaluation.

## PR #31 ternary dynamics and PR #32 Aizawa geometry reference

**Implementation status:** [PR #31](https://github.com/QSOLKCB/Trinite/pull/31) is **merged** at `507eabffce79d4702dc6a3ce53ea73cf0635ea06`, retaining a descriptive quantizer/STE instrument without a causal learning claim. **PR #32** adds the separate [Aizawa synthetic geometry benchmark](AIZAWA-REFERENCE.md) under a protocol frozen first at `c7fa03d25d21d1f3526d15cb76d5c725a798724c`, subject to review and CI. After PR #32 merges, verify post-merge checks and obtain separate approval to tag **immutable v0.2.0** on the exact resulting `main` SHA with unresolved scientific gates recorded.

**Order and boundary:** PR #30 and #31 have merged. PR #32 operates on a fixed **synthetic** reference and reads neither neural snapshots nor training gradients. Neither instrument authorizes geometry-on training, relaxes learning thresholds, changes the frozen quantizer, or establishes model superiority. Future activation-to-Aizawa or quantizer-to-geometry mapping would require a separate frozen protocol and independently bound inputs.

### PR #31 (merged) — Ternary training dynamics: is STE saturation suppressing useful updates?

- **Hypothesis, not a result:** The current `quantize()` uses detached absmean scale `alpha`, codes `round(W / alpha).clamp(-1,1)` and a masked surrogate passing gradients only where `abs(W / alpha) < 1`. The zero-code region is *not* intrinsically gradient dead; saturation outside the surrogate mask may prevent some weight updates. A masked gradient does not by itself establish that an update would have improved learning.
- **First deliverable:** New **read-only, CPU-only** diagnostics at frozen model/checkpoint/step identities: per-matrix ternary `-1/0/+1` occupancy, scale, normalized latent/threshold distribution, fraction inside/outside the STE support, realized gradient/update support, code transitions across predeclared steps and any non-finite/invalid cases. Distinguish upstream zero gradients from STE-masked gradients and optimizer state from observable code changes.
- **Controls and custody:** Keep the original random initialization, quantizer/AdamW/update/data path, context, geometry loss zero, and retained source closures unchanged. Pair seeds, tasks, exposure and budget with a separate dense comparator where valid. The new diagnostic must not mutate tensors, RNG or training order. Declare sampling steps and artifact schema before observing outcomes; bind exact source/model/checkpoint identities and retain all seeds/tasks, negative results, failures and resource overhead.
- **Gate:** Pass conformance and observer-on/off numerical parity; report *per-task* training/held-out eligibility under existing gates, not an aggregate-only improvement. Diagnostic correlations **cannot** prove saturation caused learning failure. Only a subsequent separately frozen intervention may compare a versioned alternative STE/scale against the unmodified baseline. Native training must not silently become QAT, teacher-distillation or packed-inference training.

### PR #32 (review pending) — Aizawa attractor: controlled geometry reference benchmark

- **Reference-only implementation:** Fixed parameters `a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1`, initial `(0.1,0,0)`, interval `[0,100]`, CPU binary64 classical RK4 with 20,000/40,000 update steps and 201 inclusive sampled positions. Early refinement convergence is gated; full-horizon disagreement is descriptive. SciPy `solve_ivp(RK45)` is not a hidden runtime dependency.
- **Geometry contract:** Each 201×3 sampled trajectory satisfies the existing **at most 256 point** `geometry.points()` interface; the unchanged pinned kernel computes distance, differences, Menger curvature and alignment. The canonical request, source/runtime closure, control samples, integration stream identities and report are independently recomputed by `--verify-only`.
- **Controls:** Full native trajectory, independent seeded temporal permutations, Rademacher sign null, periodic and affine analytic controls; orientation reversal is a **diagnostic only**, not an independent shuffle null. Retain all source-bound samples and metrics. This is `SIMULATION`, not learning, reasoning or a causal advantage.
- **After PR #31:** If independently verified native dense/ternary captures are available, freeze an additional *separate* mapping to compare their geometry with the synthetic reference and PR #31's code/gradient diagnostics. Never treat Aizawa states as model hidden states without a defined mapping, exposure boundary and matched controls. ETQ-101/303 graph-walk/fibre mapping is an optional separately specified follow-up, not a positive result or default Phase 5 intervention.

**Scientific boundary:** No new held-out scoring, geometry-on objective, learner modification, format selection, Phase 5 authorization or release claim is introduced by adding this roadmap.

## Post-v0.2.0 COSMO exact-oracle, triadic and Lean research (planned)

Authorised to record for later research by Trent on 2026-10-13. Use the [immutable COSMO v0.1.0 baseline](https://github.com/QSOLKCB/COSMO/releases/tag/v0.1.0), commit `51bdc9221553b06668168d1d8c21ec245b9b6707`, as a **source-pinned mathematical donor** only after PR #32 is merged, required post-merge CI is checked, and an explicitly approved **immutable Trinite v0.2.0** tag is published. PR #32 does not vendor COSMO, install Lean, change a numerical updater or claim an empirical benefit. COSMO's MIT and Trinite's MPL-2.0 attribution obligations must be carried into any eventual copied or adapted source.

- **Independent E8/Weyl oracle:** Compare COSMO `cosmo_core/e8.py`'s exact doubled-coordinate 240-root construction, lattice tests and 240 Weyl reflections with Trinite's existing E8 and embedded D4-triality reference. Predeclare the exact vector/scale conventions; Weyl reflections are **not** the same operator as the D4 triality. Retain algebraic invariants, cross-repository Git blob/source identities, counterexamples and failure modes; do not duplicate or weaken the authoritative Trinite runtime.
- **Triadic dynamics as a measured control:** COSMO `cosmo_core/triadic.py` provides a deterministic 8×8×8 = **512-cell** `{0,1,2}` lattice, synchronous periodic-neighbour updates, reproducible seeded states, loop closure and state entropy. Test whether its independently specified dynamics provides a useful *reference/null* for the PR #31 `{-1,0,+1}` occupancy/transitions and PR #32 geometry, **only after** freezing a sign-to-trit map, zero semantics, a bounded 512-cell-to-≤256-sample projection, paired seeds, nulls, comparison metric and source/exposure ledger. The lattice is neither a Trinite neural layer nor evidence of reasoning.
- **Lean theorem/implementation correspondence:** Source-pin COSMO `cosmovirus.lean`, Lean 4.33.1 and its Mathlib/trust policy. Its formal `(1,-2,1)` integer second-difference annihilation and six-state symbolic transition results can inform independently checked **integer** or finite-state specifications, with explicit tests translating those definitions to Trinite. They do **not** prove binary64 RK4, the Aizawa attractor, neural hidden-state dynamics, Trinite's surrogate-gradient behaviour or generalization.
- **Later export resilience:** COSMO's SECDED, ternary repetition-code and deterministic bytes↔DNA/FASTA storage can supply scoped corruption/recovery *software* controls for the future Phase 6 CPU export. Do not mistake error correction for cryptographic custody or biological evidence.
- **Claim maturity:** Preserve COSMO's **FORMAL / COMPUTATIONAL / SCIENTIFIC / HYPOTHESIS / SYMBOLIC** distinctions, explicit falsification and all negative outcomes. A later separate protocol must be source frozen before observing Trinite learning outcomes, with no Phase 5 authorization inherited.

**Checkpoint boundary:** Following a clean PR #32 merge, record the exact `main` SHA and required checks. Create a separately requested immutable `v0.2.0` release with source/protocol/evidence inventory and unresolved scientific gates **before** attempting this COSMO integration. This roadmap entry does not tag a release or change any empirical gate.

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

Requested by Trent Slade after PR #18 on 2026-10-09. [MATHS.md](MATHS.md) implements a separately frozen finite maths/linear algebra corpus and paired arithmetic-control training pilot. This explicit new scope permits exploratory held-out diagnostics for its own protocol while historical fold-sum and composed adequacy remain blocked. It selects no format and unlocks no geometry phase or scaling; Both complete pilots are measured and retained. The corrected pilot improves over the arithmetic control in both seeds, but only one seed beats the overall majority baseline and arithmetic retention regresses. Unique-system solving remains mixed. Learning adequacy and drift prevention are unresolved; historical gates stay blocked.


## Fixed arithmetic rehearsal development

After PR #19, the next separately frozen learning increment is [MATHS-RETENTION.md](MATHS-RETENTION.md): repeat the pooled maths pilot and compare four arithmetic/four maths examples in each batch with the same two seeds and update budget. All four cells completed and freshly verified. Rehearsal passes the arithmetic training criterion in both seeds but reduces maths accuracy; all maths training criteria remain failed. Both pooled safe tensors, histories and predictions reproduce PR #19 exactly. Actual visits, targets and training criteria remain visible. The reused test diagnostics are exploratory. No schedule is selected and historical learning adequacy, composed assessment, format selection, scaling and Phase 5 remain blocked.

## Planned retention audit and joint-learning budget comparison

Requested by Trent on 2026-10-12 after review of
[PR #21 at 84b54d2](https://github.com/QSOLKCB/Trinite/pull/21).
Status: assurance PR #21 merged; the next separately frozen budget comparison is
implemented in [MATHS-RETENTION-BUDGET.md](MATHS-RETENTION-BUDGET.md), with production
outcomes measured below. Software
verification, training performance, fresh generalization and release readiness
are separate decisions. A successful replay does not clear an accuracy gate.

The supplied independent review reports a complete four-cell production rerun
on Ryzen/AVX2 with exact within-host optimizer replay and read-only verification,
but its delivery contains the report alone. Preserve the complete audit as a
lossless source-bound archive before treating those local records as a published
evidence package: experiment request/digest, independently retained external
verification request/digest, replay summary, all model/AdamW payloads, histories,
predictions, custody, logs and an immutable member inventory. Retain the original
EPYC/AVX512 producers and negative results unchanged. Different host counts do
not justify changing an archived environment fingerprint.

The next learning question is whether a larger balanced-rehearsal training
budget improves finite maths learning while preserving arithmetic training
performance. Freeze the protocol and publish its producer before target outcomes.

| Arm | Updates | Arithmetic visits | Maths visits | Seeds |
| --- | --- | --- | --- | --- |
| Balanced rehearsal control | 4,096 | 16,384 | 16,384 | 0, 1 |
| Balanced rehearsal candidate | 8,192 | 32,768 | 32,768 | 0, 1 |

Both arms use the same four-arithmetic/four-maths batch composition, admitted
corpus, native initialization, model, optimizer and constant learning rate.
Rerun both arms under one new producer and declared environment. Freeze resource
limits, diagnostic/checkpoint schedule and stop conditions in advance. Doubling
updates increases both corpora's exposure and extends the optimizer trajectory;
this tests the larger-budget recipe, not the causal effect of maths exposure
alone, and is not a compute-matched superiority comparison.

The current flat v1 `BudgetPlan`/`MathsPlan` rejects more than 4,096 updates.
Introduce a separately versioned plan/protocol and complete runtime source
closure; preserve the frozen v1 plan, fixtures and evidence. Reuse one
authoritative updater, scorer, admission operation and safe tensor reader.

Predeclare joint-training acceptance as at least 90% complete canonical
answer-plus-EOS accuracy in each of the four arithmetic and 12 maths tasks, in
every candidate seed. Use exact admitted task counts and integer thresholds;
for arithmetic each task needs at least 60/66. All expected control and candidate
cells must complete and verify for a complete matrix, while scientific failures
and negative contrasts remain visible. Require all six v2 checks, including
complete optimizer replay, for every full-replay claim. Retain actual visits,
scored targets/input tokens, unique training diagnostics, majority comparisons,
singular/unique solve counts and drift diagnostics.

Recipe development and selection are training-only. Existing reused test
diagnostics remain exploratory and cannot select the budget, ratio, checkpoint
or winning seed. Prepare a genuinely fresh family-level confirmation protocol
before later capability comparisons, with independent exact labels, exposure
history and declared singular/unique-system and rational-solving coverage.
Resplitting public examples is not a fresh benchmark.

Original fold-sum and composed-answer adequacy need their own frozen experiments.
Finite maths success cannot silently replace those tasks. Phase 5 still needs
original learning adequacy and a separately reviewed geometry objective,
quality/cost margins, matched geometry-off/on arms and controls. Format
selection, scaling, independent replication, CPU export and release remain
separate gates. Replace proposal language with actual verified outcomes after
execution, including negative results; no candidate success is promised.

The budget comparison is now measured: all four cells completed and passed all
six replay/custody checks without pairing or worker errors on the newly declared
Xeon/AVX512 host. Arithmetic passes all training tasks. Maths aggregate counts
increase from 975 to 1,206 and 1,074 to 1,269 out of 1,998, but every maths task
fails its 90% gate in both candidate seeds. No budget is selected; joint learning
acceptance remains false. The complete source-bound packet and exact producer
are retained in [MATHS-RETENTION-BUDGET.md](MATHS-RETENTION-BUDGET.md). A new separately
frozen training-only decision is needed before further learning outcomes. Fresh
confirmation, original fold/composed adequacy, format selection, geometry,
scaling and release remain separate blocked gates. The originally reported full
Ryzen audit still requires its own producer/artifacts; these new records do not
replace it.

## Task-balanced maths scheduling increment

After merging PR #23, Trent selected further learning development on 2026-10-12.
The next separately frozen [training-only increment](MATHS-TASK-BALANCE.md)
compares ranked maths-row cycling with equal task visits at 8,192 updates in
both arms, seeds 0 and 1. Four arithmetic/four maths slots per batch, native
initialization, model, optimizer, diagnostic view and final-checkpoint rule stay
fixed. Task frequency and ordering change together; token exposure is measured,
not matched. No held-out outcomes inform this recipe or select a winner.

The implementation retains all six assurance checks and lossless evidence,
with the same every-task/every-candidate-seed 90% joint-training gate. Research
outcomes must follow publication of the frozen producer. This increment does
not clear original fold/composed adequacy, fresh confirmation, Phase 5, format,
scaling or release gates. Previous negative results and producers stay intact.

## Exact recursive 3×3×3 geometry scaffold

Requested by Trent on 2026-10-12. The separately frozen [integer-only lattice scaffold](LATTICE-SCAFFOLD.md)
implements a bounded Menger-style 20/27 retention rule at depths 0–3,
parent/child occupancy identities, face adjacency and read-only source-bound
verification. **This is an analytic address/graph reference, not a neural
mapping or enabled geometry loss.**

The mask, recursion depth, integer coordinate/index order, parent groups,
face adjacency and resource ceiling are now fixed in lattice protocol v1;
the scaffold is deterministically reconstructible without training or Torch.
Other masks or control geometries require separate versioned specifications.

The next separately frozen [activation-to-lattice observation](ACTIVATION-LATTICE.md)
uses the exact previously retained *trained native* reference capture packet,
fixed signed 64→3 projection, calibration restricted to the first source family,
and second-family measurement. It compares 20-anchor/edge-matched Menger,
nonrecursive, shuffled-topology and random-anchor conditions. This
**retrospective, training-exposed** experiment is exploratory rather than
prospective or independent, and does not train a neural mapping. Equal/negative
results and every control remain visible. A visually orderly mapping does not
establish reasoning or a mechanism; Phase 5 geometry loss stays zero until
fresh learning adequacy and a separately reviewed intervention protocol.

## Future RES=RAG / CSNP measurement integration

Requested by Trent on 2026-10-12. Evaluate
[RES=RAG / CSNP v1.1.0](https://github.com/QSOLKCB/res-rag/releases/tag/v1.1.0),
pinned to commit `053bf73d7493ec745f31806a99e522e9818c0f27`, as a future
operational measurement and receipt reference. Its RES/RAG names denote
receptive/context-grounded and generative dynamics in that specification.
Status: planned research adapter; no new training mechanism or calibrated
control is enabled.

A Trinite study must define its own observable proxies for the two states,
their common representation space, ground metric, Wasserstein estimator,
sampling windows and local calibration before measuring discrepancy. Bind
layer/token selections, any projection or lattice-anchor mapping, input/output
identities and temporal alignment in a versioned metric profile. Compare the
recursive scaffold with the matched alternative mappings rather than assuming
that the donor theory explains model behavior.

Retain signed temporal displacement `T_RAG - T_RES` separately from its
absolute magnitude, with explicit clock/units and missing-data rules. Keep
provenance-labelled research claims separate from operational thresholds;
values such as 0.42, 0.66 and approximately 2.14 need explicit reproducible local
calibration before any control use. Freeze any positive rate-pressure
denominator floor and its reciprocal-time units.

A future adapter should test the pinned RFC 8785/JCS SHA-256 receipt semantics,
version compatibility and actual verifier against declared positive/negative
cases, while retaining Trinite's existing PROVENANCE custody separately.
The donor verifier constrains a producer-labelled `governable` receipt one way;
it does not derive the entire classification label space. The four-stage
semantic cycle is an operational state model, with no consciousness inference.
An eventual `multitask_brake` experiment needs predeclared controls, timing,
estimators, observation windows, quality/cost measures and falsifiable success
criteria; its stabilizing effect is a hypothesis.

Honor the donor's CC BY 4.0 attribution and contribution records for any adapted
material. Training intake still requires Trinite's file-level rights, source and
exposure admission. Measurement, intervention, numerical calibration, security,
learning adequacy and release claims remain separate reviewed gates.


## E8/ETQ reference map and future neural observation

Requested by Trent on 2026-10-12. The [exact reference calculator](E8-REFERENCE.md)
implements the 240 doubled-coordinate roots and the pinned E8 laboratory's
embedded D4 × D4 order-three action, preserving all roots/inner products. It
declares a separate 3D receiver and +pi/2 z display rotation. The `[1,−2,1]`
stencil reuses the existing second-difference metric in a synthetic example.
The ETQ-compatible extension preserves SONIFICATION's canonical 101-site
selector and 303-state independent-fibre map, exact phase cycles, graph lift,
symbolic note labels and all seven published finite-payload commitments.
Actual donor-runtime event/graph/registry comparison agrees exactly. Status:
analytic software reference/map implemented; no neural observation or learning
benefit measured. Existing geometry and learning gates stay unchanged.

A neural study needs a separately frozen source-bound capture/projection
protocol with training-only fitting, units/alignment and matched controls before
target outcomes. Native second differences and an explicit E8 anchor receiver
are candidate diagnostics, with shuffled-order/orbit/anchor and matched random
receivers as declared controls. Exact symmetry or a pleasing projection does
not establish a learned representation. Phase 5 and its original learning
prerequisites still govern any geometry-on training experiment.

## Post-v0.1.0 CNR-001: measured connectome and ETQ-303 topology

The separately frozen [CNR-001 experiment](CNR-001.md) is a retrospective
measurement increment **after** immutable v0.1.0, never a replacement for
its failed and pending learning experiments. It compares the identical retained
trained-native 16-probe captures and original baseline scorer under eight
20-receiver/24-edge graph conditions: Menger v0.1.0, biological MaleCNS,
ETQ-101, ETQ-303, degree-preserving controls, and a seeded random graph.
The original MaleCNS 100-neuron, 3,630-connection biased subset is copied
under CC BY 4.0, source blobs pinned, and the authentic ingestion tested in
its isolated hash-locked reader workflow. Full ETQ-303 303-state event/phase
closure is verified separately from a true 303-step lifted-graph walk.

The corrected [CNR-001 v2 run 38290658293](https://github.com/QSOLKCB/Trinite/actions/runs/38290658293)
passed full genuine-input execution and read-only replay. A further
separately frozen [v3 run 38291828194](https://github.com/QSOLKCB/Trinite/actions/runs/38291828194)
passed complete pinned-reader module/import-byte attestation, original
decoder/source pins, new versioned request and analysis schema and full
read-only replay. V3 reproduced v2's negative/saturated graph-hop scores. The earlier
[historical v1 run 38288002645](https://github.com/QSOLKCB/Trinite/actions/runs/38288002645)
is retained separately: Codex exposed spurious no-op degree rewires in its
control producer. Both exact requests and analyses are now permanently
stored in the [lossless v1/v2 evidence packet](../fixtures/cnr-001-evidence/cnr001-v1-v2.tar.gz)
with explicit immutable byte commitments, alongside the separate
[v3 full evidence packet](../fixtures/cnr-001-evidence/cnr001-v3.tar.gz)
(SHA-256 fcec1f268a3985aeb921899d2cfcd8886ec91d53598faa6184761eb800a5ef5a).
The original v0.1.0 source and all three research records stay unchanged.
The corrected v2 protocol pins original v0.1.0 scorer/E8 basis Git blobs and
the transitive source closure. **No biological or ETQ arm beats the baseline:**
all block.0 hop sums remain zero with only two occupied anchors; final
Menger/ETQ-101/ETQ-303/random tie at 2, MaleCNS/revised rewired MaleCNS/
rewired ETQ-303 score 4. **The ETQ-101 rewired control is ineligible**:
its graph permits no effective rewiring under the frozen procedure.
Five of eight final vectors collide at receivers. ETQ's exact 303 event
steps are all nonedges of the Cartesian lift; the distinct genuine graph
walk visits 179 states. These are strongly saturation-limited **negative
descriptive findings**, not advantages for any structural hypothesis.

This topology observation uses arbitrary receiver slot placement, not
neurobiological homology, new learning, held-out generalization or a causal
geometry mechanism. Do not unlock Phase 5, format selection, device
benchmarks, scaling or release readiness. The historical v0.1.0 producer
and its earlier negative records remain unchanged.


CNR-001 v4 corrects installed-package baseline pin resolution and rejects
injected samples from official analyses. The real wheel boundary, authentic
v4 run/replay and historical v3 replay passed in
[run 38295236828](https://github.com/QSOLKCB/Trinite/actions/runs/38295236828).
Its [separate permanent packet](../fixtures/cnr-001-evidence/cnr001-v4.tar.gz)
retains the complete new request, analysis and receipts; original v1/v2/v3
packets remain unchanged. V4 reproduces the same negative/saturated scores,
with no model, graph-algorithm or research-gate advancement. See
[CNR-001](CNR-001.md#v4-package-and-authentic-input-boundary-correction).
