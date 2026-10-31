# Modules

Status: Phases 1–4 implement flat contracts.py, tokenizer.py, data.py, cli.py, model.py, quantizer.py, inspection.py, training.py, checkpoint.py, observation.py, experiment.py, capture.py, geometry.py, and geometry_run.py within src/trinite/. The table describes responsibility boundaries; remaining namespaces are planned. Separate distributions, plugin registries, services, and native extensions must earn their complexity through a measured need.

| Planned path | Owns | Interface / artifact | Permitted internal dependencies |
| --- | --- | --- | --- |
| src/trinite/contracts/ | Configuration, schema versions, identity and validation types | Validated configs and content identities | None |
| src/trinite/data/ | Admission, generators, curation, deduplication, family splits | Examples and split manifests | contracts, tokenizer (bound example encodings) |
| src/trinite/tokenizer/ | Byte/special-token mapping and span alignment | Token IDs, masks, source spans | contracts |
| src/trinite/model/ | Decoder blocks, dense/ternary layers, quantizer | Explicit forward outputs and named tensors | contracts; quantizer.py implements the model quantizer boundary |
| src/trinite/training/ | Loss, optimizer, batches, checkpoints, resume | Training result and checkpoint receipts | contracts, data, tokenizer, model |
| src/trinite/inspection/ | Bounded, detached hidden-state and tensor capture | Capture arrays and manifests | contracts, tokenizer, model |
| src/trinite/geometry/ | Geometry metrics and optional pinned reference adapter | Native-space metrics and protocol receipts | contracts; inspection artifact types only |
| src/trinite/evaluation/ | Scorers, baselines, ablations, uncertainty | Complete result records | contracts, data, tokenizer, model, inspection, geometry |
| src/trinite/artifacts/ | Checkpoint inventory, export, integrity checks | Tensor inventory and export bundle | contracts, model |
| src/trinite/inference/ | Offline CPU execution and generation | Outputs and resource measurements | contracts, tokenizer, model, artifacts |
| src/trinite/observation/ | PROVENANCE adapter and run evidence capture | Evidence/custody records | contracts; external PROVENANCE adapter only |
| src/trinite/cli/ | Composition and explicit backend/observer selection | CLI invocations and exit status | All modules as orchestration entrypoints |

These names are design boundaries, not an instruction to create a directory per tiny helper. Start with flat files under the named namespaces where practical; split when the interface has enough substance. Shared types belong in contracts, not a miscellaneous utils module.

## Dependency rules

`packed_ternary/` owns only bounded in-memory ternary code/byte conversion,
as [CPU-export preparation](INFERENCE.md#implemented-code-payload-preparation).
It imports only shared contract validation helpers and has no model, training,
checkpoint, filesystem or backend dependency. Its package boundary preserves
the historical flat maths/retention source inventory. A future artifact layer may use
it after separately validating complete inventory, shape, scale and identity
metadata; it is not a second safe tensor reader or an inference backend.

The detour adds qec_source.py for the pinned offline oracle, comparison_data.py for numeric admission, comparison_training.py for frozen settings/source/corpus/state construction, comparison_checkpoint.py for snapshot/replay, and comparison.py for evaluation/geometry/evidence orchestration. Both native and comparison states use training.py's authoritative optimiser/update loop; its schedule uses the admitted training length and enforces actual scored-target budgets. scripts/run_comparison.py owns fresh process orchestration. No circular imports or model-to-data/observer dependencies are added. The unmodified trinite._qec dependency retains original relative imports, source pins and separate license.

Core model/tokenizer modules cannot import observation, geometry analysis, evaluation, or CLI. Inference cannot import training or require Unsloth. Data cannot query a model or observer to select its examples. Geometry consumes detached capture artifacts and cannot mutate the model. The first geometry training intervention is orchestrated explicitly by training and may add a reviewed dependency on geometry after its differentiable contract is specified.

The model returns logits and only explicitly selected detached hidden states. CaptureSpec lives in model.py so the model does not import inspection.py. Inspection imports model/quantizer and cannot choose data or mutate model tensors. CLI imports the PyTorch/model path lazily; foundation imports remain standard-library only.

Observation is injected by orchestration through a narrow record(event, artifact identities) interface. Internal modules emit structured facts or callbacks; they do not reach into global observer state. The adapter must not use a model RNG or execute model code. No circular imports or hidden downloads.

## Planned layout outside source

| Path | Purpose |
| --- | --- |
| docs/ | All narrative contracts, plans, instructions, and later reports |
| docs/assets/ | Documentation artwork |
| configs/ | Small versioned model and run configurations |
| schemas/ | Implemented machine-readable artifact contracts |
| tests/ | Boundary, numerical, replay, and integration checks |
| fixtures/ | Tiny rights-admissible conformance artifacts |
| scripts/ | Thin workflow launchers; no second implementation of core logic |
| .github/workflows/ | Later validation and explicit training orchestration |

Large datasets, captures, checkpoints, and exports belong outside the source tree in explicitly selected artifact locations. Each retained artifact requires an identity and retrieval record; location alone is not identity.

## Implemented Phase 3 composition

training.py owns the pure numerical plan, admitted batches, target loss, AdamW and update loop. checkpoint.py composes that state with bounded safetensors/JSON serialization and validation; it imports training, never observation. experiment.py owns explicit filesystem operations and stage-level observation, including recovery/reporting. observation.py consumes detached byte artifacts and imports the frozen upstream record/verifier packages after checking their source pin; it cannot import Torch, NumPy, RNG modules or model code. CLI selection remains lazy, preserving the standard-library foundation. AST tests enforce these directions.

The unmodified upstream runtime dependency closure lives in src/provenance_core, src/provenance_verify, src/provenance_trust, src/provenance_privacy and src/provenance_transfer. These top-level names preserve upstream imports; they are a frozen dependency, not parallel Trinite implementations. Source identities, acquisition paths and license are listed in [PROVENANCE.md](PROVENANCE.md).

## Implemented Phase 4 composition

capture.py consumes the existing explicit model capture interface and tokenizer, returning detached, losslessly encoded prompt states with a strict reader. geometry.py consumes bounded coordinate lists and the unmodified frozen _geo_reference.py numerical kernel; it imports no Torch, NumPy, training or observation code. geometry_run.py is the explicit orchestration boundary: it composes capture, geometry, elected checkpoint loading and injected observation, preserving model/optimizer/progress and caller RNG. The model and numerical training loop acquire no geometry dependencies. Source pin, license and protocol are packaged artifacts; original upstream tests and contracts are retained separately. AST tests enforce these directions.

## Implemented foundations composition

`foundations_data.py` and `foundations_oracle.py` own newly generated formal
items, independent labels/parser checks and source-specific admission.
`foundations_plan.py` defines a separate bounded plan without increasing native
`RunConfig` limits. `foundations_training.py` composes fixed inputs and model
states using the authoritative native optimizer/update loop.
`foundations_checkpoint.py` shares the comparison checkpoint tensor serializer
and progress checks, adding foundations data/plan/workload binding.
`foundations.py` owns prompt-only scoring, training-only baselines, verified
learning decisions, explicit evidence retention and transactional aggregation.
`scripts/run_foundations.py` launches fixed fresh workers; it cannot accept an
arbitrary executable or shell command. CLI composition stays lazy. Boundary
tests cover every new dependency; data/oracles cannot query models or observers.
The complete historical comparison archive is retained in its original fixture
location; any foundations archive is a separately identified review exception
to the small-fixture convention, not an invitation to store evolving runs in git.

Review-only `restore_foundations_archive.py` performs bounded standard-library
lossless byte reconstruction; `verify_foundations_run.py` calls the existing
authoritative summarizer on a disposable copy and compares the retained summary.
They do not choose examples, rescore outputs or alter the scientific protocol.
The public archive stores duplicate artifacts once to reduce repository and CI
checkout transfer without changing original evidence bytes.

## Implemented convergence development

`convergence.py` composes foundations admission/scoring, the authoritative native
numerical loop, shared safe checkpoints and closed observation. Its elected
training-only data view uses train examples for every numerical diagnostic.
`scripts/run_convergence.py` owns freeze, fixed fresh workers and read-only review;
there is no held-out stage. That convergence increment preserves the preceding numerical modules and protocols. The later learning increment factors the shared scorer entrypoints without changing their computations.


## Implemented budget development and game corpus

`budget_training.py` owns frozen settings, requests, source receipts and train-only state.
`budget_plan.py` owns the explicit larger development resource bounds;
`budget_checkpoint.py` owns its source/request/train-only checkpoint profile;
`budget.py` composes admitted foundations data, native updates, milestone scoring
and closed observation. The existing comparison checkpoint module owns the
shared closed tensor reader, used by comparison, foundations and budget profiles.

No numerical model/update implementation is duplicated.

`game_data.py` and `game_oracle.py` are standard-library numeric admission/oracle
modules. Neither can import a model, observer or network code. The game corpus
is prepared for a later frozen training increment and is absent from budget
requests/inputs. Boundary tests declare these dependencies explicitly.

## Implemented matched learning/evaluation

`learning_data.py` adds a versioned complete per-item admission over unchanged
foundations facts. `learning_plan.py` delegates numerical/resource bounds to the
existing budget profile. `learning_training.py` owns the frozen protocol, request,
actual runner receipt and parent-order-preserving state. `learning_checkpoint.py`
uses the shared safe named-tensor reader with its own admitted train-only context.
`learning.py` composes native updates, the shared foundations scorer, fresh matrix
authorization and prediction-regenerating verification. `scripts/run_learning.py`
owns fixed fresh-process deadlines and conservative failure retention. The
low-level model, optimizer and observer gain no dependency on these modules.

The shared scorer accepts an explicitly admitted example list and cap, while the
historical foundations entrypoints retain their original dataset and protocol.
The new admission metadata changes identities, not numeric examples or exposure
order. See [LEARNING.md](LEARNING.md).

## Implemented generalization diagnostics and scalar preparation

`curriculum_data.py` owns parent-bound scalar projections and admission;
`curriculum_oracle.py` independently checks labels and rendered prompts without
model/observer dependencies. `generalization.py` delegates fresh model/evidence
verification to the unchanged learning summarizer, then computes bounded
descriptive coverage/component counts and closes the new observer evidence.
`scripts/run_generalization.py` owns the fixed fresh-worker deadline, disjoint
output paths, exclusive publication and partial failure retention. It never
trains or unlocks geometry. `check_research_cpu.py` extends the historical native
runner with exact ordered research-suite discovery; standalone and existing
fresh-process probes remain available.

## Implemented scalar curriculum learning

`scalar_plan.py` delegates numerical/resource bounds to the budget profile.
`scalar_training.py` owns fixed parent/projection exposure, native state construction,
source/context requests and decision lineage over unchanged scalar admission.
`scalar_checkpoint.py` composes the existing safe named-tensor/progress reader.
`scalar_learning.py` owns fresh train/test evidence verification, matched authorization,
source-bound reports and conservative scalar-only aggregation. Its gate wrapper
validates the larger projection total and delegates unchanged per-task thresholds.
`scripts/run_scalar_learning.py` owns bounded fixed fresh workers and retention.
No numerical update loop or scorer is copied; original experiment source modules
remain unchanged. The grouped research CI appends the scalar suite in order.


## External reference geometry observation

`reference_inputs.py` binds visible scalar admission, the pinned external asset
manifest, original native producer and separate frozen request.
`reference_capture.py` is the sole optional Transformers/Torch instrumentation
boundary. It consumes locally verified inputs and explicit native capture APIs.
`reference_geometry.py` owns bounded standard-library detached geometry.
`reference_evidence.py` composes closed condition and derived PROVENANCE evidence.
`scripts/run_reference_geometry.py` owns fixed fresh workers and deadlines;
`scripts/acquire_reference_model.py` is explicit fixed-revision acquisition.
The independent Fraction oracle under `scripts/reference_math_oracle.py` is an
intentional verification implementation, not a second production kernel. No
numerical training loop, native checkpoint reader or admission operation is
duplicated. See [REFERENCE-GEOMETRY.md](REFERENCE-GEOMETRY.md).

## Fold-sum exposure development

`fold_exposure_plan.py` validates a separate fixed multiplicity profile using the
existing budget constraints. `fold_exposure.py` owns train-only composition,
request/source/admission binding, safe checkpoint context and fresh report
verification. It delegates scalar admission/order/scoring, the native numerical
loop, shared named-tensor reader and closed observer; no numerical kernel or
scorer is duplicated. `run_fold_exposure.py` owns fixed fresh workers and failure
retention. It has no evaluation stage. See [FOLD-EXPOSURE.md](FOLD-EXPOSURE.md).


## Fold-order development

`fold_order_plan.py` adds the separate fixed parent/mixed ordering profile within
BudgetPlan bounds. `fold_order.py` owns source-bound requests, schedule context,
replay and training-only evidence composition. It delegates existing admission,
scalar rows/scoring, numerical updates, tensor serialization/loading and closed
PROVENANCE verification; it does not add a numerical kernel or scoring rule.
`scripts/run_fold_order.py` owns fixed fresh workers and failure retention.
Historical fold-exposure modules and evidence remain unchanged. See
[FOLD-ORDER.md](FOLD-ORDER.md).


## Maths corpus and pilot

`maths_data.py` and `maths_oracle.py` own standard-library generation/admission and independent exact checking. `maths_plan.py` delegates bounds to the existing budget profile. `maths_learning.py` composes admitted train rows, the authoritative native updater, shared safe tensor/progress reader and foundations prompt-only scorer, with its own frozen protocol and closed observer evidence. `scripts/run_maths.py` launches fixed fresh workers. Historical numerical/evidence implementations remain unchanged.


## Maths retention development

`maths_retention.py` composes the existing admitted pooled state with two ranked training streams and a separately frozen rehearsal protocol. It owns source/request/schedule/checkpoint context, actual exposure diagnostics and fresh report assembly. Admission, initialization, numerical updates, safe tensor/progress handling, prompt-only scoring, drift diagnostics and closed PROVENANCE verification delegate to their existing authoritative operations. `scripts/run_maths_retention.py` owns fixed fresh workers and failure retention. The unique training diagnostic view stays separate from scheduled rehearsal visits. See [MATHS-RETENTION.md](MATHS-RETENTION.md).


## Exact E8 and ETQ reference

`e8_reference/` owns bounded exact root generation, the embedded D4 × D4
triality, complete root orbits and a declared display projection/quarter-turn.
`etq.py` in the same package owns the compatible 101-site selector, distinct
internal/fibre labels, exact Gaussian phases, 303-event map and Cartesian graph.
The package imports only contracts and the standard library. `run_e8_reference.py` prints
analytic reference examples or the complete compatible ETQ event JSON, and delegates the synthetic second difference to
the existing pinned `geometry.metrics`. No model capture, neural mapping,
training dependency or new geometry kernel is introduced. See
[E8-REFERENCE.md](E8-REFERENCE.md).


## Retention evidence v2

The isolated `retention_v2/` package preserves v1's flat source closure. `spec.py`
owns the frozen specification, explicit research/smoke profile, invocation-local
immutable admission, scoring views and adapter to the unchanged v1 schedule.
`criteria.py` owns strict exact task/count checks using only contracts and stdlib.
`evidence.py` shares bounded immutable byte binding and fresh closed name-index
verification; it imports contracts and observation, never model/training.
`experiment.py` composes requests, safe progress/tensors, prompt-only scoring,
full native update replay, scope-aware receipts and conservative aggregation.
Its numerical dependencies remain the authoritative existing operations.
`run_maths_retention_v2.py` owns fixed workers and complete failure inventories.
No numerical kernel, dynamic executable selection or success cache is added.
See [MATHS-RETENTION-V2.md](MATHS-RETENTION-V2.md).

## Balanced rehearsal budget development

`retention_v2/budget/plan.py` validates separately versioned larger resource bounds.
`spec.py` delegates frozen admission/native state construction and defines the
fixed four-plus-four schedule and train-only scoring views. `experiment.py`
composes the authoritative updater, scorer, safe tensor reader and shared v2
evidence/criteria/exposure operations under new schemas. `archive.py` preserves
complete verified runs, externally bound requests and exact producer source with
a lossless member inventory. The fixed `run_maths_retention_budget.py` and
`archive_maths_retention_budget.py` scripts own process/archival entrypoints.
Original flat v1 and v2 producers remain byte-identical. See
[MATHS-RETENTION-BUDGET.md](MATHS-RETENTION-BUDGET.md).

## Task-balanced maths scheduling

`retention_v2/task_balance/schedule.py` owns bounded integer scheduling over
admitted rows: an unchanged ranked-row control and a round-robin maths-task
candidate. `spec.py` composes the unchanged budget plan/native state, substitutes
only scheduled training visits, and binds the complete prior and new source
closures. `experiment.py` and `archive.py` isolate the new request/checkpoint/
report/evidence identities while reusing the authoritative updater, scorer,
safe tensor reader, admission, criteria and custody operations. The fixed
`run_maths_task_balance.py` and `archive_maths_task_balance.py` scripts own the
process and archival entrypoints. No existing producer is modified or rebound.
See [MATHS-TASK-BALANCE.md](MATHS-TASK-BALANCE.md).

## Recursive integer geometry reference

`lattice/scaffold.py` independently implements the frozen 3×3×3 Menger-style
address hierarchy, 20-child occupancy rule and exact six-neighbour leaf graph.
The standalone `generate_lattice_scaffold.py` exclusively creates a canonical
source-bound artifact or independently reconstructs it for read-only verification.
This pure-standard-library module owns no trained mapping, activation fitting,
loss, geometry intervention or existing producer. See
[LATTICE-SCAFFOLD.md](LATTICE-SCAFFOLD.md).

## Retrospective native activation / lattice research

`lattice_observation/experiment.py` is a separately versioned, standard-library
only read-only analysis of the retained native trained reference capture. It
uses the existing original capture auditor and exact v1 Menger scaffold, with
source-bound requests and an external result-identity verifier. A separate
`run_lattice_observation.py` performs immutable freeze, run and verification.
No native model, loss, admission, optimizer, prior reference producer or
historical packet is modified. See [ACTIVATION-LATTICE.md](ACTIVATION-LATTICE.md).


## Immutable final-forward export candidate

`forward_export/` is the additive encode-only boundary documented in
[FORWARD-EXPORT.md](FORWARD-EXPORT.md). Its standard-library `source.py` provides
an explicit bounded read of installed source bytes. `encoder.py` accepts that
immutable captured input plus the native ternary Decoder, validates canonical
structure/bounds, delegates the existing quantizer, tensor identity and packed
codec, and serializes float exceptions/scales with `safetensors.torch.save`.
The returned members are immutable bytes; no exporter file write, checkpoint
reader, training import, numerical updater, second tensor reader, CLI, generation
path or alternative backend is added. Only tests reconstruct trusted encoder
output with the authoritative safetensors reader and native dense Decoder.
Historical flat source closures remain unchanged. This is engineering
preparation, not format selection or completed Phase 6.

## Non-displacing auxiliary engineering composition

`procedural_qec_auxiliary/inputs.py` owns isolated actual historical/current
admission and full-row bijection; `scheduler.py` owns mapped historical ranking
and separate eight-row additional updates. `spec.py` describes a reviewable,
not-scientifically-frozen candidate and a bounded engineering-only profile.
`experiment.py` composes existing native training, scorer, safe tensor reader and
closed observer, with new source-bound checkpoint/full-replay schemas.
`archive.py` retains exact source and run bytes after fresh numerical replay;
transport verification remains standard-library only. Fixed launcher scripts
provide explicit checkout-only entrypoints. Research execution is disabled.
See [the implementation boundary](PROCEDURAL-QEC-AUXILIARY-IMPLEMENTATION.md).

## Opt-in declared-state observation

Checkout-only `scripts/declared_state/` owns only a strict five-label base-three codec and immutable
raw-text syntactic observations. It imports `contracts` and the standard library;
no core numerical module imports it. It does not read tensors, run inference,
repair responses, admit data or replace exact-answer scoring. Historical flat
source closures and the recursive CNR-001 inventory remain unchanged. It is
not an installed-wheel API. See [DECLARED-STATE.md](DECLARED-STATE.md).

## Checkout-only PCGrad engineering lane

`scripts/pcgrad_v1/` owns an opt-in two-objective projection kernel and fixed
synthetic two-parameter replay harness. It reuses the native environment and
AdamW constructor/validator, but never edits or replaces the native updater.
Its canonical JSON snapshots use full-prefix replay-to-resume, not a second
native tensor loader. It is excluded from the installed package and historical
CNR source closure. [Contract and limits](PCGRAD-ENGINEERING.md).

## Checkout-only learned ternary scale engineering lane

`scripts/learned_scales_v1/` owns the separately versioned matrix-scale surrogate,
wrapper over unchanged native decoder operations, and four-update synthetic
matched learned/frozen-scale replay harness. It reuses native initialization,
AdamW constructor/validation and environment checks; native numerical sources,
PCGrad and historical closures stay unchanged. New canonical JSON checkpoints
use full-prefix replay-to-resume, not another tensor loader. Scientific execution
is hard-closed. [Contract and evidence limits](LEARNED-TERNARY-SCALES.md).

## Checkout-only fixed-code refit training pilot

`scripts/scale_refit_training_v1/` wraps unchanged native decoder operations with
existing guarded fixed-code refitting and a matched native-absmean control.
It reuses the authoritative native admission, updater, loss and greedy decoder;
new canonical snapshots resume by exact bounded prefix replay, not another
safe tensor reader. Only the expressly frozen visible-fixture pilot is enabled;
historical research gates remain unchanged. [Contract](SCALE-REFIT-TRAINING.md).

## External NCM reference feasibility

Checkout-only `scripts/ncm_reference_v1/` owns an explicitly optional, pinned
upstream Sudoku adapter, formal smoke inputs, strict scoring and source-bound
transport/replay. Its stdlib contracts and tests never run an external model.
The separately authorized CUDA boundary is not imported by native Trinite or
its installed package. No training or historical source changes.
See [the external pilot contract](NCM-REFERENCE-PILOT.md).

## Coverage-balanced addition composition

`scripts/balanced_addition_v1/` owns the new finite corpus, frozen length-matched
schedule, plan/request and six-cell orchestration described in
[BALANCED-ADDITION](BALANCED-ADDITION.md). It delegates native updates, scoring,
safe checkpoint tensors and custody to their existing authoritative modules.
Its reporting adapter reuses the historical pure failure classifier on new
source-bound raw predictions; it does not pass new producers through the old
fixed-admission checkpoint importer. No core model or numerical updater changes.

## Checkout-only position interface conformance

`scripts/position_coupling_v1/` owns an explicit-position dense forward boundary,
input-only arithmetic slot mapping and bounded synthetic parity checks. It reuses
native Decoder initialization, validation, blocks, norms and tied output without
changing historical source bytes or adding a second optimizer/scorer. The fixed
`check_position_coupling.py` emits an engineering-only source-bound report, not a
research checkpoint or learning result. [Contract and prospective study](POSITION-COUPLING.md).

## Prospective position-coupling study

`scripts/position_study_v1/` composes the frozen admission, stateless live-prefix
position adapter, native numerical updater/loss/scorer and shared safe named
tensor reader. New versioned scoring binds input-derived caps and reversed
fixed-width syntax; prediction journals preserve partial scoring. The fixed
`run_position_study.py` owns bounded preflight/freeze/run/read-only audit/summary.
Structural evidence verification is separate from optimizer/prediction replay.
No historical runtime/fixture changes or target experiment is included. See
[position study v1](POSITION-STUDY-V1.md).
## Partial balanced-addition result verification

`scripts/balanced_result_v1/` authenticates the immutable 43-file public subset
and its separately pinned derivative source-binding receipt. It reuses original
admission, score checking and descriptive maps; it does not load checkpoints,
construct models, update parameters or claim closed custody. The checker uses
the pinned CPU dependencies because authoritative inspection imports Torch.
See [the outcome and verification limits](BALANCED-ADDITION-RESULT.md).

## Partial position-study result verification

`scripts/position_result_v1/` authenticates the selected original public-data
records and the separately pinned derivative source-binding receipt. It reuses
the frozen admission, score validator and descriptive reporting, without
constructing models, loading checkpoints, updating parameters or claiming closed
custody. `check_position_result.py` is a read-only checkout entrypoint requiring
the pinned CPU imports. See [the outcome and limits](POSITION-STUDY-RESULT.md).

## Prospective positional-exposure ablation

`scripts/position_exposure_v1/` and `scripts/run_position_exposure.py` compose the
unchanged original arithmetic admission, numerical updater, scorer and safe tensor
reader into a separate twelve-cell factorial workflow. Deterministic cycle offsets
are explicit training context; evaluation stays at zero. Static admission covers
address/coarse-role pairs and retains detailed significance gaps. Source-bound
requests, cursor/checkpoint identities, raw journals and partial-failure custody
remain explicit. See [the contract](POSITION-EXPOSURE-V1.md); engineering tests
are not target results or trained numerical replay.


## Positional-exposure result preservation

`scripts/position_exposure_result_v1/` and `scripts/check_position_exposure_result.py` verify the [negative twelve-cell public subset](POSITION-EXPOSURE-RESULT.md) read-only, reusing authoritative admission/scoring/exposure/reporting and prior public-result checks. They do not construct models, load checkpoints or run training/generation. Historical study and result modules remain unchanged.


## Memory traffic accounting v1

`scripts/memory_traffic_v1/` owns integer-only, source-bound shape accounting and
an opt-in metadata observer around the unchanged native CPU forward. It does
not import training, load checkpoints, implement cache attention or change the
historical CPU ledger. The standard-library CLI imports no numerical backend;
the separate native adapter requires one explicitly supplied eval/no-grad model.
See [memory accounting](MEMORY-TRAFFIC.md) for scope and counting conventions.
