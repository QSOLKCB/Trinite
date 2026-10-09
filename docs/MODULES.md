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
