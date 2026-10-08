# Modules

Status: Phase 1 implements flat contracts.py, tokenizer.py, data.py, and cli.py within src/trinite/. The remaining namespaces below are planned. Separate distributions, plugin registries, services, and native extensions must earn their complexity through a measured need.

| Planned path | Owns | Interface / artifact | Permitted internal dependencies |
| --- | --- | --- | --- |
| src/trinite/contracts/ | Configuration, schema versions, identity and validation types | Validated configs and content identities | None |
| src/trinite/data/ | Admission, generators, curation, deduplication, family splits | Examples and split manifests | contracts, tokenizer (bound example encodings) |
| src/trinite/tokenizer/ | Byte/special-token mapping and span alignment | Token IDs, masks, source spans | contracts |
| src/trinite/model/ | Decoder blocks, dense/ternary layers, quantizer | Explicit forward outputs and named tensors | contracts |
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

Core model/tokenizer modules cannot import observation, geometry analysis, evaluation, or CLI. Inference cannot import training or require Unsloth. Data cannot query a model or observer to select its examples. Geometry consumes detached capture artifacts and cannot mutate the model. The first geometry training intervention is orchestrated explicitly by training and may add a reviewed dependency on geometry after its differentiable contract is specified.

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
