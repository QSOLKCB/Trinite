# PROVENANCE observation

Status: adapter contract proposal. Inspect and pin the actual [PROVENANCE](https://github.com/QSOLKCB/PROVENANCE) interfaces before implementation; this document invents no upstream API or verifier result.

## Observer boundary

Observe data generation/admission, initialization, training, evaluation, capture, export, and inference as explicit run stages. Bind who/what/when/where/why/how records to input/output evidence through a pinned PROVENANCE adapter. The observer is not a neural layer and is never fed back into training labels, losses, or example selection.

Use detached records and bounded artifact references. No RNG draws from model generators, forward hooks that modify tensors, hidden downloads, or changes to optimizer steps. Record enabled/disabled mode and overhead. A failed mandatory observation invalidates the run's release-evidence eligibility, reports an explicit error, and retains any recoverable failure evidence; it does not silently certify an unrecorded run.

## Minimum evidence contents

| Object | Required identity/context |
| --- | --- |
| Plan | Protocol ID/version, hypothesis, primary metrics, controls, budgets, stop rules |
| Source | Repository commit, dirty-state report, dependency lock/environment identities |
| Inputs | Dataset source/admission/split hashes, generator identities, tokenizer/config identity |
| Model | Architecture/tensor inventory, initialization or checkpoint hashes, quantizer version |
| Execution | Run/stage IDs, actor/runner, timestamps, command/config, seeds, hardware, backend/dtype, determinism scope |
| Outputs | Logs, checkpoint/capture/export/result hashes, artifact locations, stage outcome |
| Verification | Verifier implementation/version, checked scope, results, missing artifacts, unresolved errors |

Names here are requirements for a future Trinite run manifest, not field names asserted to exist in upstream PROVENANCE. The adapter maps them to the reviewed upstream schema and fails on incompatibility. Respect upstream canonical JSON, identity, event-type, derivation, and verification rules; do not create a competing almost-compatible serializer.

Derived results reference the exact inputs and procedures that produced them. Content hashes bind bytes; locations and timestamps do not replace hashes. Model inventory distinguishes latent training weights, forward codes/scales, and deployed export.

## Verification and disclosure

Verify from retained artifact bytes, not only self-reported manifests. Report integrity/custody separately from scientific validity and measured model correctness. Missing files, failed checks, unsupported versions, and unverified claims cannot be marked verified. A valid custody chain does not prove that an experiment's conclusion is true.

Default datasets are public admissible formal fixtures. If a run includes private system/user information, publish only a reviewed disclosure with redaction lineage and a stated verification limit. Do not silently remove evidence or claim the disclosed subset reconstructs withheld inputs. Full-open native releases still require DATA's available-input gate.

Observer-on/off fixture checks compare outputs, gradients/updates for tiny training runs, RNG states, example order, and exported identities. Runtime/event IDs and timestamps may differ; compute artifacts must preserve the frozen reference scope. Hash verification, error handling, and observer isolation are required before using PROVENANCE branding as an implemented integration claim.
