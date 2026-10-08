# PROVENANCE observation

Status: **Phase 3 adapter implemented against frozen upstream source; local conformance passes.** The target is [QSOLKCB/PROVENANCE at 41451bf690597a75f461299dfcbbee559c83e924](https://github.com/QSOLKCB/PROVENANCE/tree/41451bf690597a75f461299dfcbbee559c83e924). This full commit identity, rather than a moving branch or repository homepage, defines the first Phase 3 integration target.

## Pinned upstream contract

The following files were inspected at that revision. The executable record validators are the schema authority for this integration; this plan does not assume a separate JSON Schema file.

| Contract | Pinned source / conformance reference |
| --- | --- |
| Canonical JSON | [provenance_core/canonical.py](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/provenance_core/canonical.py): provenance.canonical-json.v1 |
| Content and domain identities | [provenance_core/identity.py](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/provenance_core/identity.py): raw SHA-256 and NUL-terminated semantic domains |
| Record schemas, event classes, and derivation | [provenance_core/model.py](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/provenance_core/model.py): artifact, event, manifest, and custody v1 records |
| Physical evidence bundle | [docs/BUNDLE.md](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/docs/BUNDLE.md): provenance.bundle.v1 layout, retention, references, and membership |
| Verifier API and report | [provenance_verify/verifier.py](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/provenance_verify/verifier.py) and [public exports](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/provenance_verify/__init__.py): verify_bundle(Path) → VerificationReport |
| Canonicalization, identities, and record fixtures | [tests/test_core.py](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/tests/test_core.py): canonical bytes, fixed v1 identities, DERIVED source requirements, envelope substitution rejection |
| Bundle conformance fixtures | [tests/test_verify.py](https://github.com/QSOLKCB/PROVENANCE/blob/41451bf690597a75f461299dfcbbee559c83e924/tests/test_verify.py): valid retained/digest-only bundles, open missing evidence, tampering, missing/extra files, and unresolved references |

Phase 3 must acquire this exact source revision, verify the checkout identity, and retain its source/dependency receipt. Do not silently substitute main, a newer tag, or an equivalent-looking serializer. A pin update requires review of the affected contract files and fixtures before the adapter is retested. If the revision cannot be obtained or its contract/conformance checks cannot run on the selected environment, the observer integration gate is **blocked**; native training work may proceed as explicitly unintegrated work, but Phase 3 cannot be marked complete and no PROVENANCE-backed release claim is allowed.

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

Names here are Trinite evidence requirements, not additional fields in upstream event or manifest cores. Map them through the pinned contract as follows:

- Store run metadata, configurations, logs, checkpoints, captures, and results as exact-byte artifacts using ArtifactRecord with explicit retention. Bind their identities in ManifestCore/ManifestEnvelope; extra Trinite metadata belongs in referenced artifacts, not invented upstream core fields.
- Record stages with EventCore/EventEnvelope: explicit actor, operation, input/output artifact identities, relationships, and collection status. Use OBSERVED, DECLARED, or DERIVED according to collection semantics. Trinite's scientific evidence classes in RESEARCH are separate metadata and must not be substituted into the upstream event enum.
- A DERIVED event must identify at least one input or derived_from relationship. All input/output and relationship targets must resolve under the pinned bundle contract. Preserve the envelope self-hash exclusions and upstream identity functions.
- Upstream canonical JSON rejects floating-point values. Keep measured floating-point arrays/results as hash-bound artifact bytes under their own declared format; any floating-point configuration stored in a canonical metadata artifact needs an explicit lossless string/encoding rule. Do not round or convert scientific values merely to make an upstream record serialize.
- Verify an immutable/frozen bundle with verify_bundle(Path). Record the complete report, including integrity_verified, known_missing_artifacts, checks, and errors. An open manifest can pass integrity while declaring missing evidence; such a report does not satisfy Trinite's retained-input/full-open gate. Custody/disclosure checks require their separate upstream contracts and do not follow automatically from bundle verification.

Before the Phase 3 observer milestone closes, run the pinned core and bundle conformance suites in their supported environment and compare a Trinite-generated fixture against the pinned canonical bytes/identity fixtures and verifier. Include rejection cases for tampering, missing retained bytes, unresolved derivations/references, and unsupported schema/pin changes, plus the observer-isolation checks below. Phase 3 implements these checks in tests/test_observation.py, tests/training/test_training.py and the unchanged upstream suites. Remote CI and review remain required before merge. No alternate serializer is used.

Derived results reference the exact inputs and procedures that produced them. Content hashes bind bytes; locations and timestamps do not replace hashes. Model inventory distinguishes latent training weights, forward codes/scales, and deployed export.

## Verification and disclosure

Verify from retained artifact bytes, not only self-reported manifests. Report integrity/custody separately from scientific validity and measured model correctness. Missing files, failed checks, unsupported versions, and unverified claims cannot be marked verified. A valid custody chain does not prove that an experiment's conclusion is true.

Default datasets are public admissible formal fixtures. If a run includes private system/user information, publish only a reviewed disclosure with redaction lineage and a stated verification limit. Do not silently remove evidence or claim the disclosed subset reconstructs withheld inputs. Full-open native releases still require DATA's available-input gate.

Observer-on/off fixture checks compare outputs, gradients/updates for tiny training runs, RNG states, example order, and exported identities. Runtime/event IDs and timestamps may differ; compute artifacts must preserve the frozen reference scope. Hash verification, error handling, and observer isolation are required before using PROVENANCE branding as an implemented integration claim.

## Implemented source acquisition and adapter

The GitHub API acquisition verified the fixed commit/tree and each selected source path against its Git blob SHA-1 and raw SHA-256. [provenance-pin.json](../src/trinite/provenance-pin.json) freezes commit 41451bf690597a75f461299dfcbbee559c83e924, tree 8e1764e5b805405b89d484becefdfb7eb84a59bb, and all 24 acquired files. The 20 runtime files retain their original provenance_core/verify/trust/privacy/transfer package names and exact upstream bytes. The dependency closure is standard-library only; it includes helpers imported by the verifier's public exports. No trust/privacy/transfer operation is invoked by this adapter. The unchanged core/bundle tests live under tests/upstream, and [the frozen bundle contract](upstream/PROVENANCE-BUNDLE.md) records their upstream context.

The upstream MPL-2.0 license is retained in [PROVENANCE-LICENSE.txt](../src/trinite/PROVENANCE-LICENSE.txt), included in built wheels alongside the source pin and dependency lock. All original source notices are preserved. This is an engineering dependency, not admitted training data. Pin changes require acquisition and contract review; startup rejects a changed receipt/runtime file or a namespace resolved to substitute installed source. Repository CI checks all 24 files; an installed runtime checks its 21 shipped source/license files.

BundleObserver delegates canonical serialization, artifact/event/manifest construction, identity computation and verification to that frozen implementation. Trinite payloads remain referenced byte artifacts. initialize/resume and train-and-validate are DERIVED events with explicit input/output content identities; a failed computation records training-failed. Inputs include exact admitted dataset/manifest, frozen plan, execution context and parent checkpoint bytes when resuming. Outputs include initialization, complete step history, result, final named parameter/code inventory and both checkpoint files. The context contains exact installed-source/dependency receipts, environment, local clock assurance and declared revision/dirty-state assurance. Scientific scalar values are float.hex strings; optimizer decimal settings are strings.

The collector permits at most 32 events, 64 unique artifacts, 32 MiB per artifact and 64 MiB retained content. Artifact-name lookup and source-pin context are themselves retained. finalize writes a closed manifest and prohibits further collector mutation. The full verifier report lives beside the frozen bundle, preserving its exact physical membership. This is a stable-filesystem contract, not protection against a malicious concurrent writer. Verification requires closed scope, zero known missing artifacts, no unresolved errors and integrity_verified=true. Custody/signatures and scientific truth are separate future claims.

Observer evidence eligibility means that this procedure's required observations verified without collection or computation errors. It is narrower than release eligibility: release_ready remains false until the later release inventory, fresh evaluation and archival gates close. Unsupported/missing upstream source blocks mandatory observation; --observer off records explicitly unintegrated computation and cannot close this gate. Tests compare actual upstream canonical bytes/domain identities, reject tampering/missing/unresolved/unsupported records, and compare on/off/failing observers for gradients, updates, RNG/order, result/inventory/log and checkpoint bytes.
