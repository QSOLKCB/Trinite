# PROVENANCE observation

Status: **upstream integration contract pinned; Trinite adapter not implemented or tested.** The target is [QSOLKCB/PROVENANCE at 41451bf690597a75f461299dfcbbee559c83e924](https://github.com/QSOLKCB/PROVENANCE/tree/41451bf690597a75f461299dfcbbee559c83e924). This full commit identity, rather than a moving branch or repository homepage, defines the first Phase 3 integration target.

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

Names here are requirements for a future Trinite run manifest, not additional fields in upstream event or manifest cores. Map them through the pinned contract as follows:

- Store run metadata, configurations, logs, checkpoints, captures, and results as exact-byte artifacts using ArtifactRecord with explicit retention. Bind their identities in ManifestCore/ManifestEnvelope; extra Trinite metadata belongs in referenced artifacts, not invented upstream core fields.
- Record stages with EventCore/EventEnvelope: explicit actor, operation, input/output artifact identities, relationships, and collection status. Use OBSERVED, DECLARED, or DERIVED according to collection semantics. Trinite's scientific evidence classes in RESEARCH are separate metadata and must not be substituted into the upstream event enum.
- A DERIVED event must identify at least one input or derived_from relationship. All input/output and relationship targets must resolve under the pinned bundle contract. Preserve the envelope self-hash exclusions and upstream identity functions.
- Upstream canonical JSON rejects floating-point values. Keep measured floating-point arrays/results as hash-bound artifact bytes under their own declared format; any floating-point configuration stored in a canonical metadata artifact needs an explicit lossless string/encoding rule. Do not round or convert scientific values merely to make an upstream record serialize.
- Verify an immutable/frozen bundle with verify_bundle(Path). Record the complete report, including integrity_verified, known_missing_artifacts, checks, and errors. An open manifest can pass integrity while declaring missing evidence; such a report does not satisfy Trinite's retained-input/full-open gate. Custody/disclosure checks require their separate upstream contracts and do not follow automatically from bundle verification.

Before the Phase 3 observer milestone closes, run the pinned core and bundle conformance suites in their supported environment and compare a Trinite-generated fixture against the pinned canonical bytes/identity fixtures and verifier. Include rejection cases for tampering, missing retained bytes, unresolved derivations/references, and unsupported schema/pin changes, plus the observer-isolation checks below. These are required future adapter checks, not tests performed by this documentation PR. No alternate serializer or claimed compatibility is accepted without those checks.

Derived results reference the exact inputs and procedures that produced them. Content hashes bind bytes; locations and timestamps do not replace hashes. Model inventory distinguishes latent training weights, forward codes/scales, and deployed export.

## Verification and disclosure

Verify from retained artifact bytes, not only self-reported manifests. Report integrity/custody separately from scientific validity and measured model correctness. Missing files, failed checks, unsupported versions, and unverified claims cannot be marked verified. A valid custody chain does not prove that an experiment's conclusion is true.

Default datasets are public admissible formal fixtures. If a run includes private system/user information, publish only a reviewed disclosure with redaction lineage and a stated verification limit. Do not silently remove evidence or claim the disclosed subset reconstructs withheld inputs. Full-open native releases still require DATA's available-input gate.

Observer-on/off fixture checks compare outputs, gradients/updates for tiny training runs, RNG states, example order, and exported identities. Runtime/event IDs and timestamps may differ; compute artifacts must preserve the frozen reference scope. Hash verification, error handling, and observer isolation are required before using PROVENANCE branding as an implemented integration claim.
