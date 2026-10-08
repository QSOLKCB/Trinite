# Phase 1 foundation

Status: implemented and locally verified; remote CI/review/merge are PR gates. No model has been constructed or trained.

## Implemented interfaces

| Module | Contract |
| --- | --- |
| contracts.py | Frozen validated model config, strict JSON parsing, versioned exact-byte identities |
| tokenizer.py | 259-token UTF-8 byte mapping; BOS/EOS/PAD; byte spans; answer/padding masks; explicit decode policy |
| data.py | Narrow symbolic-source admission, cyclic-counter generation, independent modulo verification, prompt parse, family rank, full replay audit |
| cli.py / __main__.py | validate-config, tokenize, generate-fixture, audit-fixture; nonzero failure status |

Model config values reject unknown/missing fields, booleans masquerading as integers, incompatible head dimensions, unsupported backends/semantics, and unbounded sizes. The reference shape calculation is 1,247,232 parameters; it is a planned inventory calculation, not a tensor count from an implemented model.

The machine-readable contracts are the executable validators and shipped reference config/dataset/manifest. Artifact JSON v1 uses sorted keys, compact UTF-8, exactly one final LF, and portable integers; duplicates, floats/nonfinite values, BOMs, and invalid Unicode are rejected. This is **Trinite's artifact format**, not a PROVENANCE integration or serializer. Float-valued future artifacts need their own explicit format.

## Validation

The suite covers module dependency limits, configuration boundaries, Unicode/spans and context limits, answer/PAD masks, exact raw-byte decoding, formal-answer validation across the small arithmetic domain, frozen fixture replay, split isolation, complete admission evidence and substitution rejection, changed-label/split/mask/source rejection, bounded artifact reads/membership/symlinks, and CLI failure behavior. LF checkout semantics are pinned for every byte-bound source/fixture extension and exercised with core.autocrlf=true for fresh and upgrade checkouts. The upgrade test binds the original contracts/tokenizer source hashes, verifies that the revision refreshes all byte-bound source files, and audits the retained target fixture. Stale CRLF source files fail with a migration diagnostic; the recovery procedure is in [GETTING_STARTED](GETTING_STARTED.md). Standalone admission checks also reject contradictory policy fields, substituted procedures/review evidence, and malformed or unpinned review dates.

All 30 checks for the corrected source pass locally with Python 3.12.14. The initial Phase 1 wheel was built with setuptools 84.0.0 without dependency downloads, installed into a clean virtual environment, and its CLI validated/audited/generated the frozen fixture. The runtime/test dependency set is empty. CI additionally exercises Python 3.11 and 3.13; remote matrix results are reported on the PR rather than inferred from local execution.

The fixture contains 24 unique formal problems, two labelled carriers each, eight complete modulus families, and 36 train / 6 validation / 6 test examples. Code receipts bind the generator/validator, JSON/config contract, and tokenizer. Admission, curation counts, item identities, token masks, and split assignments are recomputed during audit even if a producer refreshes its claimed dataset hash.

## Requirement changes and remaining gates

This increment implements foundation checks supporting TRI-I01, TRI-I03, TRI-I07, TRI-I08, TRI-I11, TRI-I12, and TRI-I14. It does not close model inspection/inference, observer isolation, or empirical research requirements. MODULES explicitly adds tokenizer as a data dependency to bind token encodings to each retained example. DATA clarifies that carrier variants are intentional grouped views, not separately deduplicated formal problems.

New Phase 1 semantics select exact GELU for the future reference model, constrain supported model-config dimensions/context, freeze loss-mask target indexing and UTF-8 byte offsets, define the 6/1/1 family rank, and define source-byte replay identity. These require fixture/contract review if changed.

Phase 2 builds and validates the actual dense/ternary model and quantizer. Phase 3 trains tiny runs and integrates the separately pinned PROVENANCE contract. The visible software fixture is not a fresh benchmark; no claim about reasoning, geometry, edge inference speed, or small-model superiority follows from these checks.
