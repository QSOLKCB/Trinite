# Invariants

These requirements apply to future implementations and releases. Their presence is not evidence that implementation checks already exist.

| ID | Requirement | Required evidence |
| --- | --- | --- |
| TRI-I01 | Every model operation, tensor class, tokenizer rule, and configuration is inspectable without executing downloaded remote code. | Source inventory, tensor inventory, and offline load/inspection checks |
| TRI-I02 | Native Trinite starts from random initialization; no undisclosed pretrained weights, teacher outputs, or distillation. | Initialization identity and complete input/checkpoint lineage |
| TRI-I03 | Training and evaluation inputs satisfy DATA's strict no-copyrighted-data admission policy. | Per-source rights basis, generation records, and split receipts |
| TRI-I04 | Ternary claims identify exactly which tensors use {-1, 0, +1} codes and which remain floating point. | Counts, scales, exceptions, and export parity |
| TRI-I05 | Baseline inference and inspection work on CPU without CUDA or an accelerator package. | Offline CPU smoke run and measured resource report |
| TRI-I06 | The provenance observer cannot change model outputs, optimization, RNG state, or selected examples. | Observer-on/off comparison and explicit failure behavior |
| TRI-I07 | Data, model, tokenizer, protocol, environment, and artifacts are bound to immutable content identities. | Hash-verified manifest and retained artifacts |
| TRI-I08 | Reproducibility names its hardware/software/numerical scope; a seed is not a universal guarantee. | Same-environment replay or documented failure/tolerance |
| TRI-I09 | Geometry is a measurement or tested intervention; it is never assumed to be the mechanism of reasoning. | RESEARCH controls, evidence class, and claim limits |
| TRI-I10 | No small-model superiority or universal hardware claim without scoped measurements. | Frozen comparators, budgets, exposure assessment, and device report |
| TRI-I11 | Evaluation families and labels remain isolated from training, tuning, and optimizer selection. | Family-level split audit, test lock, and full result retention |
| TRI-I12 | Modules own explicit interfaces and do not hide backend substitutions, data downloads, or mutable global state. | Boundary tests, explicit backend selection, and dependency review |
| TRI-I13 | Null, failed, inconclusive, and negative results remain in the evidence record. | Run outcome inventory and failure receipts |
| TRI-I14 | Contract changes name affected requirements, tests, fixtures, and compatibility. | Reviewed change record before affected results are compared |

Inspectability permits tensor and activation inspection. It does not prove semantic interpretability, absence of memorization, fairness, correctness, or a causal explanation of an answer.

## Prospective application orchestration invariants (TRI-ORCH-001)

**Proposal only, 9 November 2026.** These *additional* requirements apply to a future local chat application. They neither change original TRI-I01–TRI-I14 nor claim any implementation or passed validation. [Architecture and threat model](TRI-ORCH-001.md); [next prototype PR plan](TRI-CHAT-001.md).

| ID | Requirement | Required future evidence |
| --- | --- | --- |
| TRI-O01 | Owner policies, explicit tool allowlists and resource budgets precede model advice; neither d1 nor chat prose can grant authority. | Unauthorized-route/fault tests; policy-bound receipts |
| TRI-O02 | Deterministic controller state/replay is distinguished from probabilistic model generation and hardware-specific fresh re-inference. | Fixed-event controller replay; actual model/revision/quantization identities and scoped comparisons |
| TRI-O03 | A decision model is an advisory typed-choice/score service, not a generative teacher, exact oracle or permission engine. | Strict d1 schema/endpoint, invalid choice, fallback and abstention tests |
| TRI-O04 | External pretrained teacher/decision weights and runtime text never silently become native Trinite parameters, training labels or admitted data. | Dependency/import isolation, explicit distinct rights/data/intake approval |
| TRI-O05 | Exact formal computation is independently verified and stored separately from unverified teacher explanations. | Oracle property tests; proof of result-field immutability and mismatch rejection |
| TRI-O06 | Local transport, backend endpoints, request/response sizes, memory, time and concurrency have deny-by-default enforced bounds; no implicit external egress/download. | Loopback, Host/Origin, SSRF/redirect and resource-limit fault tests |
| TRI-O07 | User input, browser, model text, d1 decisions and external artifacts cross explicit untrusted boundaries; no model-origin executable HTML/JS/tool commands. | Prompt injection, XSS, escaping, unknown command, JSON duplicate-key and schema rejection tests |
| TRI-O08 | Actual source/config/model/runtime/route and outputs have scoped identities; observed failures/partials remain, and hashes are not origin attestation or correctness. | Versioned, redacted receipt and failure inventories, external digest anchors |
| TRI-O09 | Missing/uncertain/malformed model decisions fail safe or request clarification; no adaptive retry, unauthorized fallback or invented success. | Deterministic negative-case matrix with timeout/unavailable/incorrect-model |
| TRI-O10 | Native source/fixtures, historical experiments, MTC-001 data quarantine and Phase gates remain unmodified; new runtime application does not retrofit old numerical results. | Changed-file/source closure check, native regression and rights boundary audit |
| TRI-O11 | Local chat usability, d1 routing benefit, fresh native model learning, security and deployment must be evaluated and claimed separately. | Frozen baselines, held-out routing/false-execution/abstention metrics, owner-host resource measurements |
| TRI-O12 | No GAN training is required for conversational reasoning or exact-formal data; adversarial tests are mandatory; synthetic GAN visual/audio data is a separate versioned admissibility decision. | Tests with hostile samples, independent exact-generator/oracle checks and any later data-rights/source split protocol |

**Status:** all rows are **PROPOSED, NOT IMPLEMENTED**. A merged documentation PR grants no local model call, training, dataset admission, automatic downloads, deployment, tag or release.
