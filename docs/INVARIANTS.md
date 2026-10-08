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
