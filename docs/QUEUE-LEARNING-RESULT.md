# Queue-learning v1: negative two-seed pilot

On 1 November 2026 the owner approved the exact frozen queue-only pilot after
implementation PR #112 merged at `7a2ea15de2fd732ce88a2d1532911391ecdf663d` and
all 72 post-merge workflows / 207 checks succeeded. Both prescribed attempts
completed. **Neither model produced a numeric held-out answer.** This is a
negative learning result, not a training/runtime failure or forecasting gain.

The immutable [protocol](QUEUE-LEARNING-V1.md) was not changed: 384 requests,
256/64/64 train/validation/test episodes, grouped family splits, seeds 17 and 29
once each, final update 64 only, 24,768-parameter dense CPU float32 decoder.
No retries, search, teacher inputs, GPU, paid compute or post-outcome retuning.
Geometry loss remains zero; racing and cross-domain work are outside this run.

## Results

Every one of each seed's 128 held-out outputs was exactly token `[257]`: EOS
with an empty answer. These are **malformed**, not numeric zero or explicit `?`
abstention. Their gate-interface commitments abstain while preserving the raw
malformed status. All planned rows remain in denominators.

| Method | Validation numeric / planned | Validation exact / planned | Validation MAE | Test numeric / planned | Test exact / planned | Test MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Seed 17 | 0/64 | 0/64 | undefined | 0/64 | 0/64 | undefined |
| Seed 29 | 0/64 | 0/64 | undefined | 0/64 | 0/64 | undefined |
| Train lower median | 64/64 | 0/64 | 1032/64 = 16.125 | 64/64 | 8/64 | 1100/64 = 17.1875 |
| Train mode | 64/64 | 8/64 | 1728/64 = 27 | 64/64 | 8/64 | 1776/64 = 27.75 |
| Causal continuation | 64/64 | 8/64 | 1184/64 = 18.5 | 64/64 | 16/64 | 1092/64 = 17.0625 |

Both models' valid-output rate is 0/64 on each split; conditional numeric MAE
is also undefined because its denominator is zero. No model/baseline numeric
MAE comparison or improvement claim is eligible. The JSON preserves exact
rational pairs and nulls, including numeric exact-match 0/0. Baseline constants
use train targets only. All model and baseline predictions were committed before
the controller scored any held-out target; hashes do not prove operator blindness
or chronology. The finite synthetic family holdout is not a secret benchmark.

## Execution and verification

Each seed completed 64 updates, 512 scheduled visits to 256 unique training
rows, and **1,400 actual supervised answer/EOS tokens** (hard cap 32,768;
preflight worst-case schedule 2,048). The native `validation` diagnostic is
train-only, not held-out loss or checkpoint selection.

| Seed | Native update seconds | Whole train-child seconds | Elapsed seed lifecycle seconds | Train-child peak RSS high-water KiB |
| --- | --- | --- | --- | --- |
| 17 | 1.095 | 6.437 | 16.823 | 339220 |
| 29 | 1.323 | 6.816 | 13.971 | 342080 |

No worker exited unsuccessfully or exceeded the 120-second entire train-child
or 900-second elapsed lifecycle limit. The retained parent run wall is about
34.641 seconds and parent peak RSS is 240972 KiB. RSS is process high-water,
not whole-machine memory; timings are local observations, not benchmarks.

Runtime: Linux x86_64, AMD EPYC 9V74, Python 3.12.14, Torch 2.8.0+cpu and the
unchanged hash-locked dependencies; one Torch compute thread, strict determinism.
Rust 1.90.0 built both pinned crates offline/locked. Feasibility performed three
engineering gradient calls and zero optimizer steps, with a 12.139-second
conservative estimate. Local engineering checks passed: 19 normal + 19 optimized
contract tests, four native zero-update tests, and actual exposed gate integration.

The unchanged `run_queue_learning_v1.py verify` passed after execution, and a
second read-only review passed independently in the same environment. Verification
re-executes the authoritative queue observation/target operations, validates all
640 method/case commitments and receipts, regenerates metrics, and checks the
safe checkpoint structures and identities. **It does not replay AdamW updates,
prove human approval cryptographically, establish chronology/blindness, certify
learning adequacy or demonstrate independent-host numerical replication.**

## Retention and anchors

The [public evidence package](../fixtures/queue-learning-result-20261101-v1/README.md)
retains the complete frozen inventory, both attempts and safe tensor checkpoints,
raw journals, all baseline/model commitments and score/verification receipts,
resource/process logs, independent anchor journals and operator approval declaration.
Private conversation metadata is excluded. Existing fixtures and producer sources
are unchanged. A source snapshot and both original executables are included;
byte hashes are not compiler/build attestation.

| Identity | SHA-256 |
| --- | --- |
| Freeze | `71e56232314dc32bb0b55dff9b15b3587a44d35b7f1703962466aa3e0f7b90c2` |
| Preparation anchors | `99ab87c9dd4f8fd4cd4f6088dc0143476b851afbd10e37e3dee9a91724d3be93` |
| Feasibility | `b0eacb194bfa3732032d28db0b1055efdf9a34995139108084dad93e8d59bbee` |
| Checkpoint anchors | `5003f199abeaa3205d66338c72421aa6344d60ca0e8cb6158181737bddfc131b` |
| Prediction barrier | `161e917035a5d6432847f4bea7c05472c6b4ad49cbeb81faa469c9e514f94435` |
| Run inventory | `c4a7a84b34475bced8c7b89c1e3e0c4a8dbcf891b246e31df41e8830163dcef9` |

Affected requirements: TRI-I01–I03/I05/I07/I08/I11–I14. This additive evidence
preservation changes no numerical semantics, installed API, admission policy,
historical negative result, Phase 5 gate, release gate or future-run authorization.
Any later diagnostic or revised training study requires its own explicit scope
and prospective source-bound protocol; this result is not grounds to rerun or tune
this frozen pilot.
