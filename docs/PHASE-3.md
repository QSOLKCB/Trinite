# Phase 3: native training, safe replay and observation

Status: implemented with local conformance. Remote CPU CI and PR review remain merge gates. This phase addresses TRI-I01–I04, I06–I08, I11–I14; it does not close the reasoning, deployment or full-open release milestones.

## Frozen procedure and measured scope

[TRAINING.md](TRAINING.md) specifies the complete procedure in [tiny-training.json](../configs/tiny-training.json). The plan identity is sha256:b1c8d80cce1b54df0c0c561d05a721d2ff99d8b04a1cf13d9c723a7d5437024d. Dataset identity remains sha256:1a96cdf11a757cbf3c8dcbe2ea47f9220beb60f0a25fc0aca0401e9ad98bc383; its manifest remains sha256:e4bf897acafd81fd02486da676f315c9569e454ca8b1542c523702f62eaae4c1. No Phase 1 source, admission record, split or retained fixture was changed.

Both seed-0 lanes use the same 6,752 latent parameters at initialization, 36 training examples in the same order, four examples per update, single-tensor AdamW settings, 60 updates and 480 scored targets. Their shared initialization identity is sha256:688c2378aaedb9e18fe7bda5afc5d465f8243a6406b29f8f3dea06c30089994c. The default 1,247,232-parameter reference configuration is preserved; this deliberately smaller fixture keeps training conformance inexpensive.

| Lane | Initial training target loss | Final training target loss | Final validation target loss |
| --- | --- | --- | --- |
| Dense | 5.546834 | 1.583891 | 1.271456 |
| Ternary | 5.550732 | 1.646247 | 1.336578 |

These are observed local float32 CPU losses, rounded here for readability; run artifacts retain exact float.hex values. Both closed retained-input/output bundles verified with the pinned upstream verifier. The six validation examples are software-visible, family-separated fixtures, not fresh confirmatory evidence. No test labels enter training, validation or selection. There is no claimed significance, reasoning improvement, model superiority or generalisation benchmark. The loss-decrease regression threshold (final below 70% of initial) was added after exploratory smoke execution; it is a software regression assertion, not a preregistered empirical claim. Geometry loss is zero.

Local environment: Linux x86_64, CPython 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5, safetensors 0.6.2 and the complete hash-locked dependency closure, one Torch thread and strict deterministic algorithms. Each report records CPU model/features/kernel capability, backend, interpreter/kernel/package versions, time/observer overhead and Linux process-lifetime peak RSS. The local paired measurements ran consecutively in one process, so startup/warmup differs; elapsed time and RSS do not establish comparative speed or memory performance. Hosted CPU smoke is not a laptop/SBC benchmark.

## Demonstrated checks

| Boundary | Evidence |
| --- | --- |
| Existing contracts/data | 36 standard-library tests, including admission/splits, exact fixture/source replay, legacy CRLF upgrade and five observer-mapping tests |
| Model/quantizer | All 17 Phase 2 numerical, forward, gradient, alias, mask and capture checks remain required |
| Upstream acquisition | All 24 vendored files match the frozen Git blob/raw-content identities; 20 runtime files, license, two original suites and bundle contract preserved byte for byte |
| Upstream conformance | 51 original core/bundle checks: 50 pass locally, one skips because this container is root; hosted CI runs as an ordinary user and must exercise that permission test |
| Native training | 16 tests cover frozen bounds/settings, independent shifted answer/EOS loss, no numerical test-label exposure, matched initialization/order and loss decrease in both lanes |
| Safe checkpoint | Named model/AdamW state, counters/history, source/data/plan/environment and all active global RNG states; reject stale/unsupported/duplicate/malformed/nonfinite state, mismatched inventory, symlinks and extra members |
| Replay | Both lanes reproduce complete final safetensors and metadata bytes after a step-2 interruption; same check across independent CLI processes; final order/loss/validation/inventory/result/log identities agree |
| Isolation | Observer off/on/injected failure agree on every per-update gradient identity, updates/moments, RNG state, example order, logs, result, inventory and checkpoint bytes |
| Failure | Mandatory pin/collection failure preserves completed computation but returns nonzero and false evidence eligibility; compute interruption retains a valid completed-step checkpoint where recoverable |

The local upstream skip is explicit, not counted as a pass. Exact replay is scoped to a matching source/environment fingerprint, not across the three interpreter lanes. The CLI seeds global RNGs; the Python API retains caller-owned state. Documentation links, installed-wheel runtime pin/CLI behavior and the optional dependency lock are checked separately before publication.

## Implemented interfaces and remaining gates

training.py contains numerical execution; checkpoint.py handles safe state serialization; experiment.py owns explicit IO and stage orchestration; observation.py composes exact upstream record/identity/verification APIs. Foundation imports remain standard-library only. Source receipt and upstream license/pin/dependency lock ship in the wheel; no download or alternate serializer runs at inference/training time.

The CPU workflow runs foundation/upstream and model/training checks in Python 3.11/3.12/3.13. Manual tiny training verifies caller-supplied plan/dataset identities, runs both bounded lanes and retains completed/failed review artifacts for 14 days. It becomes dispatchable after merge; adding it is not evidence that a manual run has already executed. Expiring CI artifacts are not a release archive.

Every run distinguishes compute outcome, observer evidence eligibility and release readiness. Verified bundles bind bytes and derivation references, not scientific truth or custody signatures. Source commits are explicitly caller-declared and dirty state is uninspected; installed source bytes are independently hashed. The observer integration is implemented, while release_ready remains false.

Next is Phase 4 geometry measurement with synthetic controls and a pinned QSOL-GEO-REASON cross-check. Curriculum preparation remains a separately reviewed data-admission task: the YAML stress corpus, QSOL-SUBSTRATE and AUSTRALIAN-FOR-AIS plans remain in [ROADMAP.md](ROADMAP.md), with no repository text imported here. Larger training, geometry intervention, generation, packed export, edge measurements and fresh confirmatory evaluation remain later work.
