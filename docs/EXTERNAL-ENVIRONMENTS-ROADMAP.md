# External environment roadmap

Status: engineering roadmap, 1 November 2026. The race simulator merged in
[PR #105](https://github.com/QSOLKCB/Trinite/pull/105). The
[queue module](QUEUE-SIM-V1.md) merged in [PR #106](https://github.com/QSOLKCB/Trinite/pull/106).
The [inventory module](INVENTORY-SIM-V1.md) merged in [PR #107](https://github.com/QSOLKCB/Trinite/pull/107).
The paired [qutrit decoder and noisy-syndrome sensor](QUTRIT-MODULES-V1.md) merged
in [PR #108](https://github.com/QSOLKCB/Trinite/pull/108). A separate bounded
[generic sensor environment](SENSOR-NETWORK-V1.md) merged in
[PR #109](https://github.com/QSOLKCB/Trinite/pull/109). The
[music-event environment](MUSIC-EVENTS-V1.md) merged in
[PR #110](https://github.com/QSOLKCB/Trinite/pull/110). Modules are built first; training
comes later under separate protocols. None establishes a Trinite learning
improvement or cross-domain transfer.

## Boundary: the runner computes; Trinite sees observation JSON

Use small standalone CPU runners, initially Rust, outside the native model and
numerical updater. A privileged controller owns requests, seeds, latent state
and outcomes. It may export a validated observation JSON file for Trinite;
the learner must not receive privileged files or their storage paths. JSON is
an interface, not a sandbox: a future controller must enforce that separation.
This design gives the model no shell, compiler, network access or arbitrary
code-execution tool.

Each domain gets its own fixed, versioned request, observation and outcome
contracts, strict validation and bounded resources. Reuse only a minimal
explicit envelope where useful (schema version, public episode identity,
phase and observation payload), not a plugin registry or universal simulator
framework. Any public identity must not reveal the seed or hidden outcome.
Do not retrofit a shared envelope into the race-v1 schema without a new version.

Keep observed neutral values distinct from missing evidence using an explicit
observation mask. Withdrawal is an event that changes participation, not a
missing measurement: retain the roster and active/withdrawn state and recompute
relevant field interactions. Define unknown observations, invalid inputs,
simulator failures and learner abstentions separately; never silently turn them
into zero scores or successful examples. With no observed evidence, a normalized
evidence score is undefined, not neutral.

## Staged modules and their gates

| Stage | Scope | Gate before claiming useful model evidence |
| --- | --- | --- |
| 1 — Race simulation | Merged #105: bounded seeded CPU integer toy dynamics, masked ternary observations, before/after scratchings and separate privileged ranks. No real-world racing calibration. | Review and exact-commit CI establish software conformance only. A separately reviewed controller, prediction commitment and evaluation protocol are still needed. |
| 2 — Queue simulation | Merged #106: bounded FIFO arrivals, finite buffer, service capacity, delays/outages, masked integer prefix observations and separate privileged wait/overflow accounting. No learner or ternary projection yet. | Independent event/accounting oracle; unseen arrival and service families; distinguish unavailable queue measurements from an empty queue. |
| 3 — Inventory shortages | Merged #107: finite stock, lost-sales demand, exogenous replenishment/lead times, receiving disruptions, masked prefix observations and privileged conservation/stockout outcomes. No reorder actions or learner. | Conservation checks and independent baseline; unseen demand/disruption families; distinguish zero stock, unknown stock and an unavailable supplier. |
| 4 — Noisy sensor networks | QEC adapter merged #108. Generic sensing merged #109: bounded integer latent signals, explicit noise, bias/stuck faults, dropout and sensor-to-signal assignments. No learner, calibrated detector or network transport. | Independent latent-state oracle; held-out noise/fault/topology families; calibrated uncertainty and explicit unavailable sensors. |
| 5 — Music-event environments | Merged #110: bounded synthetic note intervals, timing/transposition, dropped events, channel mute and masked prefix grids with separate privileged truth. No learner or audio renderer. | Original generated material with rights records, exact event/timing oracle and held-out rhythmic/structural families. No copyrighted audio intake or audio-preservation fidelity claim. |
| 6 — Cross-environment tests | Planned only after useful within-domain evidence: test whether a declared shared representation helps a new domain. | Frozen transfer protocol, matched no-transfer controls and fresh held-out families; report negative transfer and extra compute. No assumption that a common JSON interface creates transferable skill. |

The race implementation is a synthetic rank generator, not a betting strategy,
physical racing model, reward trainer or installed Trinite capability. Its
[contract at the merged commit](https://github.com/QSOLKCB/Trinite/blob/f8a78db1a04c8dfa9720b0a92630b35e3e30b42c/docs/RACE-SIM-V1.md)
records its actual schema, deterministic scope and limitations. A seed split
alone is insufficient: new seeds can reproduce the same structural problem.

## Experimental QEC sensor/decoder branch

The [paired external modules](QUTRIT-MODULES-V1.md) use genuine GF(3) qutrit
syndromes, with a separate observation mask. Sensor output is the decoder input;
true errors and success checks stay in a privileged engineering evaluator. The
fixed single-site table returns candidate corrections, never inferred success.
Measurement corruption can cause accepted miscorrections. This branch is not
full sensor-network completion, native learning, hardware correction or a reward
loop. QEC repository core and historical Trinite evidence remain unchanged.

## Queue evaluation infrastructure before stage 6

The [queue prediction/evaluation gate](QUEUE-EVALUATION-V1.md) adds a checkout-only
prepare/commit/score/verify path with two independently retained anchors, a fixed
future overflow count target and explicit abstention. It reuses the unchanged
queue dynamics. Public synthetic conformance examples establish engineering
behavior only: no learner, training, useful within-domain evidence or completed
transfer gate. The common JSON interface does not establish transferable skill.

## Prediction first; action and reward require a separate contract

Initially ask the learner to predict a declared outcome from an observation.
A supervised loss or evaluation score does not by itself implement reinforcement
learning. Commit predictions before revealing outcomes to the evaluator. Keep
training labels separate from held-out labels and never feed future scratchings,
seeds or latent state into the prediction input.

A later action environment would need bounded allowed actions, transition
semantics, timing, reward definitions, termination and invalid-action handling.
Define rewards against independently checked outcomes, not agreement with the
same heuristic that generated observations. Freeze reward/score changes before
evaluation; retain failures and check shortcuts and reward exploitation. The
roadmap does not implement or authorize this action/reward loop.

## Evidence required for each later experiment

- Freeze source, toolchain, schema, request, budgets, split inventories and scoring
  before target outcomes. Bind exact bytes and retain logs, rejected inputs,
  partial runs and negative results; report same-environment replay separately
  from independent reproduction. Seeded execution is not a universal guarantee.
- Use an independently implemented reference/oracle and simple baseline. Do not
  make a weighted ternary evidence score the outcome generator and then claim
  that recovering it proves predictive skill.
- Split by structural train/development/test families, not random rows alone.
  Keep related seeds, counterfactuals and before/after variants together; reserve
  locked test families outside training, tuning and model/threshold selection.
- Compare binary, ternary and continuous inputs under declared matched exposure,
  missingness, model capacity and resource budgets. Ternary input evidence is
  distinct from ternary model weights; neither implies greater accuracy.
- Report ranking metrics for rankings. If a later model emits validated
  probabilities, predeclare proper scores such as Brier or log loss, calibration
  and uncertainty intervals; an evidence score is not a winning probability.
  Report abstention coverage and error together, and all failure counts.
- Measure generator, preprocessing, inference and any authorized training costs
  separately, including wall time, CPU/RAM and artifact sizes. Equal examples or
  tokens do not establish equal compute; correlated simulations are not
  independent real-world trials.

## Compatibility and authorization

This document changes no numerical code, historical producer, fixture, admission
rule or learning threshold. It supports [TRI-I03, I07, I08 and I10–I14](INVARIANTS.md)
and preserves [module boundaries](MODULES.md), [data admission](DATA.md) and the
[diagnostic-first order](ROADMAP-V1.md). Synthetic output still requires the
actual admission policy before native training.

Future modules require separate implementation PRs. Research runs, training,
new data intake, host/resource use and action/reward experiments require their
own reviewed protocols and explicit authorization. This roadmap grants none of
those permissions and advances no learning, geometry, scaling or release gate.

## Procedural dataset modules: next engineering tranche

The next separate implementation PRs add the following foundations, not a new
training result. GitHub assigns PR numbers; sequencing is not a merge claim.

| Module | Engineering role | Remaining experimental gate |
| --- | --- | --- |
| `external/ternary-evidence-v1` | Mask-aware exact rational evidence scoring; explicit support/denominator and coverage, unknown distinct from neutral. | Compare representations against independent outcomes; do not use the score as both generated truth and evidence of predictive skill. |
| `external/tombstone-adapter-v1` | Bounded C99 splice-oracle adapter with pinned source and separate public/privileged products. | Rights/admission, family splits, baseline and source-bound controller before training. |
| [context-symbol-relation-v1](CONTEXT-SYMBOL-RELATION-V1.md) | Total public context × symbol rule tables; compatible-world support/contradiction/mixed relations; same-symbol context counterfactuals. | Native task admission, lossless carrier, controlled baselines, grouped fresh families and explicit run approval. |

These modules permit small original formal cases to be generated when needed,
rather than requiring a massive static text corpus. This tranche supplies
deterministic evaluators/contracts, not a general on-demand dataset generator.
A future generator must pin version, parameters and any seeds, retain generation
and rights records and exact case identities/receipts, and reproduce each example
and oracle label. Storing a compact generator does not eliminate the need to
retain protocol commitments, split inventories, failures and verification records.

### Ordered bridge from modules to learning

1. Review each standalone module and exact-commit CI. Preserve all existing
   protocols, source-bound historical fixtures and queue-learning-v2 semantics.
2. Implement separately reviewed native task admission and a compact, lossless
   public-observation JSON adapter. Freeze token/byte budgets; reject overlength
   instead of silently dropping rules. Measure actual context offload against
   a baseline with identical information/semantics, including module input/output,
   preprocessing, serialization, inference, memory and total compute costs.
   External storage/computation is not extra parametric memory or automatically
   larger effective context.
3. Freeze procedural generators, structural family inventories, budgets, scoring,
   baselines and success/failure thresholds before held-out outcomes. Keep all
   same-table variants, counterfactuals, mask variants and index renamings in one
   split group. Distinct seeds alone do not prove unseen structure. Old public
   engineering fixtures remain exposed; reserve fresh locked evaluation families.
4. Compare supervised baselines and controlled representation/module ablations,
   with matched information, exposure and capacity and explicitly measured equal
   compute where claimed. Report exact-answer/relation errors by family, abstention,
   coverage, failures and negative transfer. A mask intervention that changes
   compatible worlds must recompute its truth label; a metadata-shortcut probe
   preserves semantics and changes irrelevant metadata. Do not conflate them.
5. Only after review, green merged code and explicit approval of a fresh frozen
   run execute native training. This roadmap executes no optimizer, automatically
   authorizes no experiment and does not advance a learning or release gate.

### Scaling curriculum with measured competence

Increase procedural problem complexity, diversity and volume in staged,
predeclared capacity studies only when measured competence justifies it. Preserve
fixed anchor benchmarks and locked family-held-out tests so easier/evolving
curricula cannot manufacture improvement. Track generator bias, duplicated
structures, catastrophic forgetting and cross-family transfer, not just throughput.
Freeze evaluation populations; never adapt online generation or tune the generator
against held-out failures and still call the result unseen confirmation.

Larger models can receive broader curricula within the existing
[model-size policy](MODEL-SIZE-POLICY.md), whose 8B total unique-parameter ceiling
is a limit, not a goal or permission to scale. More generated cases, parameters or
external information do not guarantee learning or reasoning. Information kept in
external modules and supplied at inference remains distinct from knowledge learned
in weights. Capacity, context offload and useful generalization each require their
own measured controls and fresh authorization.

### Implemented procedural admission bridge (engineering only)

[Procedural dataset v1](PROCEDURAL-DATASET-V1.md) now implements bounded
on-demand formal generation and source-specific admission for these three modules,
with whole dimension/length families, public-only lossless carriers, measured
native byte-token budgets, independent oracle checks and train-only target export.
This supersedes only the pending generator/admission portion above. Its fully
public engineering splits are not fresh unseen evidence; no model or optimizer
runs. It does not implement the later prediction-commitment controller or controlled
fresh frozen comparison, nor advance the learning, scaling or release gates.
