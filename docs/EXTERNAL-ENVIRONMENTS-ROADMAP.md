# External environment roadmap

Status: engineering roadmap, 1 November 2026. The race simulator merged in
[PR #105](https://github.com/QSOLKCB/Trinite/pull/105). The
[queue module](QUEUE-SIM-V1.md) merged in [PR #106](https://github.com/QSOLKCB/Trinite/pull/106).
The [inventory module](INVENTORY-SIM-V1.md) merged in [PR #107](https://github.com/QSOLKCB/Trinite/pull/107).
A paired [qutrit decoder and noisy-syndrome sensor](QUTRIT-MODULES-V1.md) is
implemented for review. Generic sensor networks and music remain planned. Modules are built first; training
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
| 4 — Noisy sensor networks | First QEC-focused adapter implemented for review: explicit modular syndrome corruption/dropout feeding a separate fixed five-qutrit decoder. Generic latent signals, sensor faults and network topologies remain planned. | Independent latent-state oracle; held-out noise/fault/topology families; calibrated uncertainty and explicit unavailable sensors. |
| 5 — Music-event environments | Planned: synthetic note/timing events, missing events and channel faults; predict sequences or reconstruct bounded event properties. | Original generated material with rights records, exact event/timing oracle and held-out rhythmic/structural families. No copyrighted audio intake or audio-preservation fidelity claim. |
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
