# CPU positional-exposure ablation v1

Status: prospective engineering implementation, **target experiment unrun**.
Review, exact-commit green CI and separate owner-local authorization precede the
12-cell run. This is not a learning result, numerical replay, fresh confirmation,
or a passed Phase 5/release gate. Historical protocols and negative evidence stay
unchanged. See [the original study](POSITION-STUDY-V1.md) and its
[preserved mixed/negative outcome](POSITION-STUDY-RESULT.md).

## Question and factorial controls

Does a balanced training-offset policy improve transfer under this fixed small
recipe, and does the effect of coupling depend on that policy? Use four fresh
conditions with paired seeds 0/1/2:

| Arm | Address mode | Training offsets |
| --- | --- | --- |
| sequence-fixed | ordinary sequence | zero |
| coupled-fixed | equal-significance coupling | zero |
| sequence-balanced | ordinary sequence | cycle index modulo eight |
| coupled-balanced | equal-significance coupling | cycle index modulo eight |

Every diagnostic and generation call uses offset zero. Twelve fresh cells supply
concurrent controls; historical PR77 outputs cannot replace missing controls.
Data, labels, row order, batches, parameter inventory, effective seeded initial
tensors, optimizer, update/token budgets, prompt format, decoding and evaluation
are otherwise matched. Fixed controls must have zero-offset numerical parity
with the original path. This is an offset-policy/addressing intervention, not a
carry curriculum, larger model, ternary/LUT/KV experiment or inference router.

## Unchanged arithmetic and balanced schedule

Reuse the original authoritative `position_study_v1.data` generator and admission
bytes, including rights, labels and example identities. Train 64 unordered
families/154 unique rows; in-range unseen 24/56; length3 and length4 each 12/23.
One cycle has 160 visits, including duplicate diagonal order slots. Repeat 200
cycles, batch 8, 4,000 updates. Balanced arms use offset `cycle_index % 8` for
all 20 batches in a cycle. Every cycle slot visits each offset 25 times; each
unique diagonal row visits each offset 50 times. The schedule is deterministic,
independent of labels, outputs and model RNG, and frozen before target outcomes.

Per cell: 32,000 visits, 121,600 answer/EOS scored targets, 307,200 actual live
shifted input tokens. The latter comprises 300,800 per-example predictor inputs
plus 6,400 live unscored short-row EOS inputs introduced by native batch shifting.
There is no added training example or changed padding, row ordering or token
budget. Twelve cells double aggregate research work relative to the previous
six-cell protocol; no compute-saving claim is made.

## What the coverage gate does and does not prove

Preflight enumerates the actual batch-shifted training input lookups and every
possible live generation prefix at evaluation offset zero. With cap D+2, the
last model call consumes at most D+1 emitted tokens; terminal EOS is never fed
back after generation stops. Training-only short-row EOS lookups are counted
separately, not mistaken for generation requirements.

Balanced conditions must cover every required **embedding address × coarse token
role** in train and all development strata. Coarse roles are BOS, operand digit,
plus, equals and answer digit. Address bounds, unchanged counts and exact
per-row offset balance are mandatory admission gates. Fixed controls deliberately
retain their documented gaps; they are not relabelled coverage successes.

**Fine-role deficits remain mandatory first-class report fields.** No training
example contains operand significance 3/4 or answer slots 3/4. Offsets cannot
create these roles from one-/two-digit examples. The report lists required and
missing detailed address/significance/answer-slot pairs separately, and reports
`detailed_roles_covered: false` where appropriate. A passed coarse gate does not
waive or pass a fine-role gate.

This intervention exposes previously unused embedding addresses in coarse
operand/answer roles. It does not eliminate unseen mathematical significance,
answer-length, range or role-composition shifts. Lookups are not nonzero-gradient
support, learned significance or arithmetic competence. A positive result supports
this offset policy under this recipe, not a universal explanation of old failures.

## Numerical, state and evidence boundaries

Dense CPU float32, context 64, two blocks, width 64, four heads, FF128, 86,528
parameters. Reuse the single native initialization, updater, masked loss, AdamW,
scorer and safe tensor reader. LR0.001, betas0.9/0.999, epsilon1e-8, decay0.01,
clip1, constant schedule, accumulation1. Same hash-locked CPU environment, one
thread, deterministic algorithms. No geometry loss or external/teacher data.

The offset adapter parses only the complete live prompt and emitted token count,
never gold answers, carry state or emitted values. Sequence causal order is
unchanged. Padding positions remain zero. The explicit training context takes
its offset from the retained cursor and clears it on exit/error. Only native
autograd-enabled update forwards use that offset; native no-grad diagnostics and
generation always use zero. No per-forward counter, mutable RNG or inference
sidecar chooses offsets. Native updates are composed at cycle boundaries, not
copied into a new optimizer implementation.

Requests and checkpoint behavior bind the arm, offset policy/schedule, evaluation
policy and cursor. Resume recomputes offset from the validated retained cursor;
no automatic research resume/retry is offered. Checkpoints cannot be published
inside an active training-offset context. Same-shape zero-offset parity is exact;
cross-shape tolerances remain absolute2e-6/relative1e-5; same-environment tiny
resume compares exact model/AdamW/history. Equal tensor hashes alone do not mean
equal behavior across offset policies.

Keep checkpoints0/1000/2000/3000/4000, unique-training diagnostics only there,
and development probes only after update4000. Generation remains greedy, cap
D+2 from prompt width, stop only on actual EOS. No forced EOS, repairing,
calculator fallback, election, adaptive rerun, seed replacement or extra probe.
Raw bounded per-item journals are fsynced and retained on failure. Preserve every
attempt, partial artifact, timeout and failed comparison.

## Predeclared comparisons and stopping

Four separate directed contrasts: coupled minus sequence under fixed offsets;
coupled minus sequence under balanced offsets; balanced minus fixed for sequence;
balanced minus fixed for coupled. Each uses the original family-level practical
margins, with the right-hand condition treated as candidate:

- Each seed: at least61/64 seen families and at most one-family seen regression.
- Each seed: unseen gain at least3/24 families.
- Each seed and each length stratum: at least3/12 families and gain at least2/12.

Report every condition of every contrast. No pooled, omnibus or statistically
significant success is inferred; correlated renderings are not independent
samples. Keep exact answer-plus-EOS rows and whole families, secondary as-emitted
numeric correctness, carry/no-carry strata, malformed/early/missing EOS and
correctness alongside consistency. Original authoritative reporting is reused.

If coverage admission fails, stop before target execution. If admitted but any
frozen contrast condition fails, retain the mixed/negative result without tuning
examples, budgets, thresholds or seeds. All probes remain development-exposed;
fresh confirmation is a separate later decision. No model capacity limit or
alternative causal explanation is established by a negative result.

## Bounds and prospective owner-local workflow

Each worker:1,800 seconds; cumulative updater calls960 seconds checked at frozen
milestones; native per-invocation ceiling480 seconds. Twelve workers at most21,600
seconds plus setup/audits. Per artifact16MiB, computational payload32MiB/cell,
worker log1MiB. Custody copies are extra, not a hard total-directory/RSS cap.
Record measured wall/update time and process high-water RSS without speed claims.

```sh
PYTHONPATH=src:scripts python -B scripts/run_position_exposure.py preflight
PYTHONPATH=src:scripts python -B scripts/run_position_exposure.py freeze --request /ABS/NEW/request.json
# Independently retain the printed request identity before outcomes.
PYTHONPATH=src:scripts python -B scripts/run_position_exposure.py run \
  --request /ABS/NEW/request.json --request-identity sha256:REQUEST_ID \
  --output /ABS/NEW/attempt --authorize-run
# Independently retain matrix_identity, also for incomplete matrices.
PYTHONPATH=src:scripts python -B scripts/run_position_exposure.py audit \
  --request /ABS/NEW/request.json --request-identity sha256:REQUEST_ID \
  --output /ABS/NEW/attempt --expected-identity sha256:MATRIX_ID
PYTHONPATH=src:scripts python -B scripts/run_position_exposure.py summarize \
  --request /ABS/NEW/request.json --request-identity sha256:REQUEST_ID \
  --output /ABS/NEW/attempt --expected-identity sha256:MATRIX_ID
```

Fresh paths outside checkout; request outside attempt. Do not invoke internal
workers. Audit/summarize are read-only source/environment/admission/custody/safe
structure/history/retained-metric checks. Both reconstruct seeded initial states
and initial training loss, but neither replays trained predictions or optimizer
updates. Custody and consistency are not numerical replication or host attestation.
Retain original identities, full request/source/checkpoints/logs/raw outputs and
failures. Never modify historical receipts or reseal evidence.

Affected TRI-I01–I03/I05–I08/I10–I14. Historical source closures, fixtures, study
and result documents remain byte-identical. Motivation remains
[Position Coupling v2](https://arxiv.org/abs/2405.20671v2); deterministic balanced
cycle offsets and this small model differ from that paper's scale/randomization.
This is not a paper reproduction or a claim that exposure alone fixes arithmetic.
