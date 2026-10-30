# Training-free music task coverage and retained-error audit

This is a **retrospective descriptive audit**, not another experiment. It asks
what the frozen task actually exposes, and which retained errors coexist with
those coverage gaps. It cannot identify the causal reason a model failed.
No model, checkpoint, optimizer or native tensor backend is loaded; no new
predictions, training examples for use in updates, or learning claims are made.

## Run the offline audit

From a clean checkout with Python 3.11 or later:

```sh
PYTHONDONTWRITEBYTECODE=1 python -I -B scripts/check_music_task_audit.py
```

The command accepts no selectors or alternate evidence. It authenticates the
frozen source closure and public evidence, invokes the existing authoritative
preservation checks, regenerates the finite corpus with its original codec and
independent rational/integer oracles, and joins recorded raw outputs by their
fixed identities. JSON on stdout is a newly derived report, not a resealed
historical artifact. Redirect only to a new owner-selected file outside the
checkout if saving it. Existing sources, protocols and fixtures are unchanged.

The interpreter, standard library and new audit implementation are trusted.
The isolated launcher rejects ordinary package/cache shadows and preloads
standard-library dependencies before project imports. Python startup/site code
runs before any script-level guard and is outside this boundary.

## What coverage means

Coverage is role-specific. For the original training parents a and b, tie starts
are at flattened event positions 0 and 4; rests are at 6 and 2. Swapping their
layouts crosses those templates with the other duration family, but introduces
**no new aggregate rest or tie positions**. Development parents c and d use tie
starts 1 and 5 and rests 7 and 0. A rest at 0 is not covered merely because a tie
starts at 0 in training. Absence of a feature is recorded separately.

The report separates absolute event indices, within-bar indices, boundary roles,
rational onset times, meter-conditioned onsets and full duration-family context.
Equal scalar onset times across different meters do not establish the same
context. Pitch augmentation does not add rest/tie-position support. All c/d
probes are already exposed development evidence; only two structural parents
underlie their meters and correlated pitch/tempo views.

The placement pool has 80 scheduled slots but 70 unique prompts. Ten variant-00
pairs are identical inputs; the other 30 pairs change placement. Repeating a
slot or collecting another seed is not another independent structural subject.
The finite 42 valid joint rest/tie placements are an oracle-checked support
universe only, not a new curriculum or model evaluation.

## Error interpretation

The report retains lane, arm, seed, step and rendering distinctions, including
training versus development, and computes exact answer-plus-EOS counts,
answer histograms and within-quartet contrasts from the original token streams.
Training-label correctness, prompt round-trip and actual answer/EOS loss-mask
alignment are audited separately from learned output accuracy. A simple
`8 - rest - tie` rule exists for this task; verification of that rule does not
show that the models learned it.

The retained canonical development rest-only sections all have the same raw
histogram: nine `8` answers and one `6`, each followed by EOS, against ten gold
`7` answers. The `6` case is parent c in 4/4; the report exposes every case rather
than replacing this pattern with an aggregate accuracy alone. This describes
reused public evidence across the fixed lanes, not independent replications.
For example, the original seed-17 frozen evaluation gets six of ten same-gold
tie-only cases correct while getting none of the rest-only cases correct.

Persistent rest-only failures and role-specific support gaps can motivate a
later controlled hypothesis. Their association is **not proof** of positional
memorization, a counting defect, duration confusion or any particular neural
mechanism. No new training budget, data split, intervention, selection threshold,
checkpoint election or fresh confirmation is authorized by this report.

Public reports remain partial custody: omitted requests, checkpoints and logs
stay owner-local. Recomputed scores do not attest original execution, timing,
hardware, optimization or absence of other attempts. The v2 CPU repeat's matching
journals are agreement with the earlier result, not independent replication;
its standalone unsupported device error is not a new GPU score.

## Separate CUDA availability diagnostic

See [CUDA availability](CUDA-AVAILABILITY.md). That owner-local tool observes a
fresh runtime only when explicitly run there. It is not invoked by this audit,
does not repair drivers or rerun either experiment, and cannot diagnose the
owner's current GPU from this repository or the cloud test environment.

## Compatibility and checks

This additive lane supports TRI-I03, I07, I08, I10, I11, I13 and I14 without
changing any historical contract. Normal and optimized standard-library tests
exercise support accounting, retained joins, evidence/source tampering, unsafe
paths, import shadows and claim limits. CUDA diagnostic tests use controlled
software doubles, not hardware certification. Existing numerical conformance
remains separately reported; no target study is run here.
