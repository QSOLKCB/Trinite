# Coverage diagnostic v1 and addition-first development proposal

Status: implemented **read-only standard-library diagnostic**, plus a frozen
review-only development proposal. There is no new trainer, model execution,
corpus, training request or scientific execution command in this increment.
The [negative result](UNSEEN-NEGATIVE-RESULT.md) is preserved, not repaired.

## Inputs and verification boundary

The fixed [public subset](../fixtures/unseen-negative-seed0-v1/README.md) contains
130 exact original files. The checker pins the externally retained completion
identity, verifies index and every row hash, and rejects missing/additional
members, symlinks, non-regular files, oversized files and observed changes.
Only quiescent caller-controlled local directories are supported; pre/post
metadata checks are not hostile-filesystem race containment. The checkout must
also remain quiescent; imported module paths must match the recorded source
root, and pre/post source hashes must agree.

The original historical corpus identity is pinned. The checker calls the
unchanged foundations dataset/admission/oracle path, then verifies every
retained row's example/semantic/family/task/carrier/split and gold tokens.
All 64 validation and 64 test rows are mandatory, in original order. It does
not silently reassign families or select easier examples.

This authenticates a **partial public subset**, not the missing original closed
PROVENANCE packet, environment or model tensors. Original model linkage is
reported from the private audit and labelled accordingly. No inference, logits,
loss, probability or optimizer replay occurs. Token predictions are retained
observations, not newly certified numerical output.

The new derived report binds the analysis source closure separately from the
original evaluator commit and result commitments. It recomputes complete-answer
and teacher-forced counts from tokens, separating EOS. Coverage is measured
against training rows from the **same operation**:

- family seen versus unseen;
- gold answer string familiar versus novel;
- operand values familiar versus novel;
- operand range and digit-length extrapolation;
- answer byte length represented versus absent.

These are separate flags, not interchangeable explanations. A novel answer can
have a familiar length. Range/length strata have no historical observations
here; absence of such rows is not a successful extrapolation result. The report
retains all per-row support flags and per-operation counts.

## Commands

From a reviewed checkout, Python 3.11+ and no numerical dependencies:

```sh
mkdir -p reports/coverage-diagnostic
PYTHONPATH=src:scripts python -B scripts/run_coverage_diagnostic.py report \
  --output reports/coverage-diagnostic/report.json
# Save the printed sha256 identity independently, then:
PYTHONPATH=src:scripts python -B scripts/run_coverage_diagnostic.py verify \
  --report reports/coverage-diagnostic/report.json \
  --expected-identity 'sha256:REPLACE_WITH_PRINTED_IDENTITY'
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/coverage_diagnostic_v1 -v
```

Creation is exclusive and refuses an existing path or a path inside the fixed
fixture. Verification recomputes the report and requires exact current-source
bytes and the externally retained identity. Do not overwrite an old report or
replace its expected identity to conceal a mismatch. Parent output directories
are caller-controlled; no private source directory is scanned or uploaded.

## Frozen proposal: addition first, not target execution

The report's `trinite.coverage-development-proposal.v1` selects **addition**
before any new target outcomes, because the historical failure occurs even with
familiar same-task answer strings. This choice follows exposed historical
results and is therefore development, not fresh confirmation.

The next question is whether a single predeclared training-coverage intervention
improves new-family, familiar-answer development performance. A subsequent
implementation must first admit a new versioned development corpus, freeze
family-disjoint control/candidate splits and complete coverage tables, choose
one intervention, and specify exact seeds, budgets, schedules and failure
retention. Match dense architecture, optimizer, tokenizer/carriers, initialization
seed set, update count, scheduled target/input exposure and raw decoding; if
exact exposure matching is impossible, revise the design before execution and
state the difference, rather than silently claiming a matched comparison.

Separate familiar-answer new combinations, novel answers and operand-range/
length extrapolation. Preserve every failed and successful cell. Do not add
historical validation/test rows to training while continuing to describe them
as held out. Historical scores remain exploratory even if they later improve.

Genuinely new multiple-family confirmation and its exposure review belong after
the development recipe is frozen. Merely resplitting these public finite facts
or changing IDs is not freshness. No confirmation families are generated or
exposed by this PR. Publishing this proposal does not authorize a local target
run; that needs a later reviewed implementation, frozen request and explicit
owner scope. No thresholds, checkpoint election or release gates are relaxed.

## Compatibility

Affected requirements: TRI-I01, I03, I07–I08 and I10–I14. Historical `src/`,
producer/evaluator scripts and their bound contract documents, locks, admission, tensors and source-bound fixtures
are unchanged. New checkout-only analysis imports contracts, tokenizer, actual
foundations admission and the existing standard-library packet inventory helper.
It imports no numerical backend, training loop, model, checkpoint loader or
network client. Its arithmetic coverage calculation is not a new learning task,
training scorer or replacement admission operation. CI verifies only evidence
and analysis conformance; its green status cannot change the negative result.

The original `UNSEEN-EVALUATION.md` retains its implementation-time prospective
wording because it is part of the original evaluator source commitment. The
dated result note and current roadmap/index describe the later completed run;
that historical contract is not rewritten to change its old request identity.
