# Dense seed-0 historical unseen evaluation: negative result

Recorded 26 October 2026. One inference-only evaluation, no checkpoint selection,
training updates, retries or answer repair. This is an exploratory result on
historical research-exposed splits, not fresh confirmation.

## Result and scope

| Measure | Validation | Test |
|---|---:|---:|
| Raw complete answer plus EOS | 0 / 64 | 0 / 64 |
| Each of add/subtract/multiply/divide | 0 / 16 | 0 / 16 |
| Teacher-forced target tokens, including EOS | 96 / 176 | 101 / 176 |
| Teacher-forced answer tokens, excluding EOS | 40 / 112 | 45 / 112 |
| Teacher-forced EOS | 56 / 64 | 56 / 64 |
| Distinct generated outputs | 6 | 8 |
| Semantic families | 1 | 1 |

Each split contains 32 semantic items rendered in two carriers. Rows, token
observations and carrier comparisons are correlated. There is no iid confidence
interval or pooled success gate. Correct teacher-forced tokens with gold
prefixes are not correct prompt-only answers.

The preceding private training bundle reported 264/264 raw complete training
answers at the preselected final step-4,096 checkpoint. That training result and
its source/identity bindings were independently checked as retained evidence;
the public subset here does not include training history or tensors. The
training-fit improvement and this held-out failure coexist. Neither proves
universal architectural incapacity or a specific internal memorisation mechanism.

## Coverage explains what was tested, not why it failed

The unchanged corpus holds out absolute operand family **{1,2}** for validation
and **{1,3}** for test. Training families are {0,0}, {0,1}, {0,2}, {0,3}, {1,1},
{2,2}, {2,3}, {3,3}. Operands, task symbols and carriers are familiar; the
held-out combinations are new to this checkpoint's declared training schedule.
There is no operand-range or operand-digit-length extrapolation.

- Addition/subtraction: all gold answer strings already occur in **the same
  operation's** training answers, yet every held-out row fails.
- Multiplication/division: none of the held-out gold answer strings occur in
  the same operation's training answers. This is an additional support shift,
  not a reason to discard those failures.
- All 128 generated answers occur in their operation's training answer strings.
  Multiplication emits magnitude 1 with the correct sign throughout both
  held-out splits: for example `M:2,1=` produces `1`, not `2`, and `M:1,3=`
  produces `1`, not `3`. Training-support shortcuts are plausible, not proven.
- No output was malformed or unterminated under the original byte/termination
  definition. Premature EOS was observed on 16 validation and 7 test rows.
  Well-formed wrong answers remain wrong; this is not merely an EOS failure.

The [coverage diagnostic](COVERAGE-DIAGNOSTIC.md) reproduces the support and core
counts without loading a model. Its fixed addition-first development proposal
is a next investigation, not a demonstrated fix. More epochs, a larger model,
ternary changes, geometry loss or KV caching are not justified as fixes by this
result alone.

## Original commitments and public preservation

- Evaluator: [PR #69](https://github.com/QSOLKCB/Trinite/pull/69), commit
  `4045f03f4a2d8ce2a68bc50625bf0c6c4b6208a9`.
- Producer commit: `748f3104c98dad85fb55860749835453c4634dd1`.
- Producer request: `sha256:c1caf9163f02ef245cc7365b7ab0e6b37d084ecfc977ac0a6b9671a17c160dfc`.
- Producer completion: `sha256:6eb659e5b4f1a807ad9994f60cbcf31f736e31e6d334987a24b1f797a19825a6`.
- Reported final model: `sha256:53bc540949ddd10f77e3bbc2b279ba48de8e16a515e33d3d11b02b9242198f47`.
- Evaluation request: `sha256:317285b180ca6a101385533c8fbe7ff305d488a63541c19fb833958a6ca849e7`.
- Evaluation completion: `sha256:2a4f1c03bdcf251262ee85277f731670c4aedfd925b062eac406c49db054c110`.

The [public subset](../fixtures/unseen-negative-seed0-v1/README.md) retains all
original row bytes and their original completion/index linkage. It is not a
redacted packet claimed to pass the original closed-custody verifier. Omitted
private artifacts remain committed by the original completion, unchanged.

Private read-only review checked archive/manifest hashes, 183 evaluator source
commitments, all 128 rows against admitted labels and splits, original producer
identity, unchanged before/after inventories and provenance consistency. No
numerical predictions or optimizer updates were replayed by that review.
The public checker independently verifies the narrower subset and descriptive
counts only. `learning_adequate`, `phase5_ready` and `release_ready` remain false.
