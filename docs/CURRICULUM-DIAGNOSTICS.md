# Curriculum diagnosis after the data audit

Requested by Trent Slade after merging PR #28 on 2026-10-13. This increment
tests the existing task-balanced recipe before reducing the training corpus.
The hypothesis is that curriculum, exposure and supervision constrain learning;
it does not assume that the model's capacity or optimizer is adequate.

## Audited evidence

The two original user-supplied Codex API reviews of commit
`c9f4399eca3bb380136321453ff2c6222f0bbaec` are preserved, with their supplied
checksums, in [the review packet](../fixtures/curriculum-review-c9f4399/).
Both ZIP identities and every member in both manifests were checked. The
read-only audit was rerun against that exact commit: every corpus, scheduling,
source and evidence field matched, with only the declared host environment
differing. The reproduced audit and comparison receipt are retained separately.

| Measurement | Finding | Interpretation |
| --- | --- | --- |
| Complete maths sequence | At most 27 byte tokens, including BOS/EOS; context 64 | Current examples fit; increasing context supplies no missing input |
| Maths training coverage | 1,998 rows; 999 semantic facts; 12 tasks | Both text carriers do not double independent mathematical facts |
| Task sizes | 76 to 312 rows | Uniform row visits allocate roughly four times more visits to the largest tasks |
| Actual arithmetic/maths mixture | Zero conflicting training prompts in the audit | No demonstrated contradiction in the stream actually consumed |
| Hypothetical mixed scalar workloads | 16 conflicting prompts | Namespace tasks before any future scalar mixture; historical separate runs are unaffected |
| Research model | 86,528 parameters; two blocks, width 64 | The 1.25M reference model is not the model behind these learning outcomes |

The supplied Torch follow-up records 227 required CPU/research tests passing;
467 of 468 original tests passed overall, with one CNR test-scoping error.
Its isolated one-line correction passed. PR #28 subsequently removed that local
import defect. The supplied validation is evidence of bounded numerical and
software conformance, not a full historical research-matrix rerun or proof of
learning. Its original failure log remains in the unmodified packet.

## First controlled experiment

The reported experiment used the already published
[task-balanced protocol](MATHS-TASK-BALANCE.md) from merged producer
`18c6a13d95045ae6376e6144e38cc1aaf86a127b`, under a request for its declared
original host. The recorded plan changes neither source nor frozen protocol.
Four cells compare uniform-row and task-balanced maths scheduling at seeds 0 and 1.
Both keep all 264 arithmetic rows, all 1,998 maths rows, the same native dense
model, context 64, AdamW, 8,192 updates, four arithmetic/four maths visits per
update, unique-training diagnostics and final-checkpoint selection.

Task balance changes frequency and ordering together. It matches updates and
row visits, not target tokens or compute. All four cells and all six assurance
checks must complete, including optimizer replay from native initialization.
Training-only acceptance remains at least 90% complete answer plus EOS in every
arithmetic and maths task in both candidate seeds. No test/validation model
calls, winning-seed selection or relaxed task denominator are permitted.

[PR #29](https://github.com/QSOLKCB/Trinite/pull/29) reports that all four
cells and six assurance checks completed, with ten maths tasks still failing
the joint training gate. Its body still labels the additional archival replay
as in progress. The complete original packet and independently retained
request/verifier/archive identities have not been located in the inspected
repository, PR record or available supplied artifacts. The
[reported counts and exact closure checklist](MATHS-TASK-BALANCE.md#reported-matrix-and-unresolved-archival-closure)
distinguish that attributed result from independently verified archival closure.
Do not infer that the original evidence is lost, refresh it, or treat the
merged implementation as a complete published research record.

## Ordered follow-up decisions

1. Assess the complete task-balanced matrix before choosing another recipe.
   A positive aggregate change cannot substitute for the per-task joint gate.
2. If joint learning remains inadequate, freeze a separate full-versus-core
   curriculum experiment. The supplied proposal targets 32 semantic facts per
   maths task (768 carrier rows total), preserving every available training
   family, both carriers, and applicable sign/zero/invalid/singular/rational
   categories. Fail selection if coverage cannot fit; never silently relax it.
   Keep arithmetic unchanged. Score all original unique training rows and
   report selected/omitted rows separately. The smaller core is a hypothesis,
   not a demonstrated improvement or a pure estimate of dataset-size effects.
3. Diagnose fold-sum in a separately frozen sum-only versus mixed-task study.
   Declare whether total updates or sum exposure is matched: either choice
   leaves other differences. Preserve the original composed-output gate.
4. If fitting remains inadequate, separately compare exact procedural traces
   or model capacity. Traces need independently verified labels and explicit
   extra-token accounting. Context expansion requires an actual longer task.
5. Use genuinely fresh family-level confirmation after training development.
   Known held-out sets, new carriers of old facts, or reshuffling known families
   cannot establish fresh generalization.

These are subsequent experiments, not automatically selected recipes. The
present increment changes no tokenizer, data admission, numerical updater,
model capacity, loss or context. Geometry loss remains zero; original
fold/composed adequacy, format selection, scaling, Phase 5 and release gates
remain separate.

Affected requirements: TRI-I02/I03/I05/I07/I08/I11/I12/I13/I14. All prior
protocols, producer source closures, immutable v0.1.0 and negative evidence are
preserved. Evidence integrity, reproducible computation and learning adequacy
are reported separately.
