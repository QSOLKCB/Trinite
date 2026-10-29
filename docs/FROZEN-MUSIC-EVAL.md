# Frozen musical-family evaluation v1

**Prospective implementation and software conformance, not a target result.**
This separate checkout-only evaluation follows the
[preserved positive train-only exposure result](MUSIC-EXPOSURE-RESULT.md).
It changes no historical source, admission, fixture, installed API or numerical
operation. Affected requirements: TRI-I01/02/03/05/06/07/08/11/12/13/14.

## Fixed endpoints and question

Evaluate **both prescribed final step-640 checkpoints, seeds 17 and 29**, from
the [original exposure producer](MUSIC-EXPOSURE.md). Never select a seed,
intermediate endpoint or replacement checkpoint. There are zero new updates,
no resume or training route, no optimizer construction, no teacher or arithmetic
bridge, and no outcome-dependent tuning, retry, repair or automatic promotion.

The question is whether the already trained finite note/rest and tie/reattack
contrasts transfer to the existing structural parents `c` and `d`. The original
training parents `a` and `b` remain disjoint. Exact prompts do not overlap across
the regenerated splits. These are existing visible development families, not a
fresh independent confirmation sample. The eight-event shortcut
`8 - rest_presence - tie_presence` remains sufficient; even perfect results
would not establish arbitrary-length counting, general musical understanding,
arithmetic transfer, recursive self-improvement, or a learning/release gate.

## Original custody and model-only import

The original producer is pinned to commit
`c239acb6e7b0b5c49798debbae92ca1e1264a64c`, tree
`2bf799bf24edca25b784be36c60fe921011042d1`. Supply its clean local checkout,
complete retained run and **independently retained original request and matrix
SHA-256 identities**. A report label, arbitrary checkpoint path or public
reports-only handoff cannot substitute for these artifacts. No weights or
source are downloaded implicitly.

A separate source-bound evaluation request binds the original source,
environment, complete artifact inventory, final model identities, safe payload
and metadata identities, and new protocol/data/source closure. Original identity
checks run against the original producer, not an evolved checkout. Native safe
tensor decoding and slot validators remain authoritative; optimizer tensors are
validated as retained structure, never installed into an optimizer. Only native
model tensors are restored for inference. No pickle or unsafe loading fallback
is available. Source caches, executable shadows and unsafe paths fail admission. Git
replacement objects and ambient Git selectors cannot redirect the pinned
source checks.

Keep the original run read-only and quiescent. Before/after inventories detect
observed changes, not hostile concurrent-writer attacks. Do not overwrite,
reseal or repair original evidence. Use fresh external request and output paths;
keep machine-specific environments, logs and raw checkpoints private unless
separately authorized for publication.

## Mandatory two-seed replay barrier

Before **any** held-out model call, each fixed final checkpoint must reproduce
all 40 original training predictions, raw token for raw token, and all 40 must
be exact answer plus EOS. This is 80 prompt-only inference replays. Both seeds
must pass the barrier; a mismatch or missing input stops the attempt, retains
its failure and partial journals, and authorizes no repair, checkpoint election
or additional training. This is an inference admission check, not independent
learning evidence or optimizer replay.

Only then evaluate all 120 held-out prompts for each seed. Gold answers, oracle
results, family/sibling identifiers and contrast labels are not inference
inputs. The unchanged native CPU float32 greedy decoder receives only the
prompt and fixed cap64, including EOS. Do not force EOS, constrain the grammar,
repair output, supply a gold prefix, try a different prompt or retry failures.
Retain every generated token, including malformed and cap-ended outputs. Model
state, gradients, module modes and RNG boundaries are guarded.

## Units, metrics and controls

Per seed, the held-out inventory is:

| View | Rows | Quartets | Contrast edges |
| --- | ---: | ---: | ---: |
| Canonical primary | 40 | 10 | 40 |
| Pitch siblings | 40 | 10 | 40 |
| Tempo siblings | 40 | 10 | 40 |

There are **two held-out structural parents**, each rendered in five meters.
Pitch and tempo siblings share their parent/contrast lineage. Neither the 120
rows, repeated meter renderings nor the two seeds are independent subjects.
Do not pool them into a larger statistical sample or call the second seed an
independent replication. Report raw descriptive counts for each seed, with
canonical primary results separate from correlated siblings.

Primary metrics are exact full answer plus EOS, complete four-variant quartets,
and both-correct contrast edges. Also retain changed-output edges, EOS/cap
termination, output diversity, per-parent/meter/task/variant results and
canonical-to-pitch/tempo consistency. Consistently wrong outputs are not
successes; sensitivity without correct answers is not correct event counting.
Variant-only strata contain no complete quartet and report row metrics only.

The unchanged global, task/meter, header, length and header-plus-length majority
controls fit **unique canonical training rows only**, including their fixed
unknown-key fallback. The exact formal oracle is an offline scoring ceiling,
never a generation input or repair channel. Both retained positives and
negatives remain part of the evidence record.

## Frozen resources and failure boundary

The complete attempt is 80 training replay plus 240 held-out decodes:
**320 decodes and at most 20,480 native forwards**. EOS may reduce actual work;
equal caps do not establish equal compute. Preflight and freeze admission make
zero model forwards. The admission worker is limited to 600 seconds and the
inference worker to 900 seconds. Worker logs and journals are limited to 1 MiB
each; requests to 4 MiB; the complete evaluation attempt to 8 MiB. The original
run has a separate 128 MiB inventory bound, native checkpoints a 16 MiB bound,
and the generic safe reader an outer 32 MiB per-file ceiling. Payload checks are
not a whole-process RAM cap or a live disk quota.

On custody, source, loader, replay, model-state, timeout or decoding failure,
retain an incomplete/failed attempt and all available journals/logs. Do not
silently retry, extend a cap, substitute a checkpoint, exclude a failing seed,
or promote partial scores. Failure before the two-seed admission barrier means
held-out evaluation remains **unrun**, not zero-scoring.

## Owner-local workflow

Review and green CI for the exact evaluation commit precede separately explicit
owner-local inference authorization. Implementation and conformance do not
constitute that authorization. Use the hash-locked CPU environment and fixed
launcher `python -I -B scripts/run_frozen_music_eval.py`:

- `preflight`: regenerate the finite inventory and evaluation contract without
  loading target checkpoints or executing a model.
- `freeze`: supply `--original-source`, `--original-run`,
  `--original-request-identity`, `--original-matrix-identity` and a fresh external
  `--request`. Retain the new printed request identity independently.
- `run`: supply the original inputs again, frozen `--request`, its independent
  `--request-identity`, fresh `--output` and `--authorize-inference` after the
  separate owner authorization. This makes one bounded attempt.
- `audit`: use the request and its identity, output and independently retained
  `--expected-identity`. This is read-only source/input/artifact/score consistency,
  not fresh inference, optimizer replay, hardware attestation or independent-host
  replication. It checks the retained original-input binding structurally; it
  does not reopen the original tensor packet or attest that recorded inference
  occurred. An audit does not turn a failed attempt into a completed run.

Synthetic oracle outputs in tests are scoring conformance, not model outputs.
Run normal and optimized standard-library contracts, locked native synthetic
mechanics, affected historical suites, required aggregates, dependency checks,
documentation links and historical-byte preservation before publishing review.
No merge, tag, release or target inference is authorized by this document.
