# Prospective attacks-only musical contrast study v1

**Preparation and conformance only. No target experiment has been run.** This
new checkout-only lane follows preservation of the frozen arithmetic probe's
negative exploratory result from [PR88](https://github.com/QSOLKCB/Trinite/pull/88).
It does not replace that evidence, the positive
one/four exposed-example fit, or any earlier failure. Requirements affected:
TRI-I01/02/03/05/07/08/11/12/13/14. No installed API or historical source changes.

## Question, not an assumed cure

Can the unchanged small CPU decoder fit musical examples whose answer cannot be
read from their meter, prompt length, or answer position, and then respond to
new event arrangements? This is a new attacks-only curriculum feasibility study,
not an ablation proving that contrasts improve learning. Fresh initialization is
used; it is **not music-pretraining transfer**, and there is no numerical bridge,
pretrained teacher, audio feature, E8 geometry, new loss weighting or model growth.
Counting attacks from the event markers is the intended task, not cheating.
Every score has eight events and at most one rest and one tie: the simple rule
8 minus rest-presence minus tie-presence solves this finite task. Success would
not establish general counting over arbitrary lengths or richer musical structure.

The earlier tiny fit did not establish general musical understanding. The frozen
arithmetic probe did not demonstrate arithmetic transfer. Its unfamiliar notation
and absence of a numerical bridge prevent attributing failure to music itself.
This experiment stays with music before attempting that bridge.

## Original finite corpus and structural separation

Four structural parents use distinct duration-weight partitions
(1,1,2,4), (1,2,2,3), (1,1,3,3) and (2,2,2,2) in each of two complete bars, with eight events per score. Each parent is
instantiated in 3/4, 4/4, 5/8, 7/8 and 8/16. Durations are exact rational shares
of each bar. Two entire parents are training; two entirely different rhythm
patterns are held out. Removing pitch, tempo and header information still leaves
disjoint parent rhythm structures. This is only **two held-out structural
families**, not 40 independent generalization problems.

Every parent/meter group has four versions: note versus rest at one position,
and a valid tie versus reattack at a disjoint adjacent pair. The tie edit changes
both linked flags and retains matching pitch. All other fields are identical.
Answers are 8,7,7,6 attacks. Each quartet has exactly equal prompt length, header
and answer-start position, using the unchanged fixed-width compact music codec.
Parent/group/variant IDs, split and labels never enter the model prompt.

The training set has40 canonical rows; the held-out set has40 canonical rows.
Each held-out row also has two separate frozen invariant siblings: pitch60 to72,
and tempo120 to180. These maintain encoded lengths; no combined transformation
or augmentation sweep is performed. Thus held-out evaluation covers120 rows.
All meter and invariant versions remain in their parent's split. The complete
160-row inventory is regenerated and independently verified at admission.

Exact rational/core labels and the independent integer-tick oracle must agree;
the independent compact parser round-trips every field. The original native and
music admissions remain unchanged: this lane admits only its exact finite new
corpus. No donor compositions, recordings, teacher answers or third-party assets
are imported. Optional audio rendering is not part of this experiment.

## Frozen budget and interpretation boundary

Two fresh native dense seeds17/29, CPU float32, context256, one block, width32,
four heads, feed-forward64, AdamW LR0.001. Per seed:

- 64 updates, batch8:512 scheduled visits,1024 answer/EOS targets.
- 40 unique training rows:32 rows receive13 visits,8 receive12 visits.
- 34,812 actual shifted live input tokens;69,624 across the two seeds.
- The schedule cycles complete four-variant groups. Aggregate target answers are
  128 sixes,256 sevens,128 eights, so majority-seven achieves50% raw accuracy.
- Initial and final train-only raw greedy diagnostics, cap64. Only final step64
  determines fit; no checkpoint election, early success promotion or rerun.

This is **much less per-row exposure** than the previous four-example diagnostic
(128 visits per row). It also has a different task mix and target-token count.
Equal64-update budgets are not equal compute or a causal comparison. Failure is
failure to fit this declared curriculum at this cap, not proof of a capacity
limit, a broken model or inability to learn music. More training remains a later
hypothesis, never an automatic retry.

Per-seed cumulative update ceiling120seconds, each subprocess900seconds,
1MiB worker logs,16MiB per artifact,128MiB total retained payload.
The aggregate payload ceiling is checked after writing and during audit; it is
not a live filesystem quota. Fixed checkpoint sizes are much smaller, but an
oversized failed attempt is retained rather than deleted to manufacture success. At most four
workers (two training, two conditional inference) give aggregate ceilings
240 update seconds and3600 worker seconds. These are ceilings, not predicted
runtime or whole-process RAM caps. Initial/final train evaluation gives160
independent decodes across both seeds; gated held-out evaluation adds240. Total
maximum400 decodes ×64 generated tokens =25,600 decode forwards. Training and
teacher-forced loss forwards are separate from this decode bound.

## Both-seed gate and final-only held-out policy

Run both prescribed training cells. Require **all40 canonical training rows exact
answer plus EOS in both seeds** before either seed performs any held-out model
call. A failed, interrupted, timed-out or nonfitting cell leaves held-out stages
explicitly unrun. Do not select the better seed. Corpus regeneration for admission
may inspect all formal labels; it must not feed held-out data to updates or
training diagnostics. Native `validation` is an alias for train data, not a
validation split. There is no model-visible validation set or tuning loop.

After both pass, evaluate exactly the two final checkpoints on their frozen
120-row held-out inventory. No training, gold prefix, solver answer, forced EOS,
answer repair, prompt replacement or extra seed is allowed. Retain every raw
prediction, including cap failures and partial journals after errors.

Primary descriptive outcomes are complete four-version groups and contrast
edges with **both answers correct**, alongside raw exact+EOS. Output changes
alone do not establish correct sensitivity. Invariant siblings report identical
raw outputs separately from all-three-correct: a constant wrong answer can be
perfectly invariant. Report canonical rows, each sibling stratum and every meter
separately. A count/EOS success is not general reasoning or arithmetic competence.

The post-generation exact formal oracle is a scoring ceiling, never inference
assistance. Global, task/meter, header, length and header+length lookup majorities
are fitted only on unique TRAIN rows, with explicit global-majority fallback for
unknown keys. They cannot solve a complete contrast quartet; each gives50% raw
accuracy here. Comparisons are descriptive, not statistical proof from two seeds.
There is no automatic promotion or later-stage unlock even if all results are
positive. Preserve the matrix, then separately plan a new experiment.

## Commands and custody

Use a clean reviewed exact commit with green CI and the hash-locked CPU
environment. Commands use `python -I -B scripts/run_music_contrast.py`:
The isolated launcher rejects import caches, extension shadows and links before
project imports; ambient PYTHONPATH is ignored. Tests still use explicit
PYTHONPATH as shown below.

- `preflight`: regenerate finite corpus, validate controls and budgets; no model.
- `freeze --request /fresh/outside/request.json`: source/environment/protocol and
  admission-bound request; independently retain the printed identity.
- `run --request ... --request-identity sha256:... --output /fresh/run --authorize-run`:
  requires separate owner-local execution authorization after review/merge/CI.
- `audit --request ... --request-identity sha256:... --output /run --expected-identity sha256:...`:
  read-only source/retained-score consistency and safe tensor structure, not
  numerical inference or optimizer replay. Retain the run identity independently.

Exclusive paths outside the source tree, no symlink ancestors, no overwrites or
resealing historical evidence. Preserve failures and not-attempted stages. A
successful structural audit is not successful learning. Private environments,
paths, checkpoints and logs are not automatically approved for public upload.
Reports-only handoffs remain an incomplete subset for independent custody audit.

## Conformance

Run new standard-library contracts normally and under `-O`, pinned native
conformance, upstream and relevant historical suites. Fake oracle positives test
the scorer only, not the target model. Small synthetic update tests must not run
the target curriculum matrix. Verify no historical tracked bytes were changed.

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_contrast_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_contrast_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_contrast_native -v
```
