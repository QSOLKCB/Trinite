# Music-only compact training and evaluation pilot v1

**Prospective implementation, not a training result.** This isolated checkout lane
extends the [symbolic music preparation](SYMBOLIC-MUSIC.md) with lossless compact
notation, a source-specific admission and a frozen CPU dense development pilot.
No model inference or target training is performed to implement this PR. Musical
tasks come first; E8, geometry losses and an arithmetic bridge remain off.
Historical source files, admissions, fixtures and negative results stay unchanged.
Affected requirements: TRI-I01/02/03/05/07/08/11/12/13/14. This separate admission
does not alter the Phase 1 validator or reclassify the old conformance receipt.

## Formal source and firewall

Only the exact original 160 generated monophonic formal scores are admitted;
no composition, recording, teacher output, donor text or external asset is used.
The prospective music admission independently regenerates the source-specific
rights record, exact formal evidence, parser/oracle checks and 960 task projections.
The commissioned generator software remains MPL-2.0; its license is not the
rights basis for data. This bounded formal-fact decision is not a legal guarantee
or blanket permission for future synthetic music. A changed source, plan, codec,
rights field or item requires a new admission/request identity.

The input API accepts exactly `{task, score}` with the old strict score fields.
It rejects hidden targets, family IDs, receipts and extra fields. Only the compact
prompt and answer text reach ByteTokenizer. IDs, parent lineage, manifests,
formal evidence and family/split fields remain audit metadata. Answer and EOS
positions alone carry loss. Training diagnostics use unique train rows; neither
validation nor test rows enter updates, loss diagnostics or model selection.

## Lossless notation

`M1;<numerator>/<denominator>;<comma groups>;<quarter-QPM>;<bar>/<bar>...`
is an ASCII score. Events use five characters: two base64url digits for pitch
(128 means rest), one each for reduced duration numerator minus one and denominator
minus one, then the decimal tie flag `tie_in + 2*tie_out`. Alphabet order is
`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_`. Bar delimiters,
notated meter, accent grouping, exact durations, rest/pitch, ties and tempo survive
roundtrip. An independently implemented parser reconstructs a raw score and the
integer-tick oracle validates it; writer/core-parser agreement alone is insufficient.

A prompt appends `?<task>=` and selects attacks, rests, ties, bar_boundaries,
group_onsets or duration. Counts and reduced fraction lists are canonical targets.
The existing native 256-token cap is unchanged: overlength rejects, never truncates.
The current 960 projections occupy 45–147 tokens including BOS/answer/EOS; prompts
occupy 42–116 bytes. The frozen 64-new-token decoder cap is independent of gold
answers and fits every prompt within context. Other valid large scores can be
losslessly encoded but explicitly fail native sequence admission.

## Frozen pilot and limits

This tiny fixed smoke/learnability budget cannot meaningfully refute the music-learning approach. Training fit is a prerequisite before interpreting held-out transfer.

Two fixed random initialization seeds, 17 and 29; CPU float32 dense native decoder,
context256, one block, width32, four heads, feed-forward64; AdamW LR0.001;
64 updates, batch8, 512 scheduled visits over 480 unique train projections;
32,768 target-token ceiling and120 update seconds per cell. Each cell has a
900-second whole-worker deadline covering initialization, scoring and writing.
No pretrained initialization, altered native updater, backend fallback or geometry.
The exact hash-locked native CPU environment is required and frozen in the request.

Eight parent families supply 80/40/40 scores (480/240/240 projections), not
960 independent problems. This is an exploratory small-data pilot, not powered
or fresh generalisation evidence. All families share generator motifs and have
been public software-test inputs. Meter/group-dependent boundaries, onsets and
duration are easy readout tasks; attacks/rests/ties require event structure.
Report those groups separately, not one inflated reasoning score.

Score every row for the untrained and final models, retaining all malformed,
wrong, EOS-missing and successful raw token outputs. Include train-only per-task
majority-answer and task/meter-mode baselines, whole-answer-plus-EOS and component counts by task,
meter and family, with training fit, validation and test separated. Fixed
exploratory criteria: each task train >=90%; each task heldout >=80% and >=10
percentage points above train-majority, in both seeds. These descriptive flags
never clear learning, confirmation, Phase5 or release gates. No adaptive reruns,
checkpoint election, answer repair or changing criteria after seeing outcomes.

## Workflow and evidence scope

`PYTHONPATH=src:scripts python -B scripts/run_music_learning.py preflight`
performs standard-library admission/roundtrip/mask checks without model execution.
`freeze --request /outside/checkout/request.json` records exact source, corpus,
admission, environment and protocol before outcomes; retain its printed identity
independently. Review, exact-commit green CI and explicit owner execution approval
must precede `run --request ... --request-identity sha256:... --output /fresh/run
--authorize-run`. Internal workers also require the authorization flag and pinned
controller directory. Run never overwrites an existing attempt. Keep original
failures and partial attempts; interruption is not permission to restart or select.

`audit` and `summarize` require the request identity plus independently retained
matrix identity (`--expected-identity`). They are read-only source/admission,
custody, safe tensor/checkpoint and raw-metric consistency checks, **not actual
inference replay, optimizer replay, hardware attestation or learning adequacy**.
No command claims numerical replay. Outputs must be fresh outside the source and
request paths with nonsymlink ancestors. Reads/writes are bounded and no-follow;
pre/post inventories detect observed mutation, not an atomic hostile-writer snapshot.
Use an exclusively owned quiescent artifact directory. Retain all logs and
failed/timed-out/interrupted/not-attempted cells; no success-only matrix filtering.
