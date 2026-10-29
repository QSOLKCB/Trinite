# Controlled pitch replacement study v1

**Prospective implementation, not a target result.** This additive, CPU-first
study follows the [negative frozen musical-family development result](FROZEN-MUSIC-EVAL-RESULT.md)
and preserves the [positive exposed-training fit](MUSIC-EXPOSURE-RESULT.md).
No historical producer, fixture, admission or numerical primitive changes.
Affected TRI-I01/02/03/05/06/07/08/11/12/13/14. Target training and inference require
separate owner authorization after review and green exact-commit CI.

## Question and single intervention

Does replacing half of the canonical training visits with label-preserving
uniform pitch-72 renderings improve the prescribed rest/tie contrast behavior
and pitch consistency on the existing development parents, without losing
canonical training fit? Rest placement, valid ties, duration partitions, meter,
tempo, answers and loss masks are not separately changed. Rest-only failure is
a diagnostic stratum, not a second intervention.

Use both fresh native seeds 17 and 29 in both arms; no imported checkpoints,
pretraining, best-seed selection, intermediate checkpoint election or resume.
The deterministic cell order is control-17, pitch-17, pitch-29, control-29.
Within each seed the initial model identity must match across arms, while model
and optimizer state are independently constructed. Seed reuse is not independent
replication. The new control is evaluated descriptively; historical reproduction
is not presumed from a shared seed.

| Property | Control | Pitch replacement |
| --- | --- | --- |
| Training parents | a, b | a, b |
| Ordered 80-row cycle | canonical 40 at pitch 60, same 40 at pitch 60 | canonical 40 at pitch 60, same structures at pitch 72 |
| Distinct prompt surfaces | 40 | 80 |
| Visits per structural row | 128 at pitch 60 | 64 at pitch 60 + 64 at pitch 72 |
| Total visits / cell | 5,120 | 5,120 |
| Answer/EOS targets / cell | 10,240 | 10,240 |

Both pitch values use the existing fixed-width compact codec. Uniformly recolor
all non-rest notes, including both ends of a tie; rests remain rests. Regenerate
and cross-check the rational labels and independent integer-tick oracle. Exact
prompt lengths, semantic row order, target counts, rest/tie distribution and
native answer-plus-EOS loss are matched between arms at every scheduled visit.

This is a **replacement allocation**, not extra training exposure and not equal
unique surface data. Canonical exposure is halved in the treatment. The fixed
60-then-72 blocks end with pitch 72, so exposure allocation and block recency
are part of the intervention. The design does not isolate a mechanism called
pitch invariance or prove that diversity alone caused an effect.

## Fixed native training and measurement

The unchanged dense CPU float32 model has width 32, one block, four heads,
feed-forward width 64 and context 256. Preserve the native AdamW settings,
learning rate 0.001, clipping, answer/EOS loss and 640 updates with batch size 8.
There are four cells: 2,560 updates, 20,480 structural visits and 40,960 target
tokens in total. Common canonical training rows are the validation/loss
measurement inventory every 64 updates, never a selection criterion. Initial
loss uses each arm's 80-row pool and may differ legitimately; paired initial
model identity, not initial loss equality, is required.

Only the final step-640 checkpoint is used. Final diagnostics cover the same
80 a/b rows for every cell: 40 canonical and 40 pitch-72 rows. The latter are
unexposed nuisance probes in the control, but exposed training surfaces in the
treatment. Do not call both arms' diagnostic inventory 80 unique training rows.

All four cells must complete with valid request/source/checkpoint bindings,
matching paired initialization identities, and complete diagnostic journals
before **any** development decode. Training accuracy is deliberately **not**
a gate in this new protocol: a treatment that damages fit remains a result,
not an excluded cell. Any execution, custody, loader or model-state failure
stops development and retains an incomplete matrix. Never repair, retry,
substitute a checkpoint, extend a cap, or select a successful subset.

After that integrity barrier, evaluate all 120 existing c/d rows for each cell:
40 canonical, 40 pitch and 40 tempo. All four final endpoints are retained.
There are 320 diagnostic plus 480 development decodes, **800 total**, each with
native prompt-only greedy cap 64: at most **51,200 greedy forwards**. Training
and loss-evaluation forwards are additional work, not included in this decoder
ceiling. No gold prefix, oracle output, labels or contrast metadata enter the
inference input; no grammar constraint, forced EOS or output repair is allowed.

## Frozen interpretation

Report every cell and paired treatment-minus-control differences for each seed;
do not pool seeds as independent subjects or select the better result.
Canonical development exact answer plus EOS, complete quartets and both-correct
contrast edges are primary descriptive outcomes. Each view has 40 rows,
10 quartets and 40 edges. Keep pitch and tempo siblings separate. Also report
changed-output edges, EOS/cap terminations, raw output diversity, parent/meter/
variant strata, rest-only performance, and canonical-to-pitch/tempo pairs.
Identical wrong outputs are consistency, not success; changed outputs alone
are not correct sensitivity. Report canonical training-fit losses alongside
any development gain.

Global, task/meter, header, length and header-plus-length majority controls fit
only the original 40 unique canonical training rows, with their fixed unknown-key
fallback. The exact formal oracle is an offline scoring ceiling, never a model
input. No accuracy pass threshold, statistical significance rule, automatic
promotion or follow-on experiment is introduced. `completed` denotes execution
and artifact completion, not that the intervention worked. Zero, negative,
mixed and fit-degrading paired differences are all retained without suppression.

An ancillary **historical control reproducibility diagnostic** compares each
new control's development outputs against the immutable PR95 journal, restricted
to exact matching example identities, seed and final step. Report raw equality
counts separately for canonical, pitch and tempo views. This is offline
comparison only: no old output enters a training or inference input, no equality
gate is imposed, and mismatches neither suppress results nor trigger a retry or
checkpoint election. Environment differences can explain divergence; it is not
automatically a software defect, a causal finding or independent replication.
The contemporaneous paired comparison remains primary.

Only **two structural development parents**, c and d, are present. Their prior
outputs were inspected in PR95, so these are exposed development probes, not a
fresh blinded confirmation set. Pitch 72 is explicitly trained in the treatment:
c/d at pitch 72 measures transfer to different structural parents at a supported
pitch, **not unseen-pitch generalization**. Neither two pitches nor five meters
establish arbitrary pitch, arbitrary-length counting or general music. The
finite shortcut `8 - rest_presence - tie_presence` remains sufficient. This
study cannot establish arithmetic/music-to-mathematics transfer, a causal neural
mechanism, self-correction, QEC benefit, RSI or release readiness. No extra
parents, pitches, curriculum factors or arithmetic probes are added.

## Resource, custody and execution boundary

Each training cell has a cumulative 120-second update ceiling (including native
loss validation) and a separate 900-second whole-worker deadline. Development
has one 900-second worker deadline. Four training ceilings plus development sum
to 4,500 worker-seconds; this is **not** a wall-clock deadline for the controller,
admission and audits. Logs and journals are bounded at 1 MiB, requests at 4 MiB,
checkpoint payloads and other artifacts at 16 MiB, and retained run bytes at
128 MiB. Byte bounds are
not a whole-process RAM cap. Safe native tensor loading has no pickle fallback.

Use a clean committed exact checkout, a compatible hash-locked CPU environment,
and fresh external non-link request/output paths. The trusted virtualenv Python
executable may be its ordinary symlink; invoke that venv path directly rather
than resolving it into a different system environment. Keep original evidence
quiescent. Source checks pin the effective Git worktree, disable replacement-object and
ambient Git-selector redirection, and reject tracked-byte tampering, runtime caches
and import shadows before native imports. A changed source, protocol or
numerical environment needs a new request, not resealed historical evidence.

The entrypoint requires `python -I -B scripts/run_music_pitch_intervention.py`.
A script guard cannot undo Python startup/site code already run without `-I`.
Read the actual command help before use. Freeze the printed request identity
independently before outcomes. A separately authorized single `run` must retain
all attempts and failures; no automatic rerun follows an unsuccessful call.
Read-only `audit` checks source/request/retained bytes, safe structure and score
consistency. It does not execute fresh inference or replay optimizer updates,
attest hardware, prove absence of other attempts or turn an incomplete run into
a completed one. Full inference and optimizer replay are not claims of this
lane's audit. Preserve the full owner-local packet; reports-only exports cannot
substitute for it.

## Owner-local commands (only after separate authorization)

Use the same clean committed checkout and compatible CPU environment throughout.
`REQUEST` and `OUTPUT` below are fresh absolute external non-link paths; their
parent directories must already exist. Preserve each printed identity separately.
Preflight and freeze perform no model forward or optimizer construction.

```sh
python -I -B scripts/run_music_pitch_intervention.py preflight
python -I -B scripts/run_music_pitch_intervention.py freeze --request "$REQUEST"
# Only after explicit owner authorization, using the independently retained identity:
python -I -B scripts/run_music_pitch_intervention.py run \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --authorize-run
# Read-only; use the independently retained printed matrix identity:
python -I -B scripts/run_music_pitch_intervention.py audit \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --expected-identity "$MATRIX_ID"
```

Capture run and audit exit codes separately. Do not use `&&` or shell `errexit`
to skip failure handling. If a failed run retains a matrix and its printed
identity, audit that incomplete attempt without rerunning it. If an interruption
prevents a matrix/identity from being produced, retain the partial directory and
logs as incomplete rather than inventing or resealing a report. Internal worker
actions are controller plumbing, not supported ways to select a cell or bypass
the four-cell study. Never use them for a separate target invocation.

Normal and optimized contracts, native synthetic mechanics, adversarial source/
artifact boundaries, affected historical suites and required CPU aggregates
must pass their applicable gates. Synthetic outputs are software conformance,
not target study evidence. Implementation publication authorizes neither a
training run nor merge, readiness transition, tag or release.
