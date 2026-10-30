# Training-family rest/tie placement swap v1

**Prospective implementation, not a target result.** This new CPU-first lane
follows the [mixed/negative pitch-replacement result](MUSIC-PITCH-RESULT.md).
It leaves every historical source closure, fixture and numerical contract
unchanged. A separately authorized [CPU/CUDA device trial](MUSIC-DEVICE-TRIAL.md)
characterizes one control workload; it does not cross backend with this treatment.
Review, green exact-commit CI and explicit owner authorization precede target runs.

## Smallest controlled question

Does breaking the association between a training duration family and its fixed
rest/tie placement improve the existing structural contrast behavior? Keep
pitch 60, tempo 120, duration partitions, meter, labels and native answer/EOS
loss unchanged. This is a joint placement-template intervention, not separately
identified rest versus tie effects or an exhaustive position curriculum.

The original a/b layout templates are a: tie start 0, rest position 6; b: tie
start 4, rest position 2 (zero-based flattened eight-event positions). The
replacement swaps these two templates between the same a/b duration families.
Tie endpoints remain adjacent, and the rest never overlaps either endpoint.
Both exact rational labels and the independent integer-tick oracle must agree;
all variants retain `8 - rest_presence - tie_presence` attacks. No c/d duration
family or its layout is added to training. Regeneration checks family grouping,
no exact training/development prompt overlap, carrier round trips, prompt length,
answer/EOS masks and matched per-batch semantic/target exposure.

Four fresh cells run in fixed order control-17, swap-17, swap-29, control-29.
Both arms repeat the same 80-slot cycle: original canonical 40 followed by
original 40 (control) or swapped-layout 40 (treatment). Each semantic row has
128 visits. Treatment uses 64 visits per layout, but **10 no-rest/no-tie rows
have identical prompts in both layouts**. Thus control has 40 unique prompts;
treatment has **70, not 80**. Ten unchanged prompts receive 128 visits each;
60 distinct feature-present prompts receive 64 each. Canonical exposure halves
only for the 30 feature-present rows. Fixed original-then-swap block recency is
part of the intervention; a result does not isolate “diversity” as its mechanism.

## Fixed native CPU experiment

Use seeds 17 and 29, paired exact initial model identities and independent
model/optimizer states. The unchanged dense CPU float32 decoder remains width
32, one block, four heads, feed-forward 64, context 256. Keep native AdamW,
learning rate 0.001, clipping, answer/EOS loss, batch eight and 640 updates.
Each cell has 5,120 visits and 10,240 targets; four cells total **2,560 updates,
20,480 visits and 40,960 targets**. Common canonical-40 validation every 64
updates is observation, not selection. Initial pool loss can differ legitimately;
paired initial model identity, not loss equality, is the requirement.

Use only final step 640. Common a/b diagnostics have **80 layout slots / 70
unique prompts** per cell, original 40 plus swapped 40. Control's changed-layout
surfaces are unexposed; treatment's are trained. There is no accuracy fit gate:
all four safe completed endpoints and complete diagnostics must pass an integrity
barrier before development, but lost training fit remains an outcome. Any failed
execution, source, checkpoint, loader or model-state check retains an incomplete
study; never select a successful subset, retry, resume, repair or extend a cap.

Evaluate all 120 existing c/d rows per cell: 40 canonical, 40 pitch and 40 tempo.
These are previously inspected development probes, not a new blinded holdout.
There are **320 diagnostics + 480 development = 800 decodes**, greedy cap 64,
at most **51,200 greedy forwards**. Training/loss forwards are additional.
No gold prefix, oracle answer, metadata or output repair enters inference.

Report all per-seed treatment-minus-control canonical exact+EOS, complete
quartets and both-correct/changed-output edges, plus rest-only strata, training
fit and pitch/tempo diagnostics. Each development view has 40 rows, 10 quartets
and 40 edges. For original↔swapped training pairs, separate **30 changed-input
pairs** from **10 identical-input pairs**; do not inflate placement robustness
with trivial duplicate prompts. Raw equality, including consistently wrong
answers, is not correct transfer. Controls fit only the original canonical-40
training rows; the oracle is offline scoring only.

Only two structural development parents, c and d, exist; meter renderings,
siblings and reused seeds are not independent subjects. Pitch 72 is untrained
in both arms here. This does not test all rest/tie positions, arbitrary music,
arithmetic transfer, a neural mechanism or statistical significance. No accuracy
pass threshold, automatic promotion, checkpoint election or release claim is
introduced. `completed` denotes execution/artifact completion, not success.

## Resource and custody boundary

Each cell has a 120-second cumulative update/loss-validation ceiling and a
900-second whole-worker deadline; development has one 900-second worker.
The sum is 4,500 worker-seconds, not a whole-controller deadline. Logs/journals
are 1 MiB, requests 4 MiB, checkpoint/other artifacts 16 MiB, retained run
128 MiB. File bounds are not a process-RAM cap. No pickle or unsafe fallback.

Use clean committed exact source and the existing hash-locked CPU environment,
with fresh external non-link evidence paths. Invoke a trusted venv's absolute
lexical `bin/python` directly (ordinary executable symlinks are allowed), not a
resolved system interpreter. The launcher requires `-I -B`; its guard cannot
undo Python startup/site code already executed without isolation. Source checks
bind the effective worktree and actual blobs, reject caches/import shadows and
Git redirection, and freeze source/environment/admission/schedule identities.
Independently retain printed request and matrix identities before using them.

```sh
python -I -B scripts/run_music_placement.py preflight
python -I -B scripts/run_music_placement.py freeze --request "$REQUEST"
# Only with separate explicit owner authorization, exactly once:
python -I -B scripts/run_music_placement.py run \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --authorize-run
python -I -B scripts/run_music_placement.py audit \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --expected-identity "$MATRIX_ID"
```

Read actual help; all evidence paths are fresh absolute paths with existing
parents. Capture run and audit exit codes separately: no `&&`/errexit skipping
failure handling. Audit a retained incomplete matrix with its independently
printed identity; if interruption leaves no matrix identity, retain partial
artifacts without inventing or resealing one. Internal workers are not supported
ways to choose a cell. Full raw evidence stays local. Read-only audit checks
source-bound retained bytes, safe structure and scores, not fresh inference,
optimizer replay, hardware, timing or absence of other attempts.

Affected TRI-I01/02/03/05/06/07/08/11/12/13/14. Synthetic conformance is not a
production experiment. No old protocol, learning gate or evidence is superseded.
