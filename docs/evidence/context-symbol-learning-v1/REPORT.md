# Context-symbol learning v1: stopped after the final fit checkpoint

The owner-approved run executed **128 native optimizer updates / 2,432 supervised
tokens** on the fixed eight-row training-only fit set. Its step-128 checkpoint
generated **8/8 exact canonical answers including EOS**, up from 0/8 at step 0.
The mandatory independent validation subprocess then rejected its runtime before
restoring the checkpoint. The controller stopped: **main seeds 17/29 were not
attempted, no heldout predictions were committed and no heldout labels/scores
were exposed**. No retry, resume, source edit or budget expansion occurred.

This is a stopped study with a tiny-pool fit observation. It is not a completed
main experiment, an unseen-generalization result, or evidence of improvement over
the declared baselines. The post-run read-only audit below does not replace the
failed mandatory gate or authorize main execution.

## Producer, approval and immutable identities

| Identity | Value |
| --- | --- |
| Trained producer commit | `da53aa21c7e3ee22787394596aa7ead75b0f080e` |
| Frozen manifest SHA-256 | `eae9d110caa7503a48e2f7b9ea80e815a42b8e37a5fee1f168902eccda588cdb` |
| Preparation anchor SHA-256 | `8c40c751f4d33ab801e1a4acabe8abcdbb4f0bba10377620983f25b1260d13c0` |
| Feasibility receipt SHA-256 | `6983b720b21597c4fc7d4ea38e0a79caa65b6ae55d347731e769fd34cc5ae41d` |
| Producer source-closure ZIP SHA-256 | `35515c6626de991d94a3ff8075d5f2c6d571c9c59ebc7d1c3737ffb3d8f1309b` |
| Owner declaration SHA-256 | `e7d69e6d5abfc08d8ad9383aae81e8626e7777f496bfc0776fc4b0db37e83f6f` |
| Sealed final run-anchor SHA-256 | `5e3a38c0a89cfcddc7fdae5ed849b9878407420501b3e15d9e37964f377a6cfc` |
| Fit tensor SHA-256 | `d6e219e707c983362c30826933cdd2ec7aea164d86993ab49338995a9af04e52` |
| Fit metadata SHA-256 | `06aec19368644ac44329d8a376740defa744a071b2119a91186ce186f192a83e` |

The **evidence commit is the Git commit containing this deposit**, separately
reported on its PR and in the publication response. It is not the trained
producer SHA. The producer is the actual merged #121 commit, with all 78
main-push CI workflows verified green before execution. The canonical owner
declaration records the explicit approval of this exact producer/freeze/schedule;
it contains no raw private chat, message IDs or transcripts. Such a declaration
does not cryptographically prove human authorship or chronology.

`payload.zip` retains the frozen protocol, data/admission, all dependency and
runtime identities, 215-file source archive, feasibility, native oracle binary,
approval, exclusive execution claim, all run files, checkpoint/model/AdamW
tensors, raw fit outputs, full planned heldout inventory, logs, resource receipts
and read-only verification records. It preserves the empty failed-validation
directory as well as file bytes. `manifest.json` hashes the payload, companion
files and all 86 payload files; its own bytes are bound by the published evidence
commit. The producer's own independently bound inventories remain
unchanged. A local `manifest.json` is not a substitute for checking the exact
published evidence Git commit.

## Frozen schedule and actual execution

CPU float32 dense decoder; **24,768 unique parameters**, context 256, byte
vocabulary 259, one block/width 32/four heads/FF64, tied embeddings/output.
Unchanged native AdamW, LR .001, batch 8, betas .9/.999, epsilon 1e-8, decay .01,
clip norm 1; geometry loss zero. Answer bytes and EOS only receive shifted loss.
The fresh table-family protocol has 648 rows (360/144/144), eighteen table orbits,
maximum 98 complete native tokens and zero truncation. Exact and declared
semantic overlap checks exclude both #119 and the recorded exposed development
families. See the retained [historical protocol](../../CONTEXT-SYMBOL-LEARNING-V1.md)
and `frozen/protocol.json` inside the payload; its implementation-time “unrun”
status is preserved, while this report records the later execution.

| Planned cell | Planned updates / targets | Actual updates / targets | Outcome |
| --- | --- | --- | --- |
| Fit seed 17, final step 128 | 128 / 2,432 | **128 / 2,432** | Native worker completed; 8/8 exact+EOS; mandatory independent subprocess failed |
| Main seed 17, final step 256 | 256 / 4,900 | 0 / 0 | Not attempted |
| Main seed 29, final step 256 | 256 / 4,900 | 0 / 0 | Not attempted |
| Total reserved ceiling | **640 / 12,232** | **128 / 2,432** | Freeze consumed; no retry/resume |

The fit uses eight distinct consistent training prefixes spanning all three
classes, repeated for 1,024 visits. It measures optimization on that fixed tiny
pool, not fit on all 360 training rows or transfer to heldout families. Native
train-target mean NLL fell from 5.540152 to 0.432709. These are answer/EOS-token
losses, not all-prompt accuracy. Exact decoding at the final checkpoint, not
teacher-forced loss or early success, supplied the worker's 8/8 observation.

## Why the required gate failed

The retained `run/fit-validation-worker.json` contains exit code 2 and:

```text
ContractError: training requires the pinned CPU backend, one thread, and strict determinism
```

Source inspection finds that the fresh `_validate-fit` CLI branch invokes the
guarded restore API without first calling `numerical.configure()`. Thread
environment variables are inherited, but Torch's process-local deterministic
algorithm setting is not configured in that fresh interpreter. A read-only
fresh-process probe finds **only `deterministic=false` differs from the frozen
environment**, and observes the same guard error. The failed CLI was not retried.

The original strict guard remains intact. The producer checkout and run bytes
were not patched or resealed. A separately recorded **post-run read-only audit**
uses the declared frozen configuration, authoritative closed safetensors reader
and native greedy decoder to restore the retained checkpoint and regenerate its
eight training answers. It confirms 8/8 including EOS, zero additional optimizer
updates and unchanged model/run bytes. This verifies the retained checkpoint's
training-only inference; it does **not** turn the failed mandatory subprocess
into a passed lifecycle gate, and no main seeds were subsequently run.

The primary frozen verifier checks source/custody/data/schedule/approval and the
stopped summary. Its code skips checkpoint reconstruction for cells whose
overall status is `failure`; therefore its generic scope string alone must not
be read as a failed-cell tensor audit. The separate native post-run audit is the
evidence for that narrower checkpoint assertion.

## Full planned denominators and comparisons

| Method | Validation planned / valid outputs | Test planned / valid outputs | Scores |
| --- | --- | --- | --- |
| Learned main 17 | 144 / 0 | 144 / 0 | Unscored; not attempted |
| Learned main 29 | 144 / 0 | 144 / 0 | Unscored; not attempted |
| Training-only majority | 144 / 0 | 144 / 0 | Constant `0` fitted on all 360 train rows; heldout inference not attempted |
| Context-ignoring shortcut | 144 / 0 | 144 / 0 | Not attempted |
| Exact native oracle reference | 144 / 0 | 144 / 0 | No heldout privileged calls; reference unscored |

All **1,152 planned model/baseline cells** remain in `run/predictions.json` with
`not_attempted` status. There are no malformed heldout model outputs because
there were no heldout outputs. Full-denominator status/coverage counts are
known; exact accuracy, per-class recall, macro-F1, family accuracy and
context-switch correctness remain **null**, not invented zero errors or scores.
The native summary also retains the oracle reference's planned denominators.
No heldout target file, committed-prediction file or external prediction barrier
exists. Read-only replay must preserve that absence and not disclose labels.

No baseline comparison supports an improvement claim here. The majority uses
training labels only; the shortcut deliberately ignores context masks. The exact
oracle is a correctness reference with task rules, not a learned competitor.
Learner updates, a constant and finite-rule evaluation have unequal compute.
Two declared seeds and correlated public table variants would not establish
statistical significance even in a completed study.

## Timing, resources and retained failures

The immutable raw receipts retain hex timings, sampled process-tree RSS,
wait4 high-water/CPU measurements, bounded stdout/stderr and cleanup information.
All limits remain 120 s native-update phase / 900 s full attempt / 2,700 s total;
1 GiB per-process RSS / 2 GiB aggregate observed RSS / 4 GiB hard per-process AS.
The update phase took approximately 2.517 s; the whole stopped study approximately
7.452 s. Whole-study observed peaks were 617.19 MiB process / 639.24 MiB aggregate.
No budget violation is recorded; the validation guard caused the stop.

The outer supervisor has no progress path and reports `observed_updates=0`;
that is not the numerical-update total. The inner fit supervisor observes
**128 / 2,432**, agreeing with checkpoint step, cursor, history and AdamW slots.
The actual numerical total is 128. RSS sampling cannot rule out transient
aggregate peaks and is not host-wide memory or hardware attestation.

The earlier failed zero-update preparation on producer
`4b31e17f3bf943c6dae3cd6b84f490a262ea6a97` is included unchanged under
`failed-preparation/`: 635 public-observation receipts, ESRCH failure logs,
original source archive and original failed-preparation inventory/anchor.
It performed zero optimizer updates and produced no valid freeze. Its retained
anchor is `0859f296e4b49359c8bc866a561f3474d335a4b962fef1d6a63a3a2d13831cf3`.
The separately authorized fresh preparation and the one approved scientific run
are distinguished from that failure. No failure, checkpoint or partial directory
has been overwritten or replaced by a later success.

## Read-only replay

Check out the exact **published evidence commit** first. From the repository root:

```sh
python3 -B docs/evidence/context-symbol-learning-v1/verify.py
```

This Linux/Python 3.11+ standard-library path validates the deposited hashes,
both source snapshots/failed-preparation anchors, safely extracts the frozen
producer into temporary storage and invokes its **`verify --scores-only`**.
It regenerates/checks data and recomputes the stopped denominators/status/null
summary without training, native model loading, network or heldout disclosure.
The expected receipt says `verified=true`, `complete=false`, `heldout_scored=false`.
It checks checkpoint **bytes** via the external run anchor, not tensor semantics.

For matching-environment native safe-load and training-only fit inference:

```sh
"$PINNED_CPU_PYTHON" -B docs/evidence/context-symbol-learning-v1/verify.py --native
```

Use the exact producer CPU dependency environment, interpreter/native library
hashes and runtime specified in the payload. This mode also invokes the retained
post-run audit; expected `fit_exact=8`, `fit_denominator=8`, zero verification
optimizer updates, and mandatory lifecycle gate still failed. Environment
mismatches reject rather than substitute a backend. The included original
context-oracle binary is hash-checked before use. Temporary copies are discarded;
the deposit and sealed run stay byte-identical. The verifier exposes no training
or resume option. Exact original CLI arguments and exit codes are retained in
`preparation-commands.json` and `run-command.json`; all fixed source entrypoints
are in the retained closure ZIP.

Custody, source/input binding, safe tensor structure and final inference are
**not independent optimizer replay, independent-host optimizer replication or
proof of operator blindness**. The post-run audit cannot repair chronology or
the mandatory gate. Public construction and a tiny repeated training pool do
not establish unseen reasoning, context offload, compression or learning gains.

Publication contains original/public commissioned formal facts, the resulting
native weights, authorized source software and verification evidence. Software
licensing and numeric-data admission remain separate. No credentials, raw private
chat metadata/transcripts, uploaded Jung book or image is included; Jung remains
the existing bibliographic inspiration only. Source software retains MPL-2.0;
see [COPYING](COPYING) and the repository [LICENSE](../../../LICENSE).

This deposit is for independent review. Do not merge or self-approve it.
