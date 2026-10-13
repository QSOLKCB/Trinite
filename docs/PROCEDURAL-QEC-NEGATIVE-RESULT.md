# QEC procedural supervision v2 — retained negative learning result

**Observed 2026-10-13.** This is an additive report of the completed
[GitHub Actions four-cell research run](https://github.com/QSOLKCB/Trinite/actions/runs/38355170260)
on source commit `93522bf8f18fe38970ceafdd2ace792f8922c382`. The
original protocol, run, ZIP, histories, source receipts and model outputs
remain unchanged. This is training/development evidence, not held-out learning
adequacy or independent numerical replication.

## Native execution / preservation status

- Run 38355170260, attempt 1: all four 4,096-update dense CPU cells completed,
  full original native AdamW trajectories replayed, and six assurance checks
  passed in each cell. Source used Python 3.12.15 and CPU Torch 2.8.0+cpu.
- All four cells returned `training_gate_pass=false`, `learning_adequacy=false`,
  `held_out_scored=false`, `phase5_ready=false`, and selected no model/arm.
- Original full research ZIP SHA-256:
  `ac3b3627a27bcf6d43f89b31eeabaf1859949ae939784da06ba49916b495fdca`.
  Inventory SHA-256:
  `180e89a892508a630f45255e44805a7487e5b5542ce97b5f6298700fe37e3736`.
  Original summary:
  `a6d53d4adff35f806226333891a38dd472506752c3d22d70cc96c76e8e2e0a68`.
  Experiment request:
  `35dfbaafd2f85e5c6836bab1837bdc99c18a8b003ca7c46273f2f4d265ce37b1`.
  Original dataset:
  `120b95ac68825b50b92dc0210d1f142a30980bfe5acedcfafe6d8f3cdadf83c4`.
- [Download the original Actions packet](https://github.com/QSOLKCB/Trinite/actions/runs/38355170260/artifacts/11794010350).
  Its GitHub retention expires 11 January 2027 at 06:09:41 UTC; retain the
  original ZIP and externally held checksums permanently before then.
  A green job or Actions artifact is not a permanent public research archive.

## Original final-answer + EOS training accuracy

| Task | Seed 0 final-only | Seed 0 procedural | Seed 1 final-only | Seed 1 procedural |
| --- | ---: | ---: | ---: | ---: |
| Add | 63/66 | 66/66 | 64/66 | 64/66 |
| Subtract | 64/66 | 62/66 | 66/66 | 64/66 |
| Multiply | 64/66 | 64/66 | 66/66 | 61/66 |
| Divide | 64/66 | 65/66 | 66/66 | 55/66 |
| Fold-sum | 786/1150 | 702/1150 | 930/1150 | 607/1150 |
| GF(3) | 54/54 | 47/54 | 50/54 | 45/54 |

The preregistered 90% threshold requires at least 1035/1150 fold examples,
49/54 GF(3), and 60/66 per arithmetic task **in both procedural seeds**.
Neither arm passes fold sum; both procedural cells fail GF(3), and procedural
seed 1 also fails division. The procedural recipe does not demonstrate
improved final-answer learning. Arithmetic prompts include the new `R/`
namespace, and these figures cannot be pooled with historic unprefixed tasks.

## Confirmed scheduler defect and v3 repair boundary

At the v2 source, the auxiliary mode and row cursor both depend on `i // 2`:
prefix always selects even indexes, trace selects odd indexes. Exact
v2 replay reproduces this sampling bug. In 4096 updates, each auxiliary
mode sees only **575/1150** distinct fold rows and **27/54** distinct GF(3)
rows. The finite budget allows 1024 auxiliary visits per stream, so even
the corrected v3 scheduler can visit only 1024 distinct fold rows (126
unvisited) while covering all 54 GF(3) rows. The old bug is not sufficient
to explain the poor trace fitting on exposed rows. All old measurements
remain valid for the old flawed schedule; no labels or outcomes are repaired
retroactively.

V3 is a **different prospectively frozen producer** with separate
experiment/request/summary schemas and all source-bound evidence. The
original v2 release/checkpoint/archive cannot be edited to call it fixed.

## Historical teacher-forced diagnostic interpretation

V2 reported `final_train_loss_hex` over **all final/prefix/trace development
examples**, including auxiliary outputs the final-only control never learned.
That single scalar is a mixed-objective diagnostic, not final-answer loss.
Observed mixed values were final-only seed 0/1: 5.504531/6.159865, versus
procedural 0.583738/0.584695. The procedural arm fitted more trace-format
tokens while original answer+EOS accuracy was worse; do not claim
superiority or stability from those mixed losses.

The new v3 report names this diagnostic honestly and adds separate
final-only, per-mode and per-task teacher-forced loss measures. It also
distinguishes accuracy on actually visited and unvisited (but already
development-family) auxiliary rows. No v2 numeric field is rewritten.

## Additive source-closure supplement

Codex verified that the v2 inner research ZIP retained only the 23
**declared** source files, not the complete runtime dependency closure.
A standalone import failed on missing `trinite.foundations_data` despite
the archive transport checksum passing. This is a completeness problem,
not evidence of a failed original training replay.

The [supplement workflow](../.github/workflows/procedural-qec-source-supplement.yml)
preserves the **complete exact Git tracked source tree** for commit
`93522bf8f18fe38970ceafdd2ace792f8922c382`, pins the original
negative-result summary/request/archive identities, checks every file's
SHA-256 and size, and successfully regenerates the original formal dataset
from the extracted tree alone using isolated `python -I -S` with no Torch.
Supplement SHA-256:
`a033902d5155d7e3914cf3abf262dddd7a2fd190c922062a03fb52a0c81d8a7a`;
Git tree: `c8cc28e560729722539a2a6ee47408f50a7a802e`; 578
Git tracked members. The full-source supplement is roughly 394 MB as an
Actions transport artifact, and should be separately retained in permanent
storage alongside the original negative result. It is **additive**:
no original v2 bytes, signatures, digest or result packet is replaced.

[Source supplement job](https://github.com/QSOLKCB/Trinite/actions/runs/38365634300)
and [source supplement artifact](https://github.com/QSOLKCB/Trinite/actions/runs/38365634300/artifacts/11796249840).

The supplement demonstrates source/dataset reconstruction. It does not
establish a new independent-host native AdamW replay, mathematical
reasoning, fresh family generalization, format superiority, geometry,
Phase 5 readiness, or release acceptance.

## Required next comparison

After v3 merges, freeze a **new research request** and rerun both arms and
both seeds under the corrected producer. The final-only arm is required to
keep the original v2 schedule exactly. Do not compare a new seed/arm to
unbound historical control evidence or alter the original experiment. If
full-trace fitting improves but original complete final answers still
regress, the separate next hypothesis is non-displacing auxiliary
supervision with explicitly normalized final/aux losses. That is not a
part of v3 and must be frozen before outcomes.

Relevant invariants: TRI-I02, I03, I05, I07, I08, I11, I12, I13, I14.
