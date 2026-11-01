# Context-symbol learning v1: stopped at the seven-of-eight fit gate

The one newly owner-approved execution on producer `578d4793a02585a0bf8b86660c4fd99720db118c`
completed **128 native optimizer updates / 2,432 supervised tokens**, then failed
its mandatory final training-fit gate: **7/8 exact canonical answers including
EOS**, versus 0/8 at initialization. The controller stopped without retry.
**Both main seeds remained unattempted; no heldout predictions, labels or scores
were produced.** The independent validation subprocess repaired by PR #123 was
not reached, because the worker's own final-fit check rejected first.

This is a negative tiny-training-pool result, not completed main learning,
heldout generalization, baseline improvement, optimizer replay or a newly passed
research gate. The [previous eight-of-eight stopped deposit](../context-symbol-learning-v1/REPORT.md)
and its original source, failed validation, fixtures and evidence remain intact.

## Identities and authorization

| Identity | Value |
| --- | --- |
| Trained producer commit | `578d4793a02585a0bf8b86660c4fd99720db118c` |
| Producer tree | `215637e915c30e8fa54374f389e15be3f3a367e9` |
| Freeze SHA-256 | `f6cb7b0e254291bfdf7c2539ab9db5a3bd54ef5ae0fe696d125a132e9b2ee069` |
| Preparation anchor SHA-256 | `a1224ca2dcce2695ba6b1f89a1c1a5808db332289533bd1d0336d9e1b42fd1ef` |
| Supervised feasibility SHA-256 | `2f69ae751225393b3d998c125e960a6eadc67b19b98dca3eac3d037e70e6f9f0` |
| Source closure ZIP SHA-256 | `df817cadfc41efb7aa538cde8897f5327065645c4638321005353aaa5605e6ac` |
| Owner declaration SHA-256 | `b245a4517696a09f2fe5c85f5c4fceb7f75d6e00a35a8ea00dfccb34a6b7466a` |
| Sealed run-anchor SHA-256 | `c4ded51e926fd8857ee9c1ae8a9ab7fe7a802dcf23cd03fc509e86b1abb256ed` |
| Final fit tensor SHA-256 | `5db0ff9697959d4f8dadc4c8353f2bfbdcfb4d3e00e11386bfdf7a226e21ae8c` |
| Final metadata SHA-256 | `1ae4dc91ee9e65d1d26b4f1dac57f9a2600bbe4e34fb77b70d3a7753277d2060` |

The producer is the actual owner-merged #123 commit. **79/79 actual main-push
workflows were verified successful before preparation**, rather than relying on
PR-head CI. One fresh zero-update feasibility/freeze succeeded. The owner then
explicitly approved that full producer SHA, freeze hash and exact bounded
schedule. The canonical declaration and exclusive execution claim are retained;
the earlier approval was not reused. These operator declarations do not prove
human authorship or chronology cryptographically. No private transcript, message
identifier or raw private-chat metadata is published.

The **evidence commit is the separate Git commit containing this deposit**, named
in the PR/publication response, not the trained producer. Check that published
commit before trusting local manifest bytes. `manifest.json` binds the payload
and companion files; the independently retained preparation/run anchors continue
to bind original producer evidence. No original anchor or source archive was
rewritten to accommodate this deposit.

## Frozen plan and actual execution

CPU float32 native dense decoder; 24,768 unique parameters, context 256, byte
vocabulary 259, one block, width 32, four heads, FF64, tied embeddings/output.
AdamW LR .001, batch 8, betas .9/.999, epsilon 1e-8, decay .01, clip norm 1;
single-tensor update path and geometry loss zero. Native shifted loss receives
only answer bytes and EOS. Generation is unrestricted greedy, three-token
maximum, mandatory EOS, without repair, constrained decoding or fallback.

| Cell | Planned updates / tokens | Actual updates / tokens | Outcome |
| --- | --- | --- | --- |
| Fit seed 17 | 128 / 2,432 | **128 / 2,432** | Final 7/8; failed mandatory 8/8 gate |
| Main seed 17 | 256 / 4,900 | 0 / 0 | Not attempted |
| Main seed 29 | 256 / 4,900 | 0 / 0 | Not attempted |
| Total ceiling | **640 / 12,232** | **128 / 2,432** | Consumed authorization; no retry or resume |

Fit uses eight distinct training prefixes, repeated for 1,024 visits. The final
checkpoint was serialized, safely reloaded and decoded before the worker
rejected `final fit below 8/8; stop dependent stages`. Every generated final fit
answer had valid canonical syntax and EOS, but row **`c0631` expected `1` and
produced `-1`**. The other seven were exact. Train-target mean NLL moved from
5.540152 to approximately 0.441807. This is answer/EOS-token loss, not prompt
accuracy or performance on all 360 unique training rows.

No independent validation certificate exists. Its subprocess was never launched;
this run neither demonstrates nor disproves successful passage through PR #123's
repaired validation lifecycle. A later read-only checkpoint audit regenerated
the same 7/8 using the authoritative safe tensor reader and greedy decoder,
observed zero optimizer/update calls, and preserved all model/run bytes. That
audit is not a rerun, resume or replacement for the failed fit gate.

## Exposure, differences from the previous run and uncertainty

The frozen data and protocol bytes are identical to the previous stopped study:
648 public rows, 360/144/144 train/validation/test; nine whole table-parent
families and eighteen semantic orbits; maximum 98 complete native tokens; zero
truncation and zero declared historical/development orbit overlap. The
scientific-family stream is public and unchanged. This is **not a newly unseen
dataset or independent confirmation** of the previous fit.

Read-only comparison records identical plan, dataset identity, initialization
tensor identity, initial training loss and initial fit report, step/visit/token counts, and
unchanged numerical/data/spec/model/training source bytes. The frozen producer
and recorded environment differ. The former run recorded a Ryzen 9 5950X / AVX2,
CPython 3.12.13, Linux 7.0/glibc 2.43 and 16 interop threads. This run records an
EPYC 9V74 / AVX512, CPython 3.12.14, Linux 6.18.57/glibc 2.41 and 9 interop
threads. Both use one Torch compute thread, strict determinism and the pinned
CPU package versions. `postrun-verification/prior-comparison.json` retains the
exact comparison rather than substituting environment identities.

**The cause of 7/8 here versus 8/8 previously is not established.** Different
hardware/runtime observations are not proof of a cause or of nondeterminism.
The PR #123 production change configures the later independent validator; that
path was not reached here. No extra optimization, parameter adjustment,
source edit, seed search or retry was performed to explain or improve the result.

## Full heldout denominators remain unscored

| Method | Validation planned / outputs | Test planned / outputs | Result |
| --- | --- | --- | --- |
| Main 17 | 144 / 0 | 144 / 0 | Not attempted, unscored |
| Main 29 | 144 / 0 | 144 / 0 | Not attempted, unscored |
| Training-majority baseline | 144 / 0 | 144 / 0 | Train constant `0` only; no heldout inference |
| Context-ignoring baseline | 144 / 0 | 144 / 0 | Not attempted, unscored |
| Exact oracle reference | 144 / 0 | 144 / 0 | No heldout privileged calls |

The run preserves all 1,152 planned model/baseline cells as `not_attempted`.
Exact accuracy, recall, macro-F1, family accuracy and context-switch correctness
are null, not fabricated zero scores. No target journal/file, scoring file,
prediction commitment or external prediction barrier exists. Preparation used
native public observations for all rows and privileged targets only for training.
Public rules can mathematically determine answers; public construction is not
proof of operator blindness. No baseline improvement, statistical significance,
unseen algorithm/size/domain or psychological claim is supported.

## Resources and complete retained custody

The unchanged ceilings are 120 seconds cumulative native updates per attempt,
900 seconds whole attempt, 2,700 seconds whole study; 1 GiB process RSS, 2 GiB
sampled aggregate tree RSS and 4 GiB hard per-process address space. The fit
update phase took approximately 2.731 seconds; the stopped controller about
8.169 seconds. Whole outer-supervisor peaks were 639,926,272 bytes process and
656,179,200 bytes aggregate. No resource-limit failure was recorded. RSS sampling
cannot exclude transient peaks and is not hardware attestation or a host-wide cap.

The outer supervisor has no progress path and records `observed_updates=0`;
it is **not** the update total. The inner fit supervisor records 128 updates and
2,432 target tokens, matching checkpoint step, cursor, history and AdamW state.
The raw hex timing/resource receipts, bounded stdout/stderr, failure detail,
empty prediction journal and all planned/unattempted cells remain in the payload.

`payload.zip` contains all current frozen/feasibility/run files, exact 215-file
source archive, original anchors and claim, approval, oracle binary,
preparation/execution receipts and post-run verification/comparison. The original
pre-execution approval request is preserved as a historical request, not current
approval state. No earlier failed evidence is replaced by the new negative.
System libraries, compiler, kernel and operator honesty remain trust boundaries.

## Reproduction without training

From the exact published evidence checkout, Linux and Python 3.11+:

```sh
python3 -B docs/evidence/context-symbol-learning-v1-fit-gate-578d4793/verify.py
python3 -B docs/evidence/context-symbol-learning-v1-fit-gate-578d4793/test_verifier.py -v
```

The standard-library verifier checks deposit/ZIP/source/anchor inventories,
original progress and canonical fit-answer records, then extracts the frozen
producer and invokes its `verify --scores-only`. It recomputes data, schedules,
training targets and the stopped null-summary; it does not load model tensors,
train, call heldout truth or turn the negative into a success. Expected:
`verified=true`, `complete=false`, `heldout_scored=false`.

For exact matching-runtime native safe restoration and training-only inference:

```sh
"$PINNED_CPU_PYTHON" -B docs/evidence/context-symbol-learning-v1-fit-gate-578d4793/verify.py --native
```

Expected native fit audit: 7/8, zero verification optimizer updates, unchanged
bytes. Runtime mismatches reject. The frozen verifier itself skips native
checkpoint reconstruction for failed cells; the separately retained profiled
post-run audit supplies that narrower native fit check. Neither mode performs
independent optimizer replay, proves chronology, advances a research/release
gate or authorizes another scientific attempt. Negative/tamper checks use only
temporary copies and do not alter original evidence.

Affected requirements: TRI-I03/I07/I08/I10/I11/I13/I14; no numerical or protocol
semantics changed. Data remain original commissioned numeric facts and minimal
notation, distinct from software licensing. No corpus, teacher, Jung book text,
OCR/images, credentials or private chat enters publication. Source software
retains MPL-2.0; see [COPYING](COPYING) and [LICENSE](../../../LICENSE).
This evidence PR is for independent review, not self-approval or merge.
