# Separate matched CPU/CUDA control trial v1

**Prospective engineering pilot; actual GPU execution is unrun here.** This
separate versioned lane accompanies the [native CPU placement study](MUSIC-PLACEMENT.md).
It asks how one fixed control workload behaves and costs on CPU versus CUDA.
It does not test GPU treatment benefit, backend-by-curriculum interaction,
multiple-seed replication, or whether a GPU improves reasoning. Publication or
merge does not authorize a local target run.

## Separate environments, matched device comparison

Historical native model, batching and environment contracts are intentionally
CPU-only. They remain unchanged. Do not move a native Decoder to CUDA and
bypass its guards. A new dense device adapter reuses the original initialized
CPU weights and dense Block/RMSNorm/Linear operations, but owns explicit
validated device-aware top-level forward, batching and state checks. Both trial
cells use that same adapter and one shared versioned updater.

The canonical four-cell science study uses the original CPU lock. **Both device
trial cells use the same separately pinned torch 2.8.0+cu128 environment**:
one CPU cell and one CUDA cell, seed 17 control only. CUDA-wheel CPU versus
native CPU is a descriptive build/adapter compatibility check, not the device
speed estimate. Never change the canonical CPU environment to install this lane.
Use [the new hash-locked trial dependency file](../requirements.music-device.lock)
in a separate trusted Python 3.12 Linux x86_64 venv. Reuse an existing compatible
GPU environment rather than rebuilding unnecessarily; otherwise provisioning
is an explicit owner-local preparation step, never an automatic fallback.
The lock pins official wheel URLs/hashes and retains [metadata provenance](../scripts/music_device_trial_v1/runtime-lock.json);
metadata checks are not an installation or hardware test. No installer, driver change or download occurs implicitly.

Runtime admission verifies actual CUDA availability, one selected visible GPU,
compute capability and native `sm_120` support, and the declared 16-GiB-class
memory range; it records observed hardware rather than trusting a product name.
This first protocol deliberately supports only the owner-device architecture/memory
class, not generic GPUs: actual capability (12,0), native `sm_120`, and a broad
14–18 GiB total-memory range, not an exact 16-GiB or marketing-name equality.
Ordinary usable-memory shortfall is allowed. Other architectures/classes need a
separately reviewed protocol, not silent substitution. The owner-described RTX
5060 Ti is context, not hardware verification. An
unavailable/incompatible GPU, unsupported deterministic operation, memory failure
or parity failure produces an unsupported/failed/incomplete trial, not a negative
learning result. Never silently substitute CPU or retry with relaxed settings.
The independent native CPU study can stand if this trial is blocked.

Use float32 throughout; no AMP, TF32, compilation, fused/foreach optimizer or
autotuning. Before CUDA initialization configure `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
deterministic algorithms with errors rather than warnings, cuDNN deterministic
mode and benchmarking off, both matmul/cuDNN TF32 off, highest float32 matmul
precision and one CPU intra-op thread. Validate flags/environment throughout.
The CPU result is a one-thread reference, not a full Ryzen 5950X throughput claim.
An 8-GiB CUDA caching-allocator limit is separately enforced and reported; it is not a cap
on all driver or other-process device memory.

## Bounded parity admission, not a hidden target warmup

Before either target cell, one fixed isolated probe worker uses the same
nonmusical eight-row length-67 fixture for **one disposable update per backend**.
This is **two additional engineering updates, 16 visits and 32 targets**, separate
from target budgets. Initial tensor bytes must match exactly. The probe models
and optimizer states are discarded; RNG is restored and target cells initialize
fresh in separate cold workers. There is no GPU-only warmup, hidden retry,
compilation or tolerance tuning. Probe overhead is reported separately and is
not claimed to persist as target-worker warmup. Both target workers perform the
same declared initial-loss observation.

Predeclared elementwise absolute/relative engineering tolerances:

| Quantity | Absolute | Relative |
| --- | ---: | ---: |
| Initial logits | 1e-5 | 1e-4 |
| Initial loss | 1e-6 | 1e-5 |
| Gradients and pre-clipping gradient norm | 1e-5 | 1e-3 |
| One-update parameters | 2e-6 | 1e-4 |
| AdamW first moment | 1e-6 | 1e-3 |
| AdamW second moment | 1e-8 | 1e-3 |

Optimizer step must be exactly one; all values finite. Retain per-tensor maximum
errors and pass/fail observations. These conservative admission limits are not
already validated on the owner's GPU. Near-zero-gradient AdamW effects may fail
them: preserve failure, never widen thresholds post-outcome. CPU/GPU bitwise
training equality is not promised. Final step-640 tensors and answers may diverge;
report those outcomes descriptively, not as a final equality gate.

## Fixed target budget and measurement

Fixed order CPU then CUDA, fresh seed 17 control in each. Exact initial CPU
model tensors, independent optimizers, structural row order, labels, masks,
loss, clipping, AdamW settings and decoding are matched. Use the placement
lane's control 80-slot cycle (original 40 repeated), 640 updates and batch eight:
**1,280 target updates, 10,240 visits, 20,480 answer/EOS targets** total.

Each final endpoint receives the common **80 a/b layout diagnostics** (70 unique
prompts) and all 120 existing c/d development rows. Both safe completed endpoints
and diagnostics must pass an integrity barrier before any development decode;
there is no accuracy fit gate. Retain both regardless of accuracy. Total:
**400 decodes, cap 64, at most 25,600 greedy forwards**. Training/loss/probe
forwards are additional. Together with the independent CPU science study this
is six target cells, 3,840 updates, 30,720 visits, 61,440 targets and 1,200 decodes
(maximum 76,800 greedy forwards), plus the explicitly separate engineering probe.

Every target cell has a 120-second update ceiling and a 900-second worker
ceiling. Probe and development workers each have 900 seconds. The summed
3,600 worker-seconds are not a whole-controller deadline. Requests are 4 MiB,
logs/journals 1 MiB, artifacts 16 MiB, retained trial 128 MiB. These do not imply
whole-process RAM or whole-device memory containment.

Time cold worker end-to-end wall time, including startup/import, environment
admission, context creation, initialization, transfers, initial loss, updates,
diagnostics and checkpoint staging/serialization. Report CPU initialization and
model transfer separately, with initial loss, updates, guarded diagnostics and
serialization (including checkpoint CPU staging). Compile and additional target
warmup are explicitly zero/not used. Report development inference and read-only
audit time separately. Greedy-call timing is a nested subcomponent, not added
again to pipeline totals: it includes prompt tensor handling and synchronized
generation but excludes integrity guards and journal callbacks. Guarded
development pipeline timing checks both endpoints and can stage both models,
so it is not isolated decoder throughput. Unattributed worker-wall residual
includes startup/import/admission/handling overhead; it is not a pure startup
measurement or a bare-kernel benchmark. CUDA phases synchronize before and after measurement; asynchronous
launch latency must not masquerade as execution time. Include host/device input
and checkpoint transfers in end-to-end cost. Report probe/controller overhead,
not only update speed. A single fixed-order pair is descriptive: cache, thermal,
background load and process startup remain limitations, not benchmark significance.

## Interpretation, evidence and commands

Report CPU and CUDA exact+EOS, complete quartets, both-correct/changed-output
edges, rest-only strata, raw equality including wrong answers, canonical fit,
80-slot diagnostics and all 120 development outputs. Separate correctness,
learning outcomes and timing. The two c/d parents are previously exposed
development, with correlated views, not fresh confirmation. Neither a faster
GPU nor matching outputs proves general music, arbitrary counting, arithmetic
transfer, a neural mechanism or release readiness. No extra updates follow speed.

Freeze exact source, environment, hardware and protocol before target outcomes.
Keep independently retained request/matrix identities and complete safe tensor,
metadata, journal, attempt, parity and failure records. Audit is read-only byte,
safe-structure and retained-score checking, not optimizer/inference replay or
hardware/timing attestation. Reports-only exports cannot replace full custody.

```sh
python -I -B scripts/run_music_device_trial.py preflight
python -I -B scripts/run_music_device_trial.py freeze --request "$REQUEST"
# Separate explicit owner authorization; one controller attempt only:
python -I -B scripts/run_music_device_trial.py run \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --authorize-run
python -I -B scripts/run_music_device_trial.py audit \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --expected-identity "$MATRIX_ID"
```

Verify actual help before use. Use fresh absolute external non-link evidence
paths; invoke the trusted venv's lexical executable directly with `-I -B`, not
a resolved system Python. Freeze does not construct models/optimizers or train;
hardware admission may initialize/query CUDA metadata. Capture run/audit exits
separately, audit retained incomplete matrices when independently printed IDs
exist, and retain partial artifacts if interruption prevents a matrix. No `&&`
or errexit hiding failures, no internal worker selection, retry or resealing.

CPU synthetic tests exercise adapter/update parity and controller failure paths;
mocked CUDA admission tests are software conformance **not actual GPU tests**.
Unavailable GPU testing is explicitly unrun, never silently skipped into a green
hardware-compatibility claim. Actual owner-local hardware execution remains a
separate step after review, green exact-commit CI and explicit authorization.

The numerical policy follows the official PyTorch 2.8 [reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html),
[numerical accuracy limits](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)
and [CUDA synchronization/TF32 documentation](https://docs.pytorch.org/docs/2.8/notes/cuda.html).
Affected TRI-I01/02/03/05/06/07/08/11/12/13/14; no historical contract is relaxed.
