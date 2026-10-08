# Training contract

Status: Phase 3 implements bounded native CPU training, safetensors checkpoints/resume, and pinned observation. Tiny dense/ternary runs learn the visible formal fixture. [PHASE-3.md](PHASE-3.md) records software checks and their scope; no new reasoning benchmark is claimed.

## Quantizer v0

For each attention/feed-forward weight matrix W, require finite float32 values and compute detached absmean in float64 to avoid reduction overflow on extreme finite inputs, apply the 1e-8 floor, then round the scalar alpha to float32. All division, rounding, codes-to-weight multiplication, and forward weights use float32. Define U = W / alpha and T = clip(round_ties_to_even(U), -1, +1). The forward weight is alpha × T. Threshold ties at U=±0.5 map to zero. Non-finite values and invalid shapes/scales fail before producing a checkpoint or export.

The reference surrogate gradient through the quantized weight with respect to W is 1 when abs(U) < 1 and 0 otherwise; alpha receives no gradient through its calculation. This is a chosen straight-through estimator, not the derivative of rounding. Phase 2 tests cover ties, saturation, all-zero/subnormal matrices, extreme finite inputs, non-finite failures, and gradient boundaries against explicit expected masks and a separate scalar oracle. This revision freezes the overflow-safe float64 reduction and subsequent float32 scale rounding as v0 semantics before native training. Later changes to scale, threshold, precision, or surrogate require a versioned quantizer change. The custom autograd forward returns the actual scaled codes directly; a subtract/add STE expression could lose small values through cancellation and is not used.

Embeddings, positional weights, RMSNorm gains, activations, optimizer state, and latent training matrices remain floating point. Do not describe the entire system as integer-only or every parameter as ternary. Export must encode the same codes/scales used by the final evaluation forward pass, without an intervening lower-precision cast that changes threshold decisions.

## First supervised procedure

Train from random initialization with causal next-token cross-entropy and AdamW. [tiny-training.json](../configs/tiny-training.json) freezes the first procedure before execution. Numeric hyperparameters are finite decimal strings; measured scalar values use canonical float.hex strings, preserving exact binary values without floating-point JSON.

| Setting | Frozen tiny procedure |
| --- | --- |
| Architecture | Byte vocabulary 259, context 32, one block, width 16, two heads, FF width 32; 6,752 unique latent float32 parameters |
| Initialization | Private CPU generator, seed 0, normal standard deviation 0.02; RMSNorm gains 1; tied input/output embedding |
| Data | Exact admitted formal-v1 bytes; 36 training examples and six validation examples; test labels excluded from numerical batches |
| Order / batch | Rank example identities by SHA-256 of canonical seed/item records; cycle this fixed order, batch size four, right padding, no reshuffle |
| Loss | Shift logits/targets by one position; mean cross-entropy over answer and EOS only, two targets per example; PAD/prompt/BOS targets excluded |
| Optimizer | Single-tensor AdamW over each unique named parameter once; LR 0.003, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01; foreach/fused/AMSGrad/capturable/maximize/differentiable off |
| Schedule / clipping | Constant LR, accumulation one, global gradient norm clipped to one; fail on nonfinite norm/loss/state |
| Horizon / selection | 60 steps, 480 scored training tokens; validation at 20/40/60; final step elected, no early stopping or test evaluation |
| Backend / budget | Linux x86_64, pinned CPU float32 eager backend, one Torch thread, strict deterministic algorithms; 60 seconds per invocation checked between updates |

RunConfig rejects unsupported schedules/objectives and bounds all accepted plans to 256 steps, batch size eight, 4,096 scored training tokens, 600 seconds per invocation, and two million parameters. These are computational bounds, not a whole-process memory or hard operating-system timeout. Model forward and checkpoint byte bounds apply independently. Validation/initial/final loss evaluation and IO are outside the between-update timer; workflow jobs have their own ten-minute hard timeout. Geometry weight remains zero and no geometry module is imported.

Use dense and ternary lanes with matched initialization, data order, examples, optimizer settings, and token budgets. Record actual wall time and resource use separately; matching tokens does not imply matching compute. A subsequent compute-matched comparison needs its own stopping rules.

Geometric loss weight is zero in the baseline and first ternary run. The geometry intervention comes after capture conformance and RESEARCH's preregistration gate. Teacher distillation and adversarial/GAN training are separate future experiments, not hidden substitutes for native training.

## Reproducibility and resume

Record Python/PyTorch/OS/library versions, repository and dependency identities, hardware, device, thread counts, dtypes, kernels, initialization, every RNG state, data order, and determinism settings. CPU eager execution is the canonical first lane. Enable deterministic algorithms with failure rather than warning-only behavior; disable silent backend fallback.

Declare replay scope before running: exact replay within a frozen environment where demonstrated; tolerance-based comparisons across changed environments with preregistered metrics/tolerances. Same seed alone never certifies exact CPU/GPU or cross-version reproduction.

Resume checkpoints must retain named optimizer state, schedule/step, RNG states, data cursor, and configuration/input identities in a versioned format. Use safetensors for tensors and explicit JSON metadata; no arbitrary serialized executable objects. An interrupted-and-resumed tiny run must agree with uninterrupted reference execution under its frozen scope before resume is claimed supported.

Implemented format: a new checkpoint directory contains exactly tensors.safetensors and metadata.json, under trinite.training-checkpoint.v1. It retains unique named model tensors, named AdamW step/first/second moments, complete step and validation histories, the fixed plan/schedule, next data cursor, scored-token count, initialization/loss receipt, Python MT state and Gaussian cache, NumPy MT19937 keys/cache, and Torch CPU generator state. The tied output alias is reconstructed from the model contract, never duplicated as a separately trainable tensor.

Payloads are bounded to 32 MiB and metadata to 1 MiB. The reader rejects unsafe membership, symlinks, noncanonical/duplicate JSON keys, unknown versions/names, stale source/dependency/data/plan/environment receipts, malformed RNG state, and invalid tensor shape/dtype/finite values or optimizer counters. Tensor identities are checked in addition to the file identity. Candidate state is constructed and validated separately; global RNGs are restored only after every check succeeds. No pickle API is used. Checkpoint creation never overwrites a directory; metadata is written last. An interrupted write is rejected by the reader rather than treated as an atomic checkpoint.

Exact replay is demonstrated within the same source, dependency closure, interpreter patch version, Linux kernel/platform, CPU model/features/kernel capability, dtype, threads, and deterministic settings. Resume rejects a changed fingerprint. It does not promise cross-version, CPU-family, Windows/ARM, or GPU replay. The CLI seeds Python/NumPy/Torch globals from the plan and then restores retained states on resume. The Python API keeps caller-owned global RNG state: callers comparing independent runs must start from matching states. The numerical training loop makes no global RNG draws.

The observer runs around completed computational stages, never inside the forward/gradient/update loop. Its timing, timestamps and enabled mode live outside computational checkpoint bytes. Required observation errors preserve completed computation but return a nonzero CLI status and false observer-evidence eligibility. Compute failures retain logs and a checkpoint of completed steps when valid state can be recovered; invalid/partial state is explicitly reported as unrecoverable. Every run reports release_ready=false: verified software evidence does not close the release checklist.

## GitHub workflow plan

Pull requests run bounded CPU contract, upstream, model and tiny training suites under Python 3.11/3.12/3.13. Manual tiny training runs both matched lanes on hosted CPU workers. Its required dispatch inputs are exact plan and dataset identities; the frozen plan supplies seed/backend/horizon/time/token budgets and the workflow selects a new runner-temporary output directory. All actions are commit-pinned and repository permissions read-only. It retains successful and failed run files as review artifacts for 14 days. This is a bounded tiny procedure, not long/full-scale training; larger training remains future reviewed work.

GPU training requires an explicitly provisioned compatible runner; a workflow file does not supply a GPU. Hosted CPU CI must not be presented as a full model-training farm. Untrusted pull-request code must not run with secrets or unrestricted access on a privileged/self-hosted GPU runner. No automatic long training on each push.

Retain logs, configs, environment, admitted-data receipts, final/elected checkpoints, and failure/timeout records. GitHub artifact expiration is not archival retention; release evidence must be exported to the chosen durable destination and hash-verified. Selection rules and discarded-run counts remain visible.

## Separate comparison detour

The [frozen three-lane protocol](COMPARISON-PROTOCOL.md) adds the opt-in four-state codebook and exact numeric QEC tasks. The original ternary algorithm and native 36/6 fixture contract are preserved. Both paths use one authoritative optimiser/update loop; comparison has admitted variable-length schedules and its own safe snapshot schema. Native/checkpoint source fingerprints remain strict, so older native checkpoints replay with their original tree. No geometry loss or quantum execution is introduced.
