# Training contract

Status: native quantization-aware training design. No training has run.

## Quantizer v0

For each attention/feed-forward weight matrix W, require finite float32 values and compute a detached scalar alpha = max(mean(abs(W)), 1e-8). Define U = W / alpha and T = clip(round_ties_to_even(U), -1, +1). The forward weight is alpha × T. Threshold ties at U=±0.5 map to zero. Non-finite values and invalid shapes/scales fail before producing a checkpoint or export.

The reference surrogate gradient through the quantized weight with respect to W is 1 when abs(U) < 1 and 0 otherwise; alpha receives no gradient through its calculation. This is a chosen straight-through estimator, not the derivative of rounding. Unit tests must include ties, saturation, all-zero matrices, extreme finite inputs, non-finite failures, and gradient boundaries. Any change to scale, threshold, precision, or surrogate is a versioned quantizer change.

Embeddings, positional weights, RMSNorm gains, activations, optimizer state, and latent training matrices remain floating point. Do not describe the entire system as integer-only or every parameter as ternary. Export must encode the same codes/scales used by the final evaluation forward pass, without an intervening lower-precision cast that changes threshold decisions.

## First supervised procedure

Train from random initialization with causal next-token cross-entropy and AdamW. The first implemented run config must freeze initialization, optimizer hyperparameters, batch construction, learning-rate schedule, gradient clipping, accumulation, loss spans, token/step budget, and validation/early-stopping policy before the run. Their numerical values are not invented as evidence in this planning PR.

Use dense and ternary lanes with matched initialization, data order, examples, optimizer settings, and token budgets. Record actual wall time and resource use separately; matching tokens does not imply matching compute. A subsequent compute-matched comparison needs its own stopping rules.

Geometric loss weight is zero in the baseline and first ternary run. The geometry intervention comes after capture conformance and RESEARCH's preregistration gate. Teacher distillation and adversarial/GAN training are separate future experiments, not hidden substitutes for native training.

## Reproducibility and resume

Record Python/PyTorch/OS/library versions, repository and dependency identities, hardware, device, thread counts, dtypes, kernels, initialization, every RNG state, data order, and determinism settings. CPU eager execution is the canonical first lane. Enable deterministic algorithms with failure rather than warning-only behavior; disable silent backend fallback.

Declare replay scope before running: exact replay within a frozen environment where demonstrated; tolerance-based comparisons across changed environments with preregistered metrics/tolerances. Same seed alone never certifies exact CPU/GPU or cross-version reproduction.

Resume checkpoints must retain named optimizer state, schedule/step, RNG states, data cursor, and configuration/input identities in a versioned format. Use safetensors for tensors and explicit JSON metadata; no arbitrary serialized executable objects. An interrupted-and-resumed tiny run must agree with uninterrupted reference execution under its frozen scope before resume is claimed supported.

## GitHub workflow plan

Pull requests run bounded CPU contract checks and tiny smoke runs after workflows are implemented. Full training is manually dispatched with explicit config, dataset identities, seed, device/backend, time/token budget, and output destination.

GPU training requires an explicitly provisioned compatible runner; a workflow file does not supply a GPU. Hosted CPU CI must not be presented as a full model-training farm. Untrusted pull-request code must not run with secrets or unrestricted access on a privileged/self-hosted GPU runner. No automatic long training on each push.

Retain logs, configs, environment, admitted-data receipts, final/elected checkpoints, and failure/timeout records. GitHub artifact expiration is not archival retention; release evidence must be exported to the chosen durable destination and hash-verified. Selection rules and discarded-run counts remain visible.
