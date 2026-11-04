# TT-003 — native GF(3) teacher and source-bound owner-local training

Status: implementation candidate, no full target teacher run, release, distillation or scientific learning result. Trinite's native decoder is CPU-only. The new producer shares its actual AdamW/step/loss implementation and safe named-tensor reader, without modifying the historical Phase-3 36/6 admission, its checkpoint formats, the 8B policy, historical evidence or PROVENANCE.

## First teacher

The first teacher is a 90,624-parameter dense random-initialized causal decoder, with two blocks, width 64, four heads, feedforward width 128, byte vocabulary 259, and context 128. A bounded frozen first experiment uses 32 steps, batch size four, seed 17, AdamW learning rate 0.001, and a maximum of 640 scored answer/EOS targets. This is a small feasibility and trainability rung, not a 1M+ teacher, a general model or a trained checkpoint.

Only TT-002's 972 independently verified, admitted and already exposed TRAIN rows enter the native updater. The TT-002 dataset manifest and a separately supplied SHA-256 anchor are checked before request freeze or model allocation. The original TT-002 source file, rights decision, oracle and split constraints remain authoritative. This native adapter sorts training IDs under the frozen seed, binds its first 128 scheduled visits and exact target count, and prepares native input/attention/loss masks with the unchanged ByteTokenizer. The native training API names a list validation; TT-003 passes it eight of the TRAIN rows as a **train-only diagnostic**, never the development validation/test targets. It also records raw greedy predictions for eight exposed train rows. No independent generalization or heldout score results from this protocol.

## Four deliberate boundaries

1. **Inspect / freeze (no Torch):** independent verification of TT-002 dataset, model parameter formula and source identities, build/driver/runtime requirements and token budget. The frozen request records source and numeric input identities, rights, the actual planned training schedule and training_authorized=false. The request is written once and its SHA-256 is returned for independent retention. No model allocation is performed.
2. **Owner approval (separate local statement):** train-owner-local demands a canonical JSON approval file whose bytes and external SHA-256 are supplied separately from the frozen request. It must bind the request identity, dataset identity, owner QSOLKCB, 32 authorized updates, and the exact single owner-local research scope. The harness does not invent or issue approvals. This statement is not cryptographic identity authentication.
3. **Bounded CPU producer:** once separately authorized, it imports the pinned native CPU backend, enables one Torch thread and deterministic algorithms, and requires the original exact hash-locked Python/PyTorch dependency environment. It invokes Trinite's unchanged Decoder, TrainingState, create_optimizer, run_steps and single-tensor AdamW objective. The original scoring/optimizer protocol is not substituted by a custom numerical update. Failure preserves a separate failure receipt and cannot yield a completed run manifest. The native update deadline is checked between updates; it is not a hard whole-process timeout. An owner should additionally use OS resource/time limits for substantial experiments.
4. **Retained evidence and verification:** a successful run keeps the approved source/request, safe named model plus complete AdamW moment tensors, immutable training history, source/runtime manifest and train-only predictions. Read-only verify-run uses an *externally retained run-manifest digest*, checks all member identities and current input/source closure, restores the native tensor inventory through the existing closed safetensors reader, checks all optimizer progress and final train-only outputs and loss. It performs no optimizer updates and does not claim full optimizer replay, third-party execution, machine attestation or cross-host bitwise determinism.

Dataset distribution rights and future weights license remain UNDECIDED; MPL-2.0 covers the original teacher harness source code, not every possible dataset or weight distribution. Third-party pretrained text, QEC donor documentation, private chats and unexplained external outputs remain excluded. Existing TT-001 and TT-002 packet contracts are unchanged.

## Usage

Use a clean trusted Trinite checkout and the pinned CPU dependency environment when executing or verifying a native model. The standard-library freeze can run without Torch.

~~~sh
python3 scripts/teacher_dataset.py prepare --out /tmp/tt002-dataset-new
# Retain the printed TT002 manifest_identity outside the dataset.
python3 scripts/teacher_training.py inspect
python3 scripts/teacher_training.py freeze --dataset /tmp/tt002-dataset-new \
  --dataset-identity sha256:TT002_ANCHOR --out /tmp/tt003-request-new
python3 scripts/teacher_training.py verify-request --request /tmp/tt003-request-new \
  --identity sha256:TT003_ANCHOR --dataset /tmp/tt002-dataset-new
~~~

The owner must create the separate approval manually, using the actual anchored hashes, canonical JSON (sorted keys, compact separators, LF-terminated):

~~~json
{"approved_updates":32,"dataset_manifest_identity":"sha256:TT002_ANCHOR","owner":"QSOLKCB","request_identity":"sha256:TT003_ANCHOR","schema":"trinite.teacher-training-local-approval.v1","scope":"one-explicit-owner-local-cpu-teacher-experiment","training_authorized":true}
~~~

This example is not a valid run until both placeholders are replaced and the owner expressly approves the intended execution. Retain the approval file's own SHA-256 independently. Then, and only then, the owner can run:

~~~sh
python3 scripts/teacher_training.py train-owner-local \
  --request /tmp/tt003-request-new --identity sha256:TT003_ANCHOR \
  --dataset /tmp/tt002-dataset-new \
  --approval /tmp/tt003-approval.json --approval-identity sha256:APPROVAL_ANCHOR \
  --out /tmp/tt003-run-new
python3 scripts/teacher_training.py verify-run \
  --run /tmp/tt003-run-new --manifest-identity sha256:RUN_MANIFEST_ANCHOR \
  --request-identity sha256:TT003_ANCHOR --dataset /tmp/tt002-dataset-new
~~~

No full 32-step run is automatically performed by opening this PR or by GitHub Actions. The dedicated CI executes normal and optimized standard-library contract tests and only **one native CPU optimizer update** with closed tensor restoration, explicitly identified as engineering conformance. It verifies the pinned requirements lock with pip check. No CPU numerical benchmark, trainable teacher result or GPU hardware certification is implied by these CI checks.

## RTX 50-series CUDA 12.8+ preflight: available but never GPU training

~~~sh
python3 scripts/teacher_training.py cuda-preflight --device 0
~~~

This opt-in local command requires a working CUDA-enabled PyTorch runtime compiled with CUDA 12.8 or newer, compatible CUDA driver API, selected genuine GeForce RTX 50-series device reporting compute capability 12.0, a native sm_120 compiled backend, and a real GPU kernel result that matches a fixed CPU arithmetic reference. It refuses CUDA 12.4 and missing kernels; no nvidia-smi software label or successful allocation alone constitutes a passing preflight. When compiling kernels from source, the toolchain/nvcc must also target sm_120 with CUDA 12.8 or newer. The preflight does not download/install drivers, change system settings, train, select a GPU optimizer or fall back to CPU. Even if it passes, the returned record states gpu_training_supported=false, because the current Trinite decoder and updater are CPU-only. Implementing GPU training requires a later versioned native backend and CPU/GPU numerical parity gates.

## Next scientific milestone

Owner-approved actual teacher runs, complete negative/positive evidence, and a new evaluator with genuine privileged-process separation and exposure-controlled unseen compositional families precede any teacher-success or student-distillation claim. The TT-002 development families are already exposed and cannot be relabeled as fresh tests. Research gates constrain claims but do not forbid separately versioned exploratory studies.

Affected TRI-I01/I02/I03/I05/I07/I08/I10/I11/I12/I13/I14. Source files licensed MPL-2.0; no new mandatory runtime dependency beyond the pre-existing locked native CPU environment.
