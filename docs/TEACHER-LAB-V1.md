# TT-001 — Trinite Teacher Lab foundation

**Status:** source-bound non-training preflight and rights/candidate-data inventory. Not a trained teacher, implemented distillation pipeline, data admission or scientific qualification. All work is within Trinite; no new repository, download or hosted service.

## Interfaces

The checkout-only Python 3.11+ standard-library CLI uses the authoritative native `ModelConfig` parameter formula and the existing `model_size_policy_v1` scalar ceiling. It never imports Torch, constructs a model, reads a checkpoint, runs an optimizer, evaluates heldout data, admits training data, or claims a proof. Its accepted lane is dense *preflight only*. Later native model construction must independently check the actual parameter inventory and existing limits.

```sh
python3 scripts/teacher_lab.py inspect
python3 scripts/teacher_lab.py preflight
python3 scripts/teacher_lab.py preview-formal
python3 scripts/teacher_lab.py freeze --out /tmp/teacher-lab-fresh
# Retain the freeze-printed request_identity separately; provide it to verify:
python3 scripts/teacher_lab.py verify --run /tmp/teacher-lab-fresh --identity sha256:ACTUAL_DIGEST
PYTHONPATH=src:scripts python3 -B -m unittest discover -s tests/teacher_lab_v1 -v
```

The output is canonical Trinite JSON. Freeze writes to a private staging directory, flushes its three packet members, and atomically renames the staged directory to a non-existing destination. Write and synchronization failures clean up the private staging directory, preserving the destination for a fresh attempt. No hostile concurrent mutation of the parent directory is assumed. The published packet contains exactly `request.json`, `dataset-rights.json`, and `weights-rights.json`. Each file's exact content is bounded, and the request binds the rights records plus the model, configuration, data contract, policy, CLI and verifier source closures by SHA-256. The verifier requires a separately supplied request digest, re-reads and revalidates current source files (including the executable `src/trinite/__init__.py` package initializer, native `model.py`/`training.py` capacity guards, and this normative `TEACHER-LAB-V1.md` protocol), and detects tampering and extra files through bounded enumeration. A fresh-process regression freezes an isolated checkout, changes the initializer's executed version assignment and confirms that the original anchored packet is rejected. **This is post-import source-integrity detection, not protection against untrusted Python startup code**: an initializer can execute before the verifier checks its hash. The checkout and Python runtime must already be trusted; source closure verification does not create a sandbox. It does not verify model execution, PROVENANCE event bundles, a Git commit signature, historical optimizer replay, or another host's execution. For historical verification, checkout the original source revision; do not rewrite records or update pins to make old receipts pass.

The default model is Trinite's existing **1,247,232-parameter** dense reference configuration (a parameter count, not an allocation). Both the existing **5M constructor limit** and **2M native RunConfig training limit** remain stricter than the prospective **8B overall policy**. These limits are inspected from the actual AST structure of the existing native source without importing Torch; unexpected guard changes fail closed rather than relying on duplicated numeric constants. AST checks establish only the recognized current guard shape, not the full safety of arbitrary Python source changes. An FP32 weight-only lower bound uses 4 bytes per parameter; the FP32 weight/gradient/two-AdamW-moment planning lower bound uses 16 bytes per parameter, excluding activations, buffers and framework overhead. These figures are not resource benchmarks or a training authorization.

## First-party dataset sources and rights

`configs/teacher/source-candidates-v1.json` records immutable commits and distinct code licences for candidate sources:

- QEC v177.2 GF(3) mathematics and the later Trinite qutrit adapter (MPL-2.0 first-party source; preserve the MIT css-decoupling donor's attribution separately).
- Trinite's OMI external arithmetic/orbit module (first-party integration, upstream Apache-2.0 attribution still applies).
- OPT v1.4.0 (Apache-2.0): **optimization methods only**, not model-learning examples.

The `preview-formal` command produces nine original, exactly checked GF(3) addition facts using a second finite-field calculation as a consistency check. Every row retains its structural pair family. These are **public, training-exposed engineering smoke examples only**: `split=NONE`, status `NOT_ADMITTED_PUBLIC_ENGINEERING_ONLY`. They are not a research dataset, a holdout, a model capability benchmark, or direct import of QEC prose. The QEC revision is a mathematical convention reference, not proof of independently reproduced quantum circuits. TT-002 must supply source-specific admission, independent generation/oracles, source rights assessments, structural family splits, non-leakage and a new frozen protocol before any candidate becomes training data.

Each frozen packet deliberately includes **two separate records**. Dataset rights remain `NOT_CREATED_NOT_ADMITTED`, with origins and future review still required. Weights rights remain `NOT_CREATED` and `UNDECIDED` even though teacher code is **MPL-2.0**. A future checkpoint needs its own ancestry, data and weights rights, revisions, attribution and distribution decision. An MPL source licence is not a blanket licence for datasets, external donor code or future weights.

## OPT-inspired boundary (without speed claims)

Donor methods: [OPT v1.4.0 at 1ddd931](https://github.com/QSOLKCB/OPT/tree/1ddd93198991c2d296447693c040466ed2ed13cb), OPT-PY-001 (minimal deterministic tests), OPT-INV-001 (reuse only after proven equivalence) and OPT-PAR-001 (bounded independent work). The new focused CI lane has pinned checkout, read-only permissions, bounded timeout, no dependency download and both normal/`-O` fresh test executions. CI captures the **identity emitted by freeze** into a separate receipt before verifying the packet; hashing the packet after publication would not independently anchor it. It does not cache verifier results, bypass existing native CI or expand the training matrix. Existing OPT-guided Trinite CI sharding and grouping are preserved unchanged; inventing another shard scheduler would duplicate that work.

The only model-adjacent optimization is **checking exact model size and estimated state bytes before allocation**, reusing existing validated contracts. TT-001 makes **no numerical change** (attention kernel, precision, optimizer, batch scheduling, gradients, quantization or architecture). Actual model-path OPT changes require separate compute/quality-equivalent controls, independent parity checks, cold reference path, complete lifecycle measurements, and an explicit experiment freeze. No speedup is claimed here.

## TT-003 roadmap — prospective CUDA / RTX 50-series preflight

**Future requirement, not a TT-001 capability or training authorization.** TT-003 must keep the pinned CPU baseline and introduce a **separate opt-in GPU lane** only after a new source-bound protocol, dataset admission, explicit owner-local execution approval and a documented CPU/GPU comparison. On the intended NVIDIA RTX 50-class Blackwell hardware, the native compute capability is **sm_120**; for this project, require **CUDA Toolkit/runtime 12.8 or newer with real sm_120 support**. CUDA 12.4 does not provide native support for these cards and must be rejected, not silently allowed because a driver reports a high CUDA API version.

The TT-003 preflight shall run **before loading weights, constructing an optimizer, allocating training tensors or running a training step**:

1. Record and check the **actual GPU** identity and compute capability via the CUDA runtime; require the declared RTX 50 target's `(12, 0)` capability. Reconcile every reported device against the explicitly selected device. Do not infer hardware support from an environment variable or `nvidia-smi`'s reported maximum CUDA version.
2. Record the installed **toolkit/runtime, NVIDIA driver, PyTorch and CUDA backend builds** and their exact identities in the frozen request and execution receipt. Require toolkit/runtime **12.8+** and a compatible driver. Where compiling native code, verify the compiler/toolchain supports the requested **sm_120** target; for packaged PyTorch, verify the installed build can execute the target architecture.
3. Execute a **real, bounded GPU allocation and native kernel**, synchronize and check returned results against a pinned deterministic CPU reference. Check for missing kernels, unsupported architecture, unexpected backend substitutions and errors. A reported CUDA device or a successful allocation alone is not evidence of training compatibility.
4. Freeze a conservative device-memory and compute budget before execution; record allocation, peak memory, performance measurements, initialization and failure events separately. Validate CUDA availability again at run time and preserve any mismatch in a failed receipt.
5. **Fail closed**, with no silent CPU fallback, automatic downloads, installer/driver changes, relaxed determinism, or unreviewed PTX-JIT substitute for the specified native path. Unsupported configurations block GPU training, **not** CPU-only research. GPU CI that runs software doubles is engineering conformance only; it cannot certify hardware execution.

Only after this gate and a new authorized TT-003 experiment should teacher training, model/optimizer state, cross-backend parity and learned performance be measured. No GPU check, driver modification or benchmark is executed by TT-001.

## Certification and next work

Exploration may keep negative results; release claims still require heldout scientific gates. TT-002: reviewed formal source admission and independently implemented GF(3) generator using pinned QEC interfaces. TT-003: small native teacher training and fresh family holdouts, with the separate fail-closed CUDA 12.8+/sm_120 preflight above before any optional RTX 50-series GPU training. TT-004: three-arm student comparison (exact oracle labels, teacher-selected verified facts, and genuine soft-target distillation) with paired seeds, compute disclosure and complete evidence.

Affected boundaries: TRI-I01/I02/I03/I05/I07/I08/I11/I12/I13/I14. Historical data, model numerics, PROVENANCE adapter, protected 8B ceiling, original fixtures, previous teacher failures and CI verification remain unchanged. Merging and scientific runs require independent review and owner authorization.
