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

The output is canonical Trinite JSON. Freeze creates a non-existing directory with exactly `request.json`, `dataset-rights.json`, and `weights-rights.json`. Each file's exact content is bounded, and the request binds the rights records plus the model, configuration, data contract, policy, CLI and verifier source closures by SHA-256. The verifier requires a separately supplied request digest, re-reads and revalidates current source files, and detects tampering and extra files. It does not verify model execution, PROVENANCE event bundles, a Git commit signature, historical optimizer replay, or another host's execution. For historical verification, checkout the original source revision; do not rewrite records or update pins to make old receipts pass.

The default model is Trinite's existing **1,247,232-parameter** dense reference configuration (a parameter count, not an allocation). Both the existing **5M constructor limit** and **2M default run budget** remain stricter than the prospective **8B overall policy**. An FP32 weight-only lower bound uses 4 bytes per parameter; the FP32 weight/gradient/two-AdamW-moment planning lower bound uses 16 bytes per parameter, excluding activations, buffers and framework overhead. These figures are not resource benchmarks or a training authorization.

## First-party dataset sources and rights

`configs/teacher/source-candidates-v1.json` records immutable commits and distinct code licences for candidate sources:

- QEC v177.2 GF(3) mathematics and the later Trinite qutrit adapter (MPL-2.0 first-party source; preserve the MIT css-decoupling donor's attribution separately).
- Trinite's OMI external arithmetic/orbit module (first-party integration, upstream Apache-2.0 attribution still applies).
- OPT v1.4.0 (Apache-2.0): **optimization methods only**, not model-learning examples.

The `preview-formal` command produces nine original, exactly checked GF(3) addition facts using a second finite-field calculation as a consistency check. Every row retains its structural pair family. These are **public, training-exposed engineering smoke examples only**: `split=NONE`, status `NOT_ADMITTED_PUBLIC_ENGINEERING_ONLY`. They are not a research dataset, a holdout, a model capability benchmark, or direct import of QEC prose. The QEC revision is a mathematical convention reference, not proof of independently reproduced quantum circuits. TT-002 must supply source-specific admission, independent generation/oracles, source rights assessments, structural family splits, non-leakage and a new frozen protocol before any candidate becomes training data.

Each frozen packet deliberately includes **two separate records**. Dataset rights remain `NOT_CREATED_NOT_ADMITTED`, with origins and future review still required. Weights rights remain `NOT_CREATED` and `UNDECIDED` even though teacher code is **MPL-2.0**. A future checkpoint needs its own ancestry, data and weights rights, revisions, attribution and distribution decision. An MPL source licence is not a blanket licence for datasets, external donor code or future weights.

## OPT-inspired boundary (without speed claims)

Donor methods: [OPT v1.4.0 at 1ddd931](https://github.com/QSOLKCB/OPT/tree/1ddd93198991c2d296447693c040466ed2ed13cb), OPT-PY-001 (minimal deterministic tests), OPT-INV-001 (reuse only after proven equivalence) and OPT-PAR-001 (bounded independent work). The new focused CI lane has pinned checkout, read-only permissions, bounded timeout, no dependency download and both normal/`-O` fresh test executions. It does not cache verifier results, bypass existing native CI or expand the training matrix. Existing OPT-guided Trinite CI sharding and grouping are preserved unchanged; inventing another shard scheduler would duplicate that work.

The only model-adjacent optimization is **checking exact model size and estimated state bytes before allocation**, reusing existing validated contracts. TT-001 makes **no numerical change** (attention kernel, precision, optimizer, batch scheduling, gradients, quantization or architecture). Actual model-path OPT changes require separate compute/quality-equivalent controls, independent parity checks, cold reference path, complete lifecycle measurements, and an explicit experiment freeze. No speedup is claimed here.

## Certification and next work

Exploration may keep negative results; release claims still require heldout scientific gates. TT-002: reviewed formal source admission and independently implemented GF(3) generator using pinned QEC interfaces. TT-003: small native teacher training and fresh family holdouts. TT-004: three-arm student comparison (exact oracle labels, teacher-selected verified facts, and genuine soft-target distillation) with paired seeds, compute disclosure and complete evidence.

Affected boundaries: TRI-I01/I02/I03/I05/I07/I08/I11/I12/I13/I14. Historical data, model numerics, PROVENANCE adapter, protected 8B ceiling, original fixtures, previous teacher failures and CI verification remain unchanged. Merging and scientific runs require independent review and owner authorization.
