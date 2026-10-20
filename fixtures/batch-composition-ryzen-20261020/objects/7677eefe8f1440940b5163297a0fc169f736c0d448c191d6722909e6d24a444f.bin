# Trinite agent instructions

Follow the current user's scope. Read [README](README.md),
[operating instructions](docs/INSTRUCTIONS.md), [invariants](docs/INVARIANTS.md),
[architecture](docs/ARCHITECTURE.md), [modules](docs/MODULES.md),
[roadmap](docs/ROADMAP.md), and the affected topic contracts before editing.
Inspect any more specific `AGENTS.md` in the paths you change.

## Implementation

- Keep one authoritative numerical updater, scorer, admission operation and safe
  tensor reader. Use small explicit modules; no registry or workflow framework.
- CPU with the hash-locked dependencies is the canonical native numerical lane.
  Never substitute a backend, download model code implicitly, or use pretrained/
  teacher inputs in a native run. The isolated external NCM reference below is
  not a native model or training input. Geometry loss stays zero until its own
  reviewed protocol.
- Keep narrative documentation under `docs/` and README concise. This root file
  is the agent entrypoint; detailed instructions remain in `docs/INSTRUCTIONS.md`.
- Preserve historical protocols, fixtures, producer snapshots and negative results.
  New numerical or evidence semantics need a new version and source-bound request.
  Do not refresh fixtures, reseal original evidence or weaken guards to pass checks.
- Preserve the flat v1 maths/retention modules and their source closure. New v2
  helpers belong in `src/trinite/retention_v2/`; include their complete runtime
  dependencies and fixed launcher in the new receipt and boundary tests.
- Freeze and publish the producer/protocol before target experiment outcomes.
  Reused test families remain exploratory; never tune a ratio on their results.
- Keep unique training diagnostics separate from scheduled repeated visits.
  Test/validation examples cannot enter updates or training diagnostics.

## Current workflow entry points

Read the linked contract and actual CLI before using a lane; these checkout-only
pilots do not extend the installed native API or unlock later research phases.

- [Fixed-code scale-refit training](docs/SCALE-REFIT-TRAINING.md):
  `scripts/run_scale_refit_training.py` owns freeze/run/full training verification.
  Preserve the original four-cell report and its environment. Slightly lower
  validation loss did not improve complete-answer accuracy.
- [Frozen-model decoding](docs/FROZEN-DECODING.md): use
  `scripts/run_frozen_decoding_v2.py` for new requests, runs and verification.
  The v1 launcher and evidence remain frozen for historical reproduction only.
  Syntax-valid constant answers are not input-sensitive reasoning; inference
  replay is not optimizer replay, and equal token caps are not equal compute.
- [External NCM reference pilot](docs/NCM-REFERENCE-PILOT.md):
  `scripts/run_ncm_reference.py` is a separate pinned pretrained Sudoku feasibility
  lane under `scripts/ncm_reference_v1/`, not a Trinite capability result.

## External NCM execution boundary

- Require a reviewed, green main commit and explicit local-run authorization
  before GPU execution. Use a clean checkout of that exact commit and record its
  Git SHA separately from the launcher's actual source closure. Freeze a request
  before outcomes; keep its SHA-256 independently. CPU contract tests use software
  doubles and do not demonstrate real-checkpoint compatibility, GPU execution or
  solve rate.
- Follow the documented separate Python 3.11/CUDA environment and exact pinned
  artifacts; do not alter `requirements.lock`, drivers or system settings, use
  paid compute, or fall back to CPU/another GPU. The admitted target is one RTX
  5060 Ti 16 GB-class device with native `sm_120` support. Preserve the frozen
  time, allocator-memory and workload bounds; the allocator cap is not a
  whole-device VRAM cap.
- Run `check-assets` before loading the two pinned local checkpoints. Keep
  `weights_only=True`, finite/shape checks and upstream import/hash checks intact.
  No unrestricted pickle, custom loading allowlist, remote model-code execution,
  upstream dataset, implicit download or unsafe fallback. Permissive upstream
  licensing alone does not satisfy native data admission.
- Use fresh destinations for request, run and replay artifacts. Retain complete
  attempt directories, partial progress, failures and both archives on mismatch;
  never overwrite historical fixtures or reseal evidence. Keep each printed ZIP
  SHA-256 independently and supply it to `verify`/`replay` as documented.
- `verify` is read-only offline archive/source/metric verification, not a fresh
  GPU run or hardware attestation. `replay` actually reruns GPU inference and
  compares deterministic result/environment bytes, not observed timings or peak
  memory. Never weaken source/asset pins, budgets, equality or tolerances to pass.
- Report clue-only raw generation separately from gold-codec diagnostics. No
  solver, clue overwrite, repair, retry or answer-conditioned generation. Four
  correlated visible formal boards are a smoke test, not a held-out benchmark,
  paper reproduction, speed comparison or native learning/release evidence.

## Verification and reporting

Name affected TRI requirements and compatibility. Use actual admission and
upstream verification in evidence tests. Process fakes are suitable for testing
controller failures, not for certifying custody or numerical outputs.

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s tests/upstream -v
# Standard-library external-reference contracts, without model loading:
PYTHONPATH=scripts python -B -m unittest discover -s tests/ncm_reference_v1 -v
# With requirements.lock installed in the chosen CPU environment:
PYTHONPATH=src python scripts/check_research_cpu.py
python -m pip check
```

The historical CPU aggregate covers model, training, geometry and its declared
research suites, not every later checkout-only pilot. Also run the affected
topic suites/workflows linked above; do not treat one aggregate as complete
coverage. Never silently skip missing numerical dependencies. Check changed-doc
links/assets, conflicting status claims and diff whitespace. Never cache verifier,
admission or test success instead of running the current check.

Retention v2, budget and task-balance source maps also bind `AGENTS.md`.
The [retained budget producer snapshot](fixtures/retention-budget-v2/README.md)
preserves the original instructions. Historical replay requires the exact
producer sources and matching environment for that run, not an evolved checkout.
Preservation tests check archived instructions but still pin every runtime file.
Those live run/verify source maps continue to bind current instructions; changed
guidance requires a fresh request, never a substituted historical hash.

Verification receipts must distinguish closed custody, source/input binding,
safe structure, scheduled visits/targets, final predictions/loss and complete
optimizer replay. Full replay starts from the declared native seed and compares
model/AdamW tensors and every numerical history/diagnostic against externally
bound expected artifacts. Do not describe final inference as update replay.
Foreign environment rejection is intentional: retain the mismatch and use a new
request for a separately identified run, without patching the old fingerprint.

Retain all expected cells, failure/timeout logs and failed comparison receipts.
Read-only verification must leave all run bytes unchanged. Report completed
checks, their environment and limits; custody is not learning adequacy. Original
fold/composed learning, format selection, scaling, Phase 5 and release remain
blocked until their own gates are demonstrated. Open a reviewable PR by default;
a direct-main edit requires the user's explicit scope. Merging, tagging and
releases require the user's corresponding task. Report CI for the exact published
commit, distinguishing passed, failed, pending and unrun checks.
