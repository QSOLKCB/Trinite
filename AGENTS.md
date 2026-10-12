# Trinite agent instructions

Follow the current user's scope. Read [README](README.md),
[operating instructions](docs/INSTRUCTIONS.md), [invariants](docs/INVARIANTS.md),
[architecture](docs/ARCHITECTURE.md), [modules](docs/MODULES.md),
[roadmap](docs/ROADMAP.md), and the affected topic contracts before editing.
Inspect any more specific `AGENTS.md` in the paths you change.

## Implementation

- Keep one authoritative numerical updater, scorer, admission operation and safe
  tensor reader. Use small explicit modules; no registry or workflow framework.
- CPU with the hash-locked dependencies is the canonical numerical lane. Never
  substitute a backend, download model code implicitly, or use pretrained/teacher
  inputs in a native run. Geometry loss stays zero until its own reviewed protocol.
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

## Verification and reporting

Name affected TRI requirements and compatibility. Use actual admission and
upstream verification in evidence tests. Process fakes are suitable for testing
controller failures, not for certifying custody or numerical outputs.

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s tests/upstream -v
# With requirements.lock installed in the chosen CPU environment:
PYTHONPATH=src python scripts/check_research_cpu.py
python -m pip check
```

The combined required CPU runner includes model, training, geometry and research
suites; never silently skip missing numerical dependencies. Check changed-doc
links/assets and diff whitespace. Never cache verifier/admission/test success
instead of running the current check.

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
blocked until their own gates are demonstrated. Open a reviewable PR; merging,
tagging and releases require the user's corresponding task.
