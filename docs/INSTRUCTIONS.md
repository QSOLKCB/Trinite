# Contributor and implementing-agent instructions

Status: project operating plan. This is a documentation file, not a root AGENTS.md automatically applied to future source files. The root README directs contributors here explicitly.

## Before implementing

Read README, INVARIANTS, ARCHITECTURE, MODULES, ROADMAP, and the topic contracts relevant to the change. Inspect the current repository and any actual scoped AGENTS.md before editing. Follow the current user's scope; do not automatically implement later phases.

All narrative documents stay under docs/. Keep the root README short and the repository LICENSE at root. Build/configuration entrypoints may be added when executable implementation needs them. Preserve IDEA verbatim as historical brainstorming; record decisions in the contracts.

## Implementation rules

Use small explicit modules and one authoritative implementation per operation. Honor the dependency direction in MODULES. No service, framework, GPU requirement, native runtime, or plugin registry without an explicit need. CPU execution is the canonical first lane.

Never download or execute remote model code implicitly. Do not admit data because it is available online or permissively licensed. Follow DATA's actual rights policy. Do not seed native training with third-party pretrained weights while describing it as fully native.

Treat geometry as a hypothesis with controls. Keep simulation, software conformance, empirical observation, intervention, proof, and custody verification distinct. Do not import performance claims from BitNet, QSOL-GEO-REASON, PROVENANCE, or upstream models into Trinite's README.

## Verification and reporting

For each change, name the affected contract IDs and run meaningful checks appropriate to those boundaries. The Phase 1 suite is runnable with PYTHONPATH=src python -m unittest discover -s tests -v, or python -m unittest discover -s tests -v after package installation. After acquiring the CPU lock, run the separate required model suite with PYTHONPATH=src python -m unittest discover -s tests/model -v. It imports the pinned backend directly and never silently skips model checks. Also run tests/training and tests/geometry explicitly with the CPU lock, and tests/upstream for the frozen PROVENANCE conformance. Root discovery includes the unchanged geometry reference suite without Torch. Follow [GETTING_STARTED.md](GETTING_STARTED.md) for acquisition and offline commands. No train/serve command is advertised before its implementation exists.

The frozen formal fixture binds contracts.py, tokenizer.py, and data.py source bytes. A semantic change to those files requires deliberate fixture regeneration into a new directory, comparison, and a reviewed update of the retained fixture. Do not automatically regenerate fixtures in tests or CI to make them pass.

The retained model-v0 forward fixture is produced by the independent scalar oracle in tests/model/oracle.py; test/CI execution never replaces it. Changes to model/quantizer semantics must name their numerical compatibility and affected requirements before updating these expectations.

.gitattributes pins LF working-tree bytes for byte-bound Python and JSON/JSONL artifacts, including the retained fixtures. Preserve these rules; core.autocrlf or a platform's default newline must not change a source/fixture identity.

Documentation changes require relative-link and moved-asset checks plus review for conflicting contracts/status claims. Model changes require forward/gradient/capture checks; exports require importer rejection and parity; data changes require rights, generator correctness, deduplication, and family-split checks. Required numerical tolerances must be frozen before comparing outcomes.

PR descriptions state the concrete behavior, scope, checks actually run, and remaining gates. Update the roadmap only for demonstrated completion. Do not add a green CI badge before that workflow exists. Open reviewable PRs; release/tag/archive actions follow an explicit release task and retained evidence.
