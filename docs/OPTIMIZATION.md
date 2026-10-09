# OPT-guided performance detour

This increment follows Phase 4 and precedes Phase 5. It changes repeated-work and CI orchestration boundaries; geometry loss remains zero. OPT is guidance, not a runtime dependency or imported training data.

## Frozen target contract

Source: [OPT at ae58b8b](https://github.com/QSOLKCB/OPT/tree/ae58b8b52e1b50e5da754ecc37b62d430809b23a), including README4AI, OPTIMIZATION-PROBLEM, CATALOG and the complete selected pattern records/source notes. Baseline: Trinite main bf4ff38ef2f0e08259b29a49806b4039d837d08f (merged Phase 4).

P = (X, F, f, d, C, B, S):

| Component | Target definition |
| --- | --- |
| X | Collector-local immutable artifact reuse; explicit geometry analysis batches; CPU test-process grouping and downloaded-wheel caching |
| F | Bounded changes that retain every input/artifact/event, complete actual upstream verification, numerical and rejection semantics, every test and Python lane |
| f / d | Minimize median complete collector lifecycle, complete geometry observation and required CPU-suite wall time; report stage costs and memory tradeoffs separately |
| C | Exact bundle bytes/reports for identical inputs; exact native trajectories/metrics/controls; model/optimizer/progress/RNG unchanged; no cached verifier outcome or weaker source/input validation; same test IDs and assertions, fresh-process replay retained |
| B | At most three scoped implementation candidates, seven paired lifecycle/observation trials and three paired CPU-suite trials; no statistical or numerical threshold tuning |
| S | Stop at budget or after measured lower median latency and all correctness gates; discard any candidate whose equivalence gate fails |

Classification: categorical local search; deterministic semantics with noisy wall time; derivative-free measurement; moderate evaluation cost; equality/semantic/resource constraints; sequential execution; exact, with no approximation permitted. Acquisition-cache benefit depends on a warm external cache and is reported separately from executed test work.

Selected patterns: [OPT-INV-001](https://github.com/QSOLKCB/OPT/blob/ae58b8b52e1b50e5da754ecc37b62d430809b23a/optimizations/OPT-INV-001-invariant-driven-reuse.md), [OPT-FAN-001](https://github.com/QSOLKCB/OPT/blob/ae58b8b52e1b50e5da754ecc37b62d430809b23a/optimizations/OPT-FAN-001-shared-materialization-fanout.md), and [OPT-PY-001](https://github.com/QSOLKCB/OPT/blob/ae58b8b52e1b50e5da754ecc37b62d430809b23a/optimizations/OPT-PY-001-deterministic-test-execution.md). Donor timing/cache constants and formal claims do not transfer.

Affected requirements: TRI-I06/07/08/12/13/14. Frozen upstream PROVENANCE and geometry code, licenses, source pins, protocol, model/quantizer semantics, rights admission and retained formal/model fixtures remain the baseline contract.

## Implemented equivalence boundaries

**OBS-REUSE-001:** within one collector, an already-bound name with exactly equal immutable bytes has the same upstream ArtifactRecord/content identity under the fixed octet-stream/CONTENT_RETAINED policy. retain validates name/type/size/closed state before lookup, compares exact bytes, and reuses only a successfully published record. New aliases still use the upstream factory; changed bytes reject. One strong bytes reference per unique artifact is bounded by the existing 64-artifact/64-MiB collector limits; equal aliases make no payload copies. This trades bounded in-process retention for less hashing/allocation. Snapshots are released at finalize. No persistent/cross-collector/source-stat cache is introduced. Mutable bytearray inputs remain rejected.

This reuse affects construction only. finalize still writes the same complete manifest and rereads retained disk bytes with the actual unmodified verifier. Cached bytes never repair or conceal tampered/deleted disk members. Successful publication alone cannot establish evidence eligibility; the complete closed verification gate still applies. The existing stable-filesystem/single-owner collector contract remains; this is not a concurrent mutation/crash-recovery store or a cached verification result.

**GEO-NULL-REUSE-001:** the exact deterministic null is determined by seed 17, example identity/layer and trajectory shape. Audited prompt states are unchanged during summary computation. Materialize each family's six null paths once for the current layer/condition, then use those exact lists for every pair. Null constructions fall from 216 to 72 per lane (432 to 144 total), without changing a sign, pair, family aggregate, layer, evidence class or RNG state. Family-local maps are discarded between groups; there is no global numerical cache. Native metrics/alignment and every source-pin check still use their original paths.

**CPU-COVERAGE-001:** the combined runner discovers model, training and geometry in the same order and with the same test IDs as standalone invocations. Every case/assertion remains; failures, import errors, empty suites and required-suite skips cannot yield success. Explicit fresh-process CLI, checkpoint replay and missing-dependency tests remain subprocesses. Foundation/upstream jobs remain separate standard-library lanes. All six Python 3.11/3.12/3.13 jobs remain required.

setup-python reuses downloaded pip content keyed by its platform/interpreter and requirements.lock identity in dependency-consuming workflows. Each run installs fresh with --require-hashes --no-deps and runs pip check, then executes current source/tests. An acquisition cache hit is not a test-result hit or evidence of cold reconstruction. First/cold runs retain the original acquisition path; invalid wheel bytes must fail hash validation. No installed environment, checkpoint, model output or verifier outcome is cached.

## Local measurements

Linux x86_64 shared virtual host, AMD EPYC 9V74, CPython 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5, one Torch compute thread. Medians include the complete declared lifecycle; no tolerance or test count is reduced.

| Workload | Reference median | Optimized median | Speedup |
| --- | ---: | ---: | ---: |
| complete-geometry-observation | 1.978847 s | 1.660583 s | 1.192× |
| payload-heavy-stage | 0.040319 s | 0.023492 s | 1.716× |
| small-stage | 0.008091 s | 0.007715 s | 1.049× |
| full CPU suites (44 cases) | 39.978371 s | 36.877886 s | 1.084× |

The small-collector gain is slight and falls within the noise range; the payload-heavy fixture benefits more because repeated multi-MiB inputs no longer get rehashed. Geometry improves by removing repeated null construction while retaining all native calculations. CPU grouping amortizes imports; download-cache benefit requires a warm external cache and is not quantified here.

These are descriptive paired measurements on shared compute. A correctness suite overlapped part of the observer/geometry series, and host scheduling/load are uncontrolled. Raw samples retain the spread; no statistical significance, isolated device benchmark or universal speedup is claimed. CI times on hosted workers have different interpreter/kernel/load fingerprints.

## Characterization and reproduction

Raw hex timings and full environment/source identities are retained in [observer and geometry samples](../fixtures/performance/observer-geometry.json) and [CPU suite samples](../fixtures/performance/cpu-suites.json). These are software performance characterization, not training data, reasoning evidence, or a portable hardware promise. Benchmarks perform exact equivalence checks before publishing a result. No timing threshold is enforced in CI.

For a fresh reproduction, obtain observation.py and geometry_run.py from the declared baseline src/trinite directory into a separate local directory. The benchmark rejects changed Git blob and raw SHA-256 identities before executing that explicit local reference. It uses the frozen baseline collector/summary with the current unchanged upstream dependencies and one current-tree request/checkpoint pair, isolating the two optimized mechanisms without weakening strict checkpoint-source compatibility. It does not claim a complete old-release replay under new receipts.

```bash
PYTHONPATH=src python scripts/benchmark_optimization.py --baseline-sources /path/to/baseline/src/trinite --geometry --output /tmp/observer-geometry.json
PYTHONPATH=src python scripts/benchmark_cpu.py --output /tmp/cpu-suites.json
```

Use the hash-locked CPU environment. Each output must be new. Observer/geometry characterization uses one warmup pair and seven measured alternating pairs, charging collector construction, retention, events, finalization and actual verification, or the full paired observation. Artifact snapshots/byte comparisons happen outside the timed interval. CPU characterization uses three alternating pairs, charging all required test processes and existing isolation subprocesses. Download/acquisition time is excluded from both local series; no warm-cache CI speedup is claimed from these samples.

The CPU benchmark dispatcher accepts only four fixed suite selectors and launches fixed argument vectors with the current Python interpreter, the repository working directory and `shell=False`. Unknown selectors reject before process creation; CLI output paths and environment values are never interpolated into arguments. Its subprocess call has a narrowly scoped exception for the `dangerous-subprocess-use-audit` rule: the dynamic interpreter preserves the selected hash-locked environment, while the closed dispatcher prevents external command selection. Regression tests check every allowed argument vector and rejection of command-like selectors. No shell escaping is needed because no shell interprets the arguments.

Rollback a candidate if exact bundle/report/trajectory/metric/control parity fails, changed inputs/files escape rejection, model/RNG/checkpoint bytes change, discovery loses a case, or repeated target measurements show the tradeoff no longer improves the workload. Keep the direct cold/reference paths for comparison. Further source/verifier caching, fewer tests/seeds, numerical approximations, packed export and Phase 5 intervention are outside this detour.

## Complete research CI grouping after the learning increment

Trent requested another OPT review after PR #13. This adaptation uses
[OPT v1.4.0 at 1ddd931](https://github.com/QSOLKCB/OPT/tree/1ddd93198991c2d296447693c040466ed2ed13cb), its README4AI,
OPTIMIZATION-PROBLEM, CATALOG and complete OPT-PY-001/OPT-INV-001 records.
No OPT runtime or training data enters Trinite.

The target contract is fixed before timing the candidate:

| Component | Definition |
| --- | --- |
| X | Standalone CPU research-suite processes versus one grouped process |
| F | Exact ordered test identities, assertions, three Python lanes, existing fresh-process probes and failure/skip semantics preserved |
| f / d | Minimize complete dependency-consuming conformance wall time on the current host |
| C | No test deletion, result/verifier cache, tolerance change, changed science or package-lock weakening; every required suite executes fresh |
| B | One candidate and three alternating paired complete-suite trials, plus the direct coverage gate |
| S | Stop after the fixed trials; adopt only after complete parity and lower local median wall time; otherwise keep separate invocations |

Classification: categorical local choice; noisy wall time; derivative-free;
moderate evaluation cost; equality, semantic and resource constraints;
sequential; exact. Acquisition and the unchanged inspection commands are outside
the local timed boundary; no hosted CI speedup is assumed.

**CPU-RESEARCH-COVERAGE-001:** `scripts/check_research_cpu.py` delegates the
historical native model/training/geometry discovery to the unchanged
`check_cpu.py`, then discovers comparison, foundations, convergence, budget,
learning and generalization in standalone order. Its exact ordered test-ID list
must equal the standalone discoveries without duplicates. Empty suites, import
errors, assertion failures and required skips cannot produce success. Existing
fresh-process replay, CLI and missing-dependency checks still create subprocesses.
Standalone suite commands and the original runner remain available. Foundation
and upstream standard-library jobs remain separate. Existing download caching,
fresh hash-locked installation and `pip check` stay in place.

The candidate changes process lifetime only; it never reuses a previous test
result or verifier outcome. Reject it on test identity, assertion or isolation
failure, or if the fixed paired measurements do not improve the target median.

Three alternating full-suite pairs passed with the exact same 122 ordered cases
and no skips. On the existing shared Linux x86_64 / AMD EPYC 9V74 / CPython
3.12.14 / Torch 2.8.0+cpu host, standalone times were 163.225342, 160.341148
and 160.501597 seconds; grouped times were 153.394814, 151.756027 and
147.358382 seconds. Medians were **160.501597 s versus 151.756027 s**, a
**5.45% lower local wall time**. This met the fixed adoption rule; current CPU
CI uses the grouped research runner. All six hosted conformance jobs at
`d3d504305ef81120630066150980f9ab668abe82` also passed.

[Raw samples, source/environment identities and executed logs](../fixtures/performance/research-cpu/)
retain the characterization. The source producer is
`5664b48a00df2cbf510828b7c866db076f51cffc`. These are three descriptive pairs
on shared compute, not a statistical significance claim or a guaranteed hosted
CI improvement. No other local model process overlapped these timing trials.
Reproduce the fixed three-pair characterization with:

```sh
PYTHONPATH=src python scripts/benchmark_research_cpu.py --output /tmp/research-cpu-characterization
```

The scalar-learning increment appends `tests/scalar` to current grouped discovery;
the exact ordered standalone-equivalence gate includes that suite automatically.
The retained 122-case timing packet remains the historical measurement, not a
benchmark of the extended suite. No new speedup is claimed.
