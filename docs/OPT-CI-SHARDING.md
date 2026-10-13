# OPT-guided model-conformance CI sharding (post-v0.2.0)

**Status:** CI-only candidate under review; no numerical, model, source-protocol, training or learning-result changes.

**Frozen baseline:** [Trinite v0.2.0](https://github.com/QSOLKCB/Trinite/releases/tag/v0.2.0), commit `30350e2e82c2e3e7fec913ff4d20f3faac42f7e0`. The released tag and its negative research evidence remain immutable.

**Donor mechanism:** `OPT-PAR-001` bounded deterministic parallel execution, supported by `OPT-PY-001` deterministic test execution, from [immutable OPT v1.4.0](https://github.com/QSOLKCB/OPT/releases/tag/v1.4.0) at `1ddd93198991c2d296447693c040466ed2ed13cb`. OPT's donor speedup and worker counts do **not** transfer to Trinite.

## Measured reference and objective

The post-merge [v0.2.0 model-conformance run](https://github.com/QSOLKCB/Trinite/actions/runs/38332138644) reported **261 successful CPU research tests**, with no skips, for each interpreter:

| Python | Original sequential test-suite wall time |
| --- | ---: |
| 3.11 | 1,794.636 seconds |
| 3.12 | 1,808.813 seconds |
| 3.13 | 822.233 seconds |

These are one hosted CI sample per version, **not** medians or controlled CPU speed comparisons. The times exclude acquisition and setup. The larger replay/archival suites dominate the critical path: task-balanced training/replay, retention-budget replay, retained-custody validation, scalar learning and fold-exposure verification. Their full output, optimizer replay, error cases and assurance requirements must not be shortened.

**Optimization contract CPU-OPT-SHARD-001:** Minimize completed CI wall time on the same GitHub runner class by distributing independent *whole existing test suites* to a bounded four-job matrix per Python version. Feasibility requires identical required test identities, complete execution, all original assertions, per-suite order, authentic subprocess replay and same numerical/backend/lock requirements. No test result, verifier result, tensor, checkpoint or source-integrity verdict is cached. A faster outcome is ineligible if coverage/replay differs. Report job queueing, end-to-end wall time and total billable runner-minutes separately from test-only duration.

## Fixed four-way allocation

The unchanged `scripts/check_research_cpu.py` and `scripts/check_cpu.py` remain authoritative. They discover 19 named suites containing **261** v0.2.0 cases. `scripts/check_research_cpu_shards.py` dynamically discovers the entire authoritative suite inventory **before** running a subset, validates all unique ordered test IDs and enforces the v0.2.0 count. Each suite remains intact, including its `setUpClass` / `tearDownClass` and fresh-process tests.

| Shard | Complete suites, in original relative order |
| --- | --- |
| **0** | `task_balance`, `ternary_dynamics`, `aizawa_reference` |
| **1** | `budget`, `maths`, `maths_retention`, `retention_budget` |
| **2** | `learning`, `generalization`, `scalar`, `retention_v2` |
| **3** | `model`, `training`, `geometry`, `comparison`, `foundations`, `convergence`, `fold_exposure`, `fold_order` |

The fixed assignment is based on observed v0.2.0 suite wall-time distribution rather than equal numbers of tests. Order **within** each suite is unchanged. Cross-shard completion order is deliberately not fixed. Each hosted job owns its own isolated checkout, process and temporary directories; any newly discovered cross-suite ordering dependency invalidates the optimization candidate rather than justifying a relaxed test.

The workflow now runs all **4 shards on each of Python 3.11, 3.12 and 3.13**, for twelve complete model-test jobs, with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, the existing hash-locked `requirements.lock` acquisition and `pip check` in each job. Both real dense/ternary inspection commands still run **once per Python version** in shard 0.

The historical `model-conformance (3.11/3.12/3.13)` check names are retained as **dependent aggregate status gates**: they cannot pass unless **all twelve** shard jobs (including the inspections) succeed. The independent Python standard-library jobs remain unchanged. The increased number of short-lived runner jobs might increase aggregate billed minutes or scheduler contention; speed is not guaranteed until measured.

## Coverage, failure and reproducibility

A shard exits nonzero if the suite partition omits/repeats suites, changes their relative order, discovers a different total case count, discovers duplicate/empty test cases, fails/skips tests, or executes a different ordered list from its assigned cases. Every shard re-discovers the full authoritative suite list and emits its own **post-success** JSON execution disclosure with the full ordered ID digest, selected ID digest, current source/requirements-lock digests, count and test-only `float.hex` elapsed time. The disclosure is not a cryptographically signed independent PROVENANCE receipt.

The original sequential combined runner is **still available byte-for-byte unchanged** at its v0.2.0 Git blob SHA `d55f8736b43d80386ba77fda5369684beba00446`. Run it as the reference:

```sh
PYTHONPATH=src python scripts/check_research_cpu.py
```

To run the new partition, all four complete shards are required; executing only one is not full conformance:

```sh
PYTHONPATH=src python scripts/check_research_cpu_shards.py --shard 0
PYTHONPATH=src python scripts/check_research_cpu_shards.py --shard 1
PYTHONPATH=src python scripts/check_research_cpu_shards.py --shard 2
PYTHONPATH=src python scripts/check_research_cpu_shards.py --shard 3
PYTHONPATH=src python -m unittest discover -s tests -p 'test_opt_cpu_shards.py' -v
```

`tests/test_opt_cpu_shards.py` uses standard-library-only adversarial fixtures for missing/duplicated/unknown/misordered assignments, wrong case totals, duplicate actual test identities, absent suites, skips and incomplete execution. GitHub's pinned CPU matrix is the authoritative integration/performance measurement; those isolated unit tests alone do not establish full-suite equivalence or a speedup.

**Rollback:** Restore v0.2.0's original CI invocation if a required shard fails, full output/assurance differs, job scheduling prevents faster complete CI, or hosted cost/memory grows unacceptably. No acceptance threshold was fitted to target outcomes; negative CI results must remain visible.

**Scientific boundary:** Existing model, quantizer, optimizer, held-out evaluation, geometry loss (zero), Phase 5 authorization (blocked), negative learning evidence and deferred COSMO neural-mapping research all remain unchanged. An optimization of test orchestration is not evidence of improved model learning.

**Affected invariants:** TRI-I01, I05, I06, I07, I08, I11, I12, I13 and I14.
