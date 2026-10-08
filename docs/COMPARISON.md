# Dense, ternary and four-state comparison

This authorised detour follows OPT and defers Phase 5. Its [protocol](COMPARISON-PROTOCOL.md) was frozen before target outcomes at [3bc49f4](https://github.com/QSOLKCB/Trinite/commit/3bc49f478ce0997b0ab46d36c79b9eb7ba7319cd). The experiment compares three classical weight formats and introduces exact QEC-derived formal tasks. Quantum execution, phase embeddings, error-corrected neural weights and packed acceleration remain deferred.

Dense uses CPU float32. Ternary retains its original semantics. Four-state uses {-3,-1,+1,+3} codes and the frozen half-absmean scale/STE. Only attention/feed-forward matrices are quantized; latent storage, embeddings, normalisation, tied output and calculations remain floating point. The common context-64, width-16 model has 7,264 unique parameters. Context differs from the earlier tiny smoke; direct comparisons apply within this matched experiment.

## Reproduction

Use the repository and the existing hash-locked CPU environment. Outputs must be new:

```bash
PYTHONPATH=src python -m trinite freeze-comparison /tmp/comparison-request.json
PYTHONPATH=src python scripts/run_comparison.py --request /tmp/comparison-request.json --request-identity sha256:THE_PRINTED_REQUEST_ID --output /tmp/trinite-comparison
```

The request binds protocol, executable source/dependency identities, the fixed process runner, audited datasets and CPU environment. Source/data/environment changes reject; freeze a new request after an intentional change. Each worker uses a fresh process. All 27 training cells finish before locked test scoring; lane execution order rotates by seed. Failed/timed-out cells remain in summary.json and suppress complete comparison aggregates. Reruns use new directories.

The manual three-lane workflow runs the same matrix after merge and retains complete artifacts for 14 days; archive them explicitly for durable evidence. Required Python 3.11/3.12/3.13 CPU CI runs comparison conformance alongside all existing suites, without repeating the complete exploratory matrix on every push.

## Data and scoring

Formal arithmetic preserves the original exposed fixture. Qutrit contains 160 examples (96/32/32 train/validation/test); packed ququart contains 300 (180/60/60). Every nonidentity one-site error supplies syndrome/correction tasks in two minimal carriers. Support site is the split family; all task/carrier/error variants remain grouped. This tests limited support-site transfer within a known code, not unseen-code generalisation.

The offline QEC oracle is frozen at 32fdf9c88f4ac1b873c8acc4d3d54c50f87d6e0a, v170.1.0. Eight unmodified decoder/algebra files and LICENSE are bound by qec-pin.json. Attribution: QSOLKCB/QEC and its contributors, under the retained CC BY 4.0 terms. Trinite's MPL-2.0 does not replace this dependency license. Training consumes locally generated numeric facts/minimal symbols, never upstream prose. Admission records retain authorship, procedure, independent scalar syndrome/inverse checks, source pins, supporting numeric evidence, scope, reviewer/date and limitations. Contradictory admissions reject through full regeneration.

Generation is unconstrained greedy argmax with a workload-wide grammar-derived cap and required EOS; expected-answer tokens never enter generation context. Primary scores use final-step test exact-answer family means. Validation selects nothing. Retain per-item outputs, per-task/family accuracy, separate teacher-forced target loss, code occupancy/error, timings, process RSS, snapshots and update histories. Three-seed ranges/paired differences are descriptive, with no significance or broad capability claim. Visible software fixtures are not fresh confirmatory evidence.

## Integrity and scope

The observer collects only after numerical computation, then the actual pinned verifier rereads every closed bundle. Aggregation binds consumed result copies to the verified retained name index. Safe safetensors snapshots bind source/data/plan/environment and validate named model/AdamW tensors and variable-length target histories. Resume equality is scoped to that environment; the frozen update loop uses no stochastic operations or global RNG. Native checkpoint behavior is preserved, including its strict original-source requirement. Hash consistency establishes integrity, not scientific truth or authenticated custody.

Geometry is secondary, detached and training-only, using the existing numerical kernel. Native, one-sided hash-permutation and Rademacher outcomes remain separate. Original Phase 4 negatives and its paired-reversal limitation remain unchanged. Alignment improvement without better answers supplies no reasoning-benefit evidence. CPU timings are shared-host characterization; no packed or edge-device savings follow from float32 simulation.

## Results

The first frozen local matrix completed on 2026-10-09 (Australia/Adelaide), using implementation [d18ac4e](https://github.com/QSOLKCB/Trinite/commit/d18ac4e45df47853679585d3e2a77039a1599ba4), Python 3.12.14, locked Torch 2.8.0+cpu, one intra-op thread and an AMD EPYC 9V74 shared host. All 27 cells completed, all closed bundles passed the actual upstream verifier, no artifacts were missing and no worker errors were recorded. The unchanged request and complete cell summary are retained in [request.json](../fixtures/comparison-v1/request.json) and [summary.json](../fixtures/comparison-v1/summary.json).

Every lane/seed scored zero complete test answers, including required EOS. Test set sizes per cell are formal 6, qutrit 32 and ququart 60. The table reports three-seed means, accuracy min/max, mean teacher-forced test target loss, mean detached native geometry contrast, and the range of paired training wall-time ratios to dense. These are different measurements; loss and geometry cannot substitute for exact answers.

| Workload | Lane | Exact accuracy mean [min, max] | Test target loss | Native geometry contrast | Training wall ratio to dense |
| --- | --- | --- | --- | --- | --- |
| formal | dense | 0% [0%, 0%] | 3.1787 | -0.4902 | 1.000–1.000 |
| formal | ternary | 0% [0%, 0%] | 3.7546 | -0.4279 | 0.915–1.104 |
| formal | four-state | 0% [0%, 0%] | 3.4648 | -0.3582 | 0.957–1.180 |
| qutrit | dense | 0% [0%, 0%] | 1.5835 | -1.0440 | 1.000–1.000 |
| qutrit | ternary | 0% [0%, 0%] | 1.5980 | -1.0332 | 1.043–1.206 |
| qutrit | four-state | 0% [0%, 0%] | 1.5825 | -1.0271 | 1.059–1.248 |
| ququart | dense | 0% [0%, 0%] | 1.6884 | -0.9285 | 1.000–1.000 |
| ququart | ternary | 0% [0%, 0%] | 1.6855 | -0.9970 | 1.016–1.135 |
| ququart | four-state | 0% [0%, 0%] | 1.6855 | -0.9774 | 1.103–1.135 |

This is **inconclusive for choosing a weight format**. Equal zero scores make all five-percentage-point quality margins pass mechanically, but provide no evidence of useful equivalence, superiority or quantum advantage. All measured training wall ratios fell below the frozen 1.25 margin; these subsecond single-host observations do not establish a speedup. Every lane retained 29,056 float32 latent weight bytes: there is no measured storage saving or packed inference.

Each cell ran 60 updates. Actual scored training labels were 480 for formal; 1,075/1,080/1,078 for qutrit; and 1,320/1,319/1,322 for ququart (seeds 0/1/2). Counts, batches, initialization and plans matched across lanes within each workload/seed. Initial teacher-forced train losses were approximately 5.49–5.62 and final train losses 1.51–1.69; loss decreased without successful held-out complete answers. QEC training exact answers were also zero in every cell. Formal train correct counts were dense 0/0/12, ternary 1/0/12 and four-state 2/0/6 out of 36. This exposes a learning-adequacy floor in the tiny budget; it does not isolate quantization as a failure cause.

All 27 native training-only geometry contrasts were negative. Both one-sided permutation and Rademacher controls, every per-item prediction, code occupancy/error, timings, RSS, captures, snapshots, update histories and closed verification reports remain retained. Earlier Phase 4 observations and limitations are unchanged. No setting was altered after these target outcomes.

[Evidence inventory](../fixtures/comparison-v1/inventory.json) binds the summary and complete 9.14 MB archive. Reviewers can download the unchanged [Trinite-comparison-v1.zip](../fixtures/comparison-v1/Trinite-comparison-v1.zip) directly from this PR's tree. The archive includes all 27 cells and logs, numerical artifacts, snapshots, verified bundles, source/configuration/test/fixture files and an internal SHA-256 member inventory. The inventory's original handoff-availability statement is historical; the same archive bytes are now available here. This is retained exploratory evidence, not a published release or independent replication. Archive SHA-256: `722acbf6023c989c7470d506786d366bb82f31dc2a0e375ed84a76db60a17c1e`. The archive's implementation documentation is the pre-outcome version; this section records the subsequent outcome interpretation.

Software validation passed: 71 standard-library checks, 44 existing CPU checks, 13 comparison checks, and upstream conformance. The six hosted Python 3.11/3.12/3.13 jobs each ran their required suites without skips; upstream conformance ran all 51 cases. Local root execution skipped one upstream permissions case. The complete manual hosted matrix remains available after merge and has not been independently run.

Before a later comparison can select a format, freeze a separate basics-first learning-adequacy protocol with explicit train and held-out exact-answer/EOS gates, appropriate matched training budgets, and failure criteria. Preserve this run as the initial null result. New curriculum topics and budget decisions belong to that later reviewed increment.

## Correctness pass after the first run

The review of head `f9c5f8a` identified four aggregation/workflow/checkout defects
and a comparison-checkpoint state agreement gap. The corrected implementation
retains all 27 cell records and writes a failed summary with `aggregate_errors`
and no groups if any pairing constraint or aggregate metric is invalid. Groups
are published only after all checks succeed. Resource diagnostics now include
paired latent-byte ratios and each seed's frozen `<= 1` margin. Equal float32
storage passing that margin does not establish packed savings.

The manual workflow permits 180 minutes: the conservative 54-worker bound is
162 minutes, leaving 18 minutes for setup, aggregation and upload. Always-run
upload remains enabled; runner loss or cancellation can still prevent retention.
QEC license bytes are LF-pinned, with an existing-checkout refresh procedure in
[GETTING_STARTED.md](GETTING_STARTED.md). Snapshots reject model/plan/lane
disagreement before tensor serialization.

Regression coverage exercises success, a late pairing mismatch, malformed
metrics, both training and evaluation worker timeouts, snapshot rejection before
serialization, and an actual autocrlf checkout plus stale-license refresh. The
real-cell conformance test freezes a fresh request for corrected sources and
checks the resulting upstream evidence bundle. These checks are software
validation, not an independently repeated 27-cell capability experiment.

The original protocol, request, summary, inventory and archive are unchanged.
Their producer is `d18ac4e`, not this correctness pass. Replaying historical
snapshots requires that original source tree or the archived implementation;
current-source execution must freeze a new request. `scripts/run_comparison.py`
is unchanged, so its `RUNNER_IDENTITY` remains unchanged. No model setting,
learning budget or scientific outcome was tuned during these fixes.

