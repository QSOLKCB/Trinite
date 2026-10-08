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

The first frozen local matrix is pending; complete outcomes, including null/negative results, will be recorded here. Conformance alone demonstrates software semantics.
