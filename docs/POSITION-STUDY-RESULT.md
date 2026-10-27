# Position coupling: mixed/negative development result

Recorded **26 October 2026 UTC**, from the owner-authorized local execution of
[frozen PR #77](POSITION-STUDY-V1.md) at merged producer commit
[`050f8cc109453bb3e6d0de62b2c0dd2e93841ed3`](https://github.com/QSOLKCB/Trinite/commit/050f8cc109453bb3e6d0de62b2c0dd2e93841ed3).
All six predeclared cells completed in one controller invocation. Preservation
adds no target training, generation, retry, checkpoint selection or repair.
The frozen protocol remains unchanged as the pre-outcome specification.

**The frozen descriptive support criteria were not met.** Small in-range
row-level gains are retained, but only seed 2 meets the unseen whole-family gain
criterion, and no cell succeeds in either longer-number stratum. This is a
mixed/negative development result, not a successful generalisation fix.

## All final results

These are native greedy outputs at the fixed 4,000-update checkpoint. Primary
exactness requires the complete reversed, fixed-width answer **and actual EOS**.
A family succeeds only if every admitted operand-order/padding rendering is
correct; correlated rows are not independent trials.

| Arm / seed | Unique training rows | Seen families | Unseen rows | Unseen whole families | Three-digit rows / families | Four-digit rows / families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control 0 | 154/154 | 64/64 | 16/56 | 4/24 | 0/23; 0/12 | 0/23; 0/12 |
| Coupled 0 | 154/154 | 64/64 | 17/56 | 3/24 | 0/23; 0/12 | 0/23; 0/12 |
| Control 1 | 154/154 | 64/64 | 16/56 | 4/24 | 0/23; 0/12 | 0/23; 0/12 |
| Coupled 1 | 154/154 | 64/64 | 18/56 | 4/24 | 0/23; 0/12 | 0/23; 0/12 |
| Control 2 | 154/154 | 64/64 | 22/56 | 7/24 | 0/23; 0/12 | 0/23; 0/12 |
| Coupled 2 | 154/154 | 64/64 | 26/56 | 10/24 | 0/23; 0/12 | 0/23; 0/12 |

Every cell reached 154/154 unique-training fit by update 2,000 and retained it
at 4,000. This is training fit, not transfer evidence. The paired unseen
whole-family changes are **−1, 0 and +3**, while row changes are **+1, +2 and +4**.
Do not substitute the latter for the frozen family criteria or select seed 2
as the sole result. The secondary decoded-numeric counts equal canonical counts
in these final strata; none of the longer outputs is numerically correct.

Both arms retain the same 86,528-parameter dense CPU architecture, data,
effective initialization, padded operands/reversed answers, optimizer and
schedule. Only sequence versus coupled position addressing differs. Each cell
has 4,000 updates, 32,000 visits, 121,600 scored answer/EOS targets and 307,200
actual live shifted input tokens, including 6,400 unscored short-row EOS inputs.
The per-example predictor-input total is 300,800. These totals are distinct from
unique examples and from total experiment compute.

## Frozen criteria, not retrospective thresholds

For **each** paired seed the contract requires:

- Candidate seen fit at least 95% and regression at most 2 percentage points:
  all three seeds pass (64/64 each, no regression).
- Unseen-family gain at least 10 percentage points, requiring at least three
  additional families out of 24: seeds 0 and 1 fail; seed 2 passes.
- In each longer stratum, candidate at least 25% (three of 12 families) and
  gain at least 10 percentage points (two of 12): every seed fails both
  requirements in both strata.

Thus no seed satisfies every condition, and the all-seed support condition is
false. Conditions are descriptive, not statistical significance or an automatic
learning, confirmation, scaling or release gate.

## Where transfer breaks in this finite study

All six final cells solve all **57 carry-required training rows**. Among the
**18 unseen carry-required rows in eight families**, the correct row counts are:

| Seed | Control | Coupled |
| --- | ---: | ---: |
| 0 | 0/18 | 2/18 |
| 1 | 1/18 | 2/18 |
| 2 | 0/18 | 0/18 |

**No cell solves a complete unseen carry-required family.** The four coupled
successes across cells are three distinct prompts, not four independent task
families. This identifies poor carry transfer, not proof of its causal mechanism.
The stronger seed-2 whole-family count is entirely in the no-carry slice.

For a direct small example, coupled seeds 0/1/2 decode to **2, 3 and 4** for both
`2+2=` and `02+02=`. Control seeds 0 and 2 solve the unpadded form but decode to
2 for its padded form; control 1 solves both. Native reversed output tokens are
retained, rather than replaced with this human-readable decoding. An isolated
correct answer or consistent wrong answer is not an addition rule.

All three-digit outputs terminate prematurely; four-digit outputs are
predominantly premature EOS, with some missing EOS. The raw categories and
independent flags distinguish output termination from arithmetic correctness.
This result does not support a claim of universally flawless EOS behavior.

## Interpretation and next decision

Preserve this result before changing the experiment. Coupling under this exact
format, budget and fixed-zero offset policy did **not** reliably improve
whole-family transfer across seeds or extend successful addition to longer
operands. The result does not establish universal model incapacity, a capacity
limit, a general refutation of position coupling, or a proven alternative cause.

The frozen offsets leave later position rows untrained: coupled rows 4/5 are
unexposed, and row 3 is exposed only as an answer slot, not a training operand
digit. The control also has untrained later sequence rows. Therefore length
failure alone cannot isolate arithmetic-rule failure from positional/role
exposure. Range and length also shift together. This small adaptation does not
reproduce the larger paper's scale or randomized-offset training.

All probes are development-exposed. A future, separately frozen investigation
may distinguish positional/role exposure from carry-coverage interventions,
with matched controls and new evidence gates. It is not implemented or
executed here. The inference-router proposal is separate; calculator or larger-
model answers cannot be credited as native Trinite learning. Comparison with
the historical balanced-addition result changes format and data as well as
addressing, so it cannot isolate coupling.

## Evidence and verification boundary

The [public subset](../fixtures/position-study-result-20261026-v1/README.md)
retains **115 original files** byte-for-byte: the matrix plus 19 records for
each of the six cells. These include all five training scores/journals, all
three development scores/journals, report, description and completion. They
cover 5,232 predictions, including 1,536 at the final checkpoint; repeated
observations are not independent task cases. No cell or declared score is dropped.
This is deliberately not the complete private handover.

- Original request: `sha256:bdf6d3bfc7e05467f324e2794cfec70791b082ac08fad384b00fbcfcd6c09623`.
- Original matrix: `sha256:bcfee16fbbdddff24442027d22e32fe321dd0294479e40e78f44f0a929b5a762`.
- Original private archive SHA-256: `0eb62c73754e392399394625cf581243046fb046160556ba36167090bb754a70`.

Private static review checked archive/manifest bytes, all 239 request-bound
sources against the published producer, original identity receipts, retained
cell closures, checkpoint tensor structure/identities and histories. Independent
static recomputation checked 5,232 retained prediction records across all five
training checkpoints and three final development strata, plus perturbation
metrics. Those checks do not constitute trained-generation replay, optimizer
replay, independent-host replication or hardware attestation.

The new `source-bindings.json` is an explicitly **derivative private-audit
receipt**, not the original request. Its source-only extract cannot reconstruct
the omitted request environment. Original matrix/completion references to
omitted artifacts remain unchanged. This public subset is **partial custody**,
not closed custody or complete public checkpoint verification.

The read-only public checker reuses authoritative admission, score validation
and descriptive reporting. Its imports require the pinned CPU environment;
it does not construct a model, load checkpoint tensors or execute training or
generation. It checks the selected original bytes and recomputes their retained
output metrics, not the provenance of an independently repeated numerical run.
Historical protocols, source closures, fixtures and negative results remain
unchanged.

```sh
PYTHONPATH=src:scripts python -B scripts/check_position_result.py
PYTHONPATH=src:scripts python -B -m unittest discover \
  -s tests/position_result_v1 -v
```

Affected TRI-I03, I07, I08, I11, I13 and I14: bounded evidence preservation only.
No numerical semantics change. Original learning adequacy, fresh confirmation,
independent replication, format selection, geometry-on Phase 5 and release
remain separately gated; `learning_adequate`, `phase5_ready` and `release_ready`
remain false.
