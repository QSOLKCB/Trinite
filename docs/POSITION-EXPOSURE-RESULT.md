# Positional exposure: negative development result

Recorded **27 October 2026 UTC**, from the owner-authorized twelve-cell run of
[frozen PR #79](POSITION-EXPOSURE-V1.md) at merged producer
[`e68666973ead5bc774673570550731eb05835dc3`](https://github.com/QSOLKCB/Trinite/commit/e68666973ead5bc774673570550731eb05835dc3).
All twelve declared cells completed in one controller invocation. This
preservation adds no training, generation, retry, checkpoint selection,
curriculum change or future-experiment authorization.

**All four prespecified contrasts failed their frozen descriptive criteria.**
Balanced offsets degraded both training fit and unseen performance when
measured at the prescribed evaluation offset zero. This is a negative
**development** result for this schedule and budget, not a universal rejection
of offset methods or proof of a model capacity limit.

## Every final cell

Primary exactness requires the complete reversed, fixed-width answer and actual
EOS. A whole family succeeds only when every admitted rendering is correct.
Rows, repeated checkpoints and correlated renderings are not independent trials.

| Arm | Seed | Training families /64 | Unseen rows /56 | Unseen families /24 | Three-digit rows /23; families /12 | Four-digit rows /23; families /12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequence fixed | 0 | 64 | 16 | 4 | 0; 0 | 0; 0 |
| Sequence fixed | 1 | 64 | 16 | 4 | 0; 0 | 0; 0 |
| Sequence fixed | 2 | 64 | 22 | 7 | 0; 0 | 0; 0 |
| Coupled fixed | 0 | 64 | 17 | 3 | 0; 0 | 0; 0 |
| Coupled fixed | 1 | 64 | 18 | 4 | 0; 0 | 0; 0 |
| Coupled fixed | 2 | 64 | 26 | 10 | 0; 0 | 0; 0 |
| Sequence balanced | 0 | 59 | 9 | 1 | 0; 0 | 0; 0 |
| Sequence balanced | 1 | 28 | 2 | 0 | 1; 0 | 0; 0 |
| Sequence balanced | 2 | 49 | 6 | 1 | 0; 0 | 0; 0 |
| Coupled balanced | 0 | 13 | 8 | 0 | 0; 0 | 0; 0 |
| Coupled balanced | 1 | 15 | 3 | 1 | 0; 0 | 0; 0 |
| Coupled balanced | 2 | 31 | 4 | 0 | 0; 0 | 0; 0 |

All six fixed cells solve 154/154 unique training rows and 64/64 families.
The balanced arms fail to retain that fit at evaluation offset zero. Every
cell scores **zero complete longer-number families** and **zero complete unseen
carry-required families out of eight**. The sole correct longer row is
sequence-balanced seed 1, `556+555=1111`; the swapped-operand rendering fails.
This isolated success is not whole-family transfer.

The original score files retain all five unique-training diagnostics, final
strata, raw outputs, secondary numeric correctness, termination flags,
carry/no-carry slices and consistency **alongside correctness**. No best seed,
checkpoint or favorable metric replaces the prespecified outcomes.

## Four contrasts, without a pooled success claim

Each directed contrast independently uses the frozen family margins: right-arm
seen fit at least 95% with no more than 2 percentage points regression;
unseen gain at least 10 percentage points; and each longer stratum at least
25% correct with a gain of at least 10 percentage points. Every seed must meet
all conditions. These are descriptive bars, not significance tests.

| Right minus left | Unseen-family differences, seeds 0 / 1 / 2 | All frozen conditions |
| --- | --- | --- |
| Coupled fixed minus sequence fixed | −1 / 0 / +3 | Not met |
| Coupled balanced minus sequence balanced | −1 / +1 / −1 | Not met |
| Sequence balanced minus sequence fixed | −3 / −4 / −6 | Not met |
| Coupled balanced minus coupled fixed | −3 / −3 / −10 | Not met |

The checker recomputes each contrast separately; there is no omnibus gate.
None authorizes scaling, learning adequacy, fresh confirmation or release.

## Exposure is not arithmetic learning

Each cell used the same 86,528-parameter dense CPU model, arithmetic data,
format, seeded effective initialization, optimizer and total budget:
4,000 updates, 32,000 scheduled visits, 121,600 scored answer/EOS targets and
307,200 actual live shifted inputs. The latter includes 300,800 predictor
inputs and 6,400 unscored short-row EOS inputs. Twelve cells double the prior
six-cell aggregate workload; this was not a compute-saving experiment.

Balanced arms rotate offsets 0–7 by the frozen cycle schedule and pass the
**embedding-address × coarse-token-role** coverage gate. Detailed higher
operand-significance and answer-slot deficits remain explicitly reported.
Touching an address does not establish gradient support, a learned digit-role
mapping or a general addition algorithm. Range, length and composition still
shift together in the longer probes.

At unchanged total budget, presentations at evaluation offset zero fall from
200 to 25 per cycle slot (duplicated diagonal rows: 400 to 50); the remaining
visits move to seven other offsets. Reduced offset-zero exposure is a plausible
contributor to poorer fit, not a proven causal explanation. No automatic
training increase, retry, curriculum change or threshold relaxation follows.
A future intervention must be separately specified and frozen; current probes
remain development-exposed, not fresh confirmation.

## Fixed-control reproducibility

Private static comparison found all **30 fixed-arm checkpoint safetensors
byte-identical** to the earlier [PR77 study](POSITION-STUDY-RESULT.md), and all
**5,232 fixed-arm prediction records** identical. Full journal bytes differ
legitimately because experiment identities and addressing descriptions differ.
This is a same-environment reproduction observation, not independent-host
replication or a new capability claim.

The public checker can verify the 5,232 prediction/model-identity comparisons
against the previously bound public subset. The checkpoint byte comparison
belongs to the private audit: checkpoints are not included here. Do not turn
that private observation into a claim of complete public tensor verification.

## Evidence and limits

The [public subset](../fixtures/position-exposure-result-20261027-v1/README.md)
retains **229 original files byte-for-byte**: matrix plus 19 records per cell,
covering **10,464 predictions**, including 3,072 final-checkpoint predictions.
No original artifacts are redacted, amended or resealed. Selected records are
public synthetic arithmetic outputs, not the full private handover.

- Original request: `sha256:cba9f27710cdc85016527ad412d8d80db0655ccfe0f6f7515f9ee0814fb8ede9`.
- Original matrix: `sha256:199bd917fece38e34190e103f11aad641d07ccf9005fadf3b6631a61c4dad78e`.
- Private delivery v1 archive SHA-256: `0b245bfffebdacb0491a36b6d6fd969bb4c312dcf4dc83fe4ad4f18d8cb8bac1`.
- Private delivery v2 archive SHA-256: `63646cc27eeaa5b90d15170ad1f2a42de731f0a3de4f34128318956d1ed2142a`.

V2 corrects packaging manifest ordering and adds packaging receipts. Original
request, attempt and producer-source bytes are unchanged; both deliveries remain
retained privately. It is not a second experiment or a repair of target results.

Private review checked both archives, v2's 2,719 payload files, 1,364 retained
producer files against published Git blobs, all 256 request-bound source
commitments, original identity receipts, 1,224 cell artifacts, 60 finite safe
checkpoint structures, Adam states, schedule/cursor metadata and before/after
inventories. Independent static analysis recomputed all 10,464 retained outputs,
perturbation metrics and offset/address-role accounting. None of these checks
replayed trained generation or optimizer updates or attested the host.

`source-bindings.json` is an explicitly derivative source-only private-audit
receipt, **not the original request**. Original references to omitted request,
environment, checkpoints and other artifacts are unchanged. Their hashes are
not substitutes for the missing bytes. This is **partial public custody**,
not closed custody or full public checkpoint verification.

The checker reuses authoritative admission, retained-score validation,
exposure accounting and descriptive reporting. Imports need the pinned CPU
environment, but verification constructs no model, restores no checkpoint and
runs no training or generation. Source bindings, membership, journals, reports,
contrasts and read-only behavior are checked without numerical replay.

```sh
PYTHONPATH=src:scripts python -B scripts/check_position_exposure_result.py
PYTHONPATH=src:scripts python -B -m unittest discover \
  -s tests/position_exposure_result_v1 -v
```

Affected TRI-I03, I07, I08, I11, I13 and I14: preservation only, no numerical
semantics change. Historical protocols, source closures, fixtures and negative
results remain unchanged. Learning, fresh confirmation, Phase 5 and release
readiness remain false and independently gated.
