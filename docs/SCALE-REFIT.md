# Fixed-code scale-refit engineering candidate v1

Status: isolated bounded quantizer and numerical conformance, **not a trained
model lane**. ID: `trinite.ternary-fixed-code-scale-refit.v1`. Existing
`trinite.decoder.v0`, absmean quantizer, native updater, checkpoint readers,
protocols, fixtures and scientific activation guards are unchanged. The native
CLI does not accept this candidate as a training or checkpoint lane.

## Attribution and mathematical scope

Independently implemented from the fixed-code least-squares objective discussed
by Li et al., [Ternary Weight Networks, arXiv:1605.04711 v3](https://arxiv.org/abs/1605.04711v3).
This is an arXiv manuscript citation, not a claimed peer-reviewed venue or a
reproduction of its training benchmarks. No author-repository code is copied.
The paper's approximate support threshold is not introduced here.

For fixed baseline ternary codes `q`, minimize `sum((w-alpha*q)^2)`. When support
is nonempty, the real optimum is `sum(w*q)/sum(q*q)`. Baseline codes have the
sign of the corresponding weight, so this is the mean magnitude on nonzero
support. Empty support leaves the baseline unchanged. This improves only the
fixed-code weight approximation; it is not the globally optimal code assignment.

The new [module](../scripts/scale_refit/quantizer.py) calls the authentic historical
quantizer for code assignment. It reduces supported products in float64, rounds
the candidate scale to float32, and materializes the actual float32 code/scale
products. It accepts those products **only if their exact SSE is strictly lower**.
Ties, worse errors, or nonfinite candidates retain the original scale and
reconstruction. No threshold, sparsity, code, surrogate mask, or updater changes.
The backward still masks by `abs(w / baseline_absmean_scale) < 1`; it does not
recompute the mask using the refitted scale. Scales remain detached.

Finite binary32 values are integer multiples of `2^-149`. The guard converts
actual inputs and rounded reconstructions to those integers and sums squared
integer differences. Thus reported SSE is its numerator divided by `2^298`.
No float64 error sum, rounded subtraction, underflow, cancellation, or overflow
can spuriously certify improvement. Float64 candidate construction is not a
claim of an exactly rounded/global floating optimum; exact acceptance is the
safety boundary. Input matrices are nonempty, strided CPU float32, finite and at
most 65,536 elements. Noncontiguous strided inputs are supported. The integer
pass adds CPU/Python overhead; no inference-speed claim is made.

## Prospective engineering suite and observed results

The [matrix specification](../scripts/scale_refit/protocol.json) was published
before numerical measurement in commit
[03294d88b1a85af09e7e0987a837933c6e41164b](https://github.com/QSOLKCB/Trinite/commit/03294d88b1a85af09e7e0987a837933c6e41164b).
This freezes engineering inputs only; it does not freeze or activate a scientific
training protocol. Twelve families cover zeros, constants, symmetry, skew,
outliers, ties, rounding-neighbor values, subnormals, floor-scale support,
near-limit values, mixed exponents and the 65,536-element bound. There are no
external datasets, model weights, optimizer steps or held-out evaluations.

The [retained source-bound report](../fixtures/scale-refit-v1/report.json) records
non-increasing exact weight SSE in all twelve cases, strict improvements in eight,
and unchanged codes/sparsity throughout. The maximum increase is exactly zero;
the aggregate exact SSE is strictly lower. The hand-checkable symmetric example
`[-2,-1,-0.5,0,0.5,1,2]` moves scale 1 to 1.5 and SSE 2.5 to 1.5 (40% lower).
These are approximation results on a visible engineering suite, not learned
accuracy, general reasoning, universal quality, packed-inference or release gains.

Fixed-input `F.linear` errors are reported separately against the actual float32
dense output. They can **increase** despite lower weight SSE: the symmetric
fixture does so. Tests additionally isolate a single coordinate where baseline
output error is zero and refit error is positive. No output-error non-increase
claim is implied. Tiny output errors and extreme cases remain exact integer
numerators rather than rounded percentage summaries.

## Verification and compatibility

Run with the existing [hash-locked CPU environment](GETTING_STARTED.md):

```sh
PYTHONPATH=src:scripts python -m unittest discover -s tests/scale_refit -v
# Complete historical CPU suites plus this new candidate:
PYTHONPATH=src:scripts python scripts/check_scale_refit_cpu.py
PYTHONPATH=src:scripts python scripts/check_scale_refit.py --output /new/report.json
# Independently retain the printed sha256 identity, then recompute:
PYTHONPATH=src:scripts python scripts/check_scale_refit.py --output /new/report.json \
  --verify-only --expected-identity sha256:RETAINED_IDENTITY
```

Creation is exclusive. Verification reads without overwriting, checks the external
report identity, reruns numerical work and requires exact source, environment and
output bytes. Cross-host verification of the retained full receipt may intentionally
fail due to environment differences. CI separately checks retained numerical
rows and complete source bindings, then produces and replays a fresh local-host
receipt. Neither path silently replaces the retained record. The source closure
binds the complete native Python package, candidate, launcher, tests, specification
and CPU locks. Future changes require a newly reviewed receipt/version; historical
receipts must remain untouched. This is software conformance, not cryptographic
attestation, full optimizer replay or independent custody of an experiment.

The independent rational tests enumerate 625 small weight matrices and all 81
ternary code assignments for a separate small support oracle. This oracle is a
lower bound only; runtime support never changes. Additional tests cover exact
subnormal contributions beside extreme errors, analytic backward masks, private
RNG noninterference, input rejection, actual CLI replay, tampering and overwrite
rejection. The dedicated three-Python [CI workflow](../.github/workflows/scale-refit.yml)
runs these checks alongside unchanged required CPU/CNR/other conformance gates.
The historical 261-case aggregate and shard inventory are preserved exactly;
`check_scale_refit_cpu.py` extends that aggregate with the new required suite.

Affected requirements: TRI-I01/I04/I05/I07/I08/I10/I12/I13/I14. No scientific
learning adequacy, geometry intervention, auxiliary execution, format selection,
Phase 5, checkpoint migration, tag or release is authorized by this increment.
