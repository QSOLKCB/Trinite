# TT-003-LOGIT-FIX-001 — binary64 positive-subnormal ratio guard correction

**Status: separate corrective source revision; no target run or historical resealing.**

This is a strictly scoped engineering repair for the auxiliary binary64
EOS/rest probability-ratio cross-check used by TT-003-LOGIT-001 v2.
The original frozen diagnostic ran at commit
`a46dd431c0cc4b59277afbf5d11775033cc2dba6` and produced immutable
request `8585174ce406a7c57ffd1b3f69fb2996a7488064a3cd54789147ad00a36e52b9`,
result `afb27a799dc4fa72fd608a98c8e9eb09fc538e72cb2e0e7add9f9ac5c616121c`,
and complete 13,689-position evidence. **That result is not modified or
retroactively attributed to the patched source.**

## Defect and mathematical contract

For an FP32 input vector with 258 logits at exactly `0.0` and EOS logit
at exactly `-730.0`, the stable EOS/rest log-odds is

`-730 - log(258) = -735.5529595849216`.

The exponentiated binary64 EOS probability is positive but *subnormal*,
approximately `3.576e-320`. Although the unmodified logit vector is
perfectly valid, `log(pEOS) - log(pRest)` differs from the stable
log-space odds by about `1.21e-5` because binary64 subnormals retain
fewer significant bits. The old auxiliary check compared the two with
a fixed `1e-12` tolerance and incorrectly rejected valid input.

The repaired cross-check runs only if **both** the EOS mass and the
summed rest mass are positive *normal* binary64 numbers, i.e.
`pEOS >= sys.float_info.min` and `pRest >= sys.float_info.min`.
When either mass is zero or positive subnormal, the optional
`eos_rest_probability_check_hex` field remains `null`. This means
"not independently checked in probability space", **not** an invalid
observation or a zero/undefined stable log-odds value.

The authoritative stable log-space EOS/rest log-odds,
full 259-token normaliser, target log probabilities, rankings,
FP32 logit bytes, original `1e-12` tolerances and independently
checked logistic probability are **unchanged**. No clipping,
epsilon padding, broad tolerance relaxation, log-probability
substitution, altered model inference or alternate backend is permitted.

The regressions exercise negative and positive EOS extremes
(`-735, -730, +730, +735`), where either pEOS or pRest is positive
subnormal; normal-range `-700, 0, +700` comparisons must remain
enabled and checked. Existing underflow and FP32 roundtrip contracts
also remain mandatory.

## Source and evidence boundaries

This is a new, source-bound corrective revision layered on top of the
historical v2 implementation; it does not create a new model, weight
checkpoint, optimiser or result schema. Its source closure binds this
document, modified arithmetic and regression tests. A future
owner-authorized inference run would need an **entirely new frozen
request and SHA-256 identities** from the exact reviewed corrective
commit, even if the evidence schema name remains compatible.

Historical TT-003-LOGIT-001 v2 must be audited or replayed with the
original `a46dd431` checkout and its externally saved anchors. Do
not run a new verifier against old evidence and call any mismatch
proof of corrupted data. No target inference rerun, training,
GPU execution, upload, merge, publish or TT-004 promotion is
authorized by this fix.

Affected TRI-I01/I05/I07/I08/I10/I12/I13/I14. This corrects numerical
conformance, not claims about learning or statistical generalization.
