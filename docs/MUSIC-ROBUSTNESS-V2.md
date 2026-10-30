# Prospective music robustness v2: allocator policy and incomplete custody

These additive v2 entry points correct two PR98 boundary defects without changing
any historical v1 source, fixture, result or request. The preserved PR98 run
completed both lanes; neither failure was observed in that retained packet.
This is software robustness work, **not a new experiment or authorization to
repeat the run**. Old evidence must continue to use its original producer.

## What changes

### Exact byte-quantized CUDA allocator policy

The pinned [PyTorch 2.8 native caching allocator](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp) converts the configured fraction
to an integer byte limit, truncating `fraction * allocator_total_bytes`; its
getter divides that effective byte limit by the allocator total. An exact
comparison with the original requested fraction can reject legitimate memory
totals: for example, a 16,375-MiB total turns the requested 8-GiB policy into
8,589,934,591 effective bytes, not 8,589,934,592.

V2 checks the **exact expected byte-quantized getter value**, not a broad floating
point tolerance. It records requested and effective byte limits, the actual
allocator total and the observed fraction. The allocation policy remains at
most 8 GiB; quantization is not permission to raise the cap or accept other
fractions. Device-property memory admission and allocator-total accounting are
explicitly distinct. Nonfinite values, invalid totals and genuine cap drift fail
closed. The caching-allocator cap still is not a whole-device/process memory cap.

All other admission policy remains fixed: one visible device, capability 12.0,
native sm_120 wheel support, 14–18 GiB device class, exact separate CUDA lock,
float32, strict deterministic errors/default stream, no AMP/TF32/compile,
no fused/foreach optimizer and no silent CPU fallback. No installation, driver
change or environment repair is implicit.

### No sealed matrix with unknown attempt custody

V1 could catch an exception escaping before `attempt()` returned, then publish
an incomplete matrix identity despite missing fixed-cell records; its own audit
correctly rejected that incomplete inventory. V2 seals a matrix only when every
prescribed cell has a returned record, including the existing explicit
`not_attempted` records after a properly recorded interruption.

If setup, receipt publication or an uncaught interruption prevents a record from
returning, close the store, retain the raw partial directory, propagate nonzero
failure, and do **not** publish `matrix.json` or a matrix identity. Do not invent
an attempt receipt, fill unknown execution with fabricated status, delete the
partial run or retry. The analogous device v2 barrier covers an escaping probe, cell or development
attempt as well: preinitialized `not_attempted` labels must not conceal an
unreturned attempt or an unaccounted partial directory. Both lanes verify the
returned records and actual inventory before sealing.

Ordinary caught worker failure, timeout and recorded
interruption still retain the complete fixed record inventory and can produce
an auditable incomplete matrix. `incomplete` is not a failed learning score.

## Reuse and compatibility

V2 has distinct launcher/bootstrap/request/controller identity and binds both
its new bytes and every reused original dependency. It cannot accept a v1 request
or silently migrate/reseal old evidence. The CPU lane reuses original placement
data, scorer and numerical functions. The device lane reuses the original
adapter, loss, authoritative `update_once`, safe tensor primitives, numerical
history/probe comparison and other runtime-independent functions. Only the
runtime-aware orchestration uses corrected v2 admission. No monkeypatch of v1
globals and no second numerical update implementation is introduced.

Original numerical checkpoint/probe data formats can be retained explicitly:
their request identities bind the new v2 request/source, and are not permission
to import an old request. Reused data/admission/plan identities describe the
unchanged corpus and schedule; the new request describes changed orchestration.
V1 remains available for historical read-only custody and reproduction, not
silently replaced. The separate original CPU and CUDA dependency locks remain
unchanged.

No training rows, layouts, pitch allocation, labels, optimizer, initialization,
loss, update counts, tolerances, decoding, backend order, timing definition,
accuracy gate or selection rule changes. The original
[placement](MUSIC-PLACEMENT.md) and [device trial](MUSIC-DEVICE-TRIAL.md) budgets and
interpretation limits still apply: four native CPU cells and a separate two-cell
same-wheel CPU/CUDA comparison, not a backend-by-curriculum experiment. Existing
c/d families remain exposed development, not a fresh holdout.

## New requests only, after a separately authorized future run

Use clean exact reviewed source and separate trusted environments. Invoke the
absolute lexical venv executable directly with `-I -B`; ordinary executable
symlinks are allowed. Do not substitute its resolved system interpreter. Set
`PYTHONDONTWRITEBYTECODE=1` for child processes. Evidence paths must be fresh,
external and non-link under the unchanged path contract.

```sh
python -I -B scripts/run_music_placement_v2.py preflight
python -I -B scripts/run_music_placement_v2.py freeze --request "$REQUEST"
# Only after separate explicit owner authorization, once:
python -I -B scripts/run_music_placement_v2.py run \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --authorize-run
python -I -B scripts/run_music_placement_v2.py audit \
  --request "$REQUEST" --request-identity "$REQUEST_ID" \
  --output "$OUTPUT" --expected-identity "$MATRIX_ID"
```

The separate device lane uses the same public flags with
`scripts/run_music_device_trial_v2.py` and its own request/output/identities.
Read actual `--help`. Do not call internal worker actions, retry/resume, select a
cell or widen limits. Preflight is software-only; freeze may query CUDA metadata
but does not train. Retain printed identities independently before using them.

Capture run and audit exit codes separately, without `&&`/errexit skipping failure
handling. Audit a retained incomplete matrix only when a real independently
retained matrix identity exists. For an unsealed partial directory, report audit
unavailable and preserve raw files; do not compute/reseal a replacement identity.
The audit remains safe structure/retained-score verification, not inference or
optimizer replay, hardware/timing attestation or proof of no other attempts.

## Software conformance, not a GPU test

Regression tests include non-power-of-two totals, legitimate byte truncation,
property-versus-allocator totals, strict policy drift, first/later-cell setup and
receipt failures (including device probe/development), no identity on unsealed
failure or unknown partial directories, ordinary incomplete audits,
v1 request rejection, source/import boundaries and shared numerical-function
identity/CPU synthetic parity. Actual CUDA hardware and target 640-update studies
are unrun here. Mock runtime admission is not hardware validation.

Affected TRI-I01/02/03/05/06/07/08/11/12/13/14. This adds no learning adequacy,
release readiness, new curriculum or performance conclusion.
