# Exact integral geometry curriculum pilot v1

Status: prospective implementation, not a measured learning result. This adds
an executable bounded comparison, exact source-specific admission and conformance
checks. CI performs one training-only update, not the target quality experiment.
Publish and review the producer before running the full comparison. No old source,
fixture, architecture contract or numerical updater is changed.

## Question and narrow scope

Can the same randomly initialized small student learn geometry-to-integral setup
from a short exact curriculum? Compare geometry setup training against a matched
**arithmetic + representation control**, not a competent general model. Retain
an initial untrained score for each seed as a third reference. The control gets
the integral's explicit operands and serializes them; the geometry arm constructs
those operands from a task label, bounds and polynomial. Both receive exactly
the same integral evaluation examples. The control therefore has mathematical
and integral exposure, but no geometry task labels in its update stream.

Tasks: area under a nonnegative polynomial; volume when that polynomial is the
*supplied cross-sectional area*; and the x-coordinate of a uniform-density
lamina's centroid. Area and volume intentionally share an integration operation.
No cross-section derivation from a shape, y-centroid, trigonometry, natural-language
proof or arbitrary-function calculus is included. This is narrow symbolic mapping.

For normalized u in [0,1], densities are s*(1+b*u+c*u^2), b,c in {0,1,2}, s>0.
Expand using u=(x-h)/L with L>0. Density and mass are strictly positive on the
domain. Centroids use the global-coordinate first moment integral of x*f(x),
divided by the mass integral of f(x). Exact antiderivatives are independently
checked by Simpson quadrature in normalized coordinates, exact at these degrees.
That calculator correctness is **not model learning**.

## Admission, split isolation and exposure

Only first-party generated rational facts and minimal symbolic carriers enter.
No textbook/paper prose, teacher answers, pretrained weights, remote code or corpus
is downloaded. Admission records commissioned authorship, rights basis, scope,
procedure, complete dataset identity and exposure. Every item binds exact generator
and oracle source revisions, explicit null generation/split seeds (exhaustive,
nonrandom enumeration), the fixed split procedure and independent verifier
outcomes. Its separate semantic ordering identity preserves the originally
declared curriculum order; the full item identity includes all provenance. Regeneration rejects edits to
rows, labels, lineage and splits. This source-specific admission does not relax
[DATA](DATA.md); software licensing is not itself a data rights basis.

A family is the normalized coefficient pair (b,c), before scale, translation,
interval stretch, task projections or notation. Six training families exclude
(1,2), (2,1), (2,2). Validation is (1,2); test is the latter two. Every projection
and transform stays within its family. Train transforms (h,L,s) are (0,1,1),
(1,1,2),(0,2,1). Held-out families additionally have (-1/2,3/2,1/2), an unseen
combination of translation, interval and scale. Counts: 54/12/24 examples across
6/1/2 families. The audit rejects equal mathematical setups across splits.

These are unseen coefficient combinations for the optimizer, not unseen formula
templates. All task templates are shared. Transforms are correlated, not independent
trials. The public suite is **exploratory**, not a secret or confirmatory benchmark.
No held-out labels enter updates, periodic diagnostics, early stopping, tuning or
selection. The native updater field named `validation` is assigned training rows
only; its output is labelled `train_only_diagnostics`. Final evaluation uses all
splits at the fixed final step. There is no outcome-driven model selection.

## Matched student and fixed budget

Profile: `trinite.integral-geometry-dense-128.v1`. Unchanged native dense Decoder,
new explicit configuration: context 128, one block, width 16, two heads, FFN 32,
tied byte-token output, CPU float32. Not a PR #49 checkpoint or a change to any
historical model. No ternary speed claim. Geometry auxiliary loss stays zero;
this is a task curriculum, not the latent-geometry Phase 5 intervention.

Seeds 0/1, two arms, 32 native AdamW updates/arm, batch 4, native default optimizer
settings. Seed-hash row order, interleaved setup/evaluation. Identical ordered row
IDs/answers give 1320 scored targets for seed 0 and 1333 for seed 1, including EOS,
below the 4096 cap. Initialization, model, optimizer, steps and target exposure
match. Prompt lengths and difficulty differ: **not equal FLOPs or walltime**.
A short pilot may fail; do not tune this budget using its outcomes.

Overlength inputs fail rather than truncate. Setup grammar degree<=3, bounded
rational literals. Workload-wide caps: 97 setup tokens and 16 answer tokens including
EOS, unconstrained greedy. Unterminated/malformed predictions fail. No per-item
length hint, calculator, syntax mask, repair or retry. Four cells have a 900-second
process deadline; each training invocation has 180seconds. Failures retain logs
and all completed cells as newline-delimited JSON.

## Separate metrics

- Setup semantic correctness compares parsed rational integrands, bounds and ratio
  components. Non-reduced fractions and trailing zero coefficients are accepted.
  Equivalent antiderivatives, variable substitutions, different bounds with equal
  values and jointly rescaled ratios are outside this grammar's equivalence.
- Canonical setup measures exact notation separately.
- Oracle-setup evaluation gives the correct integral to the model for evaluation:
  a conditional diagnostic with supplied gold setup, not end-to-end reasoning.
- End-to-end parses the emitted setup and gives **that** to the model to evaluate.
  Invalid setup fails; no gold substitution. Numerical correctness alone need not
  imply correct setup, so preserve both components and all raw predictions.

Counts per task/split retain family/novel-transform labels. No aggregate closes
a Phase 5 or capability gate. Two seeds and few correlated families do not establish
a statistically reliable benefit.

## Execute only after publication/review

Use a fresh trusted checkout of the reviewed commit and unchanged hash-locked CPU
environment from [GETTING_STARTED](GETTING_STARTED.md). This executes trusted
source; it is **not** an adversarial import-cache auditor or hardware attestation.
`-I` excludes ambient PYTHONPATH/user-site startup; interpreter, installed locked
environment and checkout remain trusted. Record the Git commit separately: a
request binds actual bytes, not an independently attested source archive.

From the checkout root, choose fresh output paths outside it:

```sh
mkdir -p "$HOME/trinite-integral-reports"
request="$HOME/trinite-integral-reports/request.json"
report="$HOME/trinite-integral-reports/report.json"
request_id=$(python -I scripts/run_integral_geometry.py freeze --request "$request")
# Save request_id independently BEFORE target outcomes.
python -I scripts/run_integral_geometry.py run --request "$request" \
  --request-identity "$request_id" --output "$report"
# Save the first printed line (report identity) independently.
python -I scripts/run_integral_geometry.py replay --request "$request" \
  --request-identity "$request_id" --expected-report "$report" \
  --expected-identity 'sha256:REPLACE_WITH_RETAINED_REPORT_ID' \
  --output "$HOME/trinite-integral-reports/replay.json"
```

Final artifacts publish complete same-directory temporaries atomically without
clobbering. Each full output filename gets a unique attempt directory. Retain all
requests, statuses, logs and completed cells, including failures, before any fresh
attempt. Status distinguishes the worker exit code from the controller exit code
and records final publication as not attempted, failed or published. A successful
worker is not evidence that the requested destination was published. Collisions
and final-link I/O errors retain a failed controller receipt and the worker report.
Both worker and controller reject completed reports bound to a different supplied
request; the experiment also checks that binding before any model work and keeps
its end-of-run source/environment comparison.

Resource status records Linux RUSAGE_CHILDREN ru_maxrss in KiB before and after
the worker is reaped, including failure/timeout. This is the controller's cumulative
reaped-child high-water mark, not a subtractable counter or whole-machine memory.
A fresh CLI normally launches one worker; prior children, if present, can dominate
this value. Parent-process RSS, GPU memory and a hard memory limit are not measured
or implied. Resource observations stay outside deterministic comparison bytes.

Replay restarts at native initialization and reruns the whole protocol,
requiring identical deterministic report bytes in the bound environment. It is
not a cross-host guarantee or resumable checkpoint. Source/environment changes
require a new request; never patch old requests or discard negative records.

## Conformance and compatibility

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/integral_geometry_v1 -v
```

Pinned numerical dependencies required, never silently skipped. Tests cover exact
independent cases, centroid affine invariants, admission mutation, split isolation,
semantic vs format scoring, capacity, paired initialization and targets, one
training-only update, and no-gold-fallback behavior. CI runs 3.11/3.12/3.13 and
retains prospective requests, not target quality outcomes.
Affected requirements TRI-I01/I02/I03/I05/I07/I08/I09/I11/I13/I14. Historical
source closures, evidence, teacher policy, numerical identities and gates remain
unchanged. This does not refresh old fixtures or report a learning gain.
