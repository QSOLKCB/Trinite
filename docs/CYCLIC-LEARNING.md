# Matched cyclic representation pilot: evidence v2

Status: prospective implementation, not an executed learning-quality result.
Protocol, request and report schemas are `trinite.cyclic-learning-protocol.v2`,
`trinite.cyclic-learning-request.v2` and `trinite.cyclic-learning-report.v2`.
They require per-arm untrained scores; the earlier v1 report schema is rejected,
not silently reinterpreted. The unchanged model, generated data and package lane
retain their separate v1 identifiers. Prior prospective v1 sources remain in Git
commits cf37bb4a2b1cb757abda9a29ec632b77c6209860 and
0b53bea39c5b8bcf1c308f80500e6e211b4c6586; no target v1 experiment was run. This
checkout-only lane is separate from PR58 exact cyclic software conformance,
NEXUS donor conformance, integral geometry and ternary scale-refit experiments.
It does not unlock Phase 5 or establish geometric reasoning.

## Question and frozen scope

Can the unchanged small native dense decoder learn the same fixed Z/360 rotation
composition tasks under two specific input encodings? Decimal operands have five
ASCII digits, such as `00072`; sexagesimal operands have `01:12`. Prompts are
`r10:rot360[00072,00012]=` and `r60:rot360[01:12,00:12]=`. Both require the same
canonical three-digit decimal answer, here `084`, followed by EOS. Radix labels,
prompt bytes, byte-token counts and four target tokens have matched lengths.
Colon use, digit frequencies and positional difficulty differ. This is NOT an
isolated causal test of number base, equal wall time, or intrinsic base-60 merit.
The modulus never changes. Sexagesimal notation does not add geometric laws.

The versioned model is `trinite.cyclic-encoding-dense-64.v1`: native dense,
context 64, width 16, one block, two heads, feed-forward 32, geometry loss zero.
Dense avoids confounding this encoding comparison with a quantizer change; this
is not evidence for ternary edge inference. Native initialization seeds 0/1 are
paired across arms. All four cells use unchanged AdamW and 32 steps of batch 4,
128 visits, exactly 512 answer/EOS targets. No early stopping, selection, repair,
solver fallback, pretrained weights, teacher answers or tuning is used.

## Source-specific admission and lineage

Only the finite mathematical facts and minimal formal strings produced by
`scripts/cyclic_learning_v1/data.py` are admitted under the numeric-fact rule in
`docs/DATA.md`. This is a new source-specific admission, not transfer of PR58's
conformance input provenance into model training. No Google/Claude prose or
AI-generated answers are admitted. Generator software follows the repository
license; mathematical facts and minimal templates are the data rights basis.
The source record and every hash-bound item include origin, revision, fixed
seed/procedure, formal specification, family, both rendered prompts, answer,
rights basis, verifier outcome and lineage. The frozen request includes all 288
items, not merely their hashes, plus source identities and exact schedules.

The fixed anchor product is a=(7,31,59,61,119,179,239,301), b=(11,47,83).
Each relation a+b+(-c)=0 generates permutations and simultaneous negation.
Canonical relation orbits are ranked by SHA256 before splitting 16/4/4 families
into train/validation/test. All 12 rearrangement/mirror cases and both notations
stay together. There are 24 unique families and 192/48/48 rows; any orbit collision
fails rather than silently replacing a candidate. This prevents those explicit
correlations crossing splits, not every possible algebraic dependence. These are
public same-task family holdouts, not unseen moduli/templates or confirmatory
secret tests. Counts are matched between encodings, not stratified by answer or
wraparound frequency; all latent cases are retained for auditing these distributions.
No outcome-adapted selection occurs.

Each answer is checked both against PR58 CyclicElement addition and an independent
bounded repeated-unit-rotation counter. Rendering is independently parsed;
duplicates, invalid provenance, oracle disagreement and context overflow fail
closed. All candidate families are admitted, with zero exclusions; counts and
exact source closure are frozen. Generated exact facts are not empirical model
quality. CI tests only contracts, ordinary failure transport, and one real update
per arm. The full target experiment must wait until publication and review.

## Scoring, diagnostics, and evidence

Only unique training rows reach optimizer data and periodic loss diagnostics.
The 192 unique rows and 128 scheduled visits are distinct quantities; rank schedules
are seed-specific and shared by arms. Validation/test labels are read for scoring
only after every fixed training cell finishes, never for updates or selection.
Raw unconstrained greedy generation is capped at four tokens. Success requires
three ASCII decimal digits matching the exact answer and EOS, without repair.
Per-row generated tokens and split-level exact/canonical counts are retained.
After all four cells finish training, each seed initial model is reconstructed and
its identity checked against the recorded initialization. Raw untrained generation
is scored separately for each encoding. Final-minus-untrained correct-count deltas
are descriptive controls, not evidence of an encoding benefit by themselves.
Full replay retrains all cells and compares deterministic report bytes, including
model/optimizer tensor identities and update histories, not only inference.

Use the pinned Linux x86_64 CPU requirements.lock environment. From a trusted
checkout, freeze with `python -I scripts/run_cyclic_learning.py freeze --request
/path/request.json`. Save the printed identity. After review run the same launcher
with `run --request /path/request.json --request-identity sha256:... --output
/path/report.json`. Replay adds `replay` instead of `run`, plus `--expected-report`
and `--expected-identity`, using a fresh output path. Never overwrite old evidence. Replay retains the verified bounded expected-report
bytes and their identity inside the attempt before launching the worker; comparison
receipts remain auditable if the external baseline is subsequently moved.

The controller retains request, partial JSONL records, stdout/stderr, worker report,
publication journal and status in a unique full-output-name attempt directory.
Numerical report identity is separate from elapsed time and Linux cumulative
reaped-child peak RSS (KiB), explicitly not a per-child delta or memory cap.
Publication is atomic no-clobber; failures are not successful experiments.
If status publication fails, consult the durable journal; a complete output alone
is not a successful attempt. Total storage failure can leave only publication
intent, so absence of a receipt must remain inconclusive, never fabricated success.
The isolated launcher assumes a trusted checkout, not hostile Python/source safety.

## Contracts and resource bounds

Affected requirements: TRI-I01 inspectability, TRI-I02 native initialization,
TRI-I03 source-specific admission, TRI-I05 CPU resources, TRI-I07 immutable IDs,
TRI-I08 exact replay scope, TRI-I09 no assumed geometry benefit, TRI-I10 no
small-model superiority or universal hardware claim, TRI-I11 heldout isolation,
TRI-I12 explicit interfaces, TRI-I13 failure retention and TRI-I14 contract-change
reporting. No invariant is relaxed. This prospective correction changes the new
pilot report to include per-arm untrained baselines and replay baseline custody;
its source-bound requests must be re-frozen. No historical lane or fixture changes.
Each cell has a 180-second native deadline; the controller bounds the whole worker
at 900 seconds. Request/report/partial records are bounded to 8 MiB; journal records
are bounded to 8192 bytes. These are engineering limits, not measured performance.
The per-item generator revision binds exact generator and independent reference
source bytes, and the admission binds the complete generated content identity.
