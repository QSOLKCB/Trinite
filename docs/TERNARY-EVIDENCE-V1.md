# Masked ternary evidence v1

Status: isolated exact engineering scorer, not a racing predictor, probability,
confidence estimate, reward, native model intervention or learning result.
`external/ternary-evidence-v1/` is an opt-in CPU-only, dependency-free Rust crate.
It has no native Python dependency, network/filesystem access, trainer, random
seed, private race input, privileged target adapter or ranking operation.
Historical race, queue, numerical code and fixtures are unchanged.

## Public request and exact contract

The sole operation is `trinite-ternary-evidence-v1 score < public-request.json`.
Input is one object with exactly these required keys:

- `schema`: literal `trinite.ternary-evidence-request.v1`.
- `input_kind`: literal `generic-public-trits`. This is a caller declaration,
  **not verification of public provenance or rights**. The caller must supply
  genuinely public, admissible observations; this CLI cannot detect private
  information deliberately encoded as allowed numbers or IDs.
- `weights`: 1–64 integer weights, each in 0..1,000,000,000, positive total W.
  They apply to every entry, with factor identity determined by array position.
- `entries`: 1–64 objects with exactly `id`, `active`, `trits`, `mask`.
  IDs are unique, 1–64 ASCII alphanumeric/underscore/hyphen characters.
  `active` is a boolean. Both arrays have exactly the weights' length;
  trits are integers in {-1,0,+1}, masks are booleans. False mask requires
  canonical trit 0. There is no imputation, repair or default field.

For entry trits T and observation mask M, compute exact integers:

```
W = sum(weights)
D = sum(weights[i] for observed i)
S = sum(weights[i] * trits[i] for observed i)
coverage = D / W
score = S / D, only when active and D > 0
```

The versioned result `trinite.ternary-evidence-result.v1` preserves every entry
in **input order**, including inactive/withdrawn entries. Each row contains ID,
active flag, status, score, coverage, observed_count and factor_count. Rationals
are unreduced `{numerator,denominator}` integer objects; no floating-point
rounding is performed. Active positive-D entries have status `scored` even when
S is zero; active zero-D entries have status `abstain` and score null. Inactive
entries always have status `inactive` and score null, even with observed weight.
Their coverage and counts describe supplied evidence, not eligibility. There
is no winner, rank, tie-break, completion estimate or inferred confidence.

An observed neutral trit counts as observed and contributes its weight to D;
a missing trit does neither. Observed zero-weight factors count in
observed_count but contribute no D: observing only such factors abstains.
Coverage measures weighted observation, not accuracy, certainty or calibration.
All inactive data are still strictly validated.

Admission establishes `abs(S) <= D <= W <= 64,000,000,000 < i64::MAX`.
Each product is a bounded weight times -1, 0 or +1. No rational cross-products
are used; ranking cannot accidentally overflow or promote uncovered entries.
Private request fields, seeds, ground truth and unknown fields are rejected.
No existing race request or observation schema is accepted directly: any future
public-observation projection requires its own reviewed mapping, without
revealing privileged state. The existing [race contract](RACE-SIM-V1.md)
continues to exclude BTRM scoring from simulator dynamics.

## Wire and process limits

The copied, isolated strict JSON parser pattern accepts unescaped ASCII strings,
i64 integers, booleans, arrays and objects. It rejects duplicate keys,
unknown/missing keys, floats, null input, escapes, non-ASCII, negative zero,
leading zeros, trailing data, excessive depth (over 8), and i64 overflow.
The output's documented null score is intentional; output is not request JSON.
stdin reads at most 16,385 bytes and rejects over 16,384. Individual dimensional
limits do not override that aggregate byte cap. Caller must close stdin and
apply its own wall-clock deadline: a byte cap is not protection against a pipe
that never closes. The CLI emits exactly one result line on success, exit 2
with stderr/no result on invalid input. Early BrokenPipe on stdout is clean
exit 0; other output errors fail. No shell or generic executable is exposed.

## Build and verification

Use pinned Rust 1.90.0, with rustfmt/clippy available. No crate dependencies are
fetched; official toolchain installation is separate and explicit.

```sh
cd external/ternary-evidence-v1
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --release
cargo build --offline --locked --release
python3 tests/oracle.py target/release/trinite-ternary-evidence-v1
target/release/trinite-ternary-evidence-v1 score < example-request.json
```

The new [workflow](../.github/workflows/ternary-evidence-v1.yml) runs these
checks. The independent standard-library Python integer oracle checks actual
CLI output, original fixed synthetic examples, exhaustive small masked trits,
seeded generated cases, sign reversal, factor permutation, entry-order
permutation, zero masks, neutral observations, zero weights, maximum sums,
strict input rejection, byte determinism and closed stdout. Rust tests exercise
parser bounds and an early BrokenPipe writer directly. The original example
request/result files are checked, not regenerated during tests. These are
engineering conformance artifacts, not a held-out research benchmark.

Compatibility: new opt-in wire version only. Affected requirements:
TRI-I01/I05/I12 (inspectable isolated CPU interface), TRI-I03/I11 (public numeric
boundary, no data/training admission), TRI-I04 (exact ternary scope),
TRI-I08/I13/I14 (determinism, abstention, explicit compatibility). No native
numerical semantics, phase gate, fixture identity or model capability changes.

## Inspiration and next research boundary

MELBCUPSIM at commit `f4f3bf54644f2d0eff6e42632fbe303fe01a0a2b`
inspired separating ternary evidence from simulation. Its synthetic/name-hashed
trits do not establish calibrated prediction. This module does not copy those
name-derived features or change MELBCUPSIM. See the separately proposed
[context-dependent symbol curriculum](CONTEXT-SYMBOL-RESEARCH.md) for a possible
future learning test; scoring conformance alone does not authorize training.
