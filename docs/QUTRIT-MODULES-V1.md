# External qutrit decoder and noisy-syndrome sensor v1

Status: experimental engineering implementation for review, 1 November 2026.
Two separate dependency-free Rust CPU programs communicate through one strict
versioned JSON contract. They do not load Trinite, train a model, implement a
reward, access quantum hardware, or apply a correction to any external state.
The sensor is a **QEC-focused first adapter**, not the entire generic noisy-sensor
network stage in the [external environment roadmap](EXTERNAL-ENVIRONMENTS-ROADMAP.md).

## Boundary and source

A privileged producer/evaluator can represent an error by classical Pauli
exponents, compute its ideal syndrome and give that syndrome to the sensor.
The sensor adds explicitly supplied measurement corruption and dropout, and
emits only measured syndrome data. The decoder receives only that JSON and emits
a candidate correction or a refusal status. A privileged evaluator, kept outside
the decoder and learner, can test the residual against the stabilizer span.
There is no statevector, hidden-state inspection, model activation access,
automatic file access, controller, hardware correction or privileged evaluation
CLI. The independent engineering evaluator lives only in the test oracle.

The fixed finite construction follows QSOLKCB/QEC commit
`4ad55333f888ca33200f0e6f062cb7afd20f1eec`:

- [Cyclic five-qutrit generators](https://github.com/QSOLKCB/QEC/blob/4ad55333f888ca33200f0e6f062cb7afd20f1eec/src/qec/decoder/qutrit/codes.py).
- [Syndrome-only bounded decoder](https://github.com/QSOLKCB/QEC/blob/4ad55333f888ca33200f0e6f062cb7afd20f1eec/src/qec/decoder/qutrit/exact.py).
- [Pauli and stabilizer conventions](https://github.com/QSOLKCB/QEC/blob/4ad55333f888ca33200f0e6f062cb7afd20f1eec/src/qec/decoder/qutrit/stabilizer.py).
- [QEC qutrit contract](https://github.com/QSOLKCB/QEC/blob/4ad55333f888ca33200f0e6f062cb7afd20f1eec/docs/QUTRIT_HARMONIC_QEC.md).

The Rust finite algebra and Python oracle are independently implemented here,
with the generator constants and mathematical convention attributed to that
pinned reference. Both repositories use MPL-2.0; these files use the repository
[license](../LICENSE). No live QEC dependency, upstream source download, QEC core
modification or historical Trinite QEC fixture replacement is introduced.
QEC's true-error `correct(error)` evaluator is **not** a decoder input API.

## Build and explicit operations

Use official Rust 1.90.0, pinned in each crate, as for the
[race module](RACE-SIM-V1.md#build-and-run-explicit-opt-in).
Cargo.lock lists only the local package in each case; neither crate depends on
the other or on any third-party crate. From the repository root:

```sh
rustup toolchain install 1.90.0 --profile minimal --component rustfmt --component clippy
for module in qutrit-decoder-v1 qutrit-sensor-v1; do
  (cd external/"$module" &&
    cargo fmt --all -- --check &&
    cargo clippy --offline --locked --all-targets -- -D warnings &&
    cargo test --offline --locked &&
    cargo test --offline --locked --release &&
    cargo build --offline --locked --release)
done
python3 external/qutrit-decoder-v1/tests/oracle.py \
  external/qutrit-decoder-v1/target/release/trinite-qutrit-decoder-v1 \
  external/qutrit-sensor-v1/target/release/trinite-qutrit-sensor-v1

# Bash pipefail preserves an upstream sensor failure as a failed pipeline.
set -o pipefail
external/qutrit-sensor-v1/target/release/trinite-qutrit-sensor-v1 observe \
  < external/qutrit-sensor-v1/example-request.json | \
  external/qutrit-decoder-v1/target/release/trinite-qutrit-decoder-v1 decode
```

Each process reads one request from stdin and writes one JSON line to stdout.
The only sensor operation is `observe`; the only decoder operation is `decode`.
Malformed input or an unknown operation exits 2, with stderr and no success JSON.
A valid incomplete or unsupported syndrome is a domain outcome: exit 0 with its
explicit status and no correction. Output-I/O errors also exit 2, but an output
sink failure can leave partial bytes; callers must check exit status and validate
complete JSON before consuming it. No operation defaults to privileged access.
After explicit toolchain installation the build, test and runtime need no network.

Both local parsers admit only unescaped printable ASCII strings, decimal i64
integers, booleans, arrays and objects, with at most 16,384 request bytes and
nesting depth eight. Null, floats, escapes, non-ASCII strings, duplicate/unknown
keys, missing fields, leading zeroes, negative zero and trailing input fail.
Trit fields are integers exactly in {-1,0,+1}; booleans are not trits. This is a
restricted JSON wire grammar, not a general-purpose JSON importer.

## Genuine GF(3) code, not a missing-data ternary label

Code identifier: `cyclic-5-qutrit-v1`. The fixed [[5,1,3]]_3 stabilizer has four
rows in (x | z) order, with five entries in each half:

```text
1 0 0 1 0 | 0 1 1 0 0
0 1 0 0 1 | 0 0 1 1 0
1 0 1 0 0 | 0 0 0 1 1
0 1 0 1 0 | 1 0 0 0 1
```

For error e=(x|z), syndrome component i is g_i,x dot z minus g_i,z dot x,
modulo three. JSON -1 represents GF(3) residue 2; 0 and +1 represent themselves.
Global Pauli phase is omitted. This is classical finite-field simulation of a
qutrit stabilizer decoder, not a neural ternary-weight decoder or hardware test.

The decoder searches identity plus the 40 nonidentity one-site Pauli errors
(5 sites times 8 local exponent pairs). Their 41 syndromes are unique. On a match
it emits the **inverse** representative, negating both x and z modulo three.
It does not emit the error representative itself. There are 81 possible complete
syndromes; the remaining 40 have no candidate within this bounded table.
Runtime searches at most 41 representatives and four fixed-width checks, with
constant retained algebra state. Input/parser memory is bounded by the byte
limit; output arrays and field lengths are fixed. No random generator, floating
point, mutable global state or unbounded search is used. These are algorithmic
bounds, not a throughput, latency or memory benchmark.

## Public syndrome wire: `trinite.qutrit-syndrome.v1`

Exactly these required fields are accepted by `decode` and emitted by `observe`:

- `schema`: `trinite.qutrit-syndrome.v1`.
- `code`: `cyclic-5-qutrit-v1`.
- `syndrome`: exactly four balanced integer trits, in the generator row order.
- `observed`: exactly four booleans; false requires canonical zero placeholder.

An observed zero is valid evidence, including an all-zero syndrome. An unobserved
entry is unknown, not a measured zero, neutral physical state or successful
correction. Any missing entry produces `incomplete_observation`; v1 does not try
to infer an erased check from other checks. The decoder rejects privileged sensor
requests and extra fields such as true errors, ideal syndromes, noise, seeds,
state, outcomes or private paths. It cannot inspect data not supplied to it.

## Privileged sensor request: `trinite.qutrit-sensor-request.v1`

Exactly schema, code, `true_syndrome`, `additive_noise` and `observed` are required.
The first two have the fixed identifiers above, except the request schema is
`trinite.qutrit-sensor-request.v1`. True syndrome and additive noise each contain
four balanced trits; observed is four booleans. All supplied trits are validated
even where a measurement is unavailable.

For each observed component the sensor emits balanced(true_syndrome +
additive_noise mod 3); otherwise it emits zero with observed=false. It emits
**exactly** the public syndrome wire, with no hidden decomposition or identifier.
Different private truth/noise decompositions yielding the same measurement and
mask produce byte-identical output. Missingness hides direct readings, not
anything inferable from other public information; it is not privacy protection.

Inputs are explicit deterministic synthetic scenarios, not a sampled or calibrated
physical noise model. Data-qudit errors (which create ideal syndromes), noisy
syndrome measurements and measurement dropout are distinct phenomena. The adapter
does not simulate a quantum measurement circuit, repeated rounds, sensor topology,
fault propagation, time-correlated noise or device fault tolerance.

## Decoder result: `trinite.qutrit-decoding.v1`

Exactly schema, code, supplied syndrome, observed, status, search_assumption,
correction_x and correction_z are emitted. The search_assumption string is
`at_most_one_data_qudit_pauli`: an assumption defining the table, **not** a
measurement or certification of the unknown error weight.

- `candidate`: complete syndrome matched the bounded table; correction_x and
  correction_z each have five balanced trits. Zero syndrome yields identity.
- `incomplete_observation`: at least one check unavailable; both correction
  arrays are empty. There is no silent imputation or identity correction.
- `outside_bounded_table`: all checks available, no table match; both arrays empty.
  This does not certify an actual error weight, fault cause or impossible state.

A candidate is **not a success label or confidence**. Only a noiseless complete
syndrome and the stated at-most-one-site Pauli assumption guarantee correction.
Higher-weight errors may share the same syndrome, including nontrivial logical
errors with zero syndrome. Measurement corruption can also change a valid
syndrome into another accepted one. V1 does not correct arbitrary measurement
noise, does not estimate its probability, and deliberately retains these failures.

## Independent engineering gates and later work

The standard-library Python oracle queries the actual decoder for all 81 complete
syndromes, then independently enumerates all 59,049 Pauli vectors to compare
residuals against an independently generated stabilizer span. It checks 81
stabilizer elements, normalizer size 729, distance exactly 3 and all 40 single-site
nonidentity errors. Of the full finite space, 29,889 vectors receive a candidate,
but only 3,321 have stabilizer residuals; 648 nontrivial logical errors have zero
syndrome. These exhaustive combinatorial counts are **not probabilities, FER,
physical trial results or performance evidence**.

Additional checks cover all 16 availability masks, all 81 additive-noise patterns,
real sensor-to-decoder subprocess piping, strict malformed-input rejection,
output allowlists, private-decomposition noninterference and explicit accepted
miscorrections. BrokenPipe-safe CLI test helpers retain other write failures and
include deterministic already-exited-child regressions. Public exposed cases
are engineering conformance, not held-out Trinite learning data.

JSON contracts do not enforce OS isolation. A future controller must separate
privileged storage from learner/decoder access, bind exact source/request/output
bytes and toolchain identity, commit predictions before evaluation and enforce
process deadlines. A stdin writer that never closes can stall below the byte cap.
No such controller, native data admission, training, reward experiment, calibrated
uncertainty, real-hardware evaluation or closed PROVENANCE custody is claimed.

Affected requirements: [TRI-I01/I04/I05/I07/I08/I11/I12/I13/I14](INVARIANTS.md).
Native model and numerical updater, existing external modules, historical QEC
producers, source pins, fixtures and negative results remain unchanged. Later
research requires its own reviewed protocol, structural-family split, locked
holdout, matched baselines and explicit authorization. The generic sensor-network
and music-event modules remain future work; this paired adapter does not close
a learning, geometry, scaling or release gate.
