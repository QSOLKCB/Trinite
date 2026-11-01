# External sensor network v1

Status: additive engineering implementation for review, 1 November 2026.
The [external roadmap](EXTERNAL-ENVIRONMENTS-ROADMAP.md) stage 4 now has a
bounded generic sensing environment alongside the unchanged specialist
[qutrit sensor/decoder](QUTRIT-MODULES-V1.md). This is deterministic synthetic
measurement generation, not a trained predictor, fault detector or hardware
network. Modules come first; training comes later under separate protocols.

## Build and run

Use the same explicit official Rust 1.90.0 baseline as the prior modules.
The crate has no third-party dependencies and Cargo.lock contains only itself.

```sh
cd external/sensor-network-v1
rustup toolchain install 1.90.0 --profile minimal --component rustfmt --component clippy
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --release
cargo build --offline --locked --release
python3 tests/oracle.py target/release/trinite-sensor-network-v1

target/release/trinite-sensor-network-v1 observe < example-request.json
target/release/trinite-sensor-network-v1 simulate-privileged < example-request.json
```

After explicit toolchain installation, the build and runner need no network.
Each invocation reads one request from stdin and emits one JSON line. Operations
are explicit; no operation defaults to privileged output. Invalid input exits 2,
writes stderr and emits no partial JSON success. Nothing loads Trinite or changes
model weights. The runner creates no files, downloads nothing and accepts no
shell commands. A later controller can save only observation stdout for Trinite.
Process deadlines, file isolation, admission, prediction commitment and learner
response validation are not implemented. JSON is not an OS sandbox. In particular,
a caller that does not close stdin can stall below the byte limit.

## Privileged request: `trinite.sensor-network-request.v1`

[Example request](../external/sensor-network-v1/example-request.json).
All fields are required; duplicate/unknown fields fail, including inside sensors.
The fixed restricted JSON grammar admits unescaped printable ASCII strings, i64
decimal integers, booleans, arrays and objects. Null, floats, escapes, non-ASCII
strings, leading zeroes, negative zero, trailing input and nesting beyond eight
are rejected. The complete serialized input is limited to 16,384 bytes, including
whitespace. That byte budget applies even to otherwise dimension-valid inputs.
This parser is locally versioned; earlier module parsers are unchanged.

| Field | Meaning and bounds |
| --- | --- |
| schema | Exactly `trinite.sensor-network-request.v1` |
| latent | 2..64 tick rows, each with the same 1..4 integer signals in −1000..1000 |
| observe_ticks | Number of visible completed ticks, 1..steps−1 |
| sensors | 1..8 sensor objects, each defined below |

Each sensor contains exactly:

| Field | Meaning and bounds |
| --- | --- |
| id | Unique integer 1..65535, supplied independently of hidden state |
| signal | Zero-based index into each latent row |
| noise | One integer −100..100 per tick |
| fault | One mode per tick: 0 healthy, 1 additive bias, 2 fixed response |
| fault_value | One integer −1000..1000 per tick; must be 0 when mode is healthy |
| observed | One boolean per tick indicating available measurement |

Each schedule has exactly `steps` entries. The static topology is a bipartite
assignment from each sensor to one signal, not a communications mesh. Multiple
sensors may read the same signal and a signal may be unconnected. There is no
packet routing, delay, consensus, sensor interaction or physical unit model.
Sensor IDs and topology are public metadata; a future producer must never encode
latent answers in IDs, ordering or split choices. Outputs sort sensors by ID,
independent of input ordering. Signals keep their declared zero-based indices.

## Frozen measurement semantics

At each tick, independently for each sensor:

1. Read `ideal = latent[t][signal]`.
2. Compute the pre-noise response: healthy → ideal; bias → ideal + fault_value;
   fixed-response → fault_value.
3. Compute `measured = response + noise[t]`.
4. Export measured only if observed is true, otherwise export placeholder 0 with
   observed false. Missingness does not alter any latent state or measurement.

The fixed-response ("stuck") fault is applied **before noise**. Its visible
measurement can vary with noise; it is not a frozen ADC output after noise.
Bias is an explicit per-tick offset, not an internally integrated drift process.
There is no clipping, threshold detector, distribution sampling, RNG, floating
point or mutable global state. Ideal is bounded by ±1000, response by ±2000,
measured by ±2100; i32 arithmetic is safe and release overflow checks remain on.
At most 512 sensor readings and 256 latent samples are generated. Runtime and
retained output are O(steps × sensors); topology sorting is bounded to eight IDs.

Fault labels describe the injected mode, not guaranteed observability. A zero
bias or stuck value matching the current signal can have no visible effect.
Latent changes, bias and noise can cancel. Identical public measurements can
arise from different true states and fault modes, so perfect fault identification
is not generally possible. Dropout is independent availability, not an inferred
fault label or a known operational outage. No fault-mode flag is public.

Schedules are explicit deterministic inputs. Different valid future schedules
can follow the same visible prefix. No stochastic likelihood, calibrated
uncertainty, realistic noise spectrum or predictive advantage follows from these
examples. A future experiment must declare its family/distribution assumptions.

## Public observation versus privileged truth

`observe` emits `trinite.sensor-network-observation.v1` with exactly:

- schema, signal_count, sensors (ID and signal only), observe_ticks, horizon;
- rows for ticks 0..observe_ticks−1, each with tick, values and observed.

The values and mask vectors follow the sorted sensor IDs. Measured zero with
observed true is real observed zero; zero with observed false means unavailable.
The output contains no latent values, ideal readings, noise, fault modes,
fault values, future measurements, future dropout, hidden IDs or outcome labels.
It computes only visible prefix measurements. Valid requests with identical
public settings and measured prefixes produce identical observations even when
hidden truth differs. Valid future changes at a fixed horizon cannot change the
observation. Validation still checks the complete privileged request first.

All-false masks still expose topology, signal count, cutoff and horizon. Public
readings may reveal true state through legitimate inference; masks and output
allowlists are not a privacy guarantee. Integer measurements retain precision;
this module does not silently quantize observations to ternary. Representation
comparisons are separate from ternary model-weight claims.

`simulate-privileged` emits `trinite.sensor-network-ground-truth.v1`, privileged
true, signal_count, sorted sensors, observe_ticks, steps and every tick row.
Rows contain tick, latent and readings. Each reading has id, ideal, fault,
fault_value, response, noise and measured. These are computed truth even when a
sensor is unavailable; they are **not observed measurements**. Masks are omitted
from this truth output, so changing only observed booleans leaves privileged
bytes unchanged. Preserve the private request if replay requires those masks.

The privileged output can support later reconstruction or fault-label scoring,
but no scorer, success flag, reward, abstention policy or controller is included.
A missing reading is not a zero prediction or successful example. Correctness of
these equations does not establish that a learned model can recover their inputs.
Neither operation accesses hardware, QEC repository code or native model state.

## Engineering verification and limits

Affected requirements: [TRI-I01/I05/I07/I08/I11/I12/I13/I14](INVARIANTS.md).
Rust tests cover fault ordering, extrema, exact topology, mask invariance,
canonical sensor ordering, prefix isolation, observationally identical faults,
strict bounds and early-closing CLI stdin. The independent Python standard-library
oracle recomputes complete public/private results across schedule families,
cutoffs, masks and extrema. It retains cancellation/no-op-fault cases rather than
mistaking an injected label for a detectable effect.

These are exposed original synthetic conformance fixtures, not admitted native
training data, fresh held-out families, independent-host reproduction or real
sensor evidence. Freeze source/request/toolchain/output identities and preserve
failure receipts for later experiments. This crate does not claim closed
PROVENANCE custody. Existing native numerical code, retained evidence and the
five previous external crates remain byte-for-byte unchanged.

Later learning still needs a source-bound producer/protocol, rights/admission,
structural train/development/test families (including noise, fault and topology
families), matched baselines, frozen calibration metrics, resource accounting
and failure retention. Keep related masks and hidden-state counterfactuals in
the same split. No training, calibrated uncertainty or cross-domain transfer gate
is cleared by this software implementation.
