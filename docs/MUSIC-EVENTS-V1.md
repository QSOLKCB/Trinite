# External music-event environment v1

Status: additive engineering implementation for review, 1 November 2026.
This standalone, dependency-free Rust CPU module implements stage 5 of the
[external environment roadmap](EXTERNAL-ENVIRONMENTS-ROADMAP.md). It computes
original synthetic note grids and impaired observations; it does not train or
load Trinite, reconstruct missing music, score a learner, synthesize audio,
parse MIDI or establish audio-preservation fidelity.

## Build and run

The explicit official Rust 1.90.0 toolchain matches the existing external
modules. This crate has no third-party dependencies and is not installed into
the native Trinite runtime. After toolchain installation, all checks are offline.

```sh
cd external/music-events-v1
rustup toolchain install 1.90.0 --profile minimal --component rustfmt --component clippy
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --release
cargo build --offline --locked --release
python3 tests/oracle.py target/release/trinite-music-events-v1

target/release/trinite-music-events-v1 observe < example-request.json
target/release/trinite-music-events-v1 simulate-privileged < example-request.json
```

Both operations require explicit selection, consume one request on stdin and
emit one JSON line. Invalid input or operation exits 2 with diagnostic stderr
and no partial JSON success. Non-UTF-8 arguments are rejected without echoing
the argument. The runner creates no files and performs no network, shell, model
or hardware operation. A caller may save observation stdout for Trinite; that
controller, exclusive file creation, prediction commitment and OS access
isolation are not implemented. A future controller must enforce a wall-time
limit: stdin that never closes can stall below the byte limit.

## Privileged request: `trinite.music-events-request.v1`

[Example request](../external/music-events-v1/example-request.json).
All fields are required. Unknown and duplicate fields fail. The parser accepts
only unescaped printable ASCII strings, i64 decimal integers, booleans, arrays
and objects. Null, floats, escapes, non-ASCII strings, leading zeroes, negative
zero, trailing data and nesting beyond eight are rejected. Input is capped at
16,384 bytes; this independent cap can reject otherwise dimension-valid large
requests. It is not a general JSON text decoder.

| Field | Bounds and interpretation |
| --- | --- |
| schema | Exactly `trinite.music-events-request.v1` |
| steps | 2..128 integer ticks in the episode |
| observe_ticks | 1..steps−1 completed ticks exposed to the learner |
| channels | 1..4 channel objects; array position is public channel index |
| events | 0..128 source note events across all channels |

Each channel contains `delay` (0..8 ticks), `transpose` (−12..12 integer note
steps), and `muted`/`observed` boolean arrays of length steps. Delay and transpose
are fixed per channel; mute and measurement availability may vary per tick.
All corruption controls are privileged, not learner-visible fault labels.

Each event contains `channel` (valid channel index), `start` and `end` with
0 ≤ start < end ≤ steps, `pitch` (0..127), `velocity` (1..127), and `delivered`
(boolean). Pitch is an abstract MIDI-like integer, not a frequency or tuning
claim; velocity is a symbolic magnitude, not measured loudness. Tick duration
is an abstract unit: no tempo, sampling rate or wall-clock calibration is
implied. Within a channel, source intervals must not overlap. Touching intervals
are allowed and carry distinct onsets, even with identical pitch/velocity.
Channels may sound simultaneously. Input event order is irrelevant; privileged
event serialization sorts by channel/start. There are no stable event IDs.

The shifted pitch must remain 0..127 for **every** event, including dropped or
muted ones. Out-of-range transposition rejects the request, rather than clipping
or wrapping. Muted and unobserved rows do not excuse invalid hidden inputs.

## Exact event and impairment semantics

For source event `[start,end)`, ideal sound exists at every integer tick from
start through end−1; ideal onset is true only at start. No active event means an
actual source rest, represented as sounding=false, pitch=velocity=0, onset=false.
Pitch zero with sounding=true is a valid note, not a rest.

1. `delivered=false` drops the entire event from the transported stream. It does
   not remove it from ideal truth and does not set a measurement mask.
2. Delivered events shift both bounds by the channel's nonnegative delay and
   shift pitch by transpose. Velocity stays unchanged. The transported onset
   occurs at start+delay, not at the pristine source onset.
3. At each tick, channel mute replaces any transported note with apparent rest,
   suppressing its onset too. Unmuting during a sustained note may produce
   sounding=true/onset=false. This is a sampled note grid, not a MIDI note-on/
   note-off stream; no stuck note or reset rule is inferred.
4. The independent observation mask hides the entire transported note cell,
   including rests. Missing cells use the canonical all-zero/false placeholder
   but have observed=false. That placeholder asserts no note or rest.

Fixed per-channel delay preserves nonoverlap. There is no RNG, scheduler,
implicit event repair or audio import. Future source events cannot shift
backwards into the visible prefix. Valid requests with identical observable
prefixes and public dimensions produce identical observation bytes even when
future events/faults differ. Validation still examines the complete request;
invalid future fields cause rejection rather than prefix success.

## Public observation boundary

`observe` emits `trinite.music-events-observation.v1` with only:

- `channel_count`, `observe_ticks`, `horizon` (steps−observe_ticks).
- `rows` for ticks 0..observe_ticks−1, each with `tick` and `channels`.
- Each channel cell has `channel`, `observed`, and `note`.
- Each note has `sounding`, `pitch`, `velocity`, `onset`.

No source events, event IDs/counts/bounds, delay, transpose, delivered/muted
flags, future cells or ideal labels appear. The fixed public tick/channel grid
does not change length when a hidden note is removed. No hidden request hash or
answer-encoding identifier is emitted. Mask and public dimensions remain visible.

An observed rest could be a genuine source rest, a dropped event, delay before
an arrival or a muted note. A visible onset can distinguish adjacent equal
notes, but a masked/muted onset can hide that articulation. Different hidden
states can have identical observations. Fault detection and source reconstruction
are therefore not always identifiable. This module reports no confidence,
probability or inferred success and does not implement a detector or learner.

## Privileged truth and censoring

`simulate-privileged` emits `trinite.music-events-ground-truth.v1`,
`privileged:true`, steps, observe_ticks, canonical events, channel delay/transpose
and full tick rows. Each row contains ideal and transported notes plus the mute
flag. Observation masks are deliberately excluded so mask counterfactuals leave
truth bytes unchanged; retain the original request when reproducing observations.

`censored_transport_ticks` counts delivered event/channel tick slots at or after
steps in the shifted, **pre-mute** timeline. It is a duration sum, not an event
count, and includes entire delayed events outside the horizon as well as tails
crossing it. It excludes dropped events. Mute is undefined beyond the declared
horizon, so this count does not pretend to know future audible sound. Delayed
bounds can be recovered only from privileged source bounds and channel delay.
The episode is not extended or flushed to complete delayed notes.

Request and truth output must stay outside learner-readable storage. JSON
serialization alone is not a sandbox or a security enforcement mechanism. A
future controller must keep truth separate, commit predictions before outcome
exposure, and validate model responses. The simulator has no model access.

## Bounds, rights and verification

Maximum grid: 128 ticks × 4 channels. At most 128 source events; all delayed
bounds are ≤136. Sum of nonoverlapping source durations is at most 512 channel
ticks; censored transport duration is at most 32 channel ticks. Integer
arithmetic is bounded, with release overflow checks enabled. The simple
implementation scans up to128 events per grid cell (at most65,536 event checks
per ideal or transported full grid); output/state are bounded by the fixed
schema limits. No floating-point operations, timers or random seeds are used.

The [source/rights record](../external/music-events-v1/PROVENANCE.json) declares
original synthetic engineering inputs: the small example and formula-generated
test schedules. No song, score, recording or third-party melody was ingested.
They remain exposed conformance examples, not admitted native training data,
held-out musical families or a rights assertion about arbitrary future inputs.
Code and bundled original examples use the repository MPL-2.0 license.

Rust tests check half-open intervals, delay/transposition, loss/mute/missingness,
canonical ordering, bounds, censoring, strict JSON and CLI failures. The Python
standard-library oracle independently materializes channel timelines and checks
both complete artifacts across generated schedules, all small prefix masks,
extrema and hidden-state equivalence. BrokenPipe-safe CLI helpers preserve all
other write errors. Same-host software conformance is not independent-host
reproduction or learned musical reconstruction.

Affected requirements: [TRI-I01/I03/I05/I07/I08/I11/I12/I13/I14](INVARIANTS.md).
All prior module, native numerical and historical music curriculum bytes remain
unchanged. Later experiments need source/request/toolchain hashes, explicit
rights/admission, rhythmic/structural family splits, frozen scoring and budgets,
retained failures and matched baselines. Keep related event-order, mask, delay
and corruption variants in the same split. Training, calibrated prediction and
cross-environment transfer remain separate unexecuted gates.
