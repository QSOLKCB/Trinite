# External queue simulator v1

Status: additive engineering implementation for review, 1 November 2026.
Second module in the [external environment roadmap](EXTERNAL-ENVIRONMENTS-ROADMAP.md).
A standalone, dependency-free Rust CPU runner computes a finite FIFO queue and
exports observation JSON. It does not load Trinite, train a model, implement
rewards, or establish a forecasting improvement. Build environments first;
learning and representation comparisons require later reviewed protocols.

## Build and run

Use the pinned official Rust 1.90.0 toolchain, the same explicit baseline as the
[race module](RACE-SIM-V1.md#build-and-run-explicit-opt-in). This crate has no
third-party dependencies; Cargo.lock contains only this local package. No new
runtime is installed in Trinite or on the user's machine by this module.

```sh
cd external/queue-sim-v1
rustup toolchain install 1.90.0 --profile minimal --component rustfmt --component clippy
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --release
cargo build --offline --locked --release
python3 tests/oracle.py target/release/trinite-queue-sim-v1

target/release/trinite-queue-sim-v1 observe < example-request.json
target/release/trinite-queue-sim-v1 simulate-privileged < example-request.json
```

After explicit toolchain installation, build/tests/simulation need no network.
Each invocation consumes one bounded request from stdin and emits one JSON line
on stdout. The operation must be explicit. Invalid requests/operations exit 2
with stderr and no JSON success. There is no file creation, fallback, automatic
repair, download, model execution or training. A privileged controller may save
observation stdout using exclusive creation and expose only that `.json` file
to Trinite. Controller, storage isolation, process deadline and prediction
commitment are **not implemented**. A caller that never closes stdin can stall
below the byte limit; the future controller must bound wall time independently.

## Privileged request: `trinite.queue-request.v1`

[Example](../external/queue-sim-v1/example-request.json). Every field is required;
unknown/duplicate fields fail. The strict JSON subset permits unescaped printable
ASCII strings, signed decimal i64 integers, booleans, arrays and objects. It
rejects null, floats, escapes, non-ASCII strings, leading zeroes, negative zero,
trailing input and nesting deeper than eight. Input limit: 16,384 bytes. No
input strings are interpolated into output. The local parser is independently
versioned from the unchanged race parser; this is not a shared plugin framework.

| Field | Bounds / meaning |
| --- | --- |
| schema | Exactly `trinite.queue-request.v1` |
| capacity | 1..128 waiting-buffer slots; includes all admitted uncompleted jobs |
| initial_queue | 0..capacity jobs initially waiting, all assigned arrival tick 0 |
| arrivals | 2..256 integer counts, each 0..32; offered jobs per tick |
| service | Same length, each 0..32; nominal completions available per tick |
| outages | Same length of booleans; true sets effective service to zero |
| observe_ticks | 1..length−1; number of completed ticks visible before forecasting |
| observed | Exactly six booleans; constant measurement availability across visible ticks |

The schedules are explicit privileged inputs, not pseudorandomly generated.
Replay is deterministic from exact source/request bytes without a seed. This
makes structural arrival/service/outage families controllable by a later dataset
producer. No family split, stochastic probability distribution, workload sampler
or realistic queue calibration is implemented here. Different future schedules
can share the identical observation: arbitrary suffixes are not predictable
from a prefix. Any useful future forecasting study must declare the distribution
or family assumptions, avoid label leakage and measure uncertainty.

## Frozen FIFO transition

Ticks are integers `t=0..steps−1`. Every job is one unit and service completes
instantaneously within a tick; there are no partially served jobs, multiple
queues, abandonment, priorities or retained service tokens. Initial jobs precede
new tick-0 arrivals in FIFO order. All initial jobs have zero prehistory age;
the module cannot reconstruct unknown waiting time before the episode.

For each tick, in this exact order:

1. Offered arrivals are read. Admit `min(arrivals[t], capacity−queue_length)`.
   Drop the remainder permanently. New admitted jobs carry arrival tick `t`.
2. Effective service is zero during an outage, otherwise `service[t]`.
   Complete `min(effective_service, queue_length_after_admission)` oldest jobs.
   Unused service capacity expires; it is not carried forward.
3. For each completed job, wait is `t−arrival_tick` (same-tick service has wait 0).
   Record end-of-tick queue length and oldest pending age `t+1−arrival_tick`;
   oldest pending age is zero if the queue is empty.

Admission occurs **before** service: a full queue drops new arrivals even when
service frees slots later in the same tick. That convention is deliberate and
covered by an explicit regression test. An outage is an operational event, not
a missing measurement. A non-outage tick may still have nominal service zero.

Per tick: offered = admitted + dropped, and new queue = old queue + admitted −
served. Across the episode: initial queue + total admitted = total served +
final pending count. Maximum 8,192 offered jobs, 8,320 initial-plus-offered jobs,
32 completions per tick, 128 pending jobs and 256 rows. Wait for served jobs is
0..255; pending age is 1..256. `u32` wait sums are safely below 2,121,600, with
release overflow checks enabled. Retained state is O(256+128), execution is
O(256×32+128); output is bounded by these fixed schema/array/value limits.
No floating-point arithmetic, platform RNG or mutable global state is used.

## Observation JSON is the learner boundary

`observe` emits only:

- `schema`: `trinite.queue-observation.v1`.
- `capacity`, `observe_ticks` and `horizon` (steps−observe_ticks).
- `rows`: completed ticks `0..observe_ticks−1`, each containing exactly `tick`,
  `outage`, `values` and `observed`.

The six integer measurements in order are **end queue length, offered arrivals,
effective service capacity, completed jobs, dropped arrivals, oldest pending
age**. Their availability is the corresponding `observed` boolean; unavailable
values have a zero placeholder. Thus `values[0]=0, observed[0]=true` means an
observed empty queue, while `observed[0]=false` means unknown queue length.
Zero is not an invented success, an abstention, a neutral ternary code or an
error result. Downstream callers must read the mask, not the placeholder.

Outage status, capacity, tick labels and forecast horizon are always public
operational information; even with all six measurements masked, those covariates
remain visible. Missingness hides direct measurements, not everything inferable
from other visible counters and conservation. This is not privacy protection.
Changing masks never changes dynamics. Observations retain exact nonnegative
integer counts rather than prematurely choosing a binary/ternary quantizer;
later matched representation experiments can explicitly compare those encodings.
Input representation is separate from ternary model weights.

Only the observed prefix is simulated by `observe`. Valid requests with identical
public settings and prefix schedules but different suffix schedules produce
byte-identical observations. No future outage, arrival, service schedule, full
request, initial-state field, pending job list, wait outcome or forecast summary
is emitted. Historical counters can legitimately include past drops/services;
they are not the future forecast target. The observation supplies no public
identifier that encodes a hidden request or answer.

`simulate-privileged` emits `trinite.queue-ground-truth.v1`, `privileged:true`,
capacity, observe_ticks, steps, initial_queue and all tick rows. Each privileged
row has exactly tick, offered, admitted, dropped, nominal_service,
effective_service, outage, served, queue, wait_sum, wait_max and
oldest_pending_age. It also emits:

- `summary`: offered, admitted, dropped, served, wait_sum and wait_max over all ticks.
- `forecast`: the same totals/max over ticks **at or after observe_ticks**.
  Waiting times for jobs completed in this suffix include any waiting before
  the cutoff; this is a completion-cohort target, not suffix-only elapsed time.
- `pending_ages`: ages at the final boundary in FIFO order. These are censored
  lower bounds on future completion waits, not completed waits or zero-delay jobs.

For no completions, served=0, wait_sum=0 and wait_max=0. The latter two are empty
aggregate sentinels; a mean wait is **undefined**, not zero. No mean/probability
or score is emitted. Total completed wait is biased toward jobs completed within
the horizon; preserve pending ages/counts when evaluating delays. For an empty
queue pending_ages is empty. The simulator does not flush remaining jobs after
the declared horizon or silently discard censoring.

The request and privileged output must remain outside learner-readable storage.
JSON is not an OS sandbox. A future controller must enforce separation, commit
predictions before revealing outcomes, and reject malformed learner responses.
Neither this CLI nor its tests implement that controller or enforce test isolation.

## Verification and compatibility

Affected requirements: [TRI-I01/I05/I07/I08/I11/I12/I13/I14](INVARIANTS.md).
Rust unit/integration tests exercise bounded input grammar, direct API bounds,
FIFO timing, admission-before-service, outages, mask invariance, future-suffix
noninterference, censoring, conservation and subprocess failure behavior.
The independent Python standard-library oracle models jobs as separate records
and checks complete output, accounting, boundaries and multiple synthetic
schedule patterns. These public exposed cases are **engineering conformance**,
not held-out learning data, a benchmark or an independent-host replication.

Preserve exact crate sources, request bytes, Cargo.lock, toolchain/target identity,
command, stdout/stderr and exit status for replay; bind them by SHA-256 for a
later source-bound experiment. This module does not claim closed PROVENANCE
custody. The example/tests are original synthetic engineering inputs, not
admitted native training data. Existing native code, updater, data policy,
race-v1 bytes, retained fixtures and negative results remain unchanged.

Future training needs its own admission, source-bound producer/protocol,
structural family splits, locked test inventory, matched baselines, exposure and
compute budgets and complete failure retention. Keep related prefixes, suffix
counterfactuals and mask variants in the same split. Unseen arrival and service
families are still an **unexecuted learning gate**. No transfer, reward, learned
uncertainty, forecasting gain or model improvement is claimed by making a module.
