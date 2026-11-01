# External inventory simulator v1

Status: additive engineering implementation for review, 1 November 2026.
Stage 3 of the [external environment roadmap](EXTERNAL-ENVIRONMENTS-ROADMAP.md).
This dependency-free Rust CPU runner models finite stock, demand, exogenous
replenishment orders, lead times and supplier disruptions. It exports observation
JSON separately from privileged outcomes. It does not train Trinite, implement
reorder actions or rewards, or establish predictive improvement. Build modules
first; learning requires a separate reviewed protocol.

## Build and run

Use the same pinned official Rust 1.90.0 toolchain as the
[race module](RACE-SIM-V1.md#build-and-run-explicit-opt-in) and
[queue module](QUEUE-SIM-V1.md#build-and-run). No third-party crate dependencies;
Cargo.lock contains only this local package. No runtime is installed into Trinite.

```sh
cd external/inventory-sim-v1
rustup toolchain install 1.90.0 --profile minimal --component rustfmt --component clippy
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --release
cargo build --offline --locked --release
python3 tests/oracle.py target/release/trinite-inventory-sim-v1

target/release/trinite-inventory-sim-v1 observe < example-request.json
target/release/trinite-inventory-sim-v1 simulate-privileged < example-request.json
```

After explicit toolchain installation, build/tests/simulation need no network.
Each explicit operation reads one bounded request from stdin and emits one JSON
line to stdout. Invalid requests/operations exit 2 with stderr and no success
JSON. No automatic repair, file creation, network, model loading or training.
A privileged controller may save observation stdout to an exclusively created
file and expose only that JSON to the learner. That controller, filesystem
isolation, wall-time deadline and prediction commitment are **not implemented**.
A caller that never closes stdin can stall below the byte limit; a future
controller must bound wall time independently of the byte bound.

## Privileged request: `trinite.inventory-request.v1`

[Example](../external/inventory-sim-v1/example-request.json). Every field is
required; unknown/duplicate fields fail. Strict wire grammar permits unescaped
printable ASCII strings, signed decimal i64 integers, booleans, arrays and
objects. It rejects null, floats, escapes, non-ASCII strings, leading zeroes,
negative zero, trailing input and nesting deeper than eight. Input limit is
16,384 bytes. No request strings are interpolated into output. The local parser
is versioned independently; existing race/queue parser bytes stay unchanged.

| Field | Bounds / meaning |
| --- | --- |
| schema | Exactly `trinite.inventory-request.v1` |
| capacity | 1..128 on-hand stock units |
| initial_stock | 0..capacity units; no initial pending replenishments |
| demand | 2..256 counts, each 0..32; requested units per tick |
| orders | Same length, each 0..32; exogenous ordered units per tick |
| lead_times | Same length, each 0..32; nominal ticks until each order is due |
| supplier_available | Same length of booleans; false prevents receipts that tick |
| observe_ticks | 1..length−1 completed ticks visible before prediction |
| observed | Exactly eight booleans; constant measurement availability over visible ticks |

Orders are predetermined inputs, **not learner actions** or an adaptive reorder
policy. Schedules are explicit, deterministic and privileged, without RNG/seed.
A later producer can define structural demand/order/disruption families, but no
workload distribution, held-out split, economic calibration or dataset admission
is implemented. Arbitrary future schedules cannot be inferred from a shared
prefix; a useful forecasting study needs declared family/distribution assumptions
and uncertainty evaluation. This is a toy single-product inventory, not financial,
procurement or real supply-chain advice.

## Frozen transition and accounting

Ticks are `t=0..steps−1`. Stock has no spoilage, cost, reservations or multiple
products. Demand is lost-sales demand, with no backlog or backorders.
For each tick, in this exact order:

1. Record this tick's positive order as one pending batch with ordered_tick=t,
   due_tick=t+lead_times[t] and quantity=orders[t]. Zero orders create no batch.
   Orders can be recorded while the supplier is unavailable.
2. If supplier_available is true, deliver **all** pending batches due at or
   before t, including a current zero-lead order. Remove delivered batches.
   Otherwise deliver none and retain all pending batches. There is no receipt
   throughput cap; a recovered supplier can release the whole overdue backlog.
3. Admit received=min(delivered, capacity−opening stock). Permanently discard
   delivered excess; it is not deferred, returned or kept pending.
4. Fulfil sales=min(demand, opening stock+received). Unmet demand is lost=demand−sales.
   Record ending stock, pending-unit count and the other tick counters.

Supplier unavailability thus models a **receiving block**, not cancellation,
production failure or a dispatch-delay model. Nominal due ticks never change;
blocked due batches wait until the first available tick. Receipts precede demand:
a full store discards a delivery even if later demand creates room that same
tick. These deliberate conventions are covered by direct tests.

Per tick: demand=sales+lost, delivered=received+discarded,
new stock=old stock+received−sales, new pending=old pending+ordered−delivered.
Across the episode:

`initial_stock + total_ordered = total_sales + total_discarded + final_stock + final_pending_units`.

Maximum 8,192 ordered units, 8,192 demand units and 256 pending batches. Batch
due ticks are at most 287; an accumulated delivery/pending count fits u16
(at most 8,192), stock at most 128. Summary counts use u32, stockout tick counts
u16 (at most 256). Release overflow checks are enabled. Execution is bounded by
O(steps²), at most 65,536 pending-batch visits; retained state/output are O(steps)
with steps≤256. No floating-point arithmetic or mutable global state is used.
This is a boundedness argument, not a measured throughput or RAM benchmark.

## Observation JSON is the learner boundary

`observe` emits exactly schema=`trinite.inventory-observation.v1`, capacity,
observe_ticks, horizon=steps−observe_ticks and rows for completed prefix ticks
0..observe_ticks−1. Each row contains tick, supplier_available, values and observed.
The eight integer measurements, in order, are:

1. End-of-tick stock.
2. Demand.
3. Sales fulfilled.
4. Lost demand.
5. Received stock.
6. Discarded delivery excess.
7. Ordered units.
8. Pending units at tick end.

Each unavailable measurement has a zero placeholder and corresponding false
mask. Observed stock zero is different from unknown stock. Supplier_available
is always-public operational information, independently false or true even when
all eight measurements are missing. An unavailable supplier does not imply zero
stock. Capacity, time labels and forecast horizon remain public too. A mask hides
direct measurement, not everything inferable from visible conservation counters;
this is not a privacy guarantee. Masks never alter dynamics.

The observation keeps exact integer measurements; binary/ternary quantization is
a later explicitly compared representation, not evidence of better learning.
Input encoding is distinct from ternary model weights. No score, probability,
abstention, action, reward or missing-data repair is emitted.

Only the prefix is simulated in `observe`. Valid requests with identical public
settings and prefix schedules but different suffix demand/orders/lead times/
supplier status produce byte-identical observations. No future schedule, seed,
initial_stock field, batch due dates, privileged forecast summary or final
outcomes are emitted. Prefix order quantities may be visible; their lead times
and batch identities are not part of this observation contract. Historical lost
sales are past measurements, not the future prediction target.

## Separate privileged outcomes

`simulate-privileged` emits schema=`trinite.inventory-ground-truth.v1`,
privileged=true, capacity, observe_ticks, steps, initial_stock, all tick rows,
summary, forecast, final_stock and pending. Each full tick row contains exactly
tick, ordered, delivered, received, discarded, demand, sales, lost, stock,
pending_units and supplier_available.

summary contains full-episode totals for ordered, delivered, received, discarded,
demand, sales and lost, plus stockout_ticks. forecast contains the same fields
for ticks at or after observe_ticks. A stockout tick means **lost>0**, not merely
zero ending stock; zero demand at an empty store is not counted. Forecast receipts
can originate from orders before the observation cutoff. The suffix summary is
not a standalone inventory balance without stock/pending state at the cutoff.

pending lists undelivered positive batches in original order, each with
ordered_tick, due_tick and quantity. It preserves both not-yet-due batches
(due_tick≥steps) and overdue blocked batches (due_tick<steps). These units are
not discarded, lost demand or confirmed future receipts. No automatic horizon
flush or assumed recovery occurs. With no orders the list is empty; zero-demand
rates would be undefined, so no service ratio/mean/probability is emitted.

The request and privileged results must remain outside learner-readable storage.
JSON is not a sandbox; a future controller must enforce that separation, commit
predictions before revealing labels and validate learner responses. No learning
or test-isolation enforcement is supplied by this CLI.

## Verification and compatibility

Affected requirements: [TRI-I01/I05/I07/I08/I11/I12/I13/I14](INVARIANTS.md).
Rust unit/CLI tests cover strict grammar, direct API bounds, zero lead, receipt
ordering, receiving blocks/recovery, overflow, lost-sales semantics, pending
batches, mask invariance, future-suffix noninterference and conservation.
The independent Python standard-library oracle precomputes each batch's first
available delivery event, then computes stock through cumulative accounting,
rather than copying the Rust pending-batch transition. It checks complete output,
exposed structural schedules, all masks, counterfactuals and invalid requests.
These are software conformance cases, not held-out learning evidence.

For a later source-bound experiment, retain exact sources, Cargo.lock, request,
toolchain/target, commands, stdout/stderr and exit status with SHA-256 identities.
This module does not claim closed PROVENANCE custody, independent-host replication
or calibrated real-world performance. Example/test inputs are original synthetic
engineering data, not automatically admitted native training data.

Race/queue modules, native model/updater/admission, historical producers, fixtures
and negative results remain unchanged. Future training needs a separate reviewed
producer/protocol, actual data admission, structural family splits, locked tests,
independent baselines and matched exposure/compute budgets. Keep related masks,
prefixes and suffix counterfactuals together in one split. Unseen demand and
disruption generalization, reward/action loops and cross-domain transfer remain
unexecuted gates. A working module alone establishes none of those gains.
