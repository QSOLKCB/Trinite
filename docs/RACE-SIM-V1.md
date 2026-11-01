# External synthetic race simulator v1

Status: engineering environment and wire contract, **not native training,
reinforcement learning, a calibrated racing predictor or a learning result**.
Requested by Trent on 1 November 2026 as an external Rust module for exploring
ternary evidence, uncertainty and later reward-driven learning. This first
increment leaves every native model, optimizer, tokenizer, data admission,
historical producer/fixture and research gate unchanged.

## Why an external Rust environment?

`external/race-sim-v1/` is an opt-in, CPU-only, dependency-free Rust crate and
fixed command-line tool. It does not give a model a shell, compiler, network,
filesystem agent or general plugin runtime. A future privileged controller can
invoke it and pass only its observation projection to a learner. It is not
imported by `src/trinite` or added to Python's requirements or installed API.
The first boundary is the process's versioned JSON stdin/stdout, not an FFI or
second numerical updater. Rust supplies explicit integer semantics; language
choice alone is neither learning improvement nor a security sandbox.

This is a **fixed-duration synthetic rank simulator**, not a physical model of
horse physiology, a real track, betting odds or finish times. Its traits, track
levels and noise are arbitrary disclosed toy mechanics. No real horses, names,
form, copyrighted racing data or weather services are acquired. The illustrative
five-factor BTRM weights 30/25/20/15/10 do **not** determine results; BTRM scoring,
probabilities, rewards and training are intentionally not implemented here.

## Build and run (explicit opt-in)

Use official [Rust 1.90.0](https://blog.rust-lang.org/2025/09/18/Rust-1.90.0/) with the pinned
`rust-toolchain.toml`; the upstream release manifest is
[dated 2025-09-18](https://static.rust-lang.org/dist/2025-09-18/channel-rust-1.90.0.toml).
The version is an explicit reproducible baseline, not a claim to be newest.
Rustup downloads occur only during explicit toolchain installation. Cargo has
**zero third-party crate dependencies**; the checked-in lockfile covers the
single local package. Build and simulation require no network after installing
the toolchain. No user-host installation is performed by this module.

```sh
cd external/race-sim-v1
rustup toolchain install 1.90.0 --profile minimal --component rustfmt --component clippy
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --release
cargo build --offline --locked --release
python3 tests/oracle.py target/release/trinite-race-sim-v1

# The request and privileged output belong to the controller, not the learner.
target/release/trinite-race-sim-v1 observe-before < example-request.json
target/release/trinite-race-sim-v1 observe-after < example-request.json
target/release/trinite-race-sim-v1 simulate-privileged < example-request.json
```

The controller may save observation stdout to a fresh `.json` file and expose
only that file to Trinite. Use exclusive creation (for example shell `set -C`)
so existing observations are not overwritten. Request and privileged result
files must stay outside learner-readable storage. This PR does not connect a
model or automatically enforce filesystem permissions.

Each command accepts exactly one request, emits one JSON line on success, and
exits. Invalid input/operation exits 2 with stderr and no partial JSON success.
There is no default privileged operation, fallback, silent repair, implicit
training, file creation or download. stdin must be finite/closed: the byte bound
is not a wall-clock limit on a caller that never closes a pipe. Future controllers
must enforce an external process deadline and bounded transport independently.

For direct archive installation on x86_64 Linux, the official release manifest
records these SHA-256s (toolchain provenance, **not model evidence**):

| Official component archive (`*-1.90.0-x86_64-unknown-linux-gnu.tar.xz`) | SHA-256 |
| --- | --- |
| rustc | `48c2a42de9e92fcae8c24568f5fe40d5734696a6f80e83cc6d46eef1a78f13c9` |
| cargo | `9853db03d68578a30972e2755c89c66aec035fec641cf8f3a7117c81eec2578d` |
| rust-std | `663f4ab7945b392d5e5294dec1b050a66820a20e86f084ec37eeb0f2f7ff5569` |
| rustfmt | `7f4d38b9d782e55832bf17969ef35477703c60781545bb098eb127cc8172d1c6` |
| clippy | `5b6466419693a05365827378145014a37ae74fb2948fab390d5210a524792ed8` |

## Request contract: `trinite.race-request.v1`

See the [complete minimal example](../external/race-sim-v1/example-request.json).
All fields are mandatory; unknown/duplicate object fields fail. The supported
JSON subset contains unescaped printable ASCII strings, signed decimal i64
integers, booleans, arrays and objects; null, floats, escapes, non-ASCII strings,
leading zeroes, negative zero and trailing input fail rather than normalize.
Whitespace is JSON space/tab/CR/LF. Maximum input is 16,384 bytes and nesting
depth is eight; no input string is interpolated into output.

| Field | Exact admission |
| --- | --- |
| schema | literal `trinite.race-request.v1` |
| seed | integer 0..4,294,967,295; **privileged** |
| steps | integer 1..256; fixed synthetic duration |
| track | integer 0..2; arbitrary discrete condition, not a calibrated going scale |
| runners | 2..32 objects, each exactly `id` and `observed` |
| id | distinct integer 1..65,535, stable public identifier, not an encoded seed |
| observed | exactly five booleans, independently chosen observation mask |
| scratches | distinct known runner IDs; at least two active runners must remain |

An empty field and single-runner walkover are rejected, not represented as races.
Maximum simulation work is 32 × 256 runner steps, retained state O(32). i32
ordinary arithmetic is bounded: distance 256..25,600 at 256 steps, pace sum at
most 2,560, fatigue numerator at most 17,340. Only the documented RNG uses
wrapping u64 operations. Release overflow checks stay enabled.

## Observation versus ground truth

`observe-before` emits `trinite.race-observation.v1`, phase `before-scratches`:
all original entrants active, without disclosing the future scratching list.
`observe-after` emits the same schema, phase `after-scratches`: withdrawn
entrants remain in the observation roster with `active:false`, but are not
participants. Active starting positions are recomputed. Withdrawn entrants'
starting-position feature becomes missing, not neutral.

Every observation has only `schema`, `phase`, `steps`, `track`, `runners`.
Each runner has only `id`, `active`, `values`, `observed`. The five **toy**
features are speed, stamina, freshness, track-fit and starting-position advantage,
not a reconstruction of the earlier real-race BTRM factors. Values are in
{-1,0,+1}; `observed:false` emits a zero placeholder. A zero with `observed:true`
means neutral evidence; a zero with `observed:false` means unavailable evidence.
All-missing remains active and valid: this module emits no score to misread as
an all-missing prediction. Downstream scorers must explicitly abstain when
coverage is zero. A scratching is never represented by merely masking features.

`simulate-privileged` emits `trinite.race-ground-truth.v1`, `privileged:true`,
seed, and a ranked array of active IDs, distances and their latent states.
It exposes no win probabilities. Withdrawn runners cannot finish or win. The
observations omit seed, latent values, distances and rankings. **Never give the
request, privileged output, source-bound seed receipts or oracle internals to a
learner**: a deterministic simulator's seed plus source reconstructs the answer.
The 32-bit noncryptographic seed could also be brute-forced or matched from
public observations; withholding it is not information-theoretic secrecy. A
future learner must not have access to the generator or privileged controller.
This schema separation is a data-flow contract, not OS access isolation. No
learner/controller or filesystem sandbox is implemented in this increment.

The two observation phases permit a future controller to obtain a forecast,
apply a predeclared scratching, then obtain an updated forecast before revealing
an outcome. Prediction commitment and a separate post-commit evaluator are
future protocol requirements, not implemented or enforced by this CLI. They are
not streaming mid-race events. Counterfactuals require a
separate request with the same seed/IDs and declared changed fields; no adaptive
outcome-conditioned seed or scenario selection is implemented or authorized.

## Frozen integer mechanics

Runner input order does not matter: IDs are sorted. Random draws use a stateless
SplitMix64 finalizer, not the platform RNG. For `(seed,id,stream,step)` define:

```
counter = (id << 32) | (stream << 16) | step
z = seed + counter * 0x9e3779b97f4a7c15
z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
z = (z ^ (z >> 27)) * 0x94d049bb133111eb
word = z ^ (z >> 31)
```

All additions/multiplications above are modulo 2^64. This is not cryptographic
randomness. Mapping `% n` has tiny modulo bias; it is intentional part of v1,
not a claim of ideal independent sampling. Streams 0..5 at step 0 generate:
speed=40+word%41, stamina=32+word%65, initial fatigue=word%17, and three track-fit
values=word%17−8. Stream 6 at each step yields noise=word%7−3. IDs, not array
indices, select draws, so withdrawing a runner cannot reassign survivors' noise
or latent traits. Reusing seeds creates paired, correlated cases, not fresh
independent examples.

After scratchings, pace=floor(mean active speed), crowd=floor(active count/4),
and lane is the zero-based ascending-ID position among active entrants. At each
step `t=0..steps-1`, for each runner:

```
fatigue = min(60, initial_fatigue + floor(t * (100-stamina) / 256))
pressure = floor(max(0, pace-stamina) / 8)
start = lane if t == 0 else 0
advance = clamp(speed + track_fit[track] - fatigue - pressure - crowd
                - start + noise, 1, 100)
distance += advance
```

Higher distance ranks first; lower ID breaks an exact tie. This arbitrary tie
rule and ID-based starting lane can favor lower IDs, so IDs are not exchangeable
labels and must not be interpreted as skill. Mask changes do not affect any
latent/dynamic state. Scratchings recompute pace/crowding/lane and simulate the
new field; this is not unsupported renormalization of old winner probabilities.

Observation ternarization is negative below low, neutral within inclusive
[low,high], positive above high: speed [52,68], stamina [48,80], fatigue [4,12]
with sign reversed for freshness, current track fit [-2,2], lane [2,8] with
sign reversed. Noise/outcomes never enter this pre-race projection.

## Verification, compatibility and future gates

Affected requirements: [TRI-I01/I05/I07/I08/I11/I12/I14](INVARIANTS.md).
Tests exercise strict input rejection, active fields, observation allowlists,
masked-neutral distinction, seed/order replay, fixed RNG vectors, tie ordering,
maximum budgets and actual subprocess stdout/error behavior. A separate Python
arbitrary-precision oracle checks integer equations with runner-major summation
against Rust's step-major computation over 36 exposed cases and scratching
counterfactuals. This is software conformance, not independent-host replication,
biological validity, an unbiased statistical generator, native learning,
forecast calibration or closed PROVENANCE custody.

For replay preserve the exact committed crate sources, Cargo lock/toolchain,
request bytes and invocation mode, compiler/target identity, stdout/stderr and
exit code; bind them by SHA-256 and retain failed attempts. A seed alone is not
source/input binding. The example and tests are commissioned, synthetic,
public engineering inputs; **they are not admitted native training data**.
No new data-admission exception is created. Source changes require a new bound
request; incompatible mechanics/observations require a new protocol version.

A later reviewed experiment must define family-level train/dev/test isolation
(including seed relatives, counterfactuals and scenario variants), rights/admission,
a source-bound frozen protocol, exposure and resource budgets, model adapter,
matched continuous/binary/ternary controls, uncertainty/calibration metrics and
full result retention before any model updates. If probabilities are introduced,
use appropriate proper scoring rules and calibration checks rather than rewarding
confident guessed winners. Rewarding simulator-specific tricks or betting profit
would not establish generalization. No real-world predictive gain, financial
return or neural improvement is implied. Existing negative results and phase
limitations remain unchanged.
