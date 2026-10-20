# Exact NEXUS donor conformance v1

This is an isolated **CPU software-conformance lane**, not a learning experiment,
model component, geometry intervention, compression scheme or quantum claim.
The integral-geometry curriculum and all historical Trinite producers/fixtures
remain untouched. No model, optimizer, tokenizer or native admission is modified.

## What is compared

The MPL-2.0 [NEXUS donor subset](../fixtures/nexus-donor-v1/NOTICE.md) pins two
separate revisions: Rust runtime v1.0.0 at `24c7b5bb8b158d84e08b10ccc8fdd16b6889b7e4`
and Lean formal package 0.1.0 at `a9486c34eebd395c8060f8b2b03885eff10115c6`.
Neither is a moving branch or the unverified Zenodo deposit. Original source
bytes, complete license, Git blob identities and SHA-256s are retained locally.

The Python integer reference derives the graph from coordinates, lexicographic
edges/triangles and first-fit exhaustive edge colouring. The Rust adapter calls
actual donor `forward`/`inverse`; the Lean adapter calls the actual functional
`Lattice.forward`/`inverse` with `Runtime.plan` and snapshot materialization.
Rust wrapping u32 arithmetic is mirrored exactly, not Python unbounded addition.

For each input, acceptance compares **all registers** in four ordered vectors:
forward(input), inverse(forward(input)), inverse(input), forward(inverse(input)).
The two restoration vectors must equal input. A census, fold, hash collision,
seed equality or self-reported passed label cannot substitute for vector parity.

There are 1,732 cases and 6,928 source-bound state/register-query items:
- 2- and 6-cell rings; twelve Z4 registers per cell;
- all-zero/all-three, alternating, each single-register impulse, and three
  deterministic SHA-256-derived arbitrary state vectors per ring;
- schedule seeds 0, 7, 0x4e455853 and 0xffffffff;
- round counts 0, 1, 2, 8, plus four two-cell 128-round coverage cases.

All 64 matching/flip/face tuples must occur. This is not exhaustive coverage of
all states, seeds, schedules or rings. Zero rounds tests identity at the
step/function level: the donor public CLI's positive-round admission is not
being changed or claimed to admit zero. Arbitrary registers are injected by a
separately hashed, test-only same-module adapter; they are not a public runtime
constructor. The exact temporary source transformation and its compiled byte
identities are disclosed in the notice and result. No GPU/native renderer runs.

## Data and provenance

Each item binds source ID, generator source closure revision, seed procedure,
complete formal operands, rendered JSON prompt, expected answer, verifier
outcome, family, split procedure and parent transformation identity. Parsing the
rendered JSON must recover the exact formal specification. State and register
queries, forward/inverse tasks, all seeds/rounds and related transformations
remain **conformance-only**. There are no train/validation/test partitions and
no heldout score; even overlapping trajectories cannot cross a learning split.

The source-specific record assesses bounded generated integer facts/minimal
formal templates separately from copyrighted MPL-2.0 software. It does not
copy donor prose into items or infer blanket native admission from a license.
The generator's candidate item outcome explicitly requires live parity. The
successful runner wraps every item in a request- and full-output-bound live
Rust/Lean/Python verification receipt. A future training curriculum needs its
own reviewed admission, frozen exposures, independent heldout design and model
identity; this PR authorizes none of those steps.

## Run, retention and limits

Linux, Python 3.11+, Rust 1.90.0 and Lean 4.19.0 are explicit prerequisites.
No runtime command downloads software. Use trusted installed toolchains; their
version strings and Python/platform/machine are recorded, not hardware attested.
The checker requires isolated/no-site startup and loads captured source bytes
rather than project bytecode. Request identity is supplied independently.

```sh
PYTHONPATH=scripts python -B -m unittest discover -s tests/nexus_conformance_v1 -v
python -I -S scripts/check_nexus_conformance.py freeze --request /fresh/request.json
# Copy the printed identity, without deriving it from a later changed request:
python -I -S scripts/check_nexus_conformance.py check \
  --request /fresh/request.json --request-identity PRINTED_SHA256 \
  --output-dir /fresh/attempt --rustc /trusted/bin/rustc --lean /trusted/bin/lean
```

`freeze` and `check` use exclusive destinations. The existing parent directory
must exist. `check` retains request/input, per-process command/exit or timeout,
stdout/stderr, temporary compiled-source identities, complete output vectors,
per-item receipts and a combined `result.json` outcome. Missing result means
incomplete, never success. Completed result publication is atomic/no-clobber;
status and result cannot diverge because they are the same file. Publication
failure retains the attempt directory, but total storage failure cannot promise
a new durable failure receipt. No existing output/evidence is overwritten.

Each subprocess has a 120-second deadline. The 16 MiB log limit is a **post-run
acceptance check**, not a live disk/memory cap. Fixed suite/input bounds prevent
arbitrary experiments; no performance or whole-process hard memory guarantee is
claimed. Linux RUSAGE_CHILDREN records cumulative peak RSS including compilers
and donor workers with explicit KiB units and the pre-run high-water mark.
Elapsed times and environment are observational, outside exact output equality.

CI runs contracts on Python 3.11–3.13 and actual Rust/Lean comparisons twice.
It retains request, logs, full outputs, item receipts and combined results as
artifacts even on failure (build products are excluded). Hosted artifacts expire
after 14 days: this is not permanent outcome archival. The repository permanently
retains donor source/pins, not the live CI results; a later evidence PR can retain
requested results without changing the producer. Replay compares exact
request/input/output/item bytes, not observed timing, paths or resource peaks.
CI is engineering conformance, not a quality experiment, theorem about the Python
implementation, full Rust refinement proof, end-to-end custody attestation or
independent hardware replication. Preserve negative attempts before changing a
producer or trying a new source-bound request.

Affected requirements: TRI-I01/I03/I07/I08/I11/I12/I13/I14. CPU execution and
resource observations support this lane's TRI-I05 scope only. Geometry claims
remain constrained by TRI-I09; no research gate is unlocked.
