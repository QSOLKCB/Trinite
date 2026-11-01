# TOMBSTONE bounded splice adapter v1

Status: first-stage engineering module, 1 November 2026. **No training, model
run, held-out evaluation pilot, reward loop or native capability claim.** This
adds a standalone external JSON adapter; existing race, queue-v2, model,
optimizer, admission, historical fixtures and TOMBSTONE itself are unchanged.

## Authority and explicit immutable source pin

The upstream [QSOLKCB/TOMBSTONE](https://github.com/QSOLKCB/TOMBSTONE) C99 engine
is the only production authority for byte-splice validation, hashing and
canonical receipts. This adapter uses its unchanged safe Rust FFI and typed
`Error::Native`, not CLI stderr parsing and not a replacement splice engine.
The explicit source commit is
[`5447558afdaa3e37f933015278b64dde6abef5d5`](https://github.com/QSOLKCB/TOMBSTONE/tree/5447558afdaa3e37f933015278b64dde6abef5d5),
the Phase 9 merge inspected during initial research. Upstream main has later
Phase 10–12 development; this module intentionally does **not** follow it.
Neither repository's immutable releases are changed.

The small complete build closure is vendored unchanged under
[`external/tombstone-adapter-v1/upstream`](../external/tombstone-adapter-v1/upstream):
three C99 sources, three headers, safe Rust binding/build metadata, and its
three original compile-time fixture receipts. The full Apache-2.0
[license](../external/tombstone-adapter-v1/upstream/LICENSE) is preserved.
The [source manifest](../external/tombstone-adapter-v1/upstream-pin.json) records
original repository-relative paths and SHA-256 values. These original imported
fixtures are not rewritten. New adapter code is MPL-2.0 like Trinite. A narrow
55,733-byte closure avoids vendoring the UI/research repository while permitting
actual offline C99 execution; a caller-supplied binary would be harder to audit.
No third-party Rust crates, GUI, network service or implicit runtime download.

Phase 9's reported 35/35 obligations cover the actual C99 splice helper under
its memory assumptions, **not** this adapter, full wrapper, provider, receipt
engine, Rust ABI, compiler, machine code, model reasoning or historical events.
This PR does not rerun a proof tool or import that result as a module proof.

## Build and supported bounded launcher (Linux)

Use the existing explicit Rust **1.90.0** baseline plus C99 compiler, `ar`,
`pkg-config`, and OpenSSL development headers/library. Python 3.11+ supplies
only build identity and process transport. Installation is an explicit operator
step; no toolchain, packages or dependencies are installed by these commands.

```sh
python3 scripts/tombstone_adapter_v1.py build
python3 scripts/tombstone_adapter_v1.py check
printf '%s' '{"schema":"trinite.tombstone-request.v1","before":[0,255],"removed":[255],"after":[0],"offset":1}' \
  | python3 scripts/tombstone_adapter_v1.py observe
# Same input, privileged caller only:
printf '%s' '{"schema":"trinite.tombstone-request.v1","before":[0,255],"removed":[255],"after":[0],"offset":1}' \
  | python3 scripts/tombstone_adapter_v1.py evaluate-privileged
```

`build` first verifies imported source identities, builds offline/locked, and
writes only the disposable local `target/build-record.json` cache binding the
adapter/imported source closure and executable SHA-256. `check` and each run
verify the source and binary anew. This record is **not** an externally trusted
attestation, test-success cache or historical research receipt. A trusted,
unchanging checkout/build directory is required; hash checking is not a sandbox
against another writer replacing the executable between check and launch.
System compiler, linker, libc/OpenSSL and hardware remain trust boundaries;
the build record does not attest them or make binaries cross-host reproducible.

The launcher accepts no executable, path, shell command or caller-selected
subprocess argument. It strips inherited preload/runtime environment variables,
closes stdin, bounds combined stdout+stderr to **4,096 bytes**, gives stdin EOF
**5 seconds** and the child **5 seconds**, and kills/reaps the child process
group on failure. Input is capped at **1,024 bytes**. Two phases mean a total
transport allowance up to roughly ten seconds plus source/hash/startup overhead.
The Rust binary itself is a low-level test/build target: direct invocation has
bounded bytes/output but can wait on unclosed stdin. Use the launcher for the
wall-clock contract. No source or user file is deleted; no execution-time
artifact is written. Process memory/swap are not securely erased.

## Exact public input and observation

All five request fields are required, with no unknown or duplicate keys:
`schema`, `before`, `removed`, `after`, `offset`. The schema must equal
`trinite.tombstone-request.v1`; each array contains **0–16 integer bytes 0–255**;
`offset` is an integer **0–16 byte offset**, never a character offset. Boolean
values are not integers. Empty arrays are present data, not missing fields.
No masks exist in v1: all supplied bytes and the proposed offset are public.
Only original synthetic bytes are appropriate; **never submit real removed
plaintext, credentials or copyrighted text** to these conformance fixtures.

The strict local JSON subset accepts unescaped printable ASCII strings,
integers, arrays, objects and booleans; field validation further restricts them.
It rejects floats, null, escaped/non-ASCII strings, negative zero, leading
zeroes, i64 overflow, nesting beyond eight, trailing input and oversize input.
It does not silently canonicalize or repair rejected inputs. The parser is a
separately versioned copy of the existing queue parser, not a shared framework.

`observe` performs **no oracle call**. It returns exactly the schema
`trinite.tombstone-observation.v1` plus the four public operand fields, without
native status, labels, hashes, receipts, seed, source path or diagnostic text.
The operand relation is deliberately answerable from public bytes; no hidden
reconstruction task is implied. A future controller must expose only that
observation, not privileged files or their paths. JSON alone is not isolation.

## Privileged native result and precedence

`evaluate-privileged` checks native ABI versions, calls `witness_delete`, and
returns `trinite.tombstone-outcome.v1`, `label`, `native_code`, and `receipt`:

| Native code | Label | Meaning at the pinned implementation |
| --- | --- | --- |
| 0 | `valid-splice` | Nonempty supplied removed span matches `before[offset:offset+len(removed)]`, and `after` equals the exact remaining splice. Canonical C99 receipt included. |
| 1 | `argument` | Empty removed span; this check wins over bad length/offset. |
| 2 | `length` | Removed span longer than before, or offset outside the remaining extent. Wins over content/after-length mismatch. |
| 6 | `mismatch` | After length/content or removed bytes fail exact equality. |

Failure labels have `receipt: null`. These codes are mapped only after the
actual C99 result; production Python/Rust does not independently classify bytes.
Invalid JSON/schema is exit **2**, not a semantic label. Allocation, crypto,
ABI, receipt creation/encoding/verification, unknown native status, source/hash,
transport and timeout failures are exit **3** with no outcome JSON, never a
negative example. Diagnostics are sanitized and never echo inputs.

Full removal is valid and has empty `after`. Empty `removed` is invalid even
when before/after are both empty. Repeated `aaaa → aaa` has four valid supplied
one-byte witnesses: offset is a valid claim, **not unique historical truth**.
A valid relation is not evidence that an edit occurred, who did it, chronology,
confidentiality or secure erasure. Short-byte hashes can be dictionary-guessed.

The evaluator creates and parses the canonical receipt and checks its singleton
chain head via C99. Receipt self-hash verification does **not** re-witness raw
bytes; it checks a commitment to claimed receipt fields. General chain links do
not establish adjacent data-state continuity or authenticated history. General
receipt ingestion and multi-record chain tasks are outside this first module.

## Reproducible conformance, not a learner benchmark

```sh
cd external/tombstone-adapter-v1
cargo fmt -p trinite-tombstone-adapter-v1 -- --check
cargo clippy --offline --locked -p trinite-tombstone-adapter-v1 --all-targets -- -D warnings
cargo test --workspace --offline --locked
cargo test --workspace --offline --locked --release
cd ../..
python3 scripts/tombstone_adapter_v1.py build
python3 external/tombstone-adapter-v1/tests/oracle.py external/tombstone-adapter-v1/target/release/trinite-tombstone-adapter-v1
python3 external/tombstone-adapter-v1/tests/provider_failure.py external/tombstone-adapter-v1/target/release/trinite-tombstone-adapter-v1
python3 external/tombstone-adapter-v1/tests/verify_fixture.py
python3 -m unittest discover -s tests/tombstone_adapter_v1 -v
```

The independent Python test-only mathematical oracle checks exact splice
relations and native error precedence, embedded NUL/non-UTF8 bytes, full removal,
ambiguous repeated spans, strict input negatives, and independent canonical
receipt SHA-256 reconstruction. Tests do not become production authority.
The Linux provider test injects failures at each of the actual C99 engine's
three witnessed-splice allocation and three SHA-finalization calls, confirming infrastructure
failure rather than labels. Transport doubles test deadlines/output/cleanup,
not native custody or numeric correctness. CI separately runs GCC and Clang.
The retained [source-bound result](../fixtures/tombstone-adapter-v1/conformance.json)
binds runtime/test source bytes and the deterministic corpus/outcome digest;
`verify_fixture.py` reruns actual C99 rather than trusting cached success. Its
local environment identity is recorded separately from replay equality.
These are public, reused engineering conformance families, not held-out
learning evidence; no model sees them and no family split is claimed.

## Staged plan and non-goals

1. **This PR:** reviewed bounded adapter, pinned native source/license, public
   observation/private outcome split, strict transport and conformance.
2. **Separate future gate:** source-bound producer/admission, structural family
   splits, prediction commitment, independent labels and exposure accounting.
3. **Only after authorization:** matched representation/model experiments and
   frozen evaluation; report null/negative results and resources. No automatic
   training, evaluation pilot, neural proof or deletion inference follows.

This supports TRI-I03, I07, I08 and I10–I14 while preserving native numerical
compatibility and all historical contracts. See the [external roadmap](EXTERNAL-ENVIRONMENTS-ROADMAP.md)
and [module boundaries](MODULES.md).

**Research inspiration only:** Carl G. Jung and collaborators, *Man and His
Symbols* (1964), is noted as a conceptual reference for symbolic representation,
not empirical ML evidence, a model input, copyright/admission determination or
a justification of ternary superiority. No book text, OCR or images are copied
into this module or its fixtures.
