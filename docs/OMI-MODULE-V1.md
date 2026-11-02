# OMI external module v1 — engineering, not training

This additive module implements the first approved step on 2 November 2026:
**OMI module/admission/token ledger → separate QEC evidence adapter → separate
bounded controller**. No model construction, optimizer, training, QEC rebuild,
latent-state access or automatic retry occurs. Historical procedural v1,
PR125 diagnostics, retained 8/8 and 7/8 results and all fixtures are unchanged.
There is no demonstrated held-out gain or numerical/quantization fix here.

## Arithmetic, donor and rights

Credit: **Brian Thorne's ARCH OMI**. The donor is
[VANILLAJS OMI Lab](https://github.com/QSOLKCB/VANILLAJS/blob/9d228aed9791dccac6e62c9dbc55738cc26ac679/experiments/arch-omi/index.js),
commit `9d228aed9791dccac6e62c9dbc55738cc26ac679`, under
[Apache-2.0](https://github.com/QSOLKCB/VANILLAJS/blob/9d228aed9791dccac6e62c9dbc55738cc26ac679/LICENSE).
The local Rust arithmetic adaptation preserves attribution and the Apache license
in `external/omi-v1/LICENSE-APACHE-2.0`; the existing restricted JSON parser is
reused under the repository MPL-2.0. No upstream executable or assets are vendored.
The original pinned references are Brian's
[delta specification](https://github.com/bthornemail/omi-portal/blob/5d2ebac312a8d44413af2a67ca9061e66594d9c8/docs/03-network/omi-core-spec.md#6-delta-law-evaluator)
and [kernel](https://github.com/bthornemail/omi-portal/blob/5d2ebac312a8d44413af2a67ca9061e66594d9c8/src/omi/omicron-kernel.js).
This is VANILLAJS's four packed nibble finite-word encoding, not a claim to
implement every OMI representation or canonical six-bit activation semantics.

For unsigned 16-bit x, Δ(x) = rotl(x,1) XOR rotl(x,3) XOR rotr(x,2) XOR 0x5A3C.
Rust checks every word against arithmetic bit-parity (division, modulo and
addition, no bitwise XOR/shift), verifies permutation/inverse and Δ⁸ identity.
Minimal-period state counts are 1:8, 2:56, 4:4032, 8:61440; 8,724 distinct orbits.
Thus an always-eight classifier scores **93.75%** over uniform words. Report
per-period performance and balanced controls, never raw accuracy alone.
Eight-step identity is deliberately absent from task horizons. Shorter orbits
still yield legitimate identity cases, which must be separately reported later.
Closure is a finite arithmetic property, not reasoning, error correction or QEC.

The generated data is commissioned finite arithmetic facts and minimal carriers,
not imported expressive text, pretrained answers or private information. Code
licensing is separate from this source-specific data assessment, not legal
certification. All exposed cases are public engineering conformance, including
any split named validation/test: they are not fresh operator-blind evaluation.

## Strict JSON boundary and admission

The dependency-free CPU Rust CLI takes exactly `observe` or
`evaluate-privileged`, never a privileged default. Input keys are exactly
`schema`, `word` (integer 0..65535), `task`, `horizon` (integer 1..7).
Schema is `trinite.omi-request.v1`. Tasks `next`, `inverse`, `period` require
horizon 1; `forward`, `backward` accept mixed horizons 1..7. Next/forward return
Δ^h, inverse/backward Δ^-h, formatted as four uppercase hex characters; period
returns one decimal character. Booleans, floats, duplicate/unknown keys, null,
escapes, non-ASCII, oversized input and out-of-range values reject with exit 2
and empty stdout. `observe` only echoes the approved operands with observation
schema; it does not compute or expose answer, orbit ID, seed, split or provenance.
The privileged outcome is separate. JSON separation is not OS isolation: a later
trusted controller must keep oracle processes and private records inaccessible
to the learner. The model receives public JSON tokens only, not an executable.

Additive `scripts/omi_module_v1.py` reuses the authoritative ByteTokenizer and
bounded subprocess transport, without altering frozen procedural v1. The public
carrier `["O1","ABCD","forward",7]` plus `\n=` is lossless and canonical; strict
roundtrip restores schema. It contains only supplied operands. `public_only`
accepts an exact public schema, never target/provenance. No answer is computed
for public inspection. `admit_train` checks whole-orbit split **before** invoking
the privileged binary, checks independent arithmetic agreement, then returns
answer-only loss masks with EOS. The binary and its local build/source receipt
are rehashed around each call. A stale build rejects; infrastructure failures
are not semantic negative labels. Build receipts are disposable local bindings,
not compiler attestations. Source/binary check-to-use races remain an operator
trust boundary. This is an admission API, not a scientific execution controller.

A family is the minimum word of its complete Δ orbit. Split is SHA256 of
`omi-v1:XXXX`, first byte modulo 10: 0..7 train, 8 validation, 9 test. Every task,
horizon and inverse view inherits that family. Hash partitioning does not promise
balanced period strata or eliminate algebraic correlations between orbits.
Future evaluation must freeze inventories, inspect class distributions and use
matched exact/identity/majority controls. Training needs a new exact freeze and
explicit owner approval; this PR authorizes zero optimizer updates.

## Token and compute ledger

Counts use actual native byte tokens: JSON syntax, quotes, task names, separator,
BOS and EOS all count. Inference excludes EOS; supervised sequences include it.
The protocol permits at most 120 ASCII prompt bytes, hence 121 prefix tokens with
BOS and 126 complete tokens with four answer bytes plus EOS (unchanged context 256).
Four hex characters need **five emitted tokens**, not one. Period needs two.
The ledger reserves answers without computing them and counts full-prefix
positions `sum(P+k for k in range(G))`. These are shapes, not measured FLOPs,
latency, cache traffic or proof of compression benefit. Admission rejects rather
than truncates. No model architecture/tokenizer/context limit changes.

The future eight-round controller proposal would cap 40 full-prefix forwards and
4,920 processed prefix positions at P≤121, G=5 (615 per round). It is **not
implemented** here and is not an approval or proof of learning. It must handle
EOS/invalid outputs, budget stop, matched single-pass/equal-compute controls,
separate module/preprocessing cost, frozen schedules and retained failures.

## Reproduce offline (Rust 1.90.0, Python 3.11+)

```sh
python3 scripts/run_omi_module_v1.py build
(cd external/omi-v1 && cargo test --offline --locked && cargo fmt --check && cargo clippy --offline --locked -- -D warnings)
PYTHONPATH=src:scripts python3 -B -m unittest discover -s tests/omi_v1 -v
python3 scripts/run_omi_module_v1.py inspect --word 43981 --task backward --horizon 7
printf '%s' '{"schema":"trinite.omi-request.v1","word":43981,"task":"next","horizon":1}' | external/omi-v1/target/release/trinite-omi-v1 observe
```

Tests are exhaustive arithmetic/grouping engineering checks plus actual CLI,
malformed/overflow, public-only leakage, token masking and train-filter tests.
They perform no model imports or optimizer updates. No generated fixture is
claimed fresh held-out evidence. No new training dependency is installed.
