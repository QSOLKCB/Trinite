# Tiny native-opaque compatibility probe v2

Prospective, additive engineering lane: **one fixed non-target Needle3 attempt**.
No 64-call producer, target rows, Bonsai invocation, model downloads, training,
scored readiness or native Trinite changes. The [v1 comparison](TINY-OUTPUT.md)
and [retained negative preflight](TINY-PREFLIGHT-LOCAL-EVIDENCE.md), including
its 51-byte malformed Bonsai response, remain unchanged. The pure SSE helper
corrects future transport validation, not that scorer or result.

Affected TRI-I01/I02/I03/I05/I07/I08/I10/I11/I12/I13/I14. Software CI uses only
synthetic data and standard-library process doubles. It is not a model load,
ABI validation, supported-pair certificate, quality measurement or target run.
No CI command invokes the external SDK or native library.

## Frozen interpretation, not matched decoding

Needle internal thinking is **native-controlled**. No reviewed disable control
is available. Sampling, confidence/suppression gating, penalties and native
output-token accounting remain explicit null values with reasons and bound
source evidence. This profile makes no `thinking=false`, temperature-zero,
greedy, matched-decoding or equal-compute claim. `max_new_tokens=256` is the
requested API argument, not proof of a native token count or reasoning budget.
The 20 layers identify the CACT artifact, not a supported SDK depth control.

The immutable identities remain the previous preflight's weights, embedded RAW
tokenizer, native Linux library and seven unmodified SDK modules. SDK 3.2.1,
native distribution label 3.3.3, and model-config engine version 3.1.0 are
separate facts. A successful local load would establish only empirical
load/capture compatibility for these bytes, not vendor support for the pairing.
The native self-reported version remains unobserved; no distribution label is
silently substituted for it.

The only prompt is:

> Return the integer in field target. Record: left=5678, target=1234, right=9012.

It uses the unchanged generic integer `answer` schema through `complete()` and
an inert dictionary tool. There are no executable functions, no `run()`, no
retrieval, examples, target-row input, response repair, answer-dependent retry,
or second attempt. Constructor settings are `auto_date=False`, `stateless=True`,
empty system, generation 3, 65,536-byte native response buffer and no tool index.
The exact launch template is part of the source-bound request; CLI override
options do not exist.

## Exact local assets

Prepare an explicit **real directory**, independent of the current working
directory. Copy only the already acquired local pinned files; do not install,
download, execute archived scripts, edit source or change pins to pass a check.
The fixed relative layout is:

- `needle3.cact`: 35,335,380 bytes, SHA-256
  `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`.
- `libneedle3.so`: 1,590,584 bytes, SHA-256
  `87476ff70e80334d7c766dbf236b72f42876072df22014c6ec8a9b087b82b260`.
- `tokenizer-embedded.raw`: 109,140 bytes, SHA-256
  `9f839cd44027ea9203738897da2bac572a5cc77aec7fceacd4391fc412a3249f`.
- `sdk/needle/`: `__init__.py`, `_worker.py`, `_telemetry.py`,
  `agent/__init__.py`, `agent/tools.py`, `agent/whistle.py`, `agent/fetch.py`.
  All seven identities are fixed in `SDK_PINS`; the empty initializer is valid.
- `evidence/needle-config.json`, `evidence/needle-native-header.h.txt`,
  `evidence/needle-porting.txt`: exact read-only source evidence from the prior
  preflight's `assets/review/` members; all hashes are fixed in `ASSET_PINS`.

The SDK files can be copied from the prior reviewed source snapshot, but are
never executed from that archive. Original licenses remain authoritative:
[Needle source, Apache-2.0](https://github.com/cactus-compute/needle/tree/b4fc11b04f1852888d375c8c6dfbaca44572d014),
[model revision](https://huggingface.co/Cactus-Compute/needle3/tree/d6c7d6a2e9655505e0f8e23339356dd1705900eb).
No upstream code or binary is newly vendored by this PR.

Canonical records contain only relative safe `path`, exact `bytes`, and
`identity` as `sha256:...`. Absolute paths, traversal, symlink components,
nonregular files, duplicate entries, changed assets and unsupported profiles
are rejected. All assets and repository runtime sources are materialized into
a private attempt directory and rehashed before any SDK import. Only the
seven admitted package files enter its SDK tree. Source files, not bytecode
caches or a preinstalled package, supply the SDK.

## Local-only execution and bounds

Use a reviewed, green exact main checkout on Linux with its existing Python
3.11+ standard library. This does not authorize an installation or model run
in CI or the cloud. The user separately runs the local probe after review.

```sh
python -I -S -B scripts/run_tiny_compatibility_v2.py freeze --asset-root /absolute/assets --output request.json
# Retain the printed request identity independently before any model load.
python -I -S -B scripts/run_tiny_compatibility_v2.py probe --asset-root /absolute/assets --request request.json --request-identity sha256:... --output result.json --allow-native-execution
python -I -S -B scripts/run_tiny_compatibility_v2.py verify --packet result.json --packet-identity sha256:...
```

`freeze` hashes local assets; it does not import the SDK or load the model.
`probe` requires explicit native opt-in and the independently retained request
identity. `verify` is offline source/transport/result consistency, **not inference
replay, independent execution proof, hardware attestation or binary-closure
attestation**. It never imports Needle. Preserve failures as results, do not
change the profile, tune to the answer or run another smoke to seek success.

The supervisor enforces 120 seconds for startup and 120 seconds for the call.
After loading, the worker blocks for an exact supervisor GO handshake; the
supervisor starts its call clock before releasing it. A late queued event cannot
restart the clock. Cleanup has a single five-second maximum budget; an early
kill margin allows process reaping within that same budget. The process group is
killed after failure and after completion to include orphaned native children.
Retained response and diagnostic bytes each cap at 64 KiB. The protocol's base64
transport has a separate 128 KiB bound. Overflow retains a marked response
prefix, never a complete response. stdout/stderr prefixes are fsynced as they
arrive, so partial capture survives final publication failure.

The SDK itself starts a Python child without isolation flags. The v2 harness
therefore supplies a fixed, hash-bound `/bin/sh` launcher that accepts only the
exact staged `_worker.py --child` invocation and executes the frozen Python
with `-I -S -B`. This is explicit harness behavior, not a claim that upstream's
default launcher was already isolated. The actual interpreter, shim, shell
identity, command, nested command, environment and materialized source/assets
are retained. CPU visibility, telemetry-off, offline intent, thread counts,
locale and home/temp paths are explicit; no inherited Python search path or
loader override is forwarded.

The OS dynamic loader, standard-library extension/shared libraries, kernel and
hardware remain environmental trust assumptions. Those dependencies are not
claimed to be a fully attested binary closure. Environment variables and explicit
local asset paths do **not** constitute an OS network sandbox. The reviewed SDK
path selects explicit weights/library and disables telemetry; there is no
requested fetch, installation or network operation. Stop on incompatible local
conditions instead of adding fallback behavior.

## Durable one-attempt semantics and claims

Before native execution, atomically reserve
`.trinite-needle-compatibility-v2-attempt` under the explicit asset root and
fsync its request and reservation. This is independent of output filenames.
No retry, resume, reservation deletion, overwrite or refill path exists. Keep
all partial streams, failed captures, prepared results and failure receipts.
Final publication uses the existing fsynced temporary plus atomic no-clobber
hard-link pattern. If publication fails, recover the retained bytes for review,
not another invocation. The reservation prevents repeat invocation **within
that asset root**; arbitrary copied roots or deliberate deletion cannot be
prevented globally. Such copies are not permission to retry.

Reports separate load completion, call start, complete capture and strict answer
diagnostics. They retain complete SDK-return JSON serialization, including
unknown, suppressed, error and nonfinite metadata; this is not original native
bytes. Native internal repair is not independently observable. The unchanged
v1 scorer evaluates the retained whole response. An incorrect answer can still
be a successful ABI/load/capture observation. A failed or killed process is
never an accepted/correct attempt, even if a retained payload happens to be
correct. Unavailable tokens/memory remain null plus reasons, never zero.
Every request/result/verification reports `target_readiness=false`. There is no
caller-ready boolean and this lane cannot authorize scored comparison.

## SSE transport helper and regressions

`tiny_sse_v2.parse_sse` is a pure standard-library parser, not a live Bonsai
adapter. It bounds the complete stream and content, requires one choice index
zero, exactly one terminal `stop` or `length` before DONE, and rejects missing
terminal/DONE, duplicate terminal, trailing content/events, errors, separate
reasoning/tools, malformed types/usage and conflicting usage duplicates.
`length` labels truncation, never correctness. It returns original content
bytes without repair. The retained 51-byte Bonsai text still fails the unchanged
scorer; no additional Bonsai call is made.

Run normal and optimized tests, plus the unchanged root/upstream suites:

```sh
python -B -m unittest discover -s tests -p 'test_tiny*' -v
python -B -O -m unittest discover -s tests -p 'test_tiny*' -v
PYTHONPATH=src python -B -m unittest discover -s tests -v
PYTHONPATH=src python -B -m unittest discover -s tests/upstream -v
```

Existing source closures, AGENTS, protocols, historical fixture bytes and all
learning/Phase 5/scaling/release gates remain unchanged.
