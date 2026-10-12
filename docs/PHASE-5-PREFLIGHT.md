# Phase 5 preflight: authorization is not a learning result

**Status:** PR #30 implementation proposal; deliberately **non-authorizing**. This
adds a read-only custody/approval check in advance of a future Phase 5 controlled
geometry intervention. It does not implement a differentiable geometry loss,
execute geometry-on training, issue a runnable authorization, or claim Phase 5
acceptance. The v0.1.0 baseline, old protocol source closures, failures and
retained data are unchanged.

The [roadmap](ROADMAP.md) still requires original foundations learning adequacy,
original fold/composed-output adequacy, fresh family-level confirmation and
independent reproduction *before* the controlled geometry intervention. The
frozen foundations gate first requires 90% training exact-answer-plus-EOS for
**every** dense workload/seed/task; only then is held-out evaluation allowed.
Held-out must clear 50% per task and a 10 percentage point margin above the
training-majority baseline. The retained result fails the prerequisite. Green
CI, 227 research tests, four replayed task-balance cells, or a plausible
32-fact-per-task core selection cannot change those empirical observations.

## Implemented preflight boundary

The standalone, standard-library-only `scripts/phase5_preflight/preflight.py` module validates a **prospective**
packet and a separate externally SHA-256-pinned trust policy:

- Exact canonical JSON, strict schemas, material identities, UTC validity window
  (at most 24 hours), 128-bit minimum nonce and Git tree reference.
- A complete ordered matrix of dense and ternary arms, each with geometry-off,
  geometry-on and auxiliary-null controls at **every** paired seed; shared
  per-cell update/token/wall ceilings, frozen objective/control identifiers,
  margins and uncertainty method. A proposal's geometry-loss weight must be a
  **canonical decimal scalar** in `(0,1]` with at most six fractional digits,
  no redundant trailing zeros, exponent, sign, `NaN`, `Infinity`, or freeform
  sentinel strings. This is schema hygiene, **not** proof that the weight is
  scientifically adequate or the objective differentiable.
- Two **distinct, explicitly trusted Ed25519 public keys**: a protocol steward
  and an independent verifier. Both must validate detached OpenSSL signatures
  over the *same complete canonical packet*. A role label is not a signature,
  the signatures are not trusted because they are present in the packet, and
  the policy digest **must** come from an out-of-band source.
- An optional invocation of the existing authoritative
  `scripts/verify_foundations_run.py` against an explicitly identified retained
  root. It checks the unchanged source/environment, custody, scores and
  summary, using that verifier's disposable-copy approach. Its scope is
  **checkpoint and scoring verification, not optimizer-update replay**.
- An exclusive, canonical, SHA-256-addressed **blocked** receipt that records
  each unsatisfied predicate. Exit code 1 means the expected blocked verdict;
  2 means malformed inputs or failure to retain the receipt. Existing receipts
  are never replaced; give each assessment a fresh path.

The preflight helper lives under `scripts/phase5_preflight/`, **outside all
historically frozen `src/trinite/` inventories**. The existing flat maths source
receipt binds top-level files, and CNR-001 recursively commits to the entire
first-party source package. No frozen source-inventory, model, updater, archive,
request or historical witness is modified or resealed. This is standalone
repository tooling invoked through `scripts/check_phase5_preflight.py`, not a
new runtime dependency or exported installed model feature.

The packet schemas are `trinite.phase5-preflight-packet.v1`,
`trinite.phase5-geometry-proposal.v1` and
`trinite.phase5-trust-policy.v1`. Receipt schema is
`trinite.phase5-preflight-receipt.v1`.

```sh
PYTHONPATH=src python scripts/check_phase5_preflight.py \
  --packet /outside/repo/phase5-candidate.json \
  --signatures /outside/repo/phase5-detached-signatures.json \
  --trust-policy /outside/repo/phase5-trusted-public-keys.json \
  --trust-policy-identity sha256:EXTERNALLY_PINNED_POLICY_DIGEST \
  --foundations-run /outside/repo/foundations-run \
  --output /outside/repo/phase5-preflight-rejection.json
```

All JSON objects must use Trinite canonical JSON (sorted keys, no extra
whitespace, trailing LF). A **semantically malformed packet** (unknown schema
field, incomplete matrix, boolean seed, invalid objective weight, etc.) exits
with code **2 without writing a blocked research verdict**. A valid proposal
that cannot satisfy the learning/authorization gate exits with code **1** and
an exclusive blocked receipt. The `signatures` file is an array with one object for
each required role: `role`, `key_identity`, `algorithm: ed25519`, and
`signature_b64` (standard canonical Base64 over 64 raw Ed25519 signature bytes).
The trusted public-key policy is a *different* file listing two corresponding
role/key/PEM records; each v1 `key_identity` remains the SHA-256 identity of the
**exact ASCII PEM bytes**, but **quorum independence is checked on canonical
Ed25519 SubjectPublicKeyInfo DER bytes** (the 32-byte public keys). Validly
rewrapping the same PEM cannot create a second independent key; EC/ECDSA,
RSA, Ed448, concatenated or malformed PEM and DER are rejected before signature
verification. External policy pinning must independently enforce organizational
separation; distinct mathematical keys do not prove different people or services. **Keep signing keys outside the repository and off experiment
workers.** Packet signatures can be created independently by the two key
holders using OpenSSL's `pkeyutl -sign -rawin` over the canonical packet bytes.
The tests generate throwaway keys; no production keys or fake successful
scientific receipts are committed.

The existing `verify_foundations_run.py` emits canonical JSON **followed by
one transport LF** from its `print()` call; the new adapter validates precisely
that framing, removes only the transport LF, and strictly checks the six
returned receipt fields and both request/summary identities. It does not
silently `strip()` output or trust a simulated score. If the verifier fails or
times out, the blocked receipt retains its return status and **bounded** stdout
and stderr previews (Base64), byte counts, SHA-256 digests, and truncation flags.
The original run files and failure logs are not overwritten; complete unbounded
child logs are not embedded in the receipt. A dedicated private temporary
workspace is selected independently of user `TMPDIR`/`TMP`/`TEMP` and is
required to lie outside the retained foundations root. The child's temporary
environment is overridden to that workspace, so **its** disposable copy cannot
be created under the evidence directory either. Failure to identify safe scratch
space is an error, never permission to modify the evidence root. Assessment
timestamps and the exact signature-bundle identity are recorded, and the
request expiry is rechecked on a fresh UTC clock **after** any
foundations-verification subprocess.

`--output` must be outside the retained `--foundations-run` root, including
nested paths, `..` aliases, and symlink traversal. Receipts are created through
pinned no-follow **directory descriptors** (not a second vulnerable pathname
resolution) with exclusive creation and mode `0600`; an existing receipt is
never replaced. The receipt file is synced, followed by its parent directory
and every newly created ancestor's directory entry, **before** the identity is
printed. If durability fails, the CLI reports error code 2 instead of success.
The receipt also contains a canonical local tool/source manifest and its hash
(CLI/helper, first-party sources, lock, foundations entrypoints).
These are **disclosure identities of the verifier's local inputs**, not an
externally signed guarantee that the local verifier is trusted. A separate
reviewed trust/witness boundary must bind the verifier itself before executable
authorization is possible.

## What this cannot authorize

A content hash is not data rights approval, an independent witness, a verified
numerical computation, or a positive learning result. Even a valid two-key
signature quorum does not make a Phase 5 experiment legitimate. In this first
increment **the final authorization predicate is always false**. The verifier
explicitly rejects promotion until the following *independently implemented
and reviewed* verifiers exist:

1. Complete, separately source/environment-bound optimizer-history replay for
   the original qualifying learning matrix (the v1 foundations verifier does
   not establish this).
2. Independently verified *fresh* family-level confirmation with a preregistered
   exposure boundary, plus restored original fold/composed-answer adequacy.
3. Cryptographically verifiable external pre-outcome protocol registration,
   source/data-admission receipts and protocol gradient/objective conformance.
4. A separately permissioned, single-use execution authorization service,
   isolated geometry worker, complete terminal-cell custody ledger and
   independent outcome/causal verifier. None is shipped in this PR.

There is deliberately **no** `phase5_ready = True` path, no effect on existing
training/model files, and no geometry-on runner in this increment. A future
runner must never accept a preflight receipt as an execution token. Signed
rejections, long-term transparency-log anchoring and revocation remain later
extensions; a locally stored SHA-256 digest is not a substitute for them.

## Reproduction and checks

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_phase5_preflight.py' -v
# Full repository conformance also required in the pinned CPU environment:
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s tests/upstream -v
PYTHONPATH=src python scripts/check_research_cpu.py
```

The preflight tests use **actual OpenSSL Ed25519** sign and verify operations,
including invalid signatures, same Ed25519 key with different PEM wrapping,
actual 64-byte ECDSA signatures mislabeled as Ed25519, RSA/Ed448 rejection,
wrong external policy digest, duplicate signer, boolean matrix seeds,
incomplete lane matrix, proposal/request mismatch, expiry and post-verifier
expiry, strict historical stdout framing, bounded verifier diagnostics,
foundations-run output overlap, poisoned `TMPDIR` and child temp isolation,
symlink-parent replacement during receipt creation, directory-entry syncing,
malformed packet exit status, canonical bounded objective weight, forged
apparent learning flags, exclusive receipt retention and a cryptographically
valid proposal that remains blocked. They are cryptographic/software-conformance
checks, not evidence that Phase 5 passed.

**Affected invariants:** TRI-I02, I03, I07, I08, I09, I11, I12, I13, I14.
Original training protocols, admission rules, model/optimizer, geometry loss
(`0`) and historical fixtures are preserved byte-for-byte.
