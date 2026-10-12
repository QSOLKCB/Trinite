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

The stdlib-only `trinite.phase5_preflight` module validates a **prospective**
packet and a separate externally SHA-256-pinned trust policy:

- Exact canonical JSON, strict schemas, material identities, UTC validity window
  (at most 24 hours), 128-bit minimum nonce and Git tree reference.
- A complete ordered matrix of dense and ternary arms, each with geometry-off,
  geometry-on and auxiliary-null controls at **every** paired seed; shared
  per-cell update/token/wall ceilings, frozen objective/control identifiers,
  margins and uncertainty method.
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
whitespace, trailing LF). The `signatures` file is an array with one object for
each required role: `role`, `key_identity`, `algorithm: ed25519`, and
`signature_b64` (standard canonical Base64 over 64 raw Ed25519 signature bytes).
The trusted public-key policy is a *different* file listing two corresponding
role/key/PEM records; each `key_identity` is the SHA-256 identity of the exact
ASCII PEM bytes. **Keep signing keys outside the repository and off experiment
workers.** Packet signatures can be created independently by the two key
holders using OpenSSL's `pkeyutl -sign -rawin` over the canonical packet bytes.
The tests generate throwaway keys; no production keys or fake successful
scientific receipts are committed.

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
including invalid signatures, wrong external policy digest, duplicate signer,
incomplete lane matrix, proposal/request mismatch, expiry, forged apparent
learning flags, exclusive receipt retention and a cryptographically valid
proposal that remains blocked. They are cryptographic/software-conformance
checks, not evidence that Phase 5 passed.

**Affected invariants:** TRI-I02, I03, I07, I08, I09, I11, I12, I13, I14.
Original training protocols, admission rules, model/optimizer, geometry loss
(`0`) and historical fixtures are preserved byte-for-byte.
