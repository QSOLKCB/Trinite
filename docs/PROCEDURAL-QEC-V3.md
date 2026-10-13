# QEC procedural v3 — independent-cursor controlled comparison

**Status:** corrected implementation merged in PR #36, followed by PR #37
evidence/CI hardening and PR #38 installed-wheel compatibility; verified
source baseline `cc996e272800a5c02667001f7c96c8d11cb513cc`. The previous v2 four-cell
result was negative and remains immutable. No completed v3 full-research
evidence is yet retained here. A source-identity-induced historical control
ordering mismatch has now been confirmed; see the audit below. See [retained negative result](PROCEDURAL-QEC-NEGATIVE-RESULT.md).

## Frozen experiment

The v3 [protocol](../src/trinite/procedural_qec_v3/protocol.json) was
published **before** the new numerical producer and any new v3 result in
commit `66a8d927ee3026e47268f6fdfae5bad3c00565df`, with SHA-256
`be5e998e5e79d52ecae40bf679c8b0a686c70577c78ac4ff0af76b49d7f27b18`.
This is a distinct protocol/request/checkpoint/report/summary and source
closure, not a v2 edit or a post hoc reanalysis pretending to be a new run.

The original 86,528-parameter dense CPU model, random initialization,
native AdamW, constant LR 0.001, 4096 updates, batch eight, tokenizer,
exact admitted QEC GF(3)/fold-sum data, all arithmetic/final-answer task
counts, two seeds 0 and 1 and geometry loss zero are unchanged.

Per update the arms still consume four R/-namespaced arithmetic finals, two
original fold-sum finals and one GF(3) final. The eighth slot is either
the original final-answer-only example or a procedural prefix/trace. The
**intended protocol manipulation** is the procedural stream's row index:
each (fold/GF3, prefix/trace) stream increments its own cursor *only when
that exact mode is visited*. The final-only regression checks equality with the v2 scheduling function
when both functions receive the **current** regenerated rows. That check does
not establish equality with the historical v2 producer's actual sequence.

Under the fixed research budget each mode receives 1024 visits:
- fold prefix: 1024 distinct / 1150 available;
- fold full trace: 1024 distinct / 1150;
- GF(3) prefix: all 54 distinct / 54;
- GF(3) trace: all 54 distinct / 54.

The old v2 schedule sampled only 575/1150 and 27/54 distinct rows by
incorrectly coupling row parity to mode. The corrected code fails closed if
distinct coverage is smaller than `min(visits,available)`. A complete fold
auxiliary pass still requires at least 4600 total updates under this cadence;
increasing the budget is not part of v3. Both even and odd stream widths and
two full cycling rounds are tested. The planned manipulation is cursor behavior, not an auxiliary-loss coefficient,
exposure increase or capacity change. The historical ordering audit below
shows an additional realized change, so cursor-only causal isolation is not
established by this current producer.

## Historical control ordering audit (13 October 2026)

The source portability correction changes `data.py` and its generator-source
receipt. The generator includes that receipt's revision in each row's
`example_identity`; both schedulers seed-rank those identities. Consequently a
source-only change can reorder mathematically identical training rows.

A read-only reconstruction from historical producer
`93522bf8f18fe38970ceafdd2ace792f8922c382` exactly regenerated its independently
recorded dataset SHA-256
`120b95ac68825b50b92dc0210d1f142a30980bfe5acedcfafe6d8f3cdadf83c4`.
The current `cc996e272800a5c02667001f7c96c8d11cb513cc` producer generates
`9ea3223ebd099933f9c05a3ccb9db43e4a7d19e3c1edf5922c326c2eec16d944`.
The full mathematical row content, after removing only each row's
`generator_revision` and `example_identity`, agrees exactly. Yet the
final-only mathematical sequence differs at **32,576 of 32,768 visits in
seed 0** and **32,521 of 32,768 visits in seed 1**.

The [source-bound audit and reproduction instructions](evidence/procedural-qec-control-order/README.md)
retain the original [comparison receipt](evidence/procedural-qec-control-order/comparison.json),
whose independently recorded SHA-256 is
`7e0160ac9ca5c6241dd8001bbc8775c6dd1c51266755533705debd819d9a087b`.
Its executable check validates both fixed Git producers' actual source bytes,
regenerates and admits both complete datasets, compares every row field except
the two named identity fields, and reconstructs all four schedules. It also
reproduces the same-current-row regression-check blind spot. The dedicated
[read-only CI audit](../.github/workflows/procedural-control-audit.yml) executes
the two pinned sources and rejection tests without importing Torch or training.
Passing this audit confirms the ordering **failure**, not historical parity.

This violates the intended preservation of the historical final-only order.
It is a scientific comparison limitation even when current-source software
conformance and complete native optimizer replay pass. A newly executed
current-producer matrix must retain all outcomes and this mismatch; it cannot
be described as an isolated cursor correction relative to the immutable v2
result. Its actual host must also be reported separately from the historical
host. No source metadata, original row identities, old requests or numerical
results may be patched to manufacture historical parity.

Correcting the source-dependent ranking requires a separately reviewed,
prospectively frozen protocol/producer and fresh request. Do not modify the
frozen v3 protocol, silently change its ordering, or execute that new scientific
comparison without the applicable authorization. Geometry, learning adequacy,
format selection, scaling and release gates remain unchanged.

## Loss interpretation repair

The v2 field `final_train_loss_hex` was actually the teacher-forced
aggregate over final/prefix/trace development rows. In v3 the same kind
of diagnostic is labelled `mixed_all_modes_loss_hex`, distinct from
`final_only_loss_hex`. The report also includes
`loss_by_mode_and_task` with per-mode overall and per-task teacher-forced
losses; `schedule_coverage` with unique visited/unvisited row commitments;
and `auxiliary_visited_unvisited` with separate exact free-running correct
and example counts. All these are **training/development diagnostics**.
Even an unvisited derived row is not a novel family-level test.

The primary scientific criterion remains greedy **complete final answer
plus EOS** in six tasks: arithmetic add/subtract/multiply/divide, original
fold-sum, and GF(3). The 90% threshold applies to **every** final task in
both procedural seeds, not an overall mean or auxiliary trace accuracy.
Arithmetic retention must be reported, too. No early stopping,
winning-seed selection, test evaluation, format selection or Phase 5
authorisation follows from software conformance alone.

The arms still differ in final-answer visits and scored target tokens.
Corrected row coverage does not isolate the causal effect of "reasoning"
as a mechanism. If v3 still regresses, non-displacing separately normalised
auxiliary supervision is a *future* distinct protocol, not bundled here.

## Full source and evidence

V3 binds all first-party runtime `src` code, schemas, text licenses and
locked resources, plus the relevant training entrypoint, pyproject and
dependency lock, into the request's immutable source inventory. A
read-only replay must match every source identity and re-execute the
original native optimiser path. Retain every final model tensor and AdamW
slot, step histories, predictions, source-bound request and independently
committed verification request, PROVENANCE custody and null results. The
v3 archiver retains those files and the complete source tree with an
independent archive and inventory SHA-256. No historical v2 bundle or
CNR v1–v5 packet is overwritten.

V3 source closure also triggers a legitimate new CNR source receipt. The
CNR numerical algorithm remains frozen; a distinct CNR
`cnr001-v6.tar.gz` source-bound archive, independently checksummed and
linked to immutable v5, preserves exact **current** request/analysis/receipt
equality rather than altering old v5 evidence.

## Reproduction of the historically confounded current producer

**These are preservation/reproduction commands, not the next exact-parity
experiment. Do not launch another current-v3 matrix to establish an isolated
cursor effect or historical-control preservation.** The existing source is
retained so an explicitly requested reproduction can be inspected and replayed;
a successful numerical replay does not clear the ordering limitation above.
Any new exact-parity comparison must first pass the separate protocol/review/
authorization gate in [the next-step contract](PROCEDURAL-QEC-NEGATIVE-RESULT.md#required-next-comparison).

Use a **separate, clean detached checkout of the identified producer**, not
the latest `main` or the checkout containing this document. V3 inventories
all first-party `src` files; even an unrelated additive package changes its
source closure. With the pinned commit already fetched, choose an unused
worktree path and verify the actual checkout before importing or freezing:

```sh
git worktree add --detach /tmp/trinite-qec-v3-cc996e cc996e272800a5c02667001f7c96c8d11cb513cc
cd /tmp/trinite-qec-v3-cc996e
test "$(git rev-parse HEAD)" = cc996e272800a5c02667001f7c96c8d11cb513cc
test -z "$(git status --porcelain)"
```

Run the following commands **from that pinned checkout**, using the
[hash-locked CPU environment](GETTING_STARTED.md), `PYTHONPATH=src`, distinct
new output paths, and independently retained returned identities. A fresh
request binds the actual new environment; a different host is a separately
identified execution, not verification of the original request's fingerprint.

```sh
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_procedural_qec_v3.py \
  --freeze-request /tmp/procedural-v3-request.json --run-profile research
# Use REQUEST from the freeze receipt:
python scripts/run_procedural_qec_v3.py --run \
  --request /tmp/procedural-v3-request.json --request-identity sha256:REQUEST \
  --verification-request /tmp/procedural-v3-verifier.json \
  --output /tmp/procedural-v3-run
# Independently retain VERIFIER from the run receipt:
python scripts/run_procedural_qec_v3.py --verify-only \
  --request /tmp/procedural-v3-request.json --request-identity sha256:REQUEST \
  --verification-request /tmp/procedural-v3-verifier.json \
  --verification-request-identity sha256:VERIFIER \
  --output /tmp/procedural-v3-run
python scripts/archive_procedural_qec_v3.py --archive \
  --request /tmp/procedural-v3-request.json --request-identity sha256:REQUEST \
  --verification-request /tmp/procedural-v3-verifier.json \
  --verification-request-identity sha256:VERIFIER \
  --run /tmp/procedural-v3-run --output /tmp/Trinite-procedural-QEC-v3.zip
# Record both ARCHIVE and INVENTORY identities from the archiver.
python scripts/archive_procedural_qec_v3.py --verify-archive \
  --archive-file /tmp/Trinite-procedural-QEC-v3.zip \
  --archive-identity sha256:ARCHIVE --inventory-identity sha256:INVENTORY
```

The unchanged [manual full research workflow](../.github/workflows/procedural-qec-v3-research.yml)
checks out its selected branch/tag; it is **not pinned to `cc996e`**. Running
it on current `main` creates a different source-bound request and cannot be
described as reproduction of this identified producer. Use the local pinned
route above for an explicitly requested reproduction. A hosted reproduction
would additionally require an eligible selected ref independently verified to
resolve to this exact commit; do not assume that a dispatch UI accepts an
arbitrary commit SHA, create a new ref, or launch it from these instructions.
Current v3 is **not eligible for the next exact-historical-parity comparison**.
PR conformance runs only the small-model smoke. A corrected scientific
comparison needs a distinct reviewed and authorized protocol/producer before
any new target outcomes.

Previous negative results are permanently documented in
[PROCEDURAL-QEC-NEGATIVE-RESULT.md](PROCEDURAL-QEC-NEGATIVE-RESULT.md).
The original full source supplement and evidence remain separate from v3;
their hashes cannot be patched to certify a new producer.

Impacted invariants: TRI-I02/I03/I05/I06/I07/I08/I11/I12/I13/I14.
