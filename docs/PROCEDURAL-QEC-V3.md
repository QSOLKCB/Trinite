# QEC procedural v3 — independent-cursor controlled comparison

**Status:** corrected implementation under PR #36. The previous v2 four-cell
result was negative and remains immutable; v3 has no completed full research
outcomes yet. See [retained negative result](PROCEDURAL-QEC-NEGATIVE-RESULT.md).

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
**only changed training input** is the procedural stream's row index:
each (fold/GF3, prefix/trace) stream increments its own cursor *only when
that exact mode is visited*. The final-only stream preserves v2's sequence,
checked by an independent row-identity regression in both seeds.

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
two full cycling rounds are tested. The only planned causal manipulation is
cursor behavior, not an auxiliary-loss coefficient, exposure increase or
capacity change.

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

## Reproduction

From the [hash-locked CPU environment](GETTING_STARTED.md), use distinct
new output paths and independently retain the identities returned by
each command:

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

The separate [manual full research workflow](../.github/workflows/procedural-qec-v3-research.yml)
executes all four new research cells after source publication, full native
replay and independent archival verification; PR conformance runs only the
small-model smoke. The manual workflow must be explicitly launched only
after code review/merge, not inferred from passing CI.

Previous negative results are permanently documented in
[PROCEDURAL-QEC-NEGATIVE-RESULT.md](PROCEDURAL-QEC-NEGATIVE-RESULT.md).
The original full source supplement and evidence remain separate from v3;
their hashes cannot be patched to certify a new producer.

Impacted invariants: TRI-I02/I03/I05/I06/I07/I08/I11/I12/I13/I14.
