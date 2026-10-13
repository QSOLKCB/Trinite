# Procedural QEC research execution and lossless archive (PR #35)

**Status: research execution and archival tooling, not a completed full research result.** PR #34 established QEC GF(3)/fold-sum procedural targets and an actual two-seed/two-arm CPU smoke. Its 4,096-update research matrix remains unmeasured. This increment closes the execution/evidence workflow, without changing any model, numerical method or frozen experiment data.

## Frozen scientific boundary

[Protocol v2](../src/trinite/procedural_qec/protocol-v2.json) remains byte-identical: SHA-256 `3c44c40c10fd63026cc7f8822f9b1a0842737897e33a91afef75a6c28d70d1bb`, first published at commit `0697d5944d48517bb0ab04afc211af7a9826cd2f`. The native random initialization, CPU dense Decoder, byte tokenizer, AdamW, GF(3) oracle, loss masking, family assignments, final-step checkpoint and geometry loss zero remain unchanged.

The manually triggered [research workflow](../.github/workflows/procedural-qec-research.yml) freezes a **new host-bound research request** for the published checkout, then runs four cells: seeds 0 and 1, each with final-answer-only and procedural supervision, 4,096 updates and batch eight. It records all negative/failed cells, actual target-token exposure, complete-answer-plus-EOS accuracy for every original task, and independent prefix/full-trace accuracy. It performs fresh full native optimizer replay and a separately invoked archival replay.

## Archive contract

The new [archive CLI](../scripts/archive_procedural_qec.py) and [standalone archive verifier](../scripts/procedural_qec_archive.py) retain the *exact* original request, independently supplied verification request, complete run files, original source closure, archive-tool source and fresh replay receipt. The ZIP has a canonical member inventory with SHA-256 and byte length for **every** file, plus an independently recorded archive digest and inventory digest. Unmodified negative outcomes are eligible for archival when all numerical/custody checks complete.

Archival rejects incomplete/missing/duplicated cells, bad external commitments, changed original producer bytes, symlinks, path traversal, oversized members, forged summaries, altered artifacts, and missing native replay. Full native replay is executed **before packaging**; the independent standard-library ZIP verifier then checks committed bytes and claimed receipt structure without importing Torch. The ZIP verifier alone is not proof of numerical replay.

The source closure of this archival tool is separately recorded. It is not falsely represented as part of the frozen PR #34 trainer. No historical protocol, previous CNR packet, v0.1.0/v0.2.0 tag or previous negative evidence is overwritten.

## Run

Use the exact Linux x86_64 CPU environment and lock in [GETTING_STARTED](GETTING_STARTED.md). New paths must be distinct, unused, regular and external to the run:

```sh
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_procedural_qec.py --freeze-request /tmp/pq-request.json --run-profile research
# Retain the returned REQUEST identity externally.
python scripts/run_procedural_qec.py --run \
  --request /tmp/pq-request.json --request-identity sha256:REQUEST \
  --verification-request /tmp/pq-verification.json --output /tmp/pq-run
# Retain the returned VERIFIER identity externally.
python scripts/run_procedural_qec.py --verify-only \
  --request /tmp/pq-request.json --request-identity sha256:REQUEST \
  --verification-request /tmp/pq-verification.json \
  --verification-request-identity sha256:VERIFIER --output /tmp/pq-run
python scripts/archive_procedural_qec.py --archive \
  --run /tmp/pq-run --request /tmp/pq-request.json \
  --request-identity sha256:REQUEST \
  --verification-request /tmp/pq-verification.json \
  --verification-request-identity sha256:VERIFIER \
  --output /tmp/Trinite-procedural-QEC-research-v2.zip
# Independently retain returned ARCHIVE and INVENTORY identities.
python scripts/archive_procedural_qec.py --verify-archive \
  --archive-file /tmp/Trinite-procedural-QEC-research-v2.zip \
  --archive-identity sha256:ARCHIVE --inventory-identity sha256:INVENTORY
```

For the full experiment, use GitHub Actions → *QEC procedural full research evidence* → *Run workflow* after the workflow exists on the default branch. It is **manual by design**; automatic PR CI runs only bounded smoke/integrity conformance. The workflow uses explicit `set -euo pipefail`, source-locked dependencies and full-graph/optimizer validation, and uploads available request, run, failure, archive and checksums as a 90-day artifact **even on job failure**. This is temporary transport, not permanent research publication; archive the complete packet and externally held digest separately before discarding runner evidence.

## What passes and what remains blocked

Pure-standard-library tests employ *fabricated structural packets* for ZIP closure and tampering; they are **not numerical verification**. The locked procedural CI smoke test now creates a lossless ZIP from four actual tiny CPU training cells, then read-only verifies it. Neither a green smoke nor a complete negative research run means learning success.

Research training acceptance requires ≥90% complete original final-answer+EOS accuracy in every declared task and **both** procedural seeds, with all four research cells verified at the six specified assurance levels. All result rows, including regressions and null findings, must be retained without post hoc selection. Trace accuracy cannot replace the original final-answer criterion. Shared arithmetic visits match between arms; total final-answer visits and target-token exposure **do not**, and the comparison cannot attribute any difference solely to reasoning traces.

Fresh family-level confirmation, original composed/fold gate, independent-host replication, CPU packing, format selection, scaling, Phase 5 geometry training and release readiness remain separately blocked. No pretrained model or teacher data has been introduced.

Affected invariants: TRI-I02/I03/I05/I06/I07/I08/I11/I12/I13/I14.
