# TT-003-FIT-001-EVIDENCE-REVIEW-001 — independent, offline evidence review

**Status:** prospective additive research tooling. **Read-only.** No native
model inference, optimizer updates, new seed, checkpoint rewrite, approval,
training, TT-004 distillation, publication of evidence or model release.

## Research question and authority separation

TT-003-FIT-001 precommitted three different initialisation seeds (17/29/43),
one source-bound 72-example TRAIN panel and 256 updates each. Lower token NLL
or apparent improvement over the constant-zero and constant-one controls is
not independent evidence of correctly learned GF(3) operations. This audit
asks **which** task views generated complete correct answers; **when**
correctness was gained or lost; whether positive/negative check preferences
flipped; whether EOS collapsed output lengths; and whether vector residues
were correct under greedy versus gold-prefix conditioning.

This tool makes independent static observations of a separately supplied
archive. It does **not** reproduce the native forward pass, validate
checkpoint tensor semantics, replay AdamW updates, authenticate the producer,
prove the mathematical oracle beyond the source-bound dataset commitments,
or establish holdout generalisation. Existing archived native-forward replay
receipts are checked for consistency and reported as **previously recorded**,
never as work performed by this tool. SHA-256 proves archive byte identity
against the **external** hash; it does not attest who produced the bytes.

No raw archive, checkpoint, model weights, optimizer moments, corpus, approval
record, user directory path or private measurement output is to be committed
or uploaded into GitHub Actions. CI exercises synthetic software doubles
only, not this experiment's evidence.

## Offline procedure

From a clean trusted Trinite checkout, use the standard library only:

```sh
PYTHONPATH=scripts python3 -B scripts/run_teacher_fit_review.py \
  --archive /absolute/private/TT-003-FIT-001-COMBINED.zip \
  --sidecar /absolute/private/combined-archive.sha256 \
  --out /absolute/private/TT-003-FIT-001-EVIDENCE-REVIEW-001.json

PYTHONPATH=scripts python3 -B -m unittest discover \
  -s tests/teacher_fit_review_v1 -v
PYTHONPATH=scripts python3 -B -O -m unittest discover \
  -s tests/teacher_fit_review_v1 -v
```

Alternatively, supply `--archive-sha256 sha256:<64 lowercase hex>` directly.
An output argument creates **only a new local file** outside the source tree;
without `--out` the derived aggregate JSON is printed to stdout. No archived
files are extracted or executed. The tool never imports PyTorch, reads a
secret key, contacts GitHub, invokes a worker, or mutates the original ZIP.

## Independent contract checks

The reviewer fails closed on any disagreement in:

- Externally retained SHA-256 versus the entire archive, bounded member
  inventory (1–2,000 files, <=16 MiB each and <=256 MiB uncompressed),
  unsupported symlinks, traversal names, duplicates or ZIP CRC.
- Canonical JSON/JSONL, member byte counts and all nested SHA-256 inventories:
  original TT-002 dataset, all recorded source files, three complete native
  attempts and 15 checkpoint manifests; unexpected archived members fail.
- Exact frozen source and request consistency across seeds, full admitted
  TRAIN inventory, exact 72-row panel and 1,024 visits/3,590 target schedule;
  valid separate seed initialisation and original fixed limits.
- Execution receipts, single-use completion declarations, 256-update
  journal, cumulative native-resource budget and historical checkpoint
  history prefixes. These checks concern **retained records**, not real-time
  process attestation or an independently replayed optimizer.
- Original 259-score little-endian IEEE-754 FP32 vectors; correct index,
  gold/greedy prefixes, targets, lengths, raw greedy argmax, terminal EOS;
  all independently recomputed full-vocabulary probabilities, EOS-versus-rest
  binary log odds, target likelihood/rank, residue preference and numerical
  normalization. Probability-space checks are skipped at binary64
  underflow/subnormal saturation while stable log-space metrics are retained.
  The existing 1e-12 numerical tolerance is not relaxed.
- All full-answer classifications by task/circuit, zero and one control
  comparisons, true task-group counts, gold-prefix metric aggregates, vector
  position measures and reported synthesis checkpoint totals.

The resulting **local-only** report records 15 checkpoint summaries,
source/custody coverage, per-seed scalar/task breakdowns, greedy and gold
nonzero-residue correctness, EOS counts, correct-example transitions and
intersection of final correct vectors across seeds. It does not select a best
run or replace missing/failed evidence with a synthesized success.

## Interpreting the results

- **Complete exact answers** require the raw 259-way greedy sequence to
  match the original answer **plus EOS**. Correct teacher-forced token ranks,
  isolated residues, EOS probabilities and token NLL do not substitute.
- A correct vector on an exposed TRAIN example is evidence of limited fit;
  even repeated correct examples across seeds might be memorized or exploit
  identical formal task structure. No causal claim about the training
  mechanism or generalization follows.
- Include all seeds and all five checkpoints, **including regressions**.
  Premature EOS and positive/negative scalar-check preference swaps can
  explain large movements that an overall exact score conceals.
- No new protocol, optimizer, scaling, alternative decoder or TT-004 run is
  authorized. Further interventions require a new frozen experiment, owner
  grants and independent review on the exact proposed source.

Affected TRI-I01/I02/I03/I05/I06/I07/I08/I10/I11/I12/I13/I14.
The historical FIT source/requests, evidence archives and rights records
remain unchanged. This new review tool has its **own** source revision and
never claims to be part of the original frozen producer closure.
