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


## Hostile-review correctness fixes (P1/P2, seven findings)

Seven independently reported, executed adversarial reproductions exposed
failure modes in the original offline-only verifier. This additive correction
changes only the **review tool**, not historical FIT-001 requests, optimiser
updates, producer source, approved outputs or archive bytes:

1. Every repeated token prefix in each checkpoint (including gold/greedy and
   cross-row repeats, as well as later identical continuations) now requires
   exactly matching 259-element FP32 vector **bytes**. Internally consistent
   but mutually contradictory re-anchored observations are rejected.
2. The fixed historical source inventory is bound to the independently held
   source commit/closure digest and **1,279** required file entries, including
   required model/trainer/AGENTS files. TT-002 dataset admission requires all
   seven known member filenames, its precommitted manifest/rights digests,
   and the complete 19-file source closure with matching source identities.
   These are externally retained frozen data commitments; they do **not**
   establish authorship or authenticate a model.
3. Each recorded approval must **exactly equal** the canonical schema and
   all twelve fields from the frozen request's source, owner/scope, request,
   environment, dataset, schedule, output directory and limits. This is
   structural evidence checking, not owner signature verification.
4. Checkpoint train loss must be canonical hexadecimal, finite and
   nonnegative, with an exact match to the preserved step-zero initial loss
   or last scheduled native validation loss. The retained validation history
   must contain the expected steps in order.
5. The CLI resolves the destination's parent path before checking the
   checkout boundary, then creates the file with a trusted directory file
   descriptor and exclusive no-follow semantics. Symlinked parent aliases
   cannot divert private derived output into the checkout.
6. Machine-readable final fit conclusion and fitted-seed inventory are
   derived from verified 256-update exact counts, never hard-coded to the
   historical failure outcome.
7. Archive bytes are copied to an unlinked bounded temporary snapshot during
   SHA-256 verification, and the ZIP reader inspects **that same snapshot**.
   A replaced pathname or in-place change after capture cannot redirect
   the verified byte stream. The snapshot is private and removed on close.

CI exercises seven additional **synthetic-only** executed negative and
positive tests (16 total), in ordinary and optimized Python. An owner-local
run against the externally anchored original FIT-001 ZIP remains a separate
read-only engineering observation; the ZIP and derived output are not included
in this PR or CI.

## Further exact-head hostile review: three additional P2 corrections

Review of the seven-fix head (`5f6e7fd55c9212f664e4c691cee904746d1c66f2`)
found three additional, independently reproduced correctness defects. These
follow-up fixes are again **review-only** and leave every historical source,
frozen request, owner authorisation, checkpoint and private archive unchanged.

1. **Ancestor race on the output directory:** `O_NOFOLLOW` on the final
   directory does not protect against an earlier path component changing
   into a symlink after `.resolve()` completes. The CLI now opens every
   component of the checked canonical absolute parent relative to a retained
   directory FD starting from `/`, using `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`
   at each step. It checks device/inode and the Linux `/proc/self/fd` target
   before creating the output by exclusive no-follow `openat`. A deterministic
   test swaps `stage` for a symlink to the checkout before the vulnerable
   open and requires rejection without a checkout write. This rules out
   symlink-following substitutions; it is not a general-purpose defence
   against a privileged adversary relocating an open directory afterward.
2. **Validation-loss cross-checkpoint continuity:** Verifying each checkpoint
   only against its own final validation record permits a re-anchored packet
   to rewrite earlier losses in later checkpoint metadata. Every checkpoint
   now requires the same step-zero initial loss and an exact validation
   prefix built from the earlier independently checked checkpoint losses.
   Synthetic tests separately alter the initial value and an earlier
   validation entry while keeping each local checkpoint self-consistent;
   both are rejected by the cross-series checker.
3. **Frozen native treatment binding:** Request model/design are checked
   against the original archived and frozen-source-bound configuration
   files. The complete native training configuration is compared against an
   independent, full-field mirror of `spec.native_config`, and the complete
   resource limits against the frozen contract. Every checkpoint metadata
   configuration must match its exact frozen request. Synthetic tests mutate
   learning rate, optimiser betas, weight decay, loss, deadlines, model,
   design and checkpoint configuration and require rejection. These
   checks validate recorded configuration, not physical updater execution.

The complete review-only contract suite now contains **19** synthetic-only
tests and is exercised under normal and optimized Python. Owner-local audit
of the unchanged, externally anchored 1,512-member archive is performed
separately and excluded from the public repository and CI.

## Third exact-head hostile review: five additional P2 corrections

Review of the three-fix head (`71193986c1966c1c4f80d9762b434cc453e700ad`)
identified five further independently reproduced fail-open cases. The offline
reviewer now enforces the following **read-only recorded-evidence contracts**:

1. **Independently reconstructed selector and schedule.** The complete 972
   original TRAIN rows are structurally inventoried by family, four-trit word,
   task, check label and unique SHA-256 row identity. The exact six selected
   words are recomputed using the frozen word-domain SHA-256 rank, followed by
   the original row-domain SHA-256 rank and the full cyclic 1,024-visit ledger.
   The complete selected panel, panel metadata, selection record and SHA-256
   identities must equal this independent calculation. Re-anchoring a packet
   with two swapped equal-length vector examples cannot change the treatment.
   No archived producer Python is imported or executed by the reviewer.
2. **Checkpoint native-time trajectory.** Every retained time is canonical,
   finite, nonnegative binary64 hexadecimal. Step zero must report zero;
   later metadata times must agree exactly with their corresponding cumulative
   update-journal observations, remain ordered and within the frozen 600-second
   native limit. The completion native time must equal the final checkpoint
   and journal time; an over-budget or retrograde intermediate value fails.
3. **Retained numerical environment.** The complete archived
   `attempts/seed-N/environment.json` must match the exact CPU FP32 environment
   bound by that seed's frozen request/approval. This is a consistency check on
   captured records, not hardware attestation.
4. **Completion exposure declaration.** A completed attempt may not set
   `heldout_scored=true` (or omit the field) while claiming exposed TRAIN-only
   evidence. The flag must be explicitly `false` alongside exactly 256 updates,
   3,590 targets and `optimizer_replayed=false`.
5. **Observer/checkpoint model identity binding.** Both diagnostic observer
   identities must equal the checkpoint metadata's exact model/optimizer
   identity record, not just each other. The canonical diagnostic observation
   still requires zero updates, unchanged RNG and unchanged state.

Five additional synthetic negative/positive tests exercise these contracts,
for **24 tests in normal and optimized Python**. Review of the original
externally SHA-256-anchored 1,512-member archive is an additional local-only
read-only check. Source, approved requests, model weights, execution evidence,
prior findings and historical custody remain immutable.

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
