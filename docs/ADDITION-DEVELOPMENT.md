# Addition development v1: matched coverage experiment

Status: implemented separately versioned development protocol and bounded workflow.
The implementation has conformance tests, **not target experiment results**.
Run the target matrix only after review, green exact-commit CI and separate owner
local-run authorization. The [historical negative result](UNSEEN-NEGATIVE-RESULT.md)
and [preceding coverage diagnostic](COVERAGE-DIAGNOSTIC.md) remain unchanged.

## Question and intervention

Does expanding the variety of training operand combinations improve raw addition
answers on shared, excluded development families at matched scheduled exposure?
This is a narrow neural-learning experiment, not an exact arithmetic service:
an ordinary integer addition algorithm is the appropriate exact solution.

The control has five unordered nonnegative operand families:

`{1,4}, {2,5}, {0,9}, {3,8}, {6,7}`.

The candidate adds five families:

`{2,3}, {0,7}, {1,8}, {5,6}, {4,9}`.

Each list covers every digit 0 through 9 once and has the same ordered sums
5, 7, 9, 11, 13. Each family is rendered in both operand orders and both existing
minimal carriers, `A:a,b=` and `A(a,b):`. The control therefore has 20 unique
training rows and five families; the candidate has 40 rows and ten families.

A fixed 40-visit cycle presents the first list and then the second list. For the
control, the second list is another visit to its first list. For the candidate,
it is the added list. Corresponding slots have the same answer, carrier, prompt
length, model-input length and scored-target length. Thus corresponding batches
also match these quantities. Whole cycles additionally match left/right operand
marginals and the full model-input token histogram. The fixed slot order is
identical across initialization seeds; it is not a random reshuffle or a claim
that example identities are identical across arms.

The intervention deliberately changes unique family coverage, unique rows,
per-example repetition and input co-occurrence structure. Equal marginals do not
make all joint statistics equal or identify an internal learning mechanism.
Unique training diagnostics are reported separately from repeated scheduled
visits. Candidate training diagnostics contain twice as many rows; diagnostic
cost and whole-experiment wall time are not claimed to be matched compute.

## New exact admission and shared development strata

The new standard-library admission enumerates nonnegative integer operands
0 through 12 and two carriers, producing 338 unique rows. A unit-increment
label generator is independently checked by bounded direct integer addition;
a separate strict prompt parser checks rendered semantics. The unchanged byte
tokenizer checks actual prompt/answer masks and lengths. Rights admission covers
only these locally generated mathematical facts and minimal symbols, not prose,
teacher answers, an external dataset or blanket synthetic-data permission.

The family key is the unordered operand pair. Swapping operands and changing
carriers cannot move an item across family boundaries. Both training arms are
excluded from every development stratum, including the candidate-only families:

| Shared stratum | Definition | Families | Rows |
| --- | --- | ---: | ---: |
| Familiar answer, new combination | Remaining pairs in 0–9 with sums 5, 7, 9, 11, 13 | 9 | 36 |
| Novel answer, in range | Other pairs in 0–9, with sums absent from both training answer sets | 36 | 124 |
| Range and operand-length shift | Pairs in 0–12 with at least one operand 10–12 | 36 | 138 |

Diagonal pairs have only one distinct operand order, which explains why row
counts need not be four times family counts. All three strata are evaluated
separately on the preselected final checkpoint. The range/length stratum contains **20 familiar-answer rows in five families**
and **118 novel-answer rows in 31 families**. The admission summary and each row
record this distinction: answer novelty and operand-length shift are separate
flags, not interchangeable explanations.
The scorer's technical split label is `test`; scientifically these are public
**development** strata, not fresh confirmation tests. The 298 development rows
are correlated renderings and related arithmetic facts, not 298 independent
trials. There are no new confirmation families in this increment.

These facts can overlap earlier public arithmetic inputs. New identities,
changed ranges or family assignments do not erase prior exposure. This protocol
starts new native models from paired random initialization; it does not resume,
retrain, repair or relabel the earlier seed-0 checkpoint and its negative result.

Admission retains source, rights, dataset and family/count receipts and rejects
modified, reordered, duplicated or missing rows and altered admission records.
Training-state construction consumes only admitted training schedule rows;
training diagnostics use only the unique training view. Development rows never
enter updates, optimizer diagnostics, checkpoint choice or training loss.

## Frozen numerical and resource recipe

- Two arms × initialization seeds 0, 1 and 2: all six cells are expected.
- Dense CPU float32 anchor: context 64, two blocks, width 64, four heads,
  feed-forward width 128; 86,528 parameters. No pretrained weights.
- Existing native updater, optimizer, scorer and safe tensor machinery remain
  authoritative. The new orchestration does not implement another numerical loop.
- Batch eight; 4,000 updates; learning rate 0.001 with the anchor AdamW settings.
  Geometry loss is zero. Native reusable KV cache remains unsupported.
- 800 complete 40-visit cycles per cell: 32,000 visits, 76,800 scored answer/EOS
  targets and 284,800 unpadded model-input tokens. Actual masks and per-slot
  lengths establish these counts. Padding allocation is not counted as an
  unpadded token. The 4,000-update budget is explicitly new, not the historical
  4,096-update request with substituted metadata.
- Checkpoints at steps 0, 1,000, 2,000, 3,000 and 4,000. Training-only observations
  at these milestones; development scoring only at final step 4,000.
- Raw prompt-only greedy generation, cap five output tokens, no repair, solver
  fallback, retries, checkpoint election or early stopping. Exact formal controls
  are labelled non-neural and cannot rescue a model answer.
- Each fixed worker has a 900-second wall deadline. Cumulative update time has a
  480-second limit checked at milestone boundaries. The six-cell worker bound is
  at most 5,400 seconds, plus controller setup/audits; it is not a promise of
  measured runtime or a whole-process CPU-time quota.
- Per-artifact 16 MiB, retained payload writes 32 MiB per cell, worker log 1 MiB.
  Custody copies, matrix metadata and checkpoint imports are not a total-directory
  size or hard RSS/VRAM cap. No GPU execution or dependency download is implicit.

No threshold or score automatically sets `learning_adequate`, `phase5_ready`,
`fresh_confirmation` or `release_ready`. A positive development comparison could
motivate a later frozen recipe and genuinely new, multiple-family confirmation.
A null, failed or negative comparison is retained with all expected cells.

## Workflow and independent commitments

Use the exact reviewed checkout and pinned CPU environment. Keep the checkout
quiescent. Output parents must exist; use new paths outside the checkout and
outside the request location. Do not overwrite any earlier attempt or substitute
its external identities. Example commands below are instructions for a later
separately authorized owner run, not an instruction to run target training in CI:

```sh
PYTHONPATH=src:scripts python -B scripts/run_addition_development.py preflight
PYTHONPATH=src:scripts python -B scripts/run_addition_development.py freeze \
  --request /ABSOLUTE/NEW/REQUEST.json
# Save the printed request identity independently before any run.
PYTHONPATH=src:scripts python -B scripts/run_addition_development.py run \
  --request /ABSOLUTE/NEW/REQUEST.json --request-identity sha256:REQUEST_ID \
  --output /ABSOLUTE/NEW/ATTEMPT --authorize-run
# Save matrix_identity from the printed canonical attempt record independently.
PYTHONPATH=src:scripts python -B scripts/run_addition_development.py audit \
  --request /ABSOLUTE/NEW/REQUEST.json --request-identity sha256:REQUEST_ID \
  --output /ABSOLUTE/NEW/ATTEMPT --expected-identity sha256:MATRIX_ID
PYTHONPATH=src:scripts python -B scripts/run_addition_development.py summarize \
  --request /ABSOLUTE/NEW/REQUEST.json --request-identity sha256:REQUEST_ID \
  --output /ABSOLUTE/NEW/ATTEMPT --expected-identity sha256:MATRIX_ID
```

Freeze binds the actual source closure, protocol, data, schedule and supported
numerical environment. In its normal flow, the controller runs fixed workers sequentially and retains
completed/failed/timed-out/interrupted outcomes rather than silently dropping cells
or retrying. Outer controller, setup, source-mutation or filesystem failures can
leave partial files without a final matrix or printed identity; preserve those
files and error logs too, and do not claim a completed failure inventory. Save
stdout, stderr, exit statuses and original printed identities, including a failed
matrix's identity when one is produced. Never invoke the internal worker directly.

Audit/summarize check original externally bound evidence. Structural custody,
source/context checks, safe tensors and internally consistent retained measurements
are not independent numerical prediction replay, complete optimizer replay,
hardware attestation or independent-host replication. Run evidence may contain
local environment details and requires private review before any public upload.
Do not redact and reseal an original packet to conceal that distinction.

## Compatibility and conformance

Affected requirements: TRI-I01–I03, I05–I08 and I10–I14. This is a new checkout-only
workflow. Historical `src/`, protocols, bound contracts, checkpoints and fixtures
remain byte-identical. There is no change to old request identities or learning
gates. The new data package owns only its new finite admission; its numerical
consumer delegates the unchanged native operations.

Data/scheduling conformance runs without a numerical backend:

```sh
PYTHONPATH=src:scripts python -B -m unittest discover \
  -s tests/addition_development_v1 -p test_data.py -v
```

Engineering tests check exact labels, rights, identities, family isolation,
paired exposure and failure boundaries. Small numerical test controls, where
used, are not results from this target six-cell experiment. Preserve every target
outcome later; no target matrix is executed by this implementation increment.
