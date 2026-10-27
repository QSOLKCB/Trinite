# Music learning v1: negative training-fit pilot

The owner-authorized two-seed pilot at [PR #82 merge
73fd0fc](https://github.com/QSOLKCB/Trinite/commit/73fd0fc4c682f75496fd7e7f8d7ef11c61386061)
completed its frozen budget. **Neither model fit the training set.** This
preservation adds no training, inference, optimizer replay, rerun, checkpoint
selection, answer repair, arithmetic bridge or geometry intervention.
The [original prospective contract](MUSIC-LEARNING.md) and producer bytes remain
unchanged; this later result supersedes only their implementation-time “unrun” status.

## Observed outcome

| Seed | Updates | Scheduled visits | Unique train | Train answer+EOS | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 64 | 512 | 480 | 0/480 | 0/240 | 0/240 |
| 29 | 64 | 512 | 480 | 0/480 | 0/240 | 0/240 |

Each cell used 3,955 scheduled answer-plus-EOS targets. Every task's frozen
training and held-out accuracy criterion failed, as did every held-out baseline
margin. Component correctness was also zero. Task-only train-mode baselines
score 156/480, 88/240 and 98/240; task/meter modes score 300/480, 144/240 and
162/240 for train, validation and test respectively. Modes are learned from
unique training rows only, not held-out labels.

All **3,840 retained predictions**, including untrained and final observations,
reach the 64-token cap without EOS. Final seed 17 emits 64 slashes for 945 rows;
15 training rows emit `0` followed by 63 slashes. Seed 29 emits 64 slashes on
all 960 final rows. No malformed or failed output has been removed.

## Interpretation and next boundary

This is a **negative training-fit smoke pilot**, not an interpretable transfer
study or a refutation of music learning. The eight correlated parent families
are divided 4/2/2; their 480/240/240 projections are not independent problems.
The tiny fixed budget and public development exposure limit interpretation.
Training fit must be established before interpreting held-out transfer.

Static mask-boundary, next-token shift and no-truncation inspection found no
obvious discrepancy; that is not proof the numerical path is bug-free.
Slashes occupy about 21.315% of training answer-plus-EOS target positions and
never begin a gold answer; onsets/boundaries contribute about 77.55% of targets.
Those imbalances are diagnostic clues, **not established causes** of collapse.

The owner requested a separate train-only one/few-example overfit diagnostic,
inspecting first-answer and EOS predictions before increasing curriculum or
budget. That implementation must be a separate versioned PR and source-bound
request, not a modification of this result. No new diagnostic is executed here.
Historical negative results and all learning, confirmation, independent
replication, format-selection, geometry-on Phase 5, scaling and release gates
remain unchanged. `learning_adequate`, `phase5_ready` and `release_ready` are false.

## Evidence and verification limits

The [public subset](../fixtures/music-learning-result-20261027-v1/README.md)
contains 25 original files byte-for-byte: original matrix plus 12 records per
seed, including complete journals, all six scores and both descriptive maps.
A separately authored `source-bindings.json` is explicitly a derivative
public-source receipt, not a replacement original request.

- Request: `sha256:e534fa4beae47aade39acd1b06a71919e02120b08cbe22e36037b8dcb09b1d65`.
- Matrix: `sha256:a36434c73ee4643caed241a202d5e5b4cf6c986c453588ea73009d1467b773ad`.

The source-only derivative binds the exact 176 public producer files to the
published Git tree and regenerable public admission. It contains no private
archive commitment, checkpoint metadata, environment or host observations.

The public subset excludes requests/environment, checkpoints/tensors, host
receipts, logs, command outputs and private provenance objects. Original matrix
and completion references to absent objects remain unchanged. This is **partial
public custody**, not the complete original bundle; the omitted original request
cannot be reconstructed from its source-only derivative. Independent privacy
review covers the publication allowlist, not a promise about omitted private files.

The read-only checker binds the public originals and producer source identities,
regenerates exact admission, uses authoritative score/description validation,
checks journal coverage and all cell links, and rejects changed or missing/extra
files, symlinks and foreign imports. Use pinned CPU dependencies because the
native score checker imports Torch; no decoder is constructed or checkpoint loaded.

```sh
PYTHONPATH=src:scripts python -B scripts/check_music_learning_result.py
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_learning_result_v1 -v
```

Affected TRI-I03, I07, I08, I11, I13 and I14: additive evidence preservation and
bounded verification only. No historical source closure or numerical semantics change.
