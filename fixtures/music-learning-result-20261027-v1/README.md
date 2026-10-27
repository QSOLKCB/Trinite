# Partial public music-learning v1 negative-result evidence

See [result and interpretation](../../docs/MUSIC-LEARNING-RESULT.md).

This directory preserves 25 **byte-exact original** public-data files from the
owner's two-seed PR #82 pilot: `matrix.json`, plus `attempt.json`,
`completion.json`, `report.json`, `predictions.jsonl`, and both
`{trained,untrained}-{description,train,validation,test}.json` for each seed.
All 3,840 raw predictions and all failed criteria remain retained.

`source-bindings.json` is a newly authored **derivative public-source receipt**,
not an original request or closed-custody certificate. This README is also new.
No private archive commitment or checkpoint metadata is published.

Requests/environment, checkpoints, tensors, worker logs, private host receipts,
command outputs and provenance directories are intentionally omitted. Original
matrix/completion hashes referencing absent files are not rewritten. The public
subset cannot reconstruct the private request and is **not closed custody**,
checkpoint verification, model execution, inference replay, optimizer replay or
independent-host replication. The public checker makes no claim about omitted private artifacts.

Run `PYTHONPATH=src:scripts python -B scripts/check_music_learning_result.py`
with the repository's pinned CPU dependencies. It recomputes retained-score
consistency without constructing a decoder or loading tensors. Existing fixtures,
producer code, protocol criteria and historical source commitments are unchanged.
