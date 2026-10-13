# Non-displacing proposal: data-only planning record

This folder supports the [review-only proposal](../../PROCEDURAL-QEC-NONDISPLACING-PROPOSAL.md).
It is not an experiment, frozen protocol, request or authorization. No model is
imported or called. The current-source/historical-row mapping and native research
producer remain unimplemented.

`planning_counts.py` uses the unchanged historical audit pinned at commit
`d01ed20a11213761836ae9f022c9f1ca38c3780b`, independently checks its SHA-256
`00894610c3ebb2d61590a6320aeedd029fdfaae7b66344c137632d097a1dfa53`, and executes
those verified source bytes directly rather than an import cache. The existing
isolated audit worker verifies the complete historical producer/source and
admits the original dataset before enumerating rows. A final source check follows
the calculation. This does not rerun the complete old/current ordering audit or
any native optimizer replay.

`planning-counts.json` records four proposed schedules, ordered-subsequence
identities, per-kind/mode visits, targets, unpadded input counts and fixed padded
positions. It binds the actual proposal, planning script, original audit and
comparison receipt, and historical source/data/manifest identities. Its separate
`planning-counts.sha256` preserves the final planning receipt bytes. The root
standard-library tests pin that independently captured digest; coordinated
replacement of the receipt and sidecar is not accepted.

## Read-only reproduction

From this repository root, with Git and standard-library Python available, use a
clean checkout of exact historical producer
`93522bf8f18fe38970ceafdd2ace792f8922c382`. If that object is not already local,
fetch this public repository explicitly before creating the worktree:

```sh
git fetch https://github.com/QSOLKCB/Trinite.git 93522bf8f18fe38970ceafdd2ace792f8922c382
git worktree add --detach /tmp/trinite-historical-planning 93522bf8f18fe38970ceafdd2ace792f8922c382
python -I -S docs/evidence/procedural-qec-nondisplacing-proposal/planning_counts.py \
  --repository . \
  --historical-source /tmp/trinite-historical-planning \
  --proposal docs/PROCEDURAL-QEC-NONDISPLACING-PROPOSAL.md \
  > /tmp/trinite-planning-counts.json
cmp /tmp/trinite-planning-counts.json docs/evidence/procedural-qec-nondisplacing-proposal/planning-counts.json
(cd docs/evidence/procedural-qec-nondisplacing-proposal && sha256sum --check planning-counts.sha256)
```

Use an unused temporary worktree/output path; do not overwrite an existing
checkout or retained receipt. A failed admission, identity or count comparison
is a blocker, never permission to regenerate historical evidence. The output
redirect is the caller's explicit local write; the planning tool prints only to
standard output and leaves the source checkouts unchanged.

The counts describe the exact proposed recipe over historical admitted rows.
They are not wall-time, FLOP, peak-memory, learning or hardware-performance
measurements. The proposal remains at scientific, implementation, resource and
review gates even when these mathematical checks pass.
