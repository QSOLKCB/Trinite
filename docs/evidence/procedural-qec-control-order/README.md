# Historical procedural-QEC control-order audit

This is executable evidence for the historical-order limitation reported in
[the v3 contract](../../PROCEDURAL-QEC-V3.md#historical-control-ordering-audit-13-october-2026).
A successful audit means that it **reproduced the ordering failure**. It does
not mean that the historical control was preserved or that any learning gate
passed. Affected requirements: TRI-I07, TRI-I08, TRI-I11, TRI-I13 and TRI-I14.
Numerical, dataset, protocol, request and checkpoint semantics are unchanged.

## Retained evidence

- [comparison.json](comparison.json) is the original independent comparison
  receipt, retained without reserialization or any other byte changes.
- [comparison.json.sha256](comparison.json.sha256) independently names its
  expected SHA-256:
  `7e0160ac9ca5c6241dd8001bbc8775c6dd1c51266755533705debd819d9a087b`.
  The executable also pins this value; changing the receipt and its sidecar
  together does not authorize a replacement.
- [audit.py](audit.py) rebuilds both datasets and schedules from two fixed Git
  sources. It accepts no supplied dataset, cached schedule, expected-hash
  override or regeneration mode for the retained receipt.
- [test_audit.py](test_audit.py) exercises the real two-source reconstruction
  and rejection boundaries, using only the Python standard library and Git.

Historical producer:
`93522bf8f18fe38970ceafdd2ace792f8922c382`.
Dataset SHA-256:
`120b95ac68825b50b92dc0210d1f142a30980bfe5acedcfafe6d8f3cdadf83c4`.
Admission manifest SHA-256:
`315168dd6be4909cead81a45a4c3b5b1abb125cc69f3ca773b5145f18b0ac412`.

Current comparison producer, fixed despite later documentation commits:
`cc996e272800a5c02667001f7c96c8d11cb513cc`.
Dataset SHA-256:
`9ea3223ebd099933f9c05a3ccb9db43e4a7d19e3c1edf5922c326c2eec16d944`.
Admission manifest SHA-256:
`528cb27db98781db8148dd4f029e9a392d0d2a786d393f2ef399dcbdad95ff49`.

All 3,876 rows match after removing **only the top-level**
`generator_revision` and `example_identity` fields. The comparison preserves
the complete dataset and row schemas and every other value, including split,
family, parent, source IDs, carrier, values and nested transformation lineage.
The prompt/answer mapping must be bijective; duplicate keys or row identities,
missing/extra rows and malformed schedules are errors. All 3,876 example
identities nevertheless change, altering their seed-based rankings.

At 4,096 updates and eight rows per update, the final-only mathematical sequence
differs in 32,576/32,768 visits for seed 0 and 32,521/32,768 for seed 1.
The receipt also retains both procedural comparisons, frequency changes,
per-slot mismatch counts, first mismatches and full sequence identities.
The current v2 scheduling function and current v3 final-only function agree
when given the **same current rows**. That regression misses the historical
dataset-identity/order change; this audit reconstructs both actual datasets.
For the procedural arms, the current-row v2/v3 equality is correctly false.

## Offline reproduction

Requirements: Python 3.11 or newer and Git, with both commits already present
in the local repository. No Torch, NumPy, dependency installation or network
access is needed. Run from the repository root that contains this evidence
directory. These two commands prepare separate clean detached checkouts;
choose unused local paths and do not repurpose an existing research directory:

```sh
git worktree add --detach /tmp/trinite-control-old-93522bf 93522bf8f18fe38970ceafdd2ace792f8922c382
git worktree add --detach /tmp/trinite-control-current-cc996e cc996e272800a5c02667001f7c96c8d11cb513cc
```

Then run the read-only audit:

```sh
python -I -S -B docs/evidence/procedural-qec-control-order/audit.py \
  --historical-source /tmp/trinite-control-old-93522bf \
  --current-source /tmp/trinite-control-current-cc996e
```

Exit zero prints the exact retained canonical comparison JSON to stdout and
its verified identity to stderr. Any mismatch exits nonzero; it never rewrites
the expected evidence to make a comparison pass. `--receipt PATH` can select
another copy of the **same pinned original receipt**, with a neighboring
`PATH.sha256` sidecar. It cannot choose a different expected result.

Run the positive and negative self-tests explicitly (they are intentionally
outside the numerical `tests/` and source-bound producer inventories):

```sh
TRINITE_HISTORICAL_SOURCE=/tmp/trinite-control-old-93522bf \
TRINITE_CURRENT_SOURCE=/tmp/trinite-control-current-cc996e \
  python -I -S -B -m unittest discover \
    -s docs/evidence/procedural-qec-control-order -p test_audit.py -v
```

Missing test checkout variables fail the suite rather than skipping coverage.
Mutation tests use disposable local sparse clones, never either supplied
producer. They cover wrong commits, dirty source, `assume-unchanged` hidden
edits, ignored bytecode/import shadows, symlinks, dataset and manifest changes,
tampered receipt/sidecar, duplicate/malformed row mappings, schema preservation,
short/extra schedules, invalid row indices, changed order and duplicate JSON
keys. Optimized Python (`-O`) must still reject an invalid source without any
success output. The real audit is rerun, not replaced by a fake-process pass.

## Source and execution boundary

Before executing either producer, the audit checks both exact Git commits,
clean worktrees/indexes, and every tracked file under `src/` against the pinned
Git blob bytes. This includes the complete transitive generator/admission
closure, QEC source and license, all package initializers, protocols and
resources. These generators enumerate their inputs locally; their file reads
are source receipts and resources under `src/`, not outside fixture inputs.
Ignored/untracked files under `src/`, including existing bytecode caches, and
symlinks are rejected. Dirty source cannot hide behind Git index flags.
Use a fresh checkout if an existing numerical environment has such files;
the audit does not clean or alter it.

Each producer runs in its own `python -I -S -B` process. It regenerates the
dataset and admission manifest, checks the independently pinned identities,
and calls the producer's original `data.audit_bytes`. The v2 scheduler is
AST-extracted from its verified `experiment.py`: only `cells`, `cell_name`,
`ranked`, `schedule` and the literal `ARMS`/`SEEDS` bindings are evaluated with
that checkout's contracts. This avoids importing its model/training code.
The current v3 pure scheduler is imported from its verified source. Sources
are checked again after execution and before reporting success. The original
receipt is also rechecked. No success check is cached in the executable.

The audit performs no training, downloads, network writes, checkpoint reads or
changes, source edits, request freezes, protocol changes or research-run writes.
It only reports source/dataset/schedule reconstruction. It is not native
optimizer replay, numerical replication, closed run custody, held-out evidence,
learning adequacy, a geometry intervention or permission for another experiment.
The original producer, original receipt and all negative results remain intact.
