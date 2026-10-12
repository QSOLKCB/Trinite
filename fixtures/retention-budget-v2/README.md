# Balanced rehearsal budget evidence packet

This packet retains the completed training-only control/candidate matrix from
published producer `816cdc1cbaac24c86048d063711891713277d3d5`. The four cells passed
all six checks, including complete native model/AdamW/history replay, with no
worker or pairing errors. Joint training acceptance is false: all twelve maths
tasks fail in both candidate seeds. Original v1/v2 fixtures and producers remain
unchanged. Full scope, exact counts and commands are in
[the experiment contract](../../docs/MATHS-RETENTION-BUDGET.md).

| File | Content |
| --- | --- |
| `request.json` | Original source/data/protocol/environment-bound experiment request |
| `verification-request.json` | Independently retained external commitments for all expected artifacts |
| `summary-replay.json` | Complete four-cell full-replay summary, including negative criteria and contrasts |
| `producer-source.zip` | 156 published source/script/runtime metadata members, including pinned donor code/licences; inventory records SHA-256, byte count and Git blob |
| `evidence.zip.partNNN` | Eleven ordered lossless ZIP parts containing every run byte, logs, model/AdamW tensors, histories, predictions, custody, requests, direct producer closure and fresh replay audit; immutable member inventory |
| `archive-receipt.json` | Independently retained evidence/inventory identities |
| `readonly-audit.json` | Whole-run before/after hash comparison for archival replay |
| `software-checks.json` | Completed CI job links and scoped local checks |
| `CHECKSUMS.txt` | SHA-256 identities for every other packet member |
| `prior-assurance-source.json` | Original v1/v2 source names/bytes used by preservation regression checks |

Verify external identities before opening the archives:

```sh
cd fixtures/retention-budget-v2
sha256sum -c CHECKSUMS.txt
cat evidence.zip.part* > /tmp/trinite-budget-evidence.zip
printf '%s  %s\n' 'eff2daf7bfcc3451253974cd2ec004c9ac0f3472c7fd7dd57d15a42eec5d8809' /tmp/trinite-budget-evidence.zip | sha256sum -c -
```

`inventory.json` inside each ZIP enumerates every other member, its identity and
byte count; it deliberately excludes itself. The evidence archiver checks every
member's lossless round trip. The complete producer snapshot additionally matches
every member to its published Git blob. `producer-source.zip` preserves the full
runtime package, including the donor code bound transitively by frozen source
pins beyond the request's direct source names. It is a source snapshot, not a
wheel cache; install the unchanged hash-locked CPU dependencies separately.

For matched-host replay, extract the producer snapshot into a disposable source
checkout and the evidence archive into a separate directory. The run is under
`run/`; externally held request copies are under `external/`. Use the fixed CLI's
read-only replay command with the separately recorded request identities. An
unmatched host/software environment is intentionally rejected. Do not rewrite
an old environment fingerprint; a different host needs a new declared run.

Experiment request:
`sha256:11a740c30a67918e9fa024ec21bee08d1c72dca069911053830e6979a1e3bcbb`.
External verification request:
`sha256:9180f158f8d111e7e1962b290ed673c925a200d60fc6f6bfea3ec3355daefa3c`.

The packet is a deliberate lossless-evidence exception to the small-fixture
convention. It does not reconstruct the earlier report-only Ryzen audit or grant
fresh capability, original fold/composed adequacy, geometry, format or release
readiness. Repeated carriers and scheduled visits are not independent samples.
