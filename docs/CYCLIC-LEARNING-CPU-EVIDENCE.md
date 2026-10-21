# Cyclic encoding pilot: cloud CPU negative result

## Scope and outcome

On 21 October 2026 (UTC), the unchanged, reviewed [PR #59](https://github.com/QSOLKCB/Trinite/pull/59)
producer was run once and fully replayed once in the assistant's isolated cloud
CPU workspace. This was **not Trent's Ryzen or RTX**, GPU execution, hardware
attestation, or independent-host replication. No outcome-driven tuning, reruns,
model selection, or changes to the frozen protocol were made.

All four final cells emitted **the same `222` followed by EOS on every one of
288 inputs**. All answers were syntactically canonical, but none was correct.
The experiment demonstrates format acquisition under this budget, not
input-sensitive arithmetic, geometric reasoning, or a benefit of either encoding.

| Seed | Encoding | Initial exact train / validation / test | Final exact train / validation / test | Exact-count deltas |
| --- | --- | --- | --- | --- |
| 0 | Decimal | 0/192 · 0/48 · 0/48 | 0/192 · 0/48 · 0/48 | 0 · 0 · 0 |
| 0 | Sexagesimal | 0/192 · 0/48 · 0/48 | 0/192 · 0/48 · 0/48 | 0 · 0 · 0 |
| 1 | Decimal | 0/192 · 0/48 · 0/48 | 0/192 · 0/48 · 0/48 | 0 · 0 · 0 |
| 1 | Sexagesimal | 0/192 · 0/48 · 0/48 | 0/192 · 0/48 · 0/48 | 0 · 0 · 0 |

For each seed and encoding, initial canonical counts were 0/192, 0/48 and 0/48;
final canonical counts were 192/192, 48/48 and 48/48. Thus canonical-count deltas
were +192, +48 and +48, distinct from the zero exact-answer deltas.
Each cell used the same 7,264-parameter native dense model configuration, 32
AdamW steps, 128 scheduled visits, and 512 answer/EOS targets. There were 192
unique training rows, so scheduled visits did not cover every training row.

## Immutable bindings and replay

- Producer merge: `b4341c5a150847eaf4158574423ea5fa153c9486`.
- Producer tree: `e5650b2869673801d45ee3c4a31349958730944a`.
- Request: `sha256:28fcd946de090fc8c0fc418b17cdcca7da29b54faa32d3f1e670da92daa1af0e`.
- Run **and** replay report: `sha256:820f200ff715fcf53f8fb362c49e7f9628ac5fb0bad87d315d597d2fc4c0028b`.
- Archive commitment: `sha256:6140ab4427dede60b24f29da6d160d8d95722692faa2178f6bd4ad6b14b00613`.

The local exported tree was checked against all 1,075 Git blob identities of the
merged tree before execution. The frozen request binds 124 producer files and
all admitted inputs. Full replay retrained all four cells from their native seeds
and matched complete deterministic report bytes, including final model and
AdamW tensor identities, all update histories, initial scores and final scores.
This is stronger than inference-only replay, but it is not independent-host
replication or a guarantee across hardware/software environments.

The [compact retained packet](../fixtures/cyclic-learning-cpu-20261021/commitment.json)
contains 150 logical members in 135 deduplicated XZ objects: all 25 run-directory
files, all 124 request-bound producer source files, and the separately pinned
verified-source loader. Request, run/replay reports, partial progress, logs,
controller journals, statuses, and the replay's retained expected report remain
available. Source records carry Git blob identities as well as SHA-256 identities.
No checkpoints were serialized by this producer; the reports retain their tensor
identities, not reconstructible trained weights.

Only the incidental pip cache warning is privacy-derived: the cloud home directory
is replaced by `<cloud-user-home>`. Its original byte count and digest, public
byte count and digest, and replacement count are retained. No numerical request,
report, attempt receipt or source file was redacted or resealed. Hashes identify
bytes; they do not authenticate a physical host or establish scientific truth.

## Environment and resources

The request records Linux x86_64, CPython 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5,
all pinned CPU package versions, float32, deterministic algorithms, one numerical
thread and nine interop threads. The cloud environment reports an Intel Xeon
Platinum 8573C and AVX512. `pip check` found no broken requirements.

| Attempt | Controller elapsed time | Reaped-child RSS high-water after attempt |
| --- | --- | --- |
| Run | 24.694239621 seconds | 316,988 KiB |
| Replay | 20.014857841 seconds | 324,276 KiB |

The resource field is Linux `RUSAGE_CHILDREN` cumulative reaped-child high-water,
not a per-child delta, whole-host maximum, enforced memory cap, or fair performance
comparison. Before-attempt values are retained in each status. Timing differences
are descriptive observations, **not a speedup claim**. The unchanged worker and
per-cell limits were 900 and 180 seconds respectively; both attempts completed.

## Offline audit

From a trusted checkout, without installing Torch or running training:

```sh
python -I -S -B scripts/check_cyclic_learning_evidence.py
python -I -S -B -O scripts/check_cyclic_learning_evidence.py
python -m unittest discover -s tests -p test_cyclic_learning_evidence.py -v
```

The isolated auditor verifies the pinned commitment and object bytes, loads only
captured verified producer source, regenerates admission/items/schedules with the
recorded environment as data, recomputes score counts from recorded tokens, and
checks request/report/partial records, successful publication receipts and exact
run/replay byte equality. This is offline structural and custody verification of
the retained execution, **not a new training replay**. A fresh full numerical
replay requires the captured producer and its exact admitted environment under
the [original reviewed instructions](CYCLIC-LEARNING.md).

## Limits and compatibility

This is an exploratory same-task family holdout: 24 correlated relation families,
16/4/4 train/validation/test, 288 rows, two seeds, one modulus, one template and
one fixed short training budget. Explicit permutation/negation relatives remain
in one split; that does not eliminate all algebraic dependence. Outputs, splits
and inputs are public. Decimal and sexagesimal prompts have equal lengths but
different digit distributions, radix labels and punctuation, so this is not an
isolated causal test of base choice. Zero exact successes cannot distinguish the
encodings' attainable learning quality. No curriculum expansion or increased
budget was selected after seeing this outcome.

This additive evidence PR preserves every historical fixture and all producer
source-closure bytes. It documents TRI-I02/03/05/07/08/11/13 evidence and respects
TRI-I09/10 claim limits and TRI-I14 compatibility. It changes no native API,
training protocol, data admission, model, quantizer, integral-geometry experiment,
NEXUS lane, roadmap gate, or release status.
