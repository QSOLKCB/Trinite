# PR98 placement and matched-device result — reports only

This note is **DERIVED**, not an original execution receipt. The 42 JSON/JSONL/
audit originals under `cpu/` and `device/` are preserved byte-for-byte from the
owner's completed handoff. `source-SHA256SUMS` is the unchanged handoff manifest;
its three owner-derived prose/inventory files are deliberately not included.
`provenance-DERIVED.json` records the exact inclusion and omission boundary.

Producer: merge `798cf05a1ade30b4877177f251f71435913293a8`, tree
`fa8cce93b08f158f37e0f9b5e52a5113208ef19b`. Source archive SHA-256:
`23a20e25510f7a2c7c06aa289103b52c315118d5d53cf3b2da3b9428cd80931d`.
The archive itself is not retained here; its identity is a handoff receipt.

## Mixed/negative placement development result

| Seed | Control canonical /40 | Swap canonical /40 | Control pitch /40 | Swap pitch /40 |
| --- | ---: | ---: | ---: | ---: |
| 17 | 16 | 16 | 10 | 10 |
| 29 | 15 | 16 | 14 | 14 |

Tempo scores equal canonical scores. Every development view in all four cells
has **0/10 complete quartets**; canonical rest-only remains **0/10**. All four
cells fit both a/b diagnostic layouts 40/40 each. Even the control already
answers all 30 informative changed-layout pairs correctly, separately from the
10 identical-input pairs. Thus this template swap does not demonstrate repair
of the remaining rest/tie development failure. The one-row seed-29 gain is not
a reliable transfer or mechanism claim.

The treatment has 80 scheduled slots but only 70 unique prompt surfaces.
Canonical exposure halves only for 30 feature-present rows; fixed block recency
is part of the intervention. Pitch remains 60 in both arms. Only two previously
exposed development parents, c and d, exist; siblings/meters/seeds are not new
independent subjects or fresh confirmation.

## Descriptive matched CPU/CUDA trial

Both target cells used the same CUDA-wheel environment, adapter, seed-17
initialization and fixed workload according to the retained reports. Their
initial identities match; final identities differ. Both fit all 80 a/b
layout diagnostics. Development canonical is **CPU 16/40, CUDA 15/40**; pitch
is **10/40 both**, tempo mirrors canonical; quartets and rest-only remain zero.
Raw outputs agree on **116/120** development rows: 38/40 canonical, 40/40 pitch,
38/40 tempo. Equality includes wrong answers and is not equivalent to correctness.
The retained probe declares **47/47 checks passed** and equal initial bytes;
its tensors are absent, so this verifier cannot independently repeat the probe.

| Retained phase, seconds | Same-wheel CPU | CUDA | CUDA/CPU |
| --- | ---: | ---: | ---: |
| Updates/validation | 3.123334 | 4.151100 | 1.329061 |
| Cold worker wall | 15.347003 | 17.149789 | 1.117468 |

For this one tiny, fixed-order pair, CUDA took longer, including cold-worker
cost. This is not a general GPU benchmark or GPU learning benefit. The CPU is a
one-thread reference, not full host throughput. Cold wall includes startup,
runtime/context admission, transfers, diagnostics and serialization; residual
is not pure startup. Synchronized phase clocks are retained measurements, not
freshly executed here. Greedy timings are nested, not added again. Guarded
development can inspect/stage both models and is not isolated decoder throughput.
Provisioning and standalone audit are separate from target worker comparisons.
No backend-by-curriculum interaction was tested. Floating-point divergence is
plausible, but the cause of the final output differences is not established.

## Verification and limits

The read-only checker authenticates pinned originals and scoring sources,
regenerates the fixed corpus, and checks **1,200 retained records / 2,400 generated
tokens**, exact summaries, available artifact links, declared probe consistency
and retained timing arithmetic. It does not run models or import a numerical
backend. All six diagnostic sets are 80/80; CPU science contributes 800 decodes
and the device pair 400, with every output ending at EOS.

The subset checks **21 of 43 CPU matrix bindings** and **17 of 31 device bindings**.
The other **22 CPU / 14 device files** remain absent: requests, checkpoint tensors
and metadata, duplicate diagnostic arrays, worker logs, and device probe tensors.
The matrix and audit receipts bring the retained originals to 23 CPU and 19 device
files. This is not full custody, a re-executed source-bound audit, optimizer replay,
inference replay, or independent attestation of actual execution, hardware,
timings, runtime admission, ordering, or absence of other attempts. Owner-derived
hardware/provisioning assertions are not independently reconstructible here.

Both matrices record completed cells/development, no controller failure, and
completed audit receipts. The later missing-cell-seal bug is not present in this
packet. The owner's separately reported memory total round-trips the allocator
fraction, but the hardware receipt is omitted: do not infer general runtime
correctness from this successful packet. The confirmed late robustness issues
are separate prospective fixes, not a reason to edit/reseal/rerun this evidence.
Historical training-fit, pitch and prior negative results remain unchanged.
No accuracy threshold, release readiness, general music, arithmetic transfer,
statistical significance or self-correction claim is established.
