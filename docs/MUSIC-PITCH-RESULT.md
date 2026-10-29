# Controlled pitch replacement: mixed/negative development result

The owner reports completing the [PR96 protocol](MUSIC-PITCH-INTERVENTION.md)
at commit [`6258ecbe`](https://github.com/QSOLKCB/Trinite/commit/6258ecbeea2dd5553ebb73619f17432dd83c5ad6),
tree `6ade3340409344cccc2712962f4651580fb8f255`. This additive preservation
keeps the prospective protocol, earlier [positive training fit](MUSIC-EXPOSURE-RESULT.md)
and [negative frozen development result](FROZEN-MUSIC-EVAL-RESULT.md) unchanged.
No target training or inference was performed to prepare this increment.

**Pitch replacement improved consistency between the two pitch renderings, but
did not improve canonical structural transfer.** Most newly invariant pairs
remain wrong. The result is mixed/negative development evidence, not successful
transfer or a failed execution. The frozen protocol had no accuracy pass gate;
`completed` denotes execution/artifact completion, not that the intervention worked.

## Paired outcomes

Each view has 40 rows, 10 complete four-variant quartets and 40 contrast edges.
Exact counts mean complete answer bytes followed by EOS. Both fixed seeds and
all four final endpoints are retained, including lost training fit.

| Seed | Arm | Canonical exact | Pitch exact | Tempo exact | Canonical complete quartets | Canonical both-correct edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | Control | 16/40 | 10/40 | 16/40 | 0/10 | 6/40 |
| 17 | Pitch replacement | 14/40 | 14/40 | 14/40 | 0/10 | 4/40 |
| 29 | Control | 15/40 | 14/40 | 15/40 | 0/10 | 5/40 |
| 29 | Pitch replacement | 15/40 | 15/40 | 15/40 | 0/10 | 5/40 |

Treatment-minus-control canonical differences are **−2 and 0 exact rows**,
**−2 and 0 both-correct edges**, and zero complete-quartet gains. Pitch-view
exact differences are +4 and +1, at a pitch explicitly trained in treatment.
Every cell/view has zero complete development quartets. All five frozen
training-only majority controls score 20/40 exact per view; the models remain
below those exact-answer counts. Controls have zero complete quartets and zero
both-correct edges; they are not contrast reasoning solutions either.

## Consistency is not correctness

- Canonical-to-pitch raw equality rises from **4/40 to 40/40** in both seeds.
  Both-correct pairs rise from 1 to 14 and 15. Thus **26/40 and 25/40 treatment
  pairs are consistently wrong**. This is observed invariance over these two
  renderings, not arbitrary-pitch robustness or a proven learned mechanism.
- Canonical rest-only variant `10` remains **0/10 in every cell**. The diagnosed
  structural rest failure is not repaired by this replacement allocation.
- All canonical-to-tempo pairs remain identical in every cell, including errors.
- The two control development journals match PR95's raw outputs **120/120 per
  seed**, across all three views. This ancillary historical consistency is not
  an admission gate, fresh replication, or evidence of a causal mechanism.

## Training-fit tradeoff and fixed exposure

| Seed | Arm | Canonical a/b diagnostic | Pitch-72 a/b diagnostic |
| --- | --- | ---: | ---: |
| 17 | Control | 40/40 | 10/40 |
| 17 | Pitch replacement | 40/40 | 40/40 |
| 29 | Control | 40/40 | 10/40 |
| 29 | Pitch replacement | 38/40 | 40/40 |

Seed 29 loses perfect canonical training fit: two rest-only rows on parent `a`,
in meters 4/4 and 5/8, become wrong. Canonical complete training quartets fall
from 10/10 to 8/10. This decline was retained, not excluded by a fit gate.
Pitch-72 diagnostic rows are unexposed in control but trained in treatment;
40/40 treatment pitch fit is not unseen-pitch generalization.

The declared study uses four fresh cells, each with 640 updates, batch eight,
5,120 visits and 10,240 answer/EOS targets: 2,560 updates, 20,480 visits and
40,960 targets overall. Each structural training row receives 128 visits;
control uses pitch 60 throughout, treatment uses 64 visits each at 60 and 72.
The treatment halves canonical exposure and changes unique surfaces from 40 to
80. Fixed 60-then-72 block recency is part of the intervention. The evidence
cannot separate diversity, exposure allocation and recency as mechanisms.

All 800 retained records contain one answer digit and EOS: 320 a/b diagnostics
and 480 c/d development outputs, totaling **1,600 recorded generated tokens**
(640 diagnostic, 960 development), with no cap terminations. This verifies
journal accounting, not actual measured forwards or timing. Training and loss
forwards are additional. The four-cell completion gate, paired initialization
identity declarations and retained links are consistent with the protocol; the
public subset cannot independently attest their temporal execution.

## Scope and limits

There are only **two previously inspected structural development parents**,
`c` and `d`, with correlated meter renderings and pitch/tempo siblings. These
are exposed development probes, not fresh confirmation. Reused seeds are not
independent subjects or replications. Canonical outcomes and training-fit
tradeoffs remain primary; aggregating siblings must not disguise their failure.

These observations do not establish general music understanding, arbitrary
pitch or length, arithmetic transfer, self-correction, QEC benefit, statistical
significance or a causal neural mechanism. The finite shortcut
`8 - rest_presence - tie_presence` still suffices for the task; the outputs do
not show reliable structural use of it. No learning-adequacy, Phase 5, release
or promotion gate is cleared. No curriculum change or further run follows.

## Public evidence and read-only verification

The [fixture](../fixtures/music-pitch-result-v1/README.md) preserves **23 selected
original files byte-for-byte**, plus the original handoff checksum manifest and
clearly derived provenance. Of the matrix's **43 bound files, 21 are present and
22 absent**. The matrix and audit receipt are the other two originals. Therefore
“23 originals” is not complete run custody. The full request, checkpoint tensors
and metadata, environment commitments and raw logs remain owner-local.

The standard-library checker authenticates fixed original identities and the
unchanged scoring-source closure before project imports, regenerates exact
inventories, rescores all 800 raw records, compares all four cell summaries and
the paired matrix summary, checks every available binding, and verifies the
historical comparison against the immutable PR95 journal. It checks before/after
byte stability under a trusted interpreter and quiescent filesystem, not a hostile
concurrent writer. No target tensors, native model or optimizer are loaded.

The original audit stdout reports a completed, verified owner-local audit.
That audit checks source/request/retained structure and scores; it is not
inference or optimizer replay. This subset cannot rerun that full audit.
Request/source identities and runtime declarations remain linked owner receipts,
not independently verified omitted inputs, hardware, timing, actual execution,
or proof that no other attempts occurred. The archive and inner hashes establish
byte integrity, not independent authority for execution claims.

The owner reports one authorized controller attempt after an earlier blocked
GitHub lookup; earlier evidence was preserved. The supplied operational narrative
also records inspection-command exits 2 and 1, attributed to a directory/file
path mistake and an incorrect expected request field. Those logs are omitted:
we neither independently verify those explanations nor erase the failures, call
them native guard failures, or claim every owner command passed. The new public
note does not copy local operational metadata unnecessarily.

```sh
PYTHONDONTWRITEBYTECODE=1 python -I -B scripts/check_music_pitch_result.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -B -m unittest discover -s tests/music_pitch_result_v1 -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/music_pitch_result_v1 -v
```

Affected TRI-I07/08/11/13/14. All historical source and fixture bytes remain
unchanged; this result note adds no training capability or new study semantics.
