# TT-003-FIT-002 — fixed diagnostic and control matrix

**Status: PROPOSED / DESIGN-ONLY. Not a frozen target request.**
Companion to [FIT-002 failure-mechanism design](TEACHER-LAB-FIT-002.md).
The fixed observations here are from completed FIT-001; every proposed
read-only computation must remain source-bound, independently recomputed
and explicitly **exploratory** because these outcomes were already seen.
No GPU, CPU training, model forward inference, held-out scoring or
new frozen request is authorised by this document.

## A. Historical 15-checkpoint reference inventory

**OBSERVED from retained FIT-001 synthesis** (not repeated training):

| Seed | Updates | Exact /72 | Apply /18 | Inverse /18 | Check 1 /18 | Check 0 /18 | Premature EOS /72 | Gold-prefix NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.54316503 |
| 17 | 64 | 19 | 1 | 0 | 0 | 18 | 15 | 1.06579774 |
| 17 | 128 | 19 | 1 | 0 | 18 | 0 | 0 | 0.711519758 |
| 17 | 192 | 21 | 1 | 2 | 6 | 12 | 0 | 0.703214566 |
| 17 | 256 | 23 | 2 | 3 | 0 | 18 | 1 | 0.60108261 |
| 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.52600243 |
| 29 | 64 | 19 | 1 | 0 | 0 | 18 | 0 | 1.09144644 |
| 29 | 128 | 20 | 1 | 1 | 18 | 0 | 0 | 0.697639112 |
| 29 | 192 | 20 | 0 | 2 | 16 | 2 | 0 | 0.793626555 |
| 29 | 256 | 18 | 0 | 0 | 0 | 18 | 31 | 0.855618015 |
| 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.48753318 |
| 43 | 64 | 19 | 1 | 0 | 0 | 18 | 21 | 1.07000478 |
| 43 | 128 | 19 | 1 | 0 | 18 | 0 | 24 | 0.773257773 |
| 43 | 192 | 22 | 2 | 3 | 17 | 0 | 0 | 0.651535377 |
| 43 | 256 | 23 | 2 | 3 | 15 | 3 | 6 | 0.510283875 |

Each row represents **the same 72 exposed TRAIN examples**, scored at its
frozen checkpoint, with a strictly separate seed model. Non-final
checkpoints are diagnostics, not alternative primary endpoints.
The native final endpoint is always update 256. The fixed constant-zero
and constant-one comparators each score **18/72** overall, **18/36**
within check rows, and **0/36** complete vector answers.

**OBSERVED nonzero residue contrasts:**

| Seed | Update | Correct nonzero raw-greedy residues /106 | Correct nonzero gold-prefix best residues /106 |
| --- | ---: | ---: | ---: |
| 17 | 0 | 0 | 49 |
| 17 | 64 | 61 | 74 |
| 17 | 128 | 81 | 82 |
| 17 | 192 | 66 | 58 |
| 17 | 256 | 76 | 75 |
| 29 | 0 | 0 | 32 |
| 29 | 64 | 74 | 74 |
| 29 | 128 | 79 | 81 |
| 29 | 192 | 74 | 72 |
| 29 | 256 | 32 | 63 |
| 43 | 0 | 0 | 34 |
| 43 | 64 | 60 | 74 |
| 43 | 128 | 59 | 80 |
| 43 | 192 | 82 | 80 |
| 43 | 256 | 75 | 82 |

The **106** denominator counts nonzero expected vector residues in the 36
vector outputs per checkpoint, not full answer accuracy. A raw greedy
residue may be absent after early EOS; the verifier must retain its
original defined counting convention rather than inventing a token.
Gold-prefix best residues condition on oracle-supplied prefixes, not on
unassisted autoregressive generation. Both measures are **diagnostic
only** and neither alone establishes a learned transformation.

Position-wise expected vector-residue inventory:

| 1-based residue position | Target 0 | Target 1 | Target 2 | Majority-target count /36 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 12 | 16 | 16 |
| 2 | 8 | 10 | 18 | 18 |
| 3 | 14 | 8 | 14 | 14 (tie: 0 and 2) |
| 4 | 8 | 2 | 26 | 26 |

The position-majority **74/144** figure sums per-position controls only.
It is **not** a complete-answer result or proof of independence between
positions. Never multiply marginal frequencies to synthesize a fictitious
full-vector baseline.

## B. Analysis definitions (all 15 checkpoints, never best-of-three)

For every checkpoint and each row ID, a future offline analyst must
recalculate, retaining exact counts and denominators:

| ID | Diagnostic | Definition / denominator | Mechanism discriminated | Critical caveat |
| --- | --- | --- | --- | --- |
| D00 | Complete exact answer | Full expected ASCII answer + EOS, /72; task subtotals /18 each | All | Accuracy on exposed TRAIN only |
| D01 | Wrong length / termination | Early EOS, absent EOS, valid length, format and wrong content, /72 and by task/family | H-EOS vs content | Early termination is not proof of cause |
| D02 | EOS margin | At a **recorded** prefix, EOS logit minus largest permitted answer-residue logit; full-vocab EOS/rest log odds separately | H-EOS | Gold vs greedy prefix must be tagged; masked-EOS not native greedy |
| D03 | Check-label confusion | TP/TN/FP/FN on 36 check rows, by circuit and word, plus label margins and EOS correctness | H-SCALAR | Scalar 18/36 constant controls; check candidate must remain matched |
| D04 | Per-residue vector | 144 positions/step and 106 nonzero targets; raw greedy vs gold-prefix conditional; position 1–4 | H-VECTOR | Unavailable greedy continuation after EOS is **not** a prediction |
| D05 | Complete vector | 36 complete 4-residue+EOS answers, apply /18 and inverse /18, per family and word | H-VECTOR vs H-EOS | Five correct vectors != 72/72 fit |
| D06 | Input sensitivity | Output diversity, per-circuit/word outcomes, row-level gains/losses and correct-vector intersections | H-SCALAR/H-VECTOR | Diversity alone does not prove semantic input dependence |
| D07 | Trajectory | D00–D06 at updates 0/64/128/192/256; compute both positive and negative 64-step transitions | All | No checkpoint re-selection or adaptive extension |
| D08 | Token loss | Native loss and gold-prefix target NLL; report exact disagreement with D00 | H-NULL | Teacher-forced loss is not end-to-end correctness |
| D09 | Custody and scope | External archive hash; 1,512 files, 1,279 source members, 3 attempts, 15 checkpoints, 72 row IDs each, rights and all seed approvals | All | Hash consistency is not producer authentication |

Counts in D09 are *expected historical inventory values*, not permission
to omit a file or regenerate a manifest to pass a check. A future
tooling change requires separately reviewed implementation, and any
new read-only local run must preserve original archive bytes.

### Metric scope and non-interchangeable denominators

- Complete-accuracy denominator = **72** at each checkpoint; breakdown
  apply=18, inverse=18, positive check=18, negative check=18.
- Scalar prediction denominator = **36**; positive and negative each 18.
- Complete vector denominator = **36**, with 18 apply and 18 inverse.
- Vector residue denominator = **144** all expected trits; nonzero
  denominator = **106**; raw greedy missing-token handling must be
  preserved from the original producer and called out.
- Checkpoints = **5** per seed, **3** seeds, **15** total;
  input-checkpoint pairs = **1,080**, not independent data points.
- Exclude neither seed 29 nor non-final checkpoints. Missing or
  inconsistent data are reported as failures/unknown, not scored zero
  unless the frozen scoring contract explicitly requires zero.
- EOS/rest is a **binary log-odds** against all other 258 scores.
  Pairwise EOS-vs-correct-residue and restricted residue best-of-three
  are different quantities. Keep stable log-space math, subnormal guards
  and original numerical equality tolerances; do not conflate them.

## C. Independent controls and discriminating signatures

| Hypothesis | Relevant already-reported lead | Discriminating read-only comparison | Independent control / null interpretation |
| --- | --- | --- | --- |
| H-EOS | Seed 29 at update 256 has 31 premature EOS; seed 43 at 128 has 24 | Per-row termination vs content failures; raw vs gold-prefix EOS margin at the same actual prefix; valid-length wrong-answer inventory | If valid-length wrong vectors remain, a length-only mechanism cannot fully explain failure. Never synthesize unobserved post-EOS prefixes |
| H-SCALAR | All seeds switch from 18 correct negative checks at 64 to 18 positive at 128 | Confusion matrices, logit(1)-logit(0), candidate/family-stratified output and EOS correctness | 18/36 constant label control; per-label balance; stable candidate-specific decisions contradict a pure constant-label shortcut |
| H-VECTOR | Seeds 17/43 only 5/36 complete vectors at 256; many nonzero position matches | Per-position gold and raw accuracy vs complete 4-trit+EOS exactness; error position, circuit, overlap by row ID | Position-majority 74/144 (not sequence baseline); always-0/1 vector baseline 0/36; random chance is not assumed uniform |
| H-NULL | Seed 29 loses complete vectors and NLL rises after 128 | Which predicted signatures fail or remain observationally inseparable; completeness, confounder and identity checks | Outcome INDETERMINATE is valid; 3 seeds do not justify significance claims |

The same prefix in a native checkpoint must have the same retained 259
FP32 logit bytes wherever it is observed. Independent static verification
must reject contradictory repeated vectors even if all nested hashes were
recomputed. Both check-target and EOS probabilities must be computed from
the **full** vocabulary before interpreting probability ratios.

## D. Prospective controlled-experiment *options* (not selected)

A future protocol may be **proposed** only after reviewer acceptance of
D00–D09 and an explicit classification of H-EOS/H-SCALAR/H-VECTOR/H-NULL.
Interventions are mutually exclusive as a first-factor experiment until a
new protocol says otherwise; none are already authorised.

| Candidate | Distinct one-factor change under *future* freeze | Matched reference | What would falsify the working mechanism? |
| --- | --- | --- | --- |
| FIT-002-EOS-A | A formally specified EOS-specific training or termination calibration change; the current native loss stays immutable | Fresh baseline *with same future seeds, row schedule, budget, model, backend and metrics* | Reduced early EOS but unchanged valid-length vector/scalar accuracy; or no reduction in early EOS |
| FIT-002-SCALAR-A | Candidate-sensitive check-task treatment (e.g. controlled label/candidate balancing); no simultaneous EOS rule change | Fresh matched baseline with equal objective measurement and rights | Scalar class bias changes but matched candidate-dependent correctness does not improve |
| FIT-002-VECTOR-A | Vector-sequence credit assignment or curriculum, with a new fixed schedule/rights packet and unchanged EOS policy | Fresh matched baseline with the **newly declared** matched exposure budget; no backfilled FIT-001 run | Gold-prefix residue scores improve but complete vector outputs remain poor at same EOS/length profile |

These are **candidate research designs**, not advice to train all arms.
If any candidate changes the schedule, training objective, tokenizer,
data or source, its source-bound request and baseline form a wholly new
experiment; comparisons with FIT-001 become contextual, not a matched
randomized treatment effect. Do not call any architecture/metric change
a mere "fix" to the original run. No selection, code implementation,
owner permission, frozen training request or model execution occurs here.

## E. Decision record and adversarial review checklist

A future FIT-002-ADJUDICATE-001 report must provide one row for each
hypothesis with:

- Evidence classification (`OBSERVED`, `DERIVED`, `UNKNOWN`).
- Requested D00–D09 measured values, exact input/hash identities,
  numerator/denominator, recorded raw/gold-prefix scope and adverse
  counterexamples.
- Outcome `SUPPORTED_AS_A_LEAD`, `CONTRADICTED` or
  `INDETERMINATE` and an explicit confounder explanation.
- Independent re-computation status, negative controls and verification
  failures; no best-seed or best-checkpoint selection.
- Separate owner decision whether to fund/propose a **new** controlled
  experiment, with no inferred execution authority.

A docs-only PR must be reviewed for internal links, data/claim scope,
no historical source or hash changes, no accidental approvals,
denominator consistency, and the absence of any private raw evidence.
Passing CI or a code-review reaction is not a source-bound training grant.

**Falsifiable scope claim:** This plan can distinguish observed error
patterns; it cannot causally identify optimization dynamics from old
checkpoint and logit receipts alone. Negative/null findings are first-class.
