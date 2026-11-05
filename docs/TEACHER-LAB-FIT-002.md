# TT-003-FIT-002 — failure-mechanism investigation (design only)

**State: PROPOSED RESEARCH DESIGN / READ-ONLY ANALYSIS FIRST / NO TARGET EXECUTION AUTHORITY.**
This document defines hypotheses, discriminating observations, independent
controls and an evidence-to-intervention decision gate. It does **not** approve
a new training run, mutate a frozen request, choose a loss intervention, unlock
a model, or permit TT-004. Documentation approval is not owner authorisation.

**Scope:** the original source-bound TT-003-FIT-001 dense teacher, the 72
TRAIN-only rows and three completed single-use seed attempts, **not** held-out
generalisation. This is a prospective analysis plan constructed **after seeing
FIT-001 outcomes**; no analysis of these already-observed runs can honestly be
called a blinded or prospectively preregistered test. A *new* intervention and
any confirmatory dataset/schedule must be separately frozen and authorised.

Read [FIT-001 design](TEACHER-LAB-FIT-001.md),
[execution contract](TEACHER-LAB-FIT-001-EXECUTION.md),
[independent evidence reviewer](TEACHER-LAB-FIT-001-EVIDENCE-REVIEW.md),
[diagnostic matrix](TEACHER-LAB-FIT-002-DIAGNOSTICS.md),
[Trinite invariants](INVARIANTS.md) and
[QSOL-IBAE governance](AGENT-IBAE-GOVERNANCE.md).
Historical producer, source closure, requests, approvals, manifest and
checkpoints are **read-only**; their source revisions remain authoritative.

## 1. Evidence anchors and observation classes

**OBSERVED IN RETAINED FIT-001 EVIDENCE (as previously reported):**

- Original native teacher source commit:
  `308a0b73b210a691efbed8f71db5b92eb4d96262`.
  FIT-001 source closure:
  `sha256:2ed6c4bc5cc2aa1aaef47c789882ac88ab6bcdc2b2634e8da0255b8ecc28288d`.
- TT-002 manifest:
  `sha256:b90287c3549e2a877b4e23325b51f234f3e54eab3a48c0227f34c5f18d065687`.
  Panel identity:
  `sha256:4449fbfc0e53ca0aabdcebfa4b092f3e5ffa42b1a3d9d68e429f8e71f2ea1570`.
  Ordered visit ledger:
  `sha256:103624fa8e4a1a294174b3e43e0db1eec45102e20f08f3fe3ee4a0f346b7cbfe`.
- Original owner-local CPU FP32 environment:
  `sha256:e2d93d16ddd955f47ee34e1bd0e04b79fd3a21bd3da28fbc24f0966c8af82fe3`.
- Independently anchored combined FIT-001 archive:
  `sha256:8da627191955dd8d835cd932c50bc90e6b62d4be2673219a55dd5b020407a19f`.
  It contains three attempts and 15 complete checkpoints; it is **private**.
  The additive offline review was merged in
  [PR #139](https://github.com/QSOLKCB/Trinite/pull/139).
- All seeds consumed 256 updates / 3,590 supervised answer/EOS targets,
  used one identical 72-row TRAIN panel and completed within the frozen limits.
  Same-host restored-checkpoint forward replay was exact; a complete AdamW
  optimizer replay was **not** performed.

| Seed | Final exact /72 | Apply /18 | Inverse /18 | Check 1 /18 | Check 0 /18 | Premature EOS /72 | Gold-prefix target NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 23 | 2 | 3 | 0 | 18 | 1 | 0.601082610 |
| 29 | 18 | 0 | 0 | 0 | 18 | 31 | 0.855618015 |
| 43 | 23 | 2 | 3 | 15 | 3 | 6 | 0.510283875 |

Both fixed constant-zero and constant-one complete-answer baselines are
**18/72**, including zero complete vector answers in this panel. Seeds 17 and
43 each have five correct complete vector answers, not five newly correct
scalar checks; only two correctly answered vector *examples* overlap across
these seeds. Seed 29 has zero correct complete vectors at step 256. All three
missed the preregistered **72/72 fit endpoint**. These are measurements of an
already-exposed TRAIN panel.

**DERIVED:** EOS collapse, scalar label preference and poor complete-vector
fitting occur together in some checkpoints. This does not show which failure
caused another. The three hypotheses below are **not mutually exclusive**.
The numerical source of an observation must remain separate from our
interpretation.

**UNKNOWN:** gradient-level mechanism, counterfactual performance under a
modified objective, optimizer-update replay, behaviour on a truly unseen
distribution, and whether further training would help.

## 2. Research question and competing hypotheses

What explains the gap between decreasing teacher-forced token loss and poor
complete GF(3) answer accuracy, including the strong dependence on seed and
checkpoint?

### H-EOS — output termination / length instability

The model sometimes terminates at the wrong step even when other token scores
or residue selections might permit a valid continuation.

**Predictions to test:** prematurely emitted EOS is more frequent among failed
examples; probability or logit margin of EOS over the best permissible
non-EOS answer token rises at affected raw prefixes; high gold-prefix residue
competence coexists with raw greedy truncation; the pattern varies by seed
and checkpoint. Seed 29 at 256 (31 premature EOS cases) is a lead, not a
causal conclusion.

**Falsifiers/limitations:** wrong non-EOS residues and wrong scalar labels can
persist even with valid length and EOS. Margins at gold prefixes are
teacher-forced and must not be described as actual free-running answers.
Do **not** reconstruct a new greedy trajectory from logits after an EOS stop:
the required counterfactual prefixes were not executed in the archive.

### H-SCALAR — check-label shortcut / marginal-majority preference

The one-token `check` task is sometimes fit as a preferred scalar answer
(`0` or `1`) without adequate discrimination of the candidate and circuit.

**Predictions to test:** one scalar label dominates predictions across the
18 positive and 18 negative checks despite their equal frequencies; scalar
confusion matrix and label margins flip sharply between checkpoints or
initialisations; failures remain despite well-formed one-token outputs and
correct EOS position. Already observed: at step 64 every seed gets check 0
18/18 and check 1 0/18, while at step 128 every seed gets check 1 18/18 and
check 0 0/18. This is evidence of *output preference*, not proof of its cause.

**Falsifiers/limitations:** meaningful candidate-dependent correct/incorrect
discrimination with stable margins, balanced by family and word, weighs
against a pure label-only shortcut. The panel is engineering-exposed, and
label flips can be induced by shared logit dynamics rather than a distinct
memorisation mechanism.

### H-VECTOR — weak complete transformation fitting

The teacher improves individual 4-trit residue probabilities under gold
prefixes but rarely generates the complete exact GF(3) mapping with correct
length, order and terminal EOS.

**Predictions to test:** conditional residue accuracy materially exceeds
chance/baselines while exact complete-vector rate remains low; errors compound
across four residues and EOS, potentially with position-dependent
concentrations; a small number of correct vectors may reflect narrow
memorisation or repeated transform fragments rather than a general rule.

**Falsifiers/limitations:** if nearly every vector failure is instead caused
solely by early EOS with otherwise correct raw continuation, a length/termination
hypothesis becomes more plausible. However, gold-prefix correctness is **not**
a free-running counterfactual, and multiplying conditional accuracies assumes
independence that has not been established.

### H-NULL — insufficient fitting evidence / unseparated mechanisms

The observed trajectories may be compatible with several mechanisms at once,
or none may be identifiable from the retained observations. H-NULL remains a
valid result; **inconclusive** must not be silently converted to a chosen
intervention.

## 3. Analysis phases, authority and allowed inputs

| Phase | Permitted activity | Required gate | Prohibited |
| --- | --- | --- | --- |
| FIT-002-DESIGN-001 (this PR) | Written protocol, fixed references, decision matrix | Exact-head reviewed documentation and links | New target run, freeze, approval or changing FIT-001 |
| FIT-002-OBSERVE-001 (future) | Read-only owner-local replay of the *retained* archive through the already merged offline auditor and independently coded analysis | Check external SHA-256 first, preserve original bytes, review derived metric code | Model load, optimizer update, new prompt inference, test labels, publishing private contents |
| FIT-002-ADJUDICATE-001 (future) | Publish a **derived, nonprivate** hypothesis/control table with all failed tests and missing fields | Reviewer checks source, formulas, controls, confounders and exact data exposure | Automatic intervention selection or treating post hoc evidence as confirmatory |
| FIT-002-INTERVENTION-001 (only if separately proposed) | New source-bound experiment with one frozen intervention and matched baseline/control | New design, exact-head review/CI, clean source/rights/environment, owner approvals per run | Reusing FIT-001 approvals, altered old receipts, automatic training, TT-004 |

The public PR must contain **no private FIT-001 archive**, optimizer/weight
tensor, rights/approval packet, detailed secret path, unredacted per-row
measurement or exact private model output. Retained aggregate counts above are
already described in the research record; they are not a new evidence release.
The analysis owner must keep independent source and archive SHA-256 sidecars
outside Git and never import or execute archived Python.

## 4. Proposed read-only measurements

The required unit of analysis is the same **72 row IDs** in each of three
seeds and each of five checkpoints (0, 64, 128, 192, 256): 1,080
checkpoint-row observations. Recompute all measurements from the
archived, verified records, never from the generated synthesis prose alone.

- **P0 — end-to-end exactness (primary descriptive metric):** exact
  answer-plus-EOS for every row; 18 each in apply, inverse, check 0 and
  check 1; 36 vector and 36 scalar examples per seed/checkpoint.
  Report valid length, missing EOS, premature EOS and wrong-answer counts
  separately. Preserve all 15 checkpoints and every seed.
- **P1 — EOS/length:** 72-row premature-EOS count, by task, family and
  checkpoint; first premature-EOS position; logit margin
  `logit(EOS) - max(logit(token) for permitted non-EOS answer tokens)`
  at *recorded* raw and gold prefixes. Also report the stable full-vocabulary
  binary EOS-versus-rest log-odds. They are different measurements.
  Do not call a restricted-token or gold-prefix result normal greedy accuracy.
- **P2 — scalar label:** per-checkpoint 2x2 confusion matrix for 36 check
  examples; `P(1)`, `P(0)`, margins `logit(1)-logit(0)` at the
  scalar-answer prefix; EOS/format errors separated; per-family and
  per-word breakdown. Compare to always-0 and always-1 **18/36** scalar
  baselines before interpreting accuracy.
- **P3 — vector content:** 36 exact complete-vector answers; 144
  per-position vector residue targets (including the zero target); split
  correct nonzero residues into raw greedy (where the prediction exists)
  versus gold-prefix best-residue conditional measures; position 1–4,
  circuit, output diversity and 4-position sequence correctness.
  Report gold-prefix NLL and actual greedy completions separately.
  Do not label an EOS-masked, length-oracle probe as an unaided answer.
- **P4 — trajectories:** for each seed at each frozen checkpoint,
  compare exact P0, gold-prefix NLL, EOS metrics, scalar confusion,
  vector correctness, distinct outputs and example-level gains/losses
  (e.g. 64→128→192→256). At checkpoints where the loss improves while
  complete accuracy falls, report both, not a selected best point.
- **P5 — uncertainty:** list absent measurements, mismatched identities,
  unresolved results and all rejected/partial derivations explicitly.
  No significance testing or statistical confidence claim from three
  selected seeds. No unapproved new experiment fills a missing field.

All recomputed metrics need exact definitions, numerator/denominator,
denominator-exclusion rules, raw-vs-gold prefix distinction, hash-bound input
inventory and a separately reviewed software-double regression.

## 5. Controls and confounding barriers

1. **Two historical constant-output controls:** always `0` and always
   `1`, both 18/72 complete and 18/36 for scalar checks. Score them
   against the identical 72 expected answers, respecting answer length.
   Do not compare vector accuracy to a scalar baseline alone.
2. **Length and EOS controls:** count semantic residue errors **conditional
   on a valid recorded raw output length** independently of early-EOS
   failures. A gold-prefix or forced-length diagnostic is explicitly
   `ORACLE_CONDITIONED`, never an observed free-run improvement.
3. **Independent GF(3) oracle:** source-bound TT-002/qutrit-conformance
   mathematical checks, including inverse consistency, source/rights and
   split manifests; no model output can act as its own reference truth.
   Verification of archived identities is not new oracle execution.
4. **Label/circuit balance control:** 18 checks of each label, all S1/S2/W
   circuits, four views per selected word. Always calculate by target
   label and family before aggregating; report candidate-conditioned errors.
5. **Position-marginal control:** calculate the majority residue at each
   vector position from 36 known targets and compare positional accuracy
   against it. Do not equate 4-position marginals with any 4-token correct
   *sequence*. Use actual frozen counts rather than a uniform 1/3 guess.
6. **Fixed-exposure control:** exact same panel, 1,024 visits and 3,590
   targets for seeds 17/29/43. Include every seed and all five checkpoints;
   no early-stop substitution, new seed, checkpoint cherry-pick or
   hindsight correction of the native training plan.
7. **Adversarial evidence controls:** synthetic-only tests for mismatched
   gold/greedy vectors on identical prefixes, forged approval scope,
   incomplete source/rights inventory, changed schedule, manipulated
   native time, altered backend, held-out scoring flag and re-anchored ZIP
   mutation. Negative tests demonstrate *reviewer* correctness, not learner
   capability.
8. **Exposure barrier:** existing TT-002 validation and test are
   engineering-exposed; neither is an independent scientific holdout.
   Never use those labels, their exact solutions, or private targets to
   tune the next intervention. A confirmatory generalisation test requires
   a newly locked, structurally independent cohort, source/rights admission,
   no development exposure and a separately approved protocol.

Controls 1–7 are analysis design and (where source permits) read-only
checks. Creating new transformed prompts, changing EOS masking in a native
forward pass, running counterfactual sampling, perturbing scalar labels
or performing a controlled objective ablation is a **new operation** and
needs its own explicit review/authority; it is not authorised here.

## 6. Evidence-to-intervention adjudication gate

**First classify all three mechanisms separately** as `SUPPORTED_AS_A_LEAD`,
`CONTRADICTED` or `INDETERMINATE`, with linked source-derived figures.
Because the FIT-001 data already influenced these hypotheses, these labels
describe *exploratory evidence*, not statistical hypothesis acceptance.

- **EOS lead:** a material group of failed raw greedy answers ends early,
  with corresponding prefix EOS margin/count evidence. Report the residual
  number of **wrong content answers even when length is valid**; this
  prevents assuming a length-only fix would solve fitting.
- **Scalar lead:** label prediction imbalance persists across matched
  positive/negative examples and changes with checkpoint despite equal
  label counts. Distinguish an unconditioned preference from actually
  candidate-sensitive correct decisions.
- **Vector lead:** per-position residues are often right but the complete
  four-residue-plus-EOS sequence is rarely exact, including nonzero examples.
  Check by circuit/word, position and prefix scope.
- **Multiple/no leads:** retain all supported leads and the null state.
  A single seed or one favourable checkpoint never selects the treatment.

An intervention is **not selected** by this document. After independent
review of the read-only adjudication, a new PR must choose at most one
first treatment as an explicitly tagged *proposal*, state competing
predictions, define a matched native baseline (same data, budget,
backend and measurement inventory), freeze any new objective or decoding
rule as a different source/experiment identity, and state falsifiers,
failure handling and owner-approval requirements **before** running.
Potential treatment families are (a) EOS-specific supervision or
termination calibration, (b) candidate-sensitive scalar checks/task
balance, and (c) vector-sequence curriculum or credit assignment.
All require fresh study and may be refused; no treatment is pre-approved.

**Do not transfer FIT-001's three exhausted single-use approvals, checkpoint
identities or research result labels into a new training permission.**
Any intervention cannot claim superiority or generalisation from a
tiny, engineer-exposed panel. The observed result can remain a negative
scientific finding even if the next producer passes every engineering test.

## 7. Acceptance criteria for the design and for future analysis

**This documentation PR is complete only when:**

- H-EOS, H-SCALAR, H-VECTOR and H-NULL have separate predictions,
  confounders and evidence requirements.
- The diagnostic matrix specifies exact numerators, denominators,
  checkpoints, raw-vs-gold scopes and constant baselines.
- Scientific claims retain the three original failed exact-fit endpoints;
  no fake repeat, optimizer replay, experiment freeze, owner grant,
  training instruction, held-out comparison or model publication appears.
- All linked in-repository contracts resolve, historical contracts stay
  unchanged, and relevant CI/tests on the *exact PR head* complete.
- Independent code review sees the exact head, with no unresolved
  actionable correctness findings.

**A later read-only analysis is accepted only if** it verifies the private
archive against the **external** SHA-256, checks all 15 checkpoints and all
72 row IDs per checkpoint, recalculates every target numerator/denominator,
retains unresolved negative or absent observations, and makes *no* causal,
unseen-input or update-replay claim. Invalid/partial evidence is a recorded
failure, not permission to improvise inputs or run the learner.

## 8. Governance and invariant impact

This change is prospective narrative documentation under `docs/`,
not a mutation of native code, datasets, trainer, original request,
historical metrics, archived source or top-level `AGENTS.md`.
It does not change the original historical source closures; future
source-bound work must bind these added files as appropriate. Root and
nested AGENTS still apply. Never add `AGENTS.md` under `src/` solely to
guide this research; its current complete-source inventory forbids it.

Affected requirement mapping:

| ID | This design's check |
| --- | --- |
| TRI-I01/I02/I05/I12 | No new numerical model/backend or imported archived code; CPU baseline remains a fixed comparator |
| TRI-I03/I11 | Exact TRAIN/rights boundary; no training or held-out label use |
| TRI-I06 | Read-only diagnostics; observer state mutation prohibited |
| TRI-I07/I08 | Source, packet, original seed requests, archive digest and replay scope explicitly separated |
| TRI-I09/I10 | No geometry, mechanism or superiority claim without intervention evidence |
| TRI-I13 | All failed/negative/unknown outcomes preserved, including seed 29 |
| TRI-I14 | New doc version and future authorisation needed for any numerical treatment |

**Next permitted action:** review this design; then, only by a separately
requested owner-local read-only action, prepare FIT-002-OBSERVE-001 evidence.
**Not permitted:** additional training, frozen request construction,
owner-approval creation, altered EOS handling, model publication, or TT-004.
