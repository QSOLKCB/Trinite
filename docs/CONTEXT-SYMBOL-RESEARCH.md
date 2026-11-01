# Prospective context-dependent symbol curriculum

Status: next curriculum experiment proposal, **not implemented training or a
learning result**. The question is whether Trinite uses explicit contextual
evidence rather than memorizing one association for a symbol. It is separate
from the [masked evidence scorer](TERNARY-EVIDENCE-V1.md); that scorer does not
construct worlds, generate labels or run a model.

## Original formal task, before any run

Define finite sets of contexts, symbols and meanings, and a total explicit
mapping `(context, symbol) -> meaning` for each admitted world. Construct
original synthetic rules and a public query such as: “In this context, does
symbol K mean M?” No passage, illustration, OCR, personal conversation or
example from a book becomes training data. Rules and queries use newly
constructed formal names, not a psychological diagnostic or interpretation of
real people.

A generator must first enumerate the finite worlds compatible with the public
rules and context evidence. A public query's label is +1 if **all** compatible
worlds support it, -1 if **all** contradict it, and 0 if some support and others
contradict. Zero compatible worlds means inconsistent input and **rejection**,
not neutral, contradiction or a silently repaired case. Freeze exact finite
bounds, the full grammar and an independent exhaustive label oracle before
freezing any run request.

Missing context is not automatically label 0: other rules can force the answer
across all compatible worlds. A separate observation mask records whether an
input observation is supplied; an observed semantically neutral/mixed label
is not the same as a missing observation. Uncertainty here is a finite-world
logical property, not probability or confidence. Public model inputs contain
only rules, observations and query—never hidden labels, selected-world state,
answer-derived metadata or family identifiers.

## Controls and split design to freeze first

- Group every counterfactual mapping variant, renaming/isomorphism and related
  query from one structural family into the same split. No memorization-friendly
  sibling leaks across train/validation/test; audit structural canonicalization
  independently before any update. Family IDs belong only in controller metadata.
- Include fixed-symbol/opposite-context pairs and context-preserving renamings;
  these expose symbol-only association strategies. Include missing-context
  examples with both determined and mixed outcomes, plus inconsistent cases.
- Compare a context-aware task against an explicitly frozen context-removed
  ablation and a symbol-only baseline. Context removal can change valid labels:
  recompute them from the remaining public rules, rather than retaining hidden
  answers and calling them knowable. Separate matched informative subsets from
  genuinely ambiguous subsets in reports.
- Freeze model/seed, optimizer, update and token budgets, example exposure,
  decoding, compute accounting and stopping criteria. Equal token limits alone
  do not establish equal compute. Keep capacity/format interventions separate;
  do not claim a ternary benefit without its controlled comparison.
- Report output-grammar validity separately from exact semantic accuracy,
  per-label results, paired contextual sensitivity, renaming consistency and
  rejection of inconsistent rules. Retain all seeds, zero/negative outcomes,
  timeouts and exposure information. Tune only on the declared development
  material; use fresh structural families for confirmation.

The next implementation would be a separately reviewed generator/oracle and
admission contract, followed by a source-bound frozen request. This note does
not run training, change native admission policy, unlock Phase 5, promise
improvement, or alter historical queue/race protocols.

## Bibliographic inspiration only

C. G. Jung and collaborators, *Man and His Symbols* (1964), is a multi-author
reference for the broad idea that symbols depend on context. This is neither
an ML proof nor a technical source for the proposed labeling algorithm. No
book text or images are redistributed, and no public-domain or rights-admission
claim is made for a particular edition. The experiment above is an original
formal construction; its usefulness must be demonstrated with controls.
