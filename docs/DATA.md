# Data contract

Status: Phase 1 implements a narrow admission policy for numeric modular-addition facts and two minimal symbolic carriers. The 48 retained examples are software-conformance fixtures only; broader datasets still require their own admission evidence.

## Phase 1 admitted fixture

The source is 24 numeric triples over moduli 2–9: (0, 0), (0, modulus-1), and (1, 1) for each modulus. The generator renders each triple as infix or labelled symbols and emits a numeric answer. No scraped text, prose corpus, teacher output, or external file is consumed.

The built-in admission record identifies the exact formal source, authorship (deterministic generator, with its Codex implementation commissioned by Trent Slade / QSOL-IMC), acquisition/generation/verification/rendering steps, symbolic scope, rights basis, automated policy audit, review date, and limitation. Its supporting reference binds src/trinite/data.py by content hash and named functions; rights evidence retains all 24 formal triples and all 48 exact symbolic texts with their payload identity. Required fields, source reference, and evidence scope are validated before full fixture replay. This is not a human legal certification or a blanket exemption for synthetic data. Generator software retains the repository's MPL-2.0 license; the numerical fixture is not admitted merely because that software is licensed.

The cyclic-counter generator and independent modulo-based verifier use different procedures. A separate prompt parser checks that the rendered operands/modulus match the formal item. The audit recomputes these checks, tokenization, source receipts, admission, and split membership; self-reported hashes and a passed label alone cannot certify a modified item.

Families are whole moduli, including every selected operand pair and both carriers. A SHA-256 rank over policy/seed/family identity assigns six families to train, one to validation, and one to test: 36/6/6 examples. No item- or carrier-level random split is used. Formal problems are deduplicated once, then deliberately retained as labelled carrier variants in the same family. Exact text duplicates are forbidden.

This fixture is fully visible and used for software checks, so its test split must not be treated as a fresh confirmatory reasoning benchmark. Later model evaluation needs its own frozen held-out families and exposure/tuning controls. Changing source code or split seed produces a new generator/dataset/manifest identity rather than preserving old evidence.

validate_admission validates only this frozen Phase 1 source policy, not arbitrary future source records. It requires the admitted outcome, exact numeric rights basis and scope, declared origin/author/reviewer/limitations, the frozen generation procedure and evidence review basis, and the pinned review date in valid ISO calendar form. Nonempty contradictory labels such as rejected or all future internet text fail even when the validator is called independently. New sources or review decisions need a separate reviewed policy change.

## Strict admission

The separately authorised [comparison detour](COMPARISON.md) adds numeric QEC syndrome/correction fixtures through comparison_data.py. Source-specific admissions record commissioned generator authorship, generation/independent-verification procedures, exact source/QEC pins, supporting numeric evidence, dataset identities, symbolic scope, reviewer/date, outcome and limits. This uses the existing generated-formal-facts category; it does not relax the policy or import QEC prose. All tasks/carriers and local error variants stay in support-site families. Oracle software retains its own attributed license. Full regeneration rejects contradictory admissions, changed evidence, labels, prompts and splits.

The original “No Copyrighted Data” requirement remains strict. A permissive license permits use but does not remove copyright, so a permissively licensed copyrighted corpus is not admitted under this plan. Any relaxation would be an explicit project-contract change.

Start with generator-defined formal facts, arithmetic, truth tables, graph relations, and templated deductions whose content and provenance can be audited. Independently generated data still needs a rights assessment; “synthetic” is not an exemption. Public-domain material needs an evidenced public-domain basis for the applicable use/jurisdiction. An explicit dedication is evaluated and recorded rather than treated as a global guarantee.

Unknown rights, scraped text, inherited pretrained corpora, model-generated teacher answers, and unverifiable source lineage are excluded from the native lane. Copyrighted generator code remains software with its own license; admission concerns the resulting data and source content embedded in it. Keep prose templates minimal, formal, and audited. Do not copy textbook explanations or problem wording.

## Required records

Each source has an ID, content hash, origin/author, acquisition or generation procedure, rights basis and supporting reference, applicable scope, review outcome, and reviewer/date. Each generated item records source IDs, generator revision, seed, formal specification, family ID, rendered text, expected answer, verifier outcome, and transformation lineage.

All consumed content is within this policy: prompts, answers, reasoning traces, tokenizer inputs, training data, validation data, and evaluation data. External models used only as comparators must disclose their unknown or different training-data exposure; their corpora are not represented as Trinite-admitted data.

## Quality and split isolation

Generate answers with deterministic formal solvers where possible and verify independently of text rendering. Reject ambiguous, contradictory, duplicate, incorrectly labelled, and overlength items. Record every exclusion rule and counts before and after curation. Quality means verified correctness and useful task coverage, not selecting only examples a preferred model solves.

Group by underlying formal problem/generator family before assigning train, validation, and locked test splits. Different names, numbers, or surface carriers of the same underlying item cannot cross splits. Freeze seeded split assignment and content hashes. Deduplicate normalized structures and exact text; report remaining near-duplicate limitations.

Test labels must not enter training, geometry-objective construction, prompt tuning, early stopping, or selection of layers/metrics. Validation selects models under the frozen protocol; test is evaluated only after selection. Repeated test use requires a new held-out confirmatory set or an explicit exploratory label.

Publish admitted inputs and generators wherever the rights basis permits. A hash of unavailable data is not complete openness. Releases must disclose any missing inputs and cannot satisfy the full-open gate while required native training inputs are unavailable.

## Foundations admission

[FOUNDATIONS.md](FOUNDATIONS.md) implements the next narrowly admitted generated
formal pack: arithmetic/fractions, signed-list folds and symbolic lattice tasks.
Each source-specific record retains commissioned authorship, complete formal
items, generation/independent-oracle/parser steps, immutable supporting source
references, numeric rights basis, reviewer/date and exposure limitations.
`validate_admission()` independently regenerates the frozen expected record; a
nonempty contradictory field or changed evidence fails. Both carriers and all
related transformations remain in the same family. This admits no donor prose,
software text, benchmark questions or third-party sheets. The earlier fixtures
and data policy remain unchanged.

The matched [learning increment](LEARNING.md) uses a separate
`trinite.learning-data.v1` admission wrapper over those same independently checked
facts. It binds generator revision, split seed and transformation lineage in every
item, including the original example as `ordering_identity`. These new identities
preserve the prior texts, families, splits and numerical exposure order. The
retained parent evidence is unchanged; this is not blanket admission of generated
data or external curriculum sources.

The [scalar curriculum preparation](GENERALIZATION.md) admits bounded numeric
projections over the same learning facts: separate count/sum/squares and
traversal-value/x/y/z targets. Every child inherits its parent's semantic family
and split; both carriers and correlated projections stay together. The
independent scalar oracle/parser checks procedural labels and prompt meanings.
Exact source revision, parent example/semantic lineage, content identities and
rights evidence are retained in the new manifests. This preparation has not
been trained and admits no donor text or model output.
