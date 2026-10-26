# Trinite: diagnostic-first roadmap from v0.4.0 toward v1.0

Original research prepared 22 October 2026 (UTC+10:30), checked 21 October.
Integrated and extended on 26 October 2026 at the owner's request.

**Status: development roadmap; later diagnostic implementations and the dated negative result are linked below. This document grants no target execution authority.** This is the current recommended development order, linked from [ROADMAP.md](ROADMAP.md). The [dated assessment](research/ASSESSMENT-2026-10-22.md) supplies the original rationale; the [26 October research update](research/UPDATE-2026-10-26.md) adds later sources and the shrek_GPT review. Historical protocols, release tags, evidence and existing scientific gates remain unchanged. The workbench/model v1 distinction below is a proposed release-scope decision, not an adopted relaxation of Phase 9. Any change to project-wide release criteria requires a separately reviewed decision.

**Baseline:** immutable [v0.4.0](https://github.com/QSOLKCB/Trinite/releases/tag/v0.4.0), commit `be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a`.

## Current next increment — 26 October 2026

The next implemented learning experiment is the separately versioned
[coverage-balanced addition recipe](BALANCED-ADDITION.md): fresh paired seeds,
40 broad-output training families versus the original five-family control,
unchanged 86,528-parameter dense capacity, and exact matched visit/input/target
budgets. Shared whole-family development, seen-anchor regressions and raw
failure/pair maps are included in the same run; another diagnostic-only PR is
not needed simply to interpret its outputs.

The candidate supports all sums0–18 and both single-digit operand roles, while
shared development covers all five control answer values plus candidate-only
answer values and separate range/length shifts. Output/operand marginals,
coverage and repetition change: this is a **data-recipe comparison**, not an
isolated output-digit causal test or a demonstrated fix. Its implementation and
tiny conformance controls are not target learning evidence. Review, green
exact-commit CI and explicit owner local-run authorization precede execution;
all six outcomes and failures must be retained.

Earlier increments were [addition development](ADDITION-DEVELOPMENT.md) in PR#71,
the [all-token diagnostic](FROZEN-TOKEN-DIAGNOSTIC.md) in PR#72 and the
[failure map](FAILURE-MAP.md) in PR#73. The owner subsequently returned local
matched-coverage and frozen-diagnostic results; their negative transfer evidence
motivates the new recipe, not a proven architecture defect. Original producer
sources, bound implementation-era contracts, checkpoints and negative evidence
remain unchanged. A frozen contract's implementation-time “unrun” statement is
not a claim that the owner never ran it later. No private handover packet is
published by this roadmap update.

These public families have historical research exposure. The new experiment
remains developmental, not fresh confirmation; it does not elect old checkpoints,
resume or repair them, add geometry/ternary/KV changes, or pass a learning gate.
The milestone text below retains the earlier chronology rather than prescribing
another already-completed diagnostic or coverage run.

## Compatibility and authority

The original docs-only roadmap records plans under TRI-I01–TRI-I14; it changes no
numerical operation, admission policy, installed interface, fixture, protocol,
parameter cap or source-bound producer. Every later implementation must name
its affected requirements and freeze its own compatibility contract before
outcomes. Publishing this roadmap authorizes neither its experiments nor a
release. Historical failed and inconclusive outcomes remain evidence.

## Direction

Build a small, inspectable system with one demonstrably useful learned behavior, a stable inference interface, measured CPU resource use, and reproducible evidence. Do not make a general-purpose chatbot, every experimental geometry branch, or an external teacher's size the definition of success.

Maintain two explicit claims:

- **Workbench v1:** a stable, usable research and inference tool with documented contracts, clean installation, reproducible examples, honest failures and supported exports.
- **Trained-model v1:** a particular versioned checkpoint additionally passes a predeclared useful-task quality gate on fresh evaluation data, with a measured resource envelope.

A workbench release does not certify every model trained with it. Conversely, a model-quality result does not automatically make the software distributable. The recommended target is to satisfy both for one narrow task before presenting Trinite as a usable learned edge model. If only the workbench is ready, name and describe that release accordingly rather than moving the scientific goalposts.

## Is the problem our harness or our model?

Neither answer alone fits the evidence.

- Historical dense models passed all nine per-task training gates under the larger-budget study. The native training stack can learn. [BUDGET.md](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/BUDGET.md)
- Held-out dense outputs were well-formed but often semantically wrong. This is a generalization/composition gap, not simply JSON or EOS failure. [GENERALIZATION.md](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/GENERALIZATION.md)
- Syntax masking improved formatting but yielded constant digits matching label prevalence. Harness improvements can make an answer valid without making it reasoned. [PR #50](https://github.com/QSOLKCB/Trinite/pull/50)
- The recent 32/60/256-update pilots have distinct tasks, configurations and exposure. Their negative outcomes do not establish that the architecture is incapable at every budget.
- The teacher pilot tested selection of numeric operands, with oracle-generated labels. It did not demonstrate useful knowledge distillation or a held-out advantage. [PR #62](https://github.com/QSOLKCB/Trinite/pull/62)
- Needle/Bonsai compatibility and output failures concern their pinned external systems. They do not diagnose Trinite's own optimizer. [PRs #64](https://github.com/QSOLKCB/Trinite/pull/64) and [#66](https://github.com/QSOLKCB/Trinite/pull/66)

**Highest-leverage next step:** diagnose one native task using existing successful controls, then change one justified factor. Do not repeat generic budget doubling, invent another trainer, or add several donors simultaneously.

## Current status ledger

| Area | Status at v0.4.0 | Next decision |
|---|---|---|
| Native dense training | Positive bounded training evidence; held-out shortcomings | Identify where correct next-token behavior becomes incorrect complete answers |
| Ternary fixed-code refit | Exact weight-error improvement on the frozen suite | Test trained task quality only against a useful dense anchor |
| Learned scales / PCGrad | Engineering conformance, not demonstrated task improvement | Activate only if diagnostics justify the hypothesis |
| Frozen decoding | Syntax gains; no demonstrated arithmetic gain | Keep raw and constrained outputs separate |
| Geometry / cyclic / NEXUS references | Exact conformance; integral target study remains prospective; cyclic learning negative | Optional research, not universal v1 blockers |
| External NCM | Small correlated RTX feasibility/replay result | Do not claim broad Sudoku or native-model gains |
| Teacher intake | Verified numeric admission; no held-out advantage | Test added coverage against an exact generator before more teachers |
| Tiny-model comparison | Compatibility probes; no completed live 64-call target study | Separate optional calibration track |
| Packed ternary inference | Codec/export foundations, not a finished packed inference runtime | Import/parity first, performance second |
| Historical task-balance evidence | Reported results with missing complete archival packet | Preserve explicit gap; never recreate an original artifact |

Historical chronology belongs in linked documentation, not repeated competing “next steps.” Keep a small claim → evidence → limitation → next decision ledger current.

## Milestone 1: one canonical native diagnostic workflow

**Implementation status:** the checkout-only [native diagnostic v1 workflow](NATIVE-DIAGNOSTIC.md)
provides the first dense-arithmetic engineering slice: mechanics preflight,
source-bound freeze/run, teacher-forced versus raw-generation observations and
read-only audit/summarize. The owner subsequently ran seed 0 locally: its final checkpoint fit 264/264 training
rows, followed by 0/64 validation and 0/64 test complete answers in the separate
inference run. See the [dated negative result](UNSEEN-NEGATIVE-RESULT.md). This does
not close the broader measurement list or any learning gate. Historical numerical
producers remain unchanged.

The additive [frozen-checkpoint unseen-evaluation workflow](UNSEEN-EVALUATION.md)
prepares a separate inference-only diagnosis using the final step-4,096 model
and all historical arithmetic validation/test rows. These families are unseen
to that checkpoint's declared training schedule, but historically research-exposed:
results remain exploratory, not fresh confirmation. The original implementation performed no target evaluation; the later owner-run
result is now preserved. It does not close a scientific gate. The next implemented
step is the [read-only coverage diagnostic and addition-first proposal](COVERAGE-DIAGNOSTIC.md),
not additional training or a claimed fix.

Reuse the authoritative updater and existing positive controls. Provide one documented preflight/freeze/run/summarize/audit workflow with predictable `reports/` outputs. No new independent training implementation.

Add focused measurements:

1. Independently checked token/label shift, answer/EOS masks, causal attention, padding and reduction weighting; optimizer/tensor update and resume checks.
2. Per-position teacher-forced accuracy/loss, first-answer-token probabilities, correct-token rank and premature-EOS frequency.
3. Free-running exact answer, answer-plus-EOS, malformed/unfinished rate and output-length distribution from the same checkpoints.
4. Balanced tiny-set learning, conflicting-answer examples, counterfactual operand pairs and prompt-permutation controls.
5. Majority/constant, lookup and exact-oracle baselines, clearly labelled non-neural where applicable.
6. Gradient/update norms, clipping, tied-output behavior; for ternary arms, code occupancy, scale and STE gradient support.

**Exit gate:** independent mechanics checks pass; existing positive controls remain successful; deliberately constant/input-insensitive controls are detected; every attempted run leaves a clear completed/failed/timed-out/setup outcome. Instrumentation must not change the authoritative numerical trajectory unless explicitly versioned.

**Stop rule:** a failed mechanics check blocks longer training. If mechanics pass, poor quality is not permission to keep rewriting the harness without a specific hypothesis.

## Milestone 2: one useful native dense task

The [addition-first development workflow](ADDITION-DEVELOPMENT.md) now implements
a separately admitted nonnegative addition experiment: five versus ten unique
training families at exactly matched scheduled answer/carrier/token exposure
and operand marginals. It provides a frozen six-cell request, bounded native
training/final-development scoring, retained failures and read-only audit. The
three development strata are public exploratory facts, not fresh confirmation.
No target research cell has been executed by this implementation; this gate
remains open. The historical negative result and all of its sources are unchanged.

Select one admitted task with historical training success and a documented held-out weakness. Specify its intended user-facing value, input range, error tolerance and whether an exact algorithm is the actual better tool. A neural arithmetic module must not be represented as exact merely because its average accuracy is high.

Freeze a bounded development protocol using the successful historical dense configuration as an anchor. Distinguish interpolation, new combinations, longer inputs and new task families. Use training-only curves to choose a recipe; do not repeatedly tune on the historical test split and call it fresh confirmation.

**Exit gate:** complete training answers and input sensitivity beat trivial baselines under the declared bound. If training succeeds but fresh families fail, examine representation/coverage/inductive bias. If training is still improving, a new bounded budget study may be justified; unbounded retries are not.

## Milestone 3: one evidence-justified intervention

Choose one branch from Milestones 1–2:

- **Generation bottleneck:** investigate answer/EOS learning and prefix sensitivity. A typed output head is a new model/task interface, not an invisible harness fix.
- **Representation bottleneck:** test task-aligned positions, digit order or a residue head separately. Report the symbolic structure supplied to the model.
- **Coverage bottleneck:** compare a fixed structured curriculum against a matched exact-generator control, with all exposure differences visible.
- **Optimization bottleneck:** use measured gradients and learning curves to justify a bounded change.
- **Capacity bottleneck:** compare a small declared size ladder only after adequate optimization and input-dependence controls.

**Exit gate:** the intervention answers its declared causal question, with failures retained. Do not change representation, loss, architecture, curriculum and budget together and attribute the outcome to one feature.

## Milestone 4: fresh confirmation, then ternary comparison

Select the recipe before freezing a new confirmation set. Use at least three prospective seeds where feasible, exact counts and uncertainty appropriate to independent families. Do not treat correlated transforms as independent trials. Predeclare quality thresholds, meaningful improvement margins and resource bounds for the chosen use case.

Only after a useful dense anchor, compare native absmean ternary and guarded fixed-code refit. Add learned scales or an STE change only when supported by diagnostics. Match data, initialization policy, optimizer, update/target-token exposure, decoder and evaluation. Disclose differences that cannot be matched.

**Exit gate:** fresh held-out quality exceeds declared trivial baselines and satisfies the use-case gate. A ternary candidate must satisfy a predeclared quality-loss allowance and later demonstrate a real deployment benefit. If it does not, preserve the result and retain dense as the honest reference.

Reconstruction SSE, model quality and runtime efficiency are three separate measurements. Floating-point `F.linear` with ternary-valued weights is not packed ternary acceleration.

## Optional parallel track: small external models

Do not let this track delay native learning diagnosis. Run models sequentially on the existing machine; no Vast/H200 requirement for Trinite.

| Candidate | Purpose | Boundary |
|---|---|---|
| Qwen3.5-0.8B | Cheap external competence/control model | Apache-2.0 card; pin template and thinking mode |
| SmolLM2-135M-Instruct | Conventional tiny baseline | Apache-2.0; competence still requires local evidence |
| LFM2-350M or LFM2.5-1.2B | One hybrid-architecture contrast | Custom LFM Open License, not Apache/MIT |
| Qwen3.5-2B versus Bonsai2-2B | Similar-size product comparison | Training/runtime differ; not a causal ternarization ablation |
| Needle3 | Specialized extraction/tool-format feasibility | Mixed 2/4-bit, not ternary; preserve current negative result |
| Microsoft BitNet | Training/runtime and packed-kernel reference | Published quality/speed does not transfer automatically |

Start with one conventional small control and one contrasting model, not every size. Report raw completion, schema acceptance, exact correctness, truncation, time and memory separately. Prospective adapter fixes are legitimate; rewriting old scoring to rescue an answerless failure is not.

Teachers must justify their cost against exact balanced/stratified generators. Verified structured task or program proposals could add pedagogical information, but expanding beyond the current numeric-operand admission policy needs a separate reviewed change. Do not silently admit arbitrary teacher prose or chain-of-thought.

## Compute and KV-cache harness (requested 26 October)

Add a [capability-gated compute/KV harness](COMPUTE-KV-HARNESS-PROPOSAL.md)
around the model in two increments: uncached resource and budget observation
first, then separately versioned incremental inference only after cache
correctness gates pass. The current decoder recomputes full-prefix attention;
there is no native reusable KV interface to switch on.

The first increment belongs with Milestone 1 and later inference profiling:
measure weights/runtime/KV/scratch separately, distinguish prefill from decode,
and enforce context, batch, output and total resource ceilings. Report native
KV support as unsupported, not a measured zero. The second belongs with
Milestone 5: preserve uncached reference behavior and historical protocols,
bind cache state to exact model/prefix/backend identity, verify position/mask
and stale-state rejection, and measure actual overhead before claiming gains.
This documentation request is not permission to train or benchmark a model.

## Milestone 5: usable CPU inference and resource evidence

Complete a safe versioned checkpoint importer/exporter and offline inspector. Demonstrate reference logits/tokens/captures within the declared exact or numerical tolerance contract before optimizing. Pin the actual tensor, scale, exception and tokenizer representations.

Provide clean installation, minimal dependency profiles, a stable inference API/CLI, supported platform/Python matrix and a compact public example packet. Resolve package version `0.0.0` versus research tag version deliberately before v1.

Profile numerical work, validation, serialization and replay separately. Measure end-to-end latency, resident memory, warm-up and repeated-run variation at equal workload. Optimize only measured bottlenecks with conformance checks. Specialized ternary kernels are a separate backend, not a rename of floating inference.

The Ryzen CPU remains the reference. Any GPU student backend needs its own environment and parity/replay scope. Hardware capacity estimates are not measured maximum model sizes or throughput guarantees.

## Milestone 6: v1 release candidate and independent reproduction

**Workbench release gate:**
- A fresh supported environment installs and runs the documented example without undocumented local state.
- A second supported environment reproduces the declared results within the stated scope.
- Stable interface/schema compatibility, bounded resource behavior, license notices and offline verification are documented.
- Positive, negative and setup-failure outcomes are understandable without hunting through directories.

**Model release gate, in addition:**
- One useful task, declared input domain and fresh held-out confirmation.
- Measured quality/resource envelope and input-sensitive performance above trivial baselines.
- Published versioned checkpoint/export with source, data, protocol and representation provenance.
- Separate reporting of unaided model performance and any exact-tool-assisted system performance.

Not required merely to call the workbench stable: success of every optional geometry, NEXUS, NCM, teacher or exploratory lane. Existing broader scientific gates remain intact for the broader claims they govern. A narrower v1 scope must be explicit and reviewed.

## The next five PRs, in order

1. **Roadmap/status consolidation (this documentation increment):** record the proposed scope distinction and evidence ledger, link historical chronology and preserve archival gaps. Actual release-scope adoption remains separately reviewed.
2. **Native learning diagnostic workflow:** independent mechanics and input-sensitivity checks, raw versus teacher-forced measurements, bounded protocol.
3. **Local diagnostic evidence:** run the reviewed protocol on the Ryzen; preserve every outcome. No simultaneous external model sweep.
4. **One justified intervention:** select from diagnosed failure mode; then confirm on fresh families.
5. **Release workflow/export:** make the useful result installable and independently reproducible; profile before kernel work.

These are logical milestones, not reserved PR numbers, promised release dates or a claim that every milestone fits one commit.

## Added research directions: RSH, integer execution and model size

These are three separate hypotheses. Do not combine them in one training experiment.

### RSH: exact geometry donor first

[QSOLKCB/RSH at 4a05bed](https://github.com/QSOLKCB/RSH/tree/4a05bedb9f10bb5d331006e37373f41aa805582b) offers an exact rational inversion/reflection surface in `formal/lean/RSH/ExactSpatial.lean`, with a normalized-integer Python witness. It is a stronger first donor than floating Frenet integration or tissue interpretations.

A prospective standalone adapter should verify the forward map against an independent rational oracle, its involution/norm identities, singular-centre rejection and canonical fractions. The barycentric witness requires numerators to sum to the denominator (centred coordinates sum to zero); test wrong-sum rejection as well as valid negative numerators. The Lean vector identities have a broader domain, so do not confuse the two contracts. Round-trip checks alone are insufficient: an incorrect identity function would pass them. Preserve MPL-2.0 and the donor's differentiated author attribution, source pin and per-item provenance. The donor's theorem is not an end-to-end proof of a new Python implementation or of neural learning.

Any later RSH curriculum needs family-grouped composition/coordinate holdouts. Do not put inverse pairs or simple coordinate permutations on opposite sides of a split and call that strong generalization. Keep geometry loss and the native capacity experiment unchanged while establishing donor conformance.

### Bounded integers: inference before integer training

A genuinely integer-only forward path is plausible, but int8 alone is not enough. Current ternary codes are reconstructed into floating weights; embeddings, RMSNorm, softmax, GELU, output projection and AdamW also use floating point.

Propose a separate numeric/backend ID using ternary weights where appropriate, int8/int16 activations and proved-safe int32/int64 intermediates. Specify integer multipliers/shifts, rounding, clipping, structural masks, norm/softmax/GELU approximation and every intermediate overflow bound. Avoid hidden floating scales or fallback operators. Calibration uses training/development data, not confirmation answers.

Start with an independent arbitrary-precision oracle and bounded primitive tests, then a useful frozen checkpoint. Require exact parity between the declared integer reference and deployed backend; separately measure approximation/quality differences versus the float reference. Profile before claiming speed or memory benefit. An unbounded Python integer oracle is not an optimized machine-integer runtime.

[I-BERT](https://proceedings.mlr.press/v139/kim21d.html) provides an algorithmic precedent, but its public Python simulation still contains floating operations. Eight-bit optimizer storage also does not mean integer updates. Fully integer training changes gradients, loss, clipping and optimizer semantics and should remain optional later research, not a v1 blocker or an automatic reasoning improvement.

### Three-size experiment: does widening improve fresh task performance?

“Current model” needs an explicit anchor: recent pilots have 7,264 parameters, the historically learnable foundations model has 86,528, and the reference configuration has 1,247,232. Do not mix these roles.

Recommended main dense ladder, derived from the pinned parameter-count formula:

| Size | Blocks / heads / context | Width / feed-forward width | Parameters |
|---|---|---|---:|
| S | 2 / 4 / 64 | 64 / 128 | 86,528 |
| M | 2 / 4 / 64 | 96 / 192 | 178,944 |
| L | 2 / 4 / 64 | 128 / 256 | 304,128 |

This varies width and head dimension, not an abstract parameter-count knob. Clone implementation and protocol, not resized weights claimed to be identical. Use distinct model/configuration/checkpoint identities and recorded fresh initialization; the same seed across shapes does not imply identical weights.

Primary comparison: fixed data/order, updates, target/input exposure and decoding. Secondary comparison: a separately frozen practical compute/time budget. Larger models use more compute at equal exposure, and see less exposure at equal compute. Report both questions without pretending either isolates every factor. Allow only a predeclared equal development-tuning opportunity, then freeze recipes before fresh confirmation across at least three seeds when feasible.

Existing FoundationsPlan/BudgetPlan enforce a 100,000-parameter cap. The larger ladder needs a new bounded protocol, not relaxed historical validation. Keep RSH, integer arithmetic, teachers and quantizer changes out of this experiment. Use one established task and require exact semantic performance, input sensitivity and trivial-baseline comparisons. Better training fit or valid syntax alone is not improved reasoning; three sizes do not establish a scaling law.

An alternative cheap ladder tied to the recent tiny pilot is 7,264 / 18,624 / 53,632 parameters. Choose the ladder that answers the declared question rather than running both by default. The recommended larger ladder has modest parameter storage, but runtime, activations and complete experiment cost still need a measured preflight on the Ryzen. The existing CPU-only reference does not become a GPU backend merely because the RTX has enough VRAM.

## Subsequent research: where the new ideas fit

These additions are **prospective**, not demonstrated Trinite upgrades. The
[source-by-source research update](research/UPDATE-2026-10-26.md) records the
external evidence, limitations and exact tests needed before promotion.

| Priority | Candidate addition | Fits after / gate |
| --- | --- | --- |
| First | Full memory/latency ledger, artifact transformation lineage, export parity and interruption/review coverage | Diagnostic workflow and CPU-export work; these measurements must not alter native numerics |
| Conditional | Progressive ternary quantization, direction/magnitude-aware gradient estimation | Useful dense anchor and measured optimization hypothesis; separate matched training arms, never bundled together |
| Conditional | Thread/NUMA waiting and work-imbalance measurement | Actual CPU deployment bottleneck; equal workload and output-contract verification before speed claims |
| Later | Frozen/binary embedding experiment | Parameter inventory establishes embeddings matter; native random-code control, explicit pretrained-information boundary |
| Defer | Shared sparse-attention candidate searches | Long-context sparse search is a measured bottleneck; no assumed fit to the current small dense-attention model |

The shrek_GPT case motivates narrow tasks, independently tracked transformations
and honest resource accounting, not pretrained GPT-2 or movie-dialogue intake.
Its reported ~64 MB quantized artifact is not a 64 MB full runtime or a native
ternary model. Do not add its pruning/quantization recipe to native training
without the gates in the research update.

## Research references and reuse boundaries

- [BitNet official model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) and [MIT runtime](https://github.com/microsoft/BitNet): native quantized training and specialized inference must be distinguished.
- [Prism Bonsai2 announcement](https://prismml.com/news/bonsai-2-family): vendor claims, not locally reproduced benchmarks.
- [Needle official source](https://github.com/cactus-compute/needle): specialized system rather than a general reasoning oracle.
- [Qwen0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [LFM350M](https://huggingface.co/LiquidAI/LFM2-350M): optional external controls; pin exact revisions and licenses before execution.
- [Position Coupling](https://arxiv.org/abs/2405.20671) and [operand-length/count generalization](https://arxiv.org/abs/2410.15787): task-structured positional hypotheses, not guaranteed fixes.
- [Custom distributions/loss](https://arxiv.org/abs/2410.03569), [modular grokking analysis](https://arxiv.org/abs/2301.02679), [Grokking](https://arxiv.org/abs/2201.02177): prospective controlled hypotheses, not permission for unbounded training.
- [TinyStories](https://arxiv.org/abs/2305.07759): narrow distributions can support small models; story fluency is not exact reasoning.

See the [dated research assessment](research/ASSESSMENT-2026-10-22.md) for primary-source dates, detailed historical counts, license caveats, hardware estimates and access limitations. See the [subsequent-source ledger](research/UPDATE-2026-10-26.md) for later research candidates and their admission gates. No external models or papers were independently reproduced during this research.
