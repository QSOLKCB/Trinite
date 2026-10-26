> Dated supporting assessment imported on 26 October 2026. The report below is
> preserved as delivered on 22 October; its research-time status language is
> historical. For current planning use [ROADMAP-V1](../ROADMAP-V1.md) and the
> [subsequent research update](UPDATE-2026-10-26.md). No historical protocol,
> experimental outcome or release gate is replaced by this publication.

# Trinite after v0.4.0: a diagnostic route to a useful v1.0

Research date: 21 October 2026. Read-only assessment; no training, model execution, benchmark reruns or repository changes were performed.

## Executive assessment

**The evidence does not support choosing “the harness” or “the model” as a single culprit. There are distinct bottlenecks, and Trinite already contains evidence that separates some of them.**

1. **Native learning is possible.** Historical dense budget experiments passed every per-task training gate in all nine cells. The local teacher pilot also passed its tiny four-fact learning diagnostic. Therefore “the optimizer/model cannot learn anything” is contradicted by retained evidence.
2. **Native generalization and composition remain weak.** Historical dense models produced well-formed but incorrect held-out answers. More recent tiny pilots produced constant answers. These are not cured merely by stricter parsing, an evidence bundle, or an external teacher's larger parameter count.
3. **External runtime compatibility is a different engineering problem.** Needle's rejected/truncated call is evidence about its pinned adapter, controls, output budget and completion contract. It is not a fair measured comparison of model intelligence, and it does not diagnose Trinite's training loop.
4. **Experiment design is now the highest-leverage improvement.** There are many conformance lanes and meticulously retained negative results, but several latest studies have deliberately tiny training budgets and weak opportunities for teacher knowledge transfer. A short causal diagnostic ladder is more useful than adding another sophisticated donor or another model family immediately.
5. **A credible engineering v1.0 and a capable trained-model release are different deliverables.** Trinite can stabilize an inspectable, reproducible research workbench without claiming general reasoning. A model release additionally needs a narrowly specified useful task, genuinely fresh evaluation, and a measured quality/resource envelope.

Recommendation: concentrate the next development cycle on one native task and one canonical execution path, not a larger universal benchmark or more geometry interventions. Preserve v0.4.0 and historical protocols unchanged.

## Evidence baseline and limits

This assessment reads the local release-equivalent source/docs and the retained release research at immutable release commit `be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a`. Numerical source and historical documentation were unchanged by the final evidence-checker fixes. No claim below is a new independent reproduction.

Pinned source entry points:

- [Native model](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/model.py)
- [Quantizer](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/quantizer.py)
- [Training](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/training.py)
- [Current roadmap](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/ROADMAP.md), [invariants](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/INVARIANTS.md), [data policy](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/DATA.md)

### What the actual architecture permits us to say

The reference is an explicit CPU float32 causal decoder with RMSNorm, learned absolute positional embeddings, attention and a GELU feed-forward block. The output projection shares token embeddings and remains dense. Ternary linear layers construct float32 effective weights and call floating-point `F.linear`; there is no native packed ternary execution claim to make. Default recent pilot configurations are width 16, one block and feed-forward width 32, much smaller than the approximately 1.25-million-parameter reference configuration described elsewhere. Other historical studies used an 86,528-parameter dense model. Comparing those outcomes without matching configuration and exposure would be misleading.

The baseline ternary scale is matrix-wide, detached, and derived from absolute mean. Its straight-through estimator passes gradients only where the pre-rounded normalized magnitude is below one. Quantizer saturation/gradient coverage are sensible diagnostics, not established causes of the observed failures. Existing fixed-code scale-refit and learned-scale lanes should be reused as candidates rather than reinvented. Lower weight reconstruction error does not imply lower activation error or better held-out predictions.

Static reading of shifted tokens and scored-target masking did not reveal an obvious universal off-by-one bug. That is not a proof of correct training. Independent small numerical tests remain appropriate. Strict CPU validation, full-prefix decoding and repeated checking can also consume time, but overhead alone cannot explain a constant answer unless a bounded run's usable training/inference budget is affected. Profile before blaming checks.

### Results that must not get lost in the new roadmap

| Evidence | Observation | Consequence |
|---|---|---|
| Historical dense four-times-budget study | All nine final per-task training checks passed; arithmetic 264/264 in three seeds; fold 1,087/1,150, 1,114/1,150 and 1,060/1,150; lattice 210/210 in all three | Basic memorization/optimization is demonstrably possible. Do not restart from “the model cannot train.” |
| Held-out generalization diagnostics | Every dense held-out output ended with EOS and strict ASCII; fold count components 311–319/320 but complete answers 0–37/320; traversal 0/8 in every seed | There are semantic/compositional failures independent of malformed output. EOS fixes cannot explain them all. |
| Larger finite-maths budget | Increasing 4,096 to 8,192 updates improved aggregate mathematics counts, but every maths task still missed its 90% gate | More exposure can help, but naïve budget doubling was already tried and did not establish adequacy. |
| Task balancing | PR-reported modest aggregate improvement; ten maths tasks still failed; complete historical archival packet not located | Do not represent this as independently reverified success or repeat it without addressing the missing closure. |
| Frozen greedy versus syntax masking | Masking forced valid syntax, but predictions were constant and followed label prevalence | Harness constraints can improve format without improving reasoning. |
| Decimal versus sexagesimal pilot | Four 32-update cells gave constant `222` outputs and zero exact accuracy | No encoding benefit established; this bounded study is not a verdict that representation never matters. |
| Teacher operand pilot | Teacher chooses bounded operands; oracle supplies labels. Test results matched the constant-residue baseline in both teacher cells | This does not establish beneficial knowledge distillation. A large teacher was mostly doing a job the formal generator can do. |

Sources: [budget outcome](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/BUDGET.md), [generalization diagnostics](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/GENERALIZATION.md), [budget retention](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/MATHS-RETENTION-BUDGET.md), [task-balance status and archival limitation](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/docs/MATHS-TASK-BALANCE.md), [PR50](https://github.com/QSOLKCB/Trinite/pull/50), [PR60](https://github.com/QSOLKCB/Trinite/pull/60), [PR62](https://github.com/QSOLKCB/Trinite/pull/62).

## The diagnostic ladder: distinguish causes before changing architecture

These are proposed experiments, not executed results. Use a new versioned lane; leave frozen historical profiles unchanged. Development results remain exploratory, including when a recipe is selected from them.

### Step A: independently test the mechanics

Use tiny admitted, training-only examples with analytically inspectable targets. Test label shift, answer/EOS mask, causal masking, padding and reduction weighting against a separate straightforward implementation, not only the same helper invoked twice. Check that future-token changes cannot alter earlier logits; masked/padded positions do not contribute targets; one optimizer step changes the intended tensors; optimizer state and resumed updates agree within their declared scope.

Include gradient coverage and finite values by layer, clipping frequency, parameter update norms and tied-embedding behavior. For quantized runs include code occupancy, zero fraction, scale distribution and fraction of elements receiving STE gradients. Do not diagnose vanishing gradients from loss alone.

**Gate:** these mechanics pass before spending on longer runs. If they fail, correct the implementation under a new identity and preserve the failure. If they pass, do not keep rewriting the harness merely because quality is poor.

### Step B: separate next-token competence from free generation

For the same checkpoints and examples, record:

- Teacher-forced loss and accuracy at each answer digit and EOS, with each position separate.
- First-answer-token probabilities, EOS probability and rank of the correct digit.
- Complete free-running answer accuracy, invalid/unfinished rates and output-length distribution.
- Answer-body accuracy versus body-plus-EOS accuracy.
- A syntax-constrained arm only as a clearly separate decoder, never as evidence that the unmodified model learned syntax or semantics.

A two-digit answer followed by EOS has a predictable EOS frequency; loss averages can hide a weak first digit. The 259-token byte vocabulary and tied dense output head should be measured, not presumed defective. Good teacher-forced later digits with poor free generation suggests prefix sensitivity/exposure mismatch. Good format with wrong bodies suggests another bottleneck. High EOS probability at the first answer position deserves its own investigation.

### Step C: verify small-set learnability and input dependence

Run dense first on one fact, then a small balanced set containing conflicting outputs and counterfactual operand pairs. The existing four-fact diagnostic is a useful positive control, not something to discard. Require complete generation to learn more than a single shared answer. Report a confusion matrix, output entropy, class prevalence and accuracy against majority/constant/per-task baselines.

A prompt-permutation test keeps labels fixed while permuting prompts across examples. If performance is unchanged, the model may be exploiting prevalence or ignoring input. Also test a deliberate one-operand change with a different exact answer. For exact arithmetic, include an independent oracle as a correctness ceiling, explicitly marked non-neural. A memorization lookup baseline can clarify whether a split merely rewards retrieval.

**Gate:** tiny balanced training-set learning must succeed before a new capacity or quantization comparison. Failure is diagnostic, not permission to silently enlarge budgets until a passing checkpoint appears.

### Step D: bounded training-only learning curves

Only after A–C, predeclare a modest geometric budget ladder, final checkpoints and a maximum wall-time/resource limit. Record loss, train exact match, input dependence, examples/target tokens seen, actual update count and elapsed time at fixed milestones. Use the known successful historical dense configuration as one anchor; do not compare a new seven-thousand-ish-parameter pilot to an older wider model and attribute the difference to one feature.

This is a development experiment: selecting the next recipe from training curves is legitimate if disclosed, not confused with confirmation. Do not use repeatedly observed historical test outcomes to select budgets. A plateau is different from a still-falling curve. A three-point budget study can stop at a fixed bound and honestly conclude “not resolved within this budget.”

**Go/no-go:** a new complexity increase must answer an identified question. If tiny memorization fails, investigate mechanics/optimization. If train succeeds and fresh families fail, prioritize representation, curriculum coverage and inductive bias. If both improve with budget, consider a bounded extension rather than assuming architecture is wrong.

## A staged roadmap with operational deliverables

### Milestone 0: make the present state legible

Replace the top-level roadmap's chronological accretion with a short current decision tree. Keep historical chronology and every fixture in linked archival documents. Label each lane as implemented, smoke-only, executed-negative, executed-positive-with-limits, blocked, or missing archival closure. Keep the task-balanced historical evidence gap explicit. Make “next” mean one current next action, not several superseded paragraphs.

Deliver a capability/status ledger mapping claim → immutable evidence → limitation → next decision. This will reduce accidental repeated work more than another receipt schema. Reconcile prospective wording in implementation documents with later evidence documents using links rather than rewriting historical claims.

### Milestone 1: one reliable native diagnostic workflow

Build the ladder above around the authoritative native updater. Avoid another independent trainer. Provide one documented entry point for environment preflight, freeze, run, summarize and audit, with predictable `reports/` output. A fresh user should identify success, scientific failure, timeout and setup failure without searching nested directories.

**Acceptance:** mechanics tests pass; positive controls learn; negative controls expose constant-answer shortcuts; every attempted run has a usable outcome; offline verification works. This is an engineering milestone, not a reasoning claim.

### Milestone 2: choose one useful task and establish a dense reference

Choose a bounded native task already admitted, preferably one with controlled compositional variation and exact labels. Do not simultaneously demand success on every historical curriculum before learning anything about one task. Preserve the original broad gates for the claims they govern, but define a separate narrow product scope explicitly.

Split by meaningful difficulty/family dimensions: operand range, length, operation composition or graph structure. Report interpolation separately from extrapolation. Modulus 60 versus 61 changes the group, not merely number notation. Decimal versus base-60 should hold semantic instances fixed when testing notation; different moduli form a different experiment.

Use three seeds for confirmation when affordable, paired input families, and exact numerator/denominator counts. Variants of one formal problem are not independent samples. Freeze a fresh confirmation set after development and evaluate it only under the selected protocol. Model quality gates must be fit for the chosen use case; a blanket percentage cannot make a fallible neural arithmetic engine exact.

**Acceptance:** predeclared per-task quality, clear margin over constant/majority controls, measurable input dependence, and a bounded CPU cost envelope. If these fail, the model remains a research result, not a ready utility.

### Milestone 3: compare ternary methods only after a useful dense anchor

Run matched dense, current absmean ternary and guarded fixed-code refit first. Add learned scales or an STE variant only if diagnostics justify another arm. Match data, initialization policy, optimizer, update/target-token budgets, decoding and evaluation families; disclose unavoidable differences. Inspect code sparsity and saturation alongside quality.

Separate three questions: (1) reconstruction error, (2) trained task quality, (3) actual storage/runtime benefit. Trinite currently has evidence for aspects of (1), not a general success in (2) or packed-CPU speed in (3). A five-trit codec's exact round trip is not an optimized model runtime. Export requires tensor/scaling/exception inventory and parity before timing.

**Acceptance:** a candidate meets a predeclared quality-loss tolerance relative to the dense reference and demonstrates a real selected deployment benefit. Otherwise retain the negative result and ship the dense reference if that is the useful system.

### Milestone 4: use teachers only where they add information

The current teacher proposes `b` for prescribed `n,a`, with oracle labels. Before expanding teacher size comparisons, compare that proposal distribution with simple balanced/exhaustive or difficulty-stratified exact generators. A teacher cannot add new correct arithmetic facts beyond the oracle; it could add useful coverage, selection, or pedagogical structure, but those hypotheses need explicit controls.

Potential new lane: teacher proposes structured task compositions or bounded solution programs; an independent interpreter checks every intermediate/final step and the repository renders canonical examples. This requires a separately reviewed admission expansion. Existing approval/policy admits bounded numeric operands, not arbitrary teacher prose or chain-of-thought. Do not silently widen it.

Use matched accepted-data size, coverage and training exposure; retain duplicates, rejection/underfill and generation cost. Distinguish teacher-selected examples from teacher-written explanations and genuine soft-target distillation. Small Qwen/LFM or Needle comparisons belong in a separate external model feasibility/capability track and need not block native improvement.

### Milestone 5: stable CPU packaging and a declared GPU backend

Trent's Ryzen 5950X, 32 GB RAM and RTX 5060 Ti 16 GB are sufficient to start the bounded proposed diagnostics, subject to measured runtime. Do not promise a particular training speed. Keep the existing CPU reference as the numerical control. A GPU student backend needs its own environment, parity/tolerance and replay scope; converting devices is not the same historical experiment.

Measure time spent in numerical work, validation, serialization and replay separately. Cache only things proven unchanged within the declared execution boundary. Benchmark end-to-end time as well as kernels. External teacher runs can be sequential with CPU student work to simplify memory ownership. There is no Trinite need for Vast/H200 in this plan.

Package a versioned installable interface, offline inspector, safe checkpoint importer/exporter, minimal dependency profiles, supported Python/platform matrix and actual memory/latency measurements. Resolve the distribution-version versus research-release-tag distinction explicitly before v1.

### Milestone 6: independent scoped reproduction and release

For a research-workbench v1: require a clean installation, successful documented diagnostic workflow on a second supported environment, an offline-verifiable public example packet, explicit reproducibility scope, stable schemas/compatibility policy, complete licenses/provenance and a concise limitations page.

For a trained-model v1: additionally require a useful selected task, held-out confirmation, quality/resource envelope, an exported checkpoint with declared quantized/dense tensors, and inference parity. If the deployed product uses an exact oracle or tool router, report system performance separately from unaided model performance. That can be a valuable product without a false model-reasoning claim.

Do not make success of every optional geometry, NEXUS, E8, NCM or external-teacher lane a prerequisite for a stable research-tool release. Changing existing global release gates must be an explicit reviewed project decision, not a quiet deletion.

## What to simplify without weakening evidence

- Preserve immutable old packets, negative outcomes, source bindings and per-item lineage.
- Use fast local exploratory diagnostics with honest status; reserve expensive full optimizer replay/independent archival acceptance for milestone candidates and published claims under the applicable new protocol. Do not retroactively downgrade old verification requirements.
- Reuse one audited evidence/publication library rather than proliferating nearly identical launchers and status-handling bugs. Refactor only new/current execution boundaries with compatibility tests; historical exact source remains retained.
- Bind the minimal complete executable closure for a new lane. Binding unrelated documentation or every source file can cause unnecessary request invalidation, but excluding a true dependency would be worse.
- Keep adversarial evidence checks proportionate to the declared trust model. Hashes establish bytes, not authenticity of GPU execution or scientific validity. A local trusted-host experiment and an attacker-controlled submission verifier need not incur identical complexity.
- Separate parser compatibility from task correctness. An unsupported option or extra runtime metadata is a feasibility finding; a truncated, low-confidence response is not made correct by relaxing validation. Prospective adapter revisions are legitimate, with old failures preserved.

## Relevant research, and what it does not prove

1. [Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv, 6 January 2022](https://arxiv.org/abs/2201.02177) reports delayed generalization on small algorithmic datasets. It motivates observing training/generalization curves and bounded optimization budgets. It does not promise Trinite will grok, justify unbounded retries, or make reused test sets fresh.
2. [Cho et al., *Position Coupling*, arXiv, 31 May 2024](https://arxiv.org/abs/2405.20671) reports strong arithmetic length generalization with task-structured positional identities and gives a corresponding expressivity argument for addition. A separately versioned positional-encoding comparison may be more informative than merely increasing width once basic learning works. Its reported 200-digit setting is not directly transferable to Trinite's byte model or varied tasks.
3. [Cho et al., *Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count*, arXiv, 21 October 2024](https://arxiv.org/abs/2410.15787) studies task-specific scratchpads and multi-level position coupling. This supports testing structured computation representations, not indiscriminate teacher prose. Earlier procedural or scalar curricula in Trinite already have negative results, so a new arm must state its actual difference.
4. [Eldan and Li, *TinyStories*, arXiv, 12 May 2023](https://arxiv.org/abs/2305.07759) reports coherent output from very small models trained on a deliberately simplified synthetic distribution. It supports the importance of task/data distribution, not a claim that a tiny Trinite pilot should already reason broadly. The paper uses GPT-4-based evaluation; that is not an exact reasoning oracle. Dataset admission is a separate decision under Trinite's policy.
5. [Weiss, Goldberg and Yahav, *Thinking Like Transformers*, 2021](https://arxiv.org/abs/2106.06981), with [author RASP implementation](https://github.com/tech-srl/RASP), provides a useful way to articulate computational structure and expected depth/head requirements. A symbolic/compiled baseline can clarify task representability, but is not evidence that gradient descent discovers that algorithm.

These sources are used for bounded hypotheses, not as evidence that the recommended changes have already improved Trinite. Repository results remain the strongest evidence about the present implementation.

## Recommended immediate PR sequence

1. **Roadmap/status consolidation only:** current claim ledger, two v1 scopes, archival-gap note, one next diagnostic milestone. No new numerical results.
2. **Native learning diagnostic ladder:** independently checked mechanics, teacher-forced/free-running separation, input-sensitivity controls and bounded training-only curve protocol. Reuse the existing positive control/updater.
3. **Evidence-only local CPU diagnostic run:** retain all results, including failure. Choose the next intervention from identified failure mode rather than enthusiasm for a new model.
4. **One justified intervention:** optimization, task representation, capacity or quantizer, not all simultaneously. Fresh confirmation only after development.
5. **CPU release workflow and export:** make the useful result usable, then independently reproduce it.

A smaller coherent v1 with one demonstrated useful behavior and explicit limits is stronger than a larger collection of flawless conformance receipts around models that still output constants.

---

# Trinite: ternary and tiny-model research for a credible v1.0

Research date: 21 October 2026. Research only: no models downloaded, executed or installed; no repository changed. Hardware assumptions: Ryzen 5950X, 32 GB system RAM, RTX 5060 Ti with 16 GB VRAM. Source inspection is not independent reproduction of benchmark claims.

## Executive recommendation

For a trained-model v1.0, the proposed standard is an inspectable native model that demonstrably learns a bounded task, generalizes under a frozen split, and ships with reproducible inference and honest limitations. A workbench-only v1.0 has a separate engineering gate, as explained above; neither scope is adopted by this research. It need not mean a miniature general-purpose assistant.

Do not replace the model merely because a short recent pilot collapsed, and do not expand the harness indefinitely because it is easier to test than learning. Historical dense controls already passed training gates under a larger budget; the important unresolved problem is held-out generalization. Recent constant-output pilots and external model interface failures are different observations, not one diagnosis.

Prioritize one controlled learning experiment on a previously memorized but poorly generalized task. Borrow data-distribution and representation ideas before importing a new architecture. Run external tiny models as calibration instruments, not as substitutes for native success. Keep packed ternary deployment as a separate engineering milestone.

## What the external models actually contribute

### Microsoft BitNet: strongest reference for the training/runtime boundary

The official model is native quantization-aware training, not ordinary post-training ternarization. Its model card describes ternary weights, 8-bit activations, absmean weight scaling, and 4 trillion pretraining tokens. It explicitly warns that standard Transformers execution does not deliver the specialized speed or energy benefits. The technical report was submitted 16 April 2025 and revised 25 April 2025; arXiv labels it work in progress. Its quality comparisons remain author-reported. Sources: [model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), [technical report](https://arxiv.org/abs/2504.12285).

The [MIT-licensed runtime](https://github.com/microsoft/BitNet) documents x86 CPU kernels and later CPU optimizations, including a 15 January 2026 update. This is a useful deployment donor, not evidence that replacing a Trinite linear layer will confer its published speedups. A narrow future experiment should compare a frozen dense reference with an exported, packed ternary implementation for actual output agreement, resident memory and latency on Trent's CPU.

A [23 September 2026 issue](https://github.com/microsoft/BitNet/issues/632) reports differences among published BF16 master weights, packed trits and GGUF scales. This is an unresolved reporter finding, not independently confirmed here. It reinforces a practical requirement: pin the exact representation and conversion path; do not assume differently packaged checkpoints are byte- or numerically equivalent.

**Use:** reference architecture/quantizer study and later packed-kernel conformance. **Do not claim:** BitNet quality or speed transfers to Trinite; a two-billion-parameter pretrained model diagnoses a tiny from-scratch learner.

### Prism Ternary Bonsai 2: paired external comparison, not a drop-in quantizer

Prism's 15 October 2026 announcement describes Qwen3.5-derived 2B, 4B and 9B ternary models and vendor benchmark comparisons. It reports both strengths and losses relative to corresponding Qwen models. These are not independent local results. [Announcement](https://prismml.com/news/bonsai-2-family).

The [2B GGUF card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) is Apache-2.0. Preserve the already audited model/runtime pins from Trinite rather than substituting generic Hugging Face-generated installation examples. The earlier malformed local output remains negative evidence for that exact configuration, not proof that all Bonsai outputs fail.

**Best experiment:** same formal inputs, bounded generation and independently checked answers for Qwen3.5-2B versus pinned Bonsai 2B. Report raw output, schema acceptance, correctness, truncation, memory and latency separately. This compares products/configurations, not the causal effect of ternarization: training and runtime also differ. Do not mix vision weights into a text-only budget.

**Use:** ternary deployment behavior and an externally matched-size comparison. **Avoid:** another unbounded compatibility project before the native learning milestone.

### Needle: valuable specialization lesson, currently not a general reasoning teacher

The [official repository](https://github.com/cactus-compute/needle) is Apache-2.0 and describes a 121M-parameter specialized architecture, schema-constrained decoding, tool calling, extraction and confidence estimation. It calls the model 2-bit, not ternary. Its description explicitly trades general chat capacity for specialized tasks. Its headline size is not a substitute for measuring the exact selected file and engine.

The transferable idea is specialization: a fixed vocabulary of operations and a narrow task can support small deployment. A learned confidence head and grammar are parts of the system, not independent evidence of semantic correctness. Trinite's retained unsuccessful call must not become a success by relaxing the acceptance rules retrospectively.

**Use:** a separate future typed extraction or tool-selection comparison only if native controls and budget behavior are understood. Use capture-only paths, never a convenience API that executes proposed tools. **Do not use:** as ground-truth oracle for modular arithmetic or as evidence that Trinite needs Needle's architecture.

### Qwen: practical small external control

[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) carry Apache-2.0 model cards. Both expose explicit thinking-mode control and are non-thinking by default; the cards warn of possible repetition in thinking mode. Soft `/think` switches are not supported. Pin the actual chat template and setting rather than relying on prompt prose.

Start with 0.8B if the goal is a cheap external competence control. Use 2B specifically for comparison with Bonsai, not because more parameters guarantee better formal results. Keep generation settings and any constrained-decoding arm explicit. A greedy protocol is a new local experiment, not reproduction of a vendor's sampled benchmark.

### Liquid LFM: architecture contrast, with a different license

The [LFM2.5-1.2B-Instruct card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) describes a 1.17B hybrid convolution/attention model. It offers a useful architectural contrast to Qwen but its vendor throughput claims are not predictions for the 5950X. [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) and [LFM2-700M](https://huggingface.co/LiquidAI/LFM2-700M) are smaller candidates if size itself is the question.

These cards use **LFM Open License v1.0**, not MIT/Apache. Its [license text](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/raw/main/LICENSE) contains a commercial-use revenue threshold and redistribution conditions. Treat the model as a separately licensed optional dependency; check the exact pinned license before distribution. This research does not accept new terms or offer legal clearance.

**Use:** one LFM arm after Qwen, not all sizes at once. Suggested initial choice is 350M for a genuinely small contrast, or 1.2B if the aim is stronger teacher output. Do not infer architecture effects from differently pretrained products.

### SmolLM2: a less exotic tiny baseline

[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) is an Apache-2.0 text-generation checkpoint with conventional Transformers support. It is above 100M parameters, but makes a useful modest external reference without Needle's unusual task API. Its availability is not evidence of competence on Trinite's tasks. One pinned raw-versus-schema comparison could reveal whether an observed tiny-model failure is peculiar to a specialized interface.

## Sub-100M learning lessons worth testing

### Narrow the learning target rather than growing a general assistant

[TinyStories](https://arxiv.org/abs/2305.07759), submitted 12 May 2023 and revised 24 May 2023, reports coherent story generation below 10M parameters using a restricted synthetic corpus. The authors use GPT-4-based grading; this does not establish exact arithmetic or compositional generalization. The useful inference for Trinite is narrow, controlled data complexity, not importing teacher prose as verified mathematical truth.

[nanoGPT's character-training configuration](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py) offers a readable small dense reference: six layers, width 384, context 256. Its [code license is MIT](https://github.com/karpathy/nanoGPT/blob/master/LICENSE). Borrow an inspectable reference implementation for differential debugging only when a specific discrepancy exists. Do not restart a broad model rewrite or interpret its Shakespeare recipe as an arithmetic recipe.

### Task-structured data and loss, with separate ablations

[Making Hard Problems Easier with Custom Data Distributions and Loss Regularization](https://arxiv.org/abs/2410.03569) describes modular-arithmetic improvements using a mixed easy/hard data distribution and a problem-specific regularizer. The arXiv record says submitted 4 October 2024, v2 on 25 August 2025; the HTML body contains an inconsistent 2026 date, so use the archive version date and flag the discrepancy. Author claims are not reproduced here. [Full text and linked source](https://arxiv.org/html/2410.03569v2).

The useful prospective Trinite test is **not** a cryptographic attack. Hold the native architecture and optimizer fixed on a bounded modular task that already fits training examples. Compare uniform sampling against a predeclared mixture of low-wrap and high-wrap examples, keeping unique-example and token budgets visible. Then, separately, consider a circular output representation/loss. Never change data, representation and loss together and attribute the result to one. Exact integer/rational oracles must still judge final answers; a circular metric must not turn a near miss into an exact success.

[Gromov's Grokking modular arithmetic](https://arxiv.org/abs/2301.02679), submitted 6 January 2023, studies a two-layer network with interpretable features and provides analytic solutions for certain modular tasks. This supports a small task-specific diagnostic reference. It does **not** show that byte-autoregressive generation with Trinite's settings should behave identically. Use it to separate representational learning from serialization, not to replace the claimed task silently.

## Proposed experiment sequence

1. **Reuse the strongest existing dense training control.** Select one historically memorized task with a documented held-out failure. Retain that baseline and its actual settings, rather than announcing a generic budget increase as new work. Verify the selected lane still executes unchanged.
2. **One-variable curriculum ablation.** Compare uniform versus bounded structured sampling with frozen test families, matched token/update exposure and at least three prospective seeds if affordable. Include constant-output, lookup and shuffled-input controls. Predeclare whether the split is new combinations, length extrapolation, or new task family; these are not interchangeable.
3. **Representation ablation only if informative.** Compare byte generation with a typed/residue head as separate model IDs. This asks whether serialization impedes learning. Do not credit a head given symbolic structure as discovery of that structure.
4. **Matched dense/ternary training.** Only after a useful dense baseline, compare native absmean, guarded fixed-code refit and any learned-scale option with architecture, data, initialization policy and training budget matched. Weight reconstruction improvement is not enough; measure train and held-out answers, calibration and loss.
5. **External calibration.** Use Qwen0.8B plus one LFM or SmolLM2 arm first. Add Qwen2B/Bonsai only for the specifically paired comparison. Keep teacher proposal acceptance and student improvement separate.
6. **Packed deployment.** Prove export and kernel conformance before measuring CPU memory/latency. Floating `F.linear` with ternary-valued tensors is not packed ternary execution.

## Hardware and measurement boundaries

Sequential inference is preferable to keeping teachers resident together. Approximate BF16 weight-only lower bounds are 1.6 GB for 0.8B, 4 GB for 2B and 2.34 GB for 1.17B parameters; these are arithmetic estimates, not full memory requirements. KV cache, auxiliary weights, runtime workspace and loading peaks are additional. Use deliberately short contexts for formal tasks, not advertised maximum windows.

For native training, float32 parameters, gradients and two Adam moments alone use approximately 16 bytes per parameter: 10M means roughly 160 MB, 100M roughly 1.6 GB before activations and other buffers. This suggests small from-scratch experiments are plausible on 16 GB, but does not establish wall time or maximum size. Measure one authorized bounded run before choosing a bigger experiment.

Record process RSS, peak allocated and reserved GPU memory where available, device/runtime versions, input/output bytes and tokens, prefill and decode separately, warm-up policy and repeated-run variability. Never compare vendor tokens/s across different tokenizers as if they were identical work. Exact same-machine replay is a valuable claim; bitwise cross-device equality requires its own evidence.

## Evidence and release discipline

A v1.0 gate should require a useful held-out result above a declared trivial baseline, uncertainty across seeds, executable clean setup, a stable inference interface, and source/data/checkpoint provenance. Do not make general chat, an arbitrary parameter count, or a cloud GPU rental a gate. Preserve negative evidence and limit release claims to what the retained runs establish.

All vendor/model-card benchmark statements above remain vendor claims. Papers are cited as arXiv manuscripts unless a venue was verified; no independent reproduction was conducted. The Prism documentation endpoint was inaccessible during this pass, so its official announcement and Hugging Face card were used instead. Exact historical Trinite metrics should be quoted from its immutable evidence packets, not from this external-source survey.

---

# Trinite: a bounded-integer path, without confusing storage with execution

Research checked 21 October 2026. Scope: roadmap research, not an implemented backend or measured speed/quality result. No model code was executed, installed, or changed.

## Decision

Yes, a genuinely integer-only inference path is a reasonable research target. No, casting Trinite to int8 will not accomplish it. Prefer **ternary weights where appropriate, int8 or int16 activations, and explicitly bounded int32/int64 intermediates**. “Integer-only” need not mean “every value has eight bits.”

Keep the existing float32 model as an immutable reference. Define a separately versioned fixed-point numeric model and backend. Its rounding, saturation, scaling, approximation and overflow rules change the numerical computation. It is not automatically bitwise equivalent to the floating model, and does not inherently improve reasoning. Integer-only training is a substantially different, later research project, not a prerequisite for a useful v1.0.

## What the immutable release actually does

Inspected v0.4.0 commit be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a through the GitHub connector:

- [quantizer.py](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/quantizer.py): returns int8 codes, but reconstructs effective float32 weights with a detached floating scale. Scale reduction even uses float64 before conversion to float32. Its surrogate gradient is explicitly not the derivative of the quantized forward.
- [model.py](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/model.py): uses floating F.linear, RMSNorm/rsqrt, attention softmax, exact GELU, embeddings and tied output projection. Inputs are integer token IDs, not an integer arithmetic network.
- [training.py](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/training.py): requires the pinned CPU float32 environment and AdamW, with floating cross-entropy, gradient clipping and optimizer updates.

Therefore “we already store ternary codes” and “we have an integer runtime” are different milestones. Changing this lane in place would also invalidate its historical numeric and replay contract.

## Useful precedents and limits

### I-BERT: nonlinear operations can be approximated with integers

[Kim et al., ICML 2021, I-BERT](https://proceedings.mlr.press/v139/kim21d.html) describes integer-only inference with approximations for GELU, softmax and LayerNorm, evaluated on RoBERTa/GLUE. This establishes a research precedent, not Trinite compatibility: Trinite uses RMSNorm and a causal decoder. The paper's T4 speed results must not be transferred to a Ryzen 5950X or RTX 5060 Ti.

The [author repository](https://github.com/kssteven418/I-BERT) is identified as MIT licensed. Its [quant_modules.py](https://github.com/kssteven418/I-BERT/blob/ibert/fairseq/quantization/utils/quant_modules.py) is particularly instructive: it retains floating scaling factors and scaled tensor outputs, and its normalization includes a comment that torch.sqrt should be replaced by an integer-sqrt kernel. Thus the released Python simulation is **not by itself proof of a float-free executable**. Pin and audit any actual borrowed implementation separately, preserving notices. Do not merely rename an operation IntSoftmax and certify deployment.

### gemmlowp: explicit requantization instead of floating scales at runtime

Google's [quantization design document](https://github.com/google/gemmlowp/blob/master/doc/quantization.md) explains zero points, integer accumulation and fixed-point multiplier/shift output stages. This is a useful design reference for turning a real-valued scale ratio into a frozen integer multiplier plus shift. Real-valued calibration/export can remain outside deployment, provided that boundary is stated. Asymmetric zero points require their own correction and intermediate bounds; they are not free bookkeeping.

### Integer training exists as research, but is not an AdamW dtype toggle

[Wu et al., WAGE, first submitted 13 February 2018](https://arxiv.org/abs/1802.04680) studies discretizing weights, activations, gradients and errors using shifts and low-bitwidth representations. The [ICLR 2018 paper PDF](https://openreview.net/pdf?id=HJGXzmspb) is a relevant precedent, not evidence that this exact transformer can train equally well under those rules. The OpenReview forum page was browser-challenged during this research; the arXiv abstract was accessible. An implementation-level all-operations audit was not completed for WAGE.

[Dettmers et al., 8-bit Optimizers, version 2 dated 20 June 2022](https://arxiv.org/html/2110.02861v2), explicitly dequantizes optimizer states to 32-bit for the update, then requantizes for storage. It can save memory, but does not eliminate floating point. Its results should not be presented as integer-only training.

## Proposed numerical contract

The following is a proposed design, not a result reported by those papers.

1. **Declare scope:** “Integer-only forward inference from serialized integer parameters and integer token IDs to integer logits/token IDs.” Separately identify floating training, calibration, export and optional evaluation. A floating evaluator must never feed the authoritative deployment path. A stronger claim covering all training/export is a separate goal.
2. **Represent scales explicitly:** use bounded integer multipliers and binary shifts, or a documented fixed-point format. Serialize scale parameters, rounding rule and format version. No runtime float scale, reciprocal, logarithm or epsilon sneaks through a helper.
3. **Round deliberately:** specify ties-to-even or another fixed rule, negative division/right-shift semantics and saturation. Do not rely on language-dependent signed shifts or undefined signed overflow. Unexpected overflow should reject rather than silently wrap. Intended clipping must be counted and reported.
4. **Quantize all remaining parameters:** token/position embeddings, tied output, RMSNorm gains and constants need a declared format as much as ternary linears do. Preserve actual tying, not two independently calibrated copies.
5. **Make masks structural:** Boolean attention masks exclude entries before integer normalization. Do not encode negative infinity as an arbitrary finite minimum that could contribute probability mass.

### Accumulator bounds are more than a GEMM check

For a dot product of length D with both operands restricted to [-127,127], the absolute sum is at most 16,129 D. A zero-bias signed int32 accumulator therefore admits D at most 133,144 under that conservative bound. If both operands may include -128, a term can equal 16,384, reducing that universal positive bound to D at most 131,071. These are mathematical worst-case bounds, not measured workloads.

For ternary {-1,0,1} weights and symmetric int8 activations the corresponding bound is 127 D; allowing -128 requires 128 D. Neither bound covers later rescaling. The negation of -128 itself must be widened before evaluation.

Prove bounds for the whole expression tree: bias additions if introduced, residual sums, zero-point corrections, multiply-before-shift requantization, squared normalization inputs, reciprocal iteration products, exponent approximation polynomials, attention sums and attention-weighted values. A safe dot product followed by an overflowing int32 scaling multiplication is unsafe. Use wider intermediates where proved sufficient; otherwise constrain shapes/scales or reject the plan. Test preallocation budget checks as well as arithmetic bounds.

### RMSNorm, softmax and GELU

- RMSNorm needs bounded sums of squares, an explicitly represented nonzero epsilon and an integer square-root/reciprocal-square-root procedure. Specify the rounding at each step, iteration/domain bounds and zero-vector behavior. I-BERT LayerNorm cannot be substituted unchanged because mean subtraction changes the function.
- Attention softmax needs a deterministic integer exponential approximation or lookup, maximum subtraction, bounded summation and normalization. A fixed probability mass S can be distributed as nonnegative integers summing exactly to S with an explicit remainder/tie rule. That is a chosen approximate distribution, not exact real softmax. Ensure excluded entries remain zero, a valid row never has a zero denominator, and accumulation precision survives long contexts.
- GELU requires a bounded polynomial or lookup with documented approximation error on the supported domain. Replacing GELU with ReLU is an architecture intervention, not faithful quantization; label and test it separately.
- Greedy final-token selection can use integer argmax without a final probability softmax, with a frozen tie rule. Attention still needs its normalization. If probabilities, sampling or likelihoods are later exposed, specify their new semantics; do not claim floating-baseline probability calibration merely because token choices agree.

## Staged roadmap and acceptance gates

### A. Bounded arithmetic conformance, independent of learning

Create an isolated integer reference with machine-width checks and an independent arbitrary-precision integer/rational oracle. Exhaust small domains and add boundary fixtures for signed extrema, rounding ties, negative shifts, all-zero tensors, invalid divisors and saturating inputs. Verify integer linear/requantization first, then each nonlinear primitive. An unbounded Python integer implementation is useful as an oracle, but does not itself demonstrate bounded machine arithmetic or speed.

Gate: all accepted plans have derivable range bounds; invalid plans fail before unsafe operations; every optimized primitive exactly matches the declared integer reference. Preserve source-bound fixtures and negative outcomes.

### B. Frozen checkpoint integer inference

After establishing a useful floating checkpoint, freeze a new export/backend ID, scales and calibration set. Calibration uses admitted training/development inputs only, never held-out answers. Compare original floating logits/outputs, quantized simulation and authoritative integer execution separately. Run causal/masking, batch-padding invariance, malformed-state and replay tests. Instrument actual operator/dtype execution so a hidden float fallback fails acceptance.

Gate: exact agreement between integer reference and deployed implementation; separately preregister tolerances and task-quality criteria versus floating inference. Layerwise approximation error, clipping frequency, changed argmax decisions, exact-answer accuracy and out-of-distribution cases all matter. Passing an arithmetic oracle does not establish reasoning quality. Freeze selections before fresh confirmation data.

### C. Local performance only after correctness

Start on the 5950X with the simple integer reference and a compiled CPU candidate. Measure warm/cold latency, throughput, thread count, memory and export overhead on fixed batch/context shapes against the existing CPU float32 baseline. A pure Python integer oracle will not be a credible optimized-kernel benchmark.

The RTX 5060 Ti can be a later backend experiment, but verify runtime/kernel compatibility and integer-only operator coverage first. Small models may be dominated by launch, transfer, unpacking and nonlinear overhead. Report GPU-resident and end-to-end timings separately, synchronize timing correctly, and retain failures. Neither low storage bits nor hardware integer support guarantees speedup. No new datacenter cards are needed for these feasibility stages.

### D. Integer-only training, optional later branch

If inference quality and portability justify it, design a new training algorithm/ID. Gradients, clipping norms, loss, AdamW moments, bias corrections, square roots and tiny updates all need replacement or higher-resolution fixed-point states. Plain int8 parameter updates can round useful learning steps to zero. Stochastic rounding, if chosen, requires a pinned random stream and replay state. Test bounded SGD/momentum first as a deliberately different optimizer, not as an equivalent AdamW implementation. Keep FP baseline training and held-out evaluation intact.

## Placement in v1.0

Make a numeric-specification and conformance milestone available early, but keep full integer inference behind a demonstrated useful checkpoint. Treat full integer training as exploratory rather than blocking v1.0. This separates three questions: can the model learn, can a lower-precision numeric model preserve useful behavior, and can an actual runtime deliver a resource benefit? A single cast or combined experiment cannot answer all three.

---

# RSH donor and native capacity research for Trinite
Prepared 21 October 2026 UTC for the 22 October morning roadmap. Read-only source review; no repository modification, model training, donor execution or Lean build was performed.

## Decision in brief

Both ideas are useful, but keep them separate. RSH supplies an unusually good **exact rational geometry conformance donor**, not evidence that geometrical regularization improves a neural model. A **three-size native dense comparison** is a worthwhile diagnostic once the learning mechanics and useful-task baseline are fixed. Neither should introduce integer-only execution, a new curriculum and a new architecture simultaneously.

Trinite baseline: [v0.4.0 source](https://github.com/QSOLKCB/Trinite/tree/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a). RSH reviewed head: [4a05bedb9f10bb5d331006e37373f41aa805582b](https://github.com/QSOLKCB/RSH/tree/4a05bedb9f10bb5d331006e37373f41aa805582b), identified in its README as release 4.1.0.

## 1. What RSH actually offers

RSH separates its canonical bounded Frenet–Serret geometry, floating numerical accelerator research, tissue simulation, evidence governance and exact theorem surfaces. Its [scientific boundary](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/docs/SCIENTIFIC_BOUNDARY.md) explicitly does not establish physical mechanisms, consciousness or agency. Preserve this distinction rather than treating geometry or biological terminology as a model-quality claim.

### First donor: rational inversion/reflection

The strongest narrow match to Trinite is [ExactSpatial.lean](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/formal/lean/RSH/ExactSpatial.lean), with the corresponding [Python integer/rational witness](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/scripts/f32_sierpinski_inversive_witness.py) and [normative witness specification](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/research/f32-sierpinski-inversive-witness/SPEC.md).

For a rational centred vector c and one of three component-swapping reflections P:
- T(c) = 6 P(c) / ||c||², excluding c = 0.
- T(T(c)) = c for the same reflection.
- ||T(c)||² = 36 / ||c||².
- With r²(c) = ||c||² / 18, r²(c) r²(T(c)) = 1/9.
- Barycentric-to-centred conversion and its inverse are exact rational round trips.

The Python path uses normalized integer numerators and a positive denominator. Given barycentric numerators n and denominator d, q_i = 3n_i − d, S = Σq_i², the transformed numerators are S + 6d q_permuted and denominator 3S, followed by gcd reduction. It rejects the singular centre. Negative barycentric numerators are legitimate outside-triangle coordinates, not automatically bad data. The donor witness also requires the barycentric numerators to sum to the denominator, equivalently centred x+y+z=0. Its normalization rejects a wrong sum. The Lean ExactSpatial vector identities cover arbitrary rational 3-vectors; that broader theorem domain does not waive the planar barycentric witness contract. Include a wrong-sum regression rather than silently admitting every theorem-valid vector.

This permits independent implementation checks: a small Fraction-based vector oracle against the donor's normalized-integer formulation, pinned donor fixtures, and explicit singularity/type/canonicalization tests. The theorem source is mathematical support, but this review did not execute its build. Do not label a new Python port “Lean verified”: the donor expressly does not prove IEEE-754 or implementation refinement.

A useful first PR would contain:
1. A pinned, license-labelled donor source closure or clearly attributed narrow adaptation, plus source manifest.
2. A standalone exact adapter with an independent arithmetic oracle; no model import, training or geometry loss.
3. Per-item source revision, generator version, input/fibre, seed/split procedure if used, verifier result and exact output identities.
4. Boundary and mutation regressions: zero centre, each reflection, negative numerators, reducible fractions, denominator sign/zero, bool/float rejection, malformed canonical integers and bounded input sizes.
5. Both round-trip and independent forward checks. A wrong identity function can pass a round-trip test, so involution alone is insufficient.
6. An evidence packet and limitation ledger. Claim only donor conformance.

The source uses arbitrary-precision Python integers, not int8. Intermediates include squares and denominator products. Any bounded-integer port needs explicit overflow bounds/rejection; this does not magically make the neural model integer-only.

### Secondary donors, later

- Exact uniform-floor logical-to-rendered sampling/index bounds from [GLUBALL Sampling.lean](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/formal/lean/RSH/Gluball/Sampling.lean) can inform bounded sampling tests. The imported geometry has its own GLUBALL provenance and must retain that chain.
- The word-capacity embedding 2^32 < 3^21 is exact, but it is an address encoding, not compression or a learned ternary representation. Twenty-one trits need more than 32 bits of information capacity.
- [Reference-spine admission](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/src/rsh/reference_spine.py) is a separate analytic/floating-point surface. Closed-form geometry evaluated with exp, sin, cos and binary64 is not an exact integer donor.
- Full Frenet integration, parallel scans and tissue metrics may be valuable numerical workloads, but they are poorer first candidates for the exact Trinite curriculum. They introduce tolerances, larger dependencies and independent interpretation questions.

If later used for learning, require fresh composition/coordinate-family holdouts. Do not train and test on opposite ends of the same involution or coordinate permutations without family grouping. Compare with exact algorithms and simple symbolic baselines. A model reproducing an inverse map is not evidence of broad geometric reasoning.

### Rights and attribution

RSH's checked [LICENSE](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/LICENSE) is MPL-2.0; the witness file has an explicit MPL header. Preserve applicable covered-file licensing, notices and source availability when adapting/distributing those files. [NOTICE.md](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/NOTICE.md) credits J. Robitaille (DeltaKingZero) and Trent Slade / QSOL-IMC, with differentiated contributions. Preserve this attribution rather than treating common organizational ownership as erasing it. [FORMALIZATION.md](https://github.com/QSOLKCB/RSH/blob/4a05bedb9f10bb5d331006e37373f41aa805582b/FORMALIZATION.md) distinguishes human authors from AI proof-engineering contribution and states proof limits.

## 2. Which “current model” should be enlarged?

There is no single current parameter count. Source-derived counts from [ModelConfig.parameter_count](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/contracts.py) are:

| Role | Blocks / width / FF / heads / context | Unique parameters |
|---|---|---:|
| Tiny-training file | 1 / 16 / 32 / 2 / 32 | 6,752 |
| Recent teacher/cyclic pilot | 1 / 16 / 32 / 2 / 64 | 7,264 |
| Historical foundations anchor | 2 / 64 / 128 / 4 / 64 | 86,528 |
| Reference configuration | 6 / 128 / 512 / 4 / 256 | 1,247,232 |

Counts include tied embeddings once. Formula is B(4d² + 2df + 2d) + Vd + Cd + d. These are arithmetic derivations from pinned source, not instantiated-model measurements.

Sources: [tiny config](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/configs/tiny-training.json), [teacher student](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/scripts/teacher_intake_v1/student.py), [foundations plan](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/foundations_plan.py), [reference config](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/configs/reference.json).

### Recommended main ladder

Use the historically learnable 86,528-parameter dense configuration as the anchor. Keep two blocks, four heads, context 64, byte vocabulary 259, FF-to-width ratio2, dense lane, geometry_loss=0 and the same numerical implementation.

| Candidate | Width | FF width | Parameters |
|---|---:|---:|---:|
| S | 64 | 128 | 86,528 |
| M | 96 | 192 | 178,944 |
| L | 128 | 256 | 304,128 |

This is a width-scaling experiment, not “parameter count alone” in an architecture-independent sense. Head dimension necessarily changes (16/24/32); holding head dimension instead would change head count. State that choice and keep depth fixed. Do not silently introduce μP, new activation, position scheme or quantizer.

An optional cheap diagnostic tied literally to the recent pilot is widths 16/32/64 at one block, two heads, context 64 and FF = 2 × width: **7,264 / 18,624 / 53,632** parameters. Use this only to answer the tiny-pilot question, not to discard the known historical 86k anchor. Avoid running both ladders merely to accumulate results.

“Clone twice” should mean same implementation/protocol with distinct configuration, model and checkpoint identities, freshly initialized. Resized weights are not identical checkpoints. Same RNG seed across dimensions does not yield the same weights or matched random draw semantics. Record actual initialization identities and policy; do not invent exact paired-initialization claims.

Current [FoundationsPlan](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/foundations_plan.py) rejects models above 100,000 parameters, and [BudgetPlan](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/budget_plan.py) delegates unchanged validation to it. The larger ladder therefore needs a new bounded capacity protocol, not loosening historical contracts in place.

## 3. Fair experiment and interpretation

### Primary question: fixed exposure

“Under one declared native dense recipe and equal data exposure, does widening improve fresh compositional task accuracy?”

Freeze the same training items, byte representation, order, batch/accumulation, loss masks, answer/EOS targets, number of updates and total supervised target tokens. Also record total input/context tokens because loss-masked target tokens alone omit attention work. Keep raw greedy decoding and answer-plus-EOS acceptance fixed; no solver, syntax repair or teacher changes. Use one established task, not a simultaneous RSH curriculum addition.

Use a bounded development stage to check training curves and optimization. Compare the fixed baseline recipe honestly, then if needed allow an equal predeclared small learning-rate/regularization search budget per size using development data only. Report these as different questions: fixed recipe versus equal-tuning-opportunity best recipe. Freeze selected recipes before fresh confirmation. Same learning rate is a valid fixed-recipe comparison, not a guarantee of equally good optimization.

### Secondary question: resource budget

“Which size gives the best quality under a fixed practical compute/time budget?”

Keep this separate. At equal updates, larger models spend more compute; at equal compute, they usually see fewer tokens. Neither is universally the fair answer. Calibrate on training-only workloads, predeclare the resource budget and exposure schedule, and report accuracy versus measured CPU time, target/input tokens and updates. FLOP estimates are supplemental and must state assumptions; wall time includes framework, validation and evidence overhead. Do not stop cells adaptively when test accuracy looks good.

All three primary cells must reach their prescribed exposure for a completed fixed-exposure comparison. Timeouts are retained failures, not smaller substitute training budgets. A capped protocol extension requires a new request/version.

### Evidence requirements

Use at least three prospective seeds, preferably five if the calibrated runtime permits. Report individual seeds and exact counts, not only a mean. Group related examples into independent families; family-level uncertainty and paired prediction disagreements matter more than treating thousands of correlated transforms as independent trials. Three sizes are not enough to fit a trustworthy scaling law.

Include:
- train and validation learning curves, training exact answer and per-position loss;
- held-out same-domain families versus new compositions, longer sequences and new moduli separately;
- counterfactual input changes, permutation controls and input-sensitivity checks;
- constant/majority, lookup and exact-algorithm references;
- malformed, EOS, truncation and exact semantic error rates;
- measured peak RSS, step/inference timing, configuration and source identities, optimizer/checkpoint replay and every failed attempt.

Use fresh confirmation families generated after recipe selection under a frozen generator and committed split seed. Public historical test sets remain development evidence. For finite modular domains, exhausted or heavily inspected universes cannot simply be relabelled fresh; choose an explicitly new domain or task-composition distribution and state the changed scope.

### How to read outcomes

- Larger models fit train data better but not held-out compositions: evidence of better fit, not reasoning improvement.
- Larger models improve fresh exact composition accuracy and input sensitivity across seeds: evidence for this declared size/recipe/task regime, not general intelligence.
- All sizes underfit and still improve at the cap: budget/optimization remains unresolved; do not conclude size is useless.
- All fit training but fail fresh families: investigate representation, coverage or inductive bias before another blind size increase.
- Small wins at equal compute while large wins at equal exposure: useful, noncontradictory deployment trade-off.
- Better syntax alone does not count as reasoning.

## 4. Hardware and research boundary

The pinned [Decoder](https://github.com/QSOLKCB/Trinite/blob/be4ac55b79f88be3c1a6052d5bc61f1d31be1f4a/src/trinite/model.py) explicitly allocates/validates CPU float32 tensors. The Ryzen 5950X is the existing native reference; the RTX is not enabled merely because the model fits in VRAM. A GPU backend is separate reviewed work.

The largest proposed 304,128-parameter model has about 1.22 MB of float32 weights. Parameters, gradients and two Adam moment tensors alone are approximately 4.87 MB, excluding activations, intermediate buffers, allocator/runtime, checkpoint/evidence storage and any extra optimizer temporaries. This suggests parameter memory is not the main 32 GB constraint, but is not a measured throughput or peak-memory guarantee. Run sizes sequentially, calibrate runtime, preserve thermal/thread settings, and keep no external teacher resident during timing comparisons.

Relevant primary research:
- [Hoffmann et al., Training Compute-Optimal Large Language Models, 29 March 2022](https://arxiv.org/abs/2203.15556): supports separating parameter growth and token/compute allocation. Its 70M–16B study does not license transferring a fixed token-per-parameter rule to Trinite's sub-million formal-task models.
- [Power et al., Grokking, 6 January 2022](https://arxiv.org/abs/2201.02177): algorithmic-task generalization can appear well after training fit, so a short negative pilot cannot settle capacity. It does not promise grokking here.
- [Yang et al., Tensor Programs V, revised 28 March 2022](https://arxiv.org/abs/2203.03466v2): hyperparameter transfer across widths relies on a specified parameterization. Trinite's standard fixed-std initialization is not automatically μP; adding it would be a separate intervention.

## Proposed roadmap insertion

**Optional RSH exact donor:** admit a pinned MPL-2.0 rational inversion/reflection adapter with per-item provenance, independent oracle and strict singularity/resource tests. Exit on exact conformance only. Any learning curriculum is a later matched experiment, separate from capacity and integer-backend work.

**Capacity diagnostic after native mechanics:** freeze one dense task and the 86,528/178,944/304,128 width ladder, with fixed exposure as primary and a separately labelled compute-budget view. Use a new bounded plan, retain all failures, predeclare optimization allowance and selection, and confirm on fresh families across seeds. Promote a larger checkpoint only if its measured quality/resource trade-off justifies it. Do not equate more parameters, geometry or ternary notation with reasoning.

