# MTC-SCAFFOLD-ABLATION-001 — controlled Modelfile × harness experiment

**Status: PROPOSED, DOCS-ONLY.** This is a prospective, separately reviewable protocol, not an implemented runner, completed experiment, teacher appointment, training decision, or authorization to execute models. It does not alter the frozen Qwen principal failures or the pending [MTC-TEACHER-RECRUITMENT-001 (draft PR #156)](https://github.com/QSOLKCB/Trinite/pull/156) candidate assessment.

## Research question

How much of a fixed teacher model's apparent formal-task performance comes from its own answer generation, a system-prompt profile stored in an Ollama Modelfile, a deterministic *inference/evaluation* harness, or their combination? Once native Trinite training has been independently authorized and measured, how much assistance can be removed without losing held-out accuracy?

A Modelfile and an inference harness **do not update model weights**. They may improve the *system* while leaving the base *model* unchanged. The phrase "training wheels" refers to external scaffolding, not a claim of model learning. Later weight-update experiments require their own data/admission, optimizer, checkpoint, and replay protocols.

## Four conditions: a 2 × 2 factorial comparison

For each **separately qualified and identity-pinned candidate**, run the identical task set in four arms:

| Arm | Added Modelfile system profile | Deterministic harness | Meaning |
| --- | --- | --- | --- |
| A — stock | No | No | Baseline behaviour through the common request transport |
| B — profile | Yes | No | Effect of system-level instructions |
| C — harness | No | Yes | Effect of fixed prompt preparation and strict response handling |
| D — combined | Yes | Yes | Joint assistance and interaction |

All arms use the same **underlying weight identity**, tokenizer/template lineage, numerical host, seed/temperature, context and output-token settings, response format policy, task order plan, scoring parser, and exact independent oracle. Any model-family-specific thinking policy must be *declared and pinned before results*, held fixed across the candidate's four arms, and not silently changed to improve one arm.

- **Modelfile toggle:** a separately created *local derived model tag* with a frozen, source-hashed `SYSTEM` instruction. Freeze the complete Modelfile bytes and Ollama's effective model/template information; verify the derived model descends from the selected base weights. A derived model's distinct manifest digest is expected; it is **not** a different trained model. Do not mutate or retag the base.
- **Harness toggle:** one static, source-hashed, answer-independent prompt wrapper/step specification and response validation. The wrapper can make task operands and required output fields explicit. It must not include an expected result, example from the locked family, hints derived from the oracle, adaptive revision, or additional inference calls.
- **Common transport:** both toggles off does not imply zero transport conventions. If common JSON mode is used, call A the **unscaffolded-relative-to-protocol** arm, not an unrestricted/raw chat test. Any truly unconstrained free-text baseline is a separate descriptive study.
- **Oracle:** pure, independently executed Python exact arithmetic is used *after* collection to score all four arms, **never** to construct an answer, rewrite model output, provide feedback, or silently repair a failing arm. An explicitly oracle-assisted application can be measured later as **arm E**, outside this factorial comparison and excluded from intrinsic-model scores.

Illustrative *non-executable* profile (replace the symbolic base and freeze its actual tag/digest before creating anything):

```text
FROM <OWNER-PINNED-LOCAL-BASE-TAG>
SYSTEM """Solve the stated mathematical task independently.
Do not approve a proposed result without checking it.
Use the common output schema. Do not invent an oracle result."""
```

The implementation must determine whether the selected Ollama model's inherited template or existing system text makes the comparison non-equivalent; record that as a limitation or refuse the run. Creating the derived tag and running inference are **separately owner-authorized workstation operations**; never run `ollama pull` or `ollama create` from CI or as a side effect of planning.

## Stages and evidence classification

1. **Engineering pilot (EXPOSED):** eight fully disclosed arithmetic/claim-review probes × four arms = **32 scheduled single-call observations per candidate**. Used only to validate transport, schema, digest, timeout, token budget, parsing, and score accounting. Do not call these holdout results. Fixed failed calls stay in the denominator.
2. **Prospective confirmatory set (LOCKED):** after prompt/profile/harness choices are frozen, prepare **32 distinct formal problem families**, balanced across direct arithmetic, incorrect-claim rejection, correct-claim acceptance, and reversed-order choice. Every family and its paraphrases must stay in one split; no pilot example, template-tuning input, or visible MTC historical diagnostic may enter the locked set. The frozen schedule is **32 × 4 = 128 calls per candidate**, with a per-candidate exact-scope approval and a privately retained request identity. These are study-internal heldouts; unknown external pretraining exposure precludes asserting universal novelty or zero contamination.
3. **Training-wheels removal (FUTURE):** separately authorize native *weight training*, freeze a new checkpoint plus a training-family-only curriculum, and repeat A–D against **new, never-tuned** evaluation families. Compare pre/post **A versus A** to estimate unassisted weight-change performance; compare pre/post D and the A–D gap for scaffold dependence. Do **not** confuse external teacher Modelfile tuning with native Trinite learning.

Do not use the locked stage to select prompts, change the candidate, alter decoding, or set thresholds retrospectively. If the pilot fails technically, version a new request and retain the failure; the previous 32-case schedule is not silently rewritten. Predeclare any replacement experiment and budget before new outcomes.

## Measures and interpretation

Retain, for *every scheduled call*, candidate and arm identities, task/family, split/exposure class, frozen request bytes/hash, exact response bytes/hash, actual model tag/digests, elapsed time, token accounting if available, parsed answer/verdict, parser failure reason, truncation/transport status, and independently recomputed oracle truth. Private raw captures, authorizations, prompts and local log paths do not go to GitHub Actions.

Primary descriptive outcomes: exact arithmetic answer accuracy; **false approval rate for wrong claims**; correctly accepted true claims; forced-choice accuracy **and** independently reported numeric-answer accuracy; malformed/truncated/time-out rate; verdict/answer contradictions; non-adaptive call count. Report exact numerator/denominator for each outcome and task family. Latency and throughput are secondary measurements, with different prompt lengths and reasoning-token counts reported, not assumed equal.

For matched outcomes, calculate `B−A` (profile uplift), `C−A` (harness uplift), and `D−B−C+A` (interaction). Use per-item paired contingency tables rather than treating repeated arms as independent samples. With 32 locked families the estimates remain small-sample and domain-specific. If D improves while A does not, **the system improved, not the model**. If A later improves after actual optimizer updates on fresh heldouts, that is evidence *consistent with* learned capability, subject to checkpoint lineage and controls; no causal or generalization claim follows automatically.

Predefine a future retirement gate against *fresh heldouts* before testing, including a minimum A accuracy, a maximum false-approval rate, acceptable malformed rate, and a maximum `D−A` reliance gap. **No teacher promotion is automatic**, even if every measured gate passes; exact arithmetic authority remains Python.

## Owner, source and privacy gates for the next implementation PR

An implementation successor (separate PR, **not included here**) must provide a Python-standard-library `plan → explicit single-use authorization → capture → offline verify` workflow with:

- A fresh versioned protocol/source-closure identity covering program, both prompt templates, effective Modelfile bytes, tests, selected base and derived installed model digests, complete task/split data, order/seed/options, parser/scorer/oracle and experiment budget; record exact reviewed Git head.
- A local-only `127.0.0.1` Ollama transport, no auto-pulls or implicit external requests, no arbitrary host override, bounded request/response bytes, deadlines, zero retry/resume by default, explicit failure retention, and before/after digest checks. Preserve partial archives on interruption and refuse mismatched authorizations.
- Private owner-controlled output paths, bounded journal writes and manifest verification using independently retained SHA-256 identity. No results or raw private content uploaded to CI; hosted CI may test synthetic software doubles only.
- Independent oracle tests, tamper/digest/branch-drift failure tests, duplicate/invalid JSON tests, wrong-claim controls, prompt-answer leakage tests, and a *manifest validator that does not trust its stored scores*.
- Frozen primary comparisons and explicit **observed / derived / proposed / unknown** statuses; no edits to historical Qwen evidence, native Trinite optimizer, pretrained weight intake, rights policy, or pre-existing MTC candidate captures.

**Sequence:** review the in-progress [MTC-TEACHER-RECRUITMENT-001 (draft PR #156)](https://github.com/QSOLKCB/Trinite/pull/156) first; create an additive implementation successor; run offline conformance; obtain fresh owner authorization for each installed model, derived profile creation, and each bounded owner-local candidate run; verify; then make a separate supervision or training admission decision.

## Relationships and invariants

- [MTC-EXAMINER-CAUSE-001](MTC-EXAMINER-CAUSE-001.md): Qwen's exposed 16-call diagnostic is historical negative-control evidence, not a generalized failure theorem or substitute for prospective arms.
- [REFERENCE-GEOMETRY](REFERENCE-GEOMETRY.md): already compares stock against a SYSTEM-only *Modelfile-derived* Qwen3-0.6B profile through a **Transformers CPU adapter**, not an executed Ollama Modelfile. Its visible training examples and measurement backend do not establish this new four-arm result.
- [MASTER-TEACHER-CORPUS-V1](MASTER-TEACHER-CORPUS-V1.md) and [DATA](DATA.md): teacher responses, external weights and unknown-rights sources are quarantined, not native training inputs.
- [TRI-ORCH-001](TRI-ORCH-001.md): deterministic controllers and exact oracles must never be represented as neural reasoning or hidden authorization.

Affected requirements: **TRI-I02, I03, I07, I08, I10, I11, I13, I14** and application-boundary **TRI-O02, O04, O05, O08, O10, O11**. Historical frozen records and their original hashes remain untouched. No training, model pulls, local inference, merges, releases, or publications are authorized by this document.
