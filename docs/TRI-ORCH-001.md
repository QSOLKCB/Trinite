# TRI-ORCH-001 — local deterministic orchestration contract

**Status:** PROPOSED design, 9 November 2026. **Documentation only.** Neither code nor live teacher/model inference is implemented or authorized by this contract. The user explicitly requested architecture and governance review in this PR **before any application implementation**; the separate next PR is [TRI-CHAT-001](TRI-CHAT-001.md). This contract is not native model learning evidence.

Read [INVARIANTS](INVARIANTS.md), [ARCHITECTURE](ARCHITECTURE.md), [ROADMAP](ROADMAP.md), [AGENT-IBAE-GOVERNANCE](AGENT-IBAE-GOVERNANCE.md), [DATA](DATA.md), [TEACHER-INTAKE](TEACHER-INTAKE.md) and [MASTER-TEACHER-CORPUS-V1](MASTER-TEACHER-CORPUS-V1.md).

## Objective, model roles and non-goals

Deliver a useful local conversational **application** without asserting that native Trinite can reason or converse. The proposed controller performs deterministic **policy enforcement and orchestration**; a local generative teacher writes prose; an optional separate decision model provides advisory routing scores; first-party formal operators compute and independently check bounded exact results.

- **Native Trinite:** separately preserved randomly initialized dense/ternary/four-state research lanes, never silently replaced by pretrained external weights. Native training, geometric objectives, old fixtures, source closures and checkpoints remain untouched.
- **LiquidAI LFM3-1.7B-GGUF:** candidate *generative* conversational teacher. Its text is **unverified prose**, including an explanation surrounding independently verified arithmetic.
- **LiquidAI d1-2B-GGUF:** candidate *non-generative decision* model. It yields specific decisions/scores, **not** conversational answers, deductive proofs, owner approval or execution authorization. Its specialized llama.cpp `/v1/systemone` contract is different from a standard chat completion API. Support depends on exact runtime revision and must be checked before implementation.
- **Exact tools:** strict first-party integer or GF(3) parser/evaluator + independent oracle, not `eval`, model arithmetic, shell, unrestricted expression language or dynamic plugin loading.
- **Future:** possible native tiny learned controller and corpus/distillation work after separate rights admission, frozen experiments, fresh benchmarks and owner approval. Not authorized by this proposal.

The assembled system must not be called "native Trinite learned reasoning" or "deterministic language generation". GANs are **not** part of the chat/reasoning route. For exact formal data, independently verified deterministic generators/oracles are preferable. GAN-generated visual/audio examples may be considered only in a separately versioned, rights-admitted data experiment; adversarial **testing** of routing, tools and model messages is encouraged in TRI-CHAT-001 without any GAN training.

## Architecture and trust boundaries

```text
 Untrusted browser / chat input
            |
  loopback-only bounded HTTP API
            |
  deterministic TRINITE controller <--- owner policy / allowed tools / caps
          /       |         \
   optional d1   exact      local LFM3
    (advice)    formal      (unverified
                 oracle       prose)
          \       |         /
           typed result composer
                 |
    independent exact verification
                 |
     bounded receipt + safe UI text

 Separate lanes (not in chat call graph):
 native Trinite numerical training / research;
 MTC-001 local teacher corpus / admission;
 experimental source-bound verification and historical evidence.
```

Four planes defined in [AGENT-IBAE-GOVERNANCE](AGENT-IBAE-GOVERNANCE.md) remain non-interchangeable:

| Plane | Responsibility | Cannot grant |
| --- | --- | --- |
| Governance | Owner-authorized policies, exact tools, routes, model/runtime IDs, rights, resource limits | An ML output, probability, PR text or green CI is not authority |
| Orchestration | Validate, determine eligible routes, request optional advisory classification, select allowed execution, journal outcomes | No expansion of permissions or retries after denial |
| Execution | A specifically allowed pure formal operator or separately approved local model call | No shell, arbitrary file writes, model-proposed network requests or native training |
| Evidence | Typed outputs, digest-bound request/decision/results and independently checked exact facts | A hash is not external-origin attestation, correctness proof or new neural reasoning |

Direction: HTTP handler -> policy -> narrow backend adapters or exact operator -> verifier -> presenter. Model and native-training modules must **not** import chat server/orchestrator; user messages never select filesystem paths, model artifacts, backend URLs, arbitrary commands, keys, data admission or training. No circular imports or duplicate native numerical updater.

## Proposed request/decision/result contracts (not yet implemented)

Design namespace, to be precisely frozen and validated before the first live call:

| Schema | Content |
| --- | --- |
| `trinite.orch.request.v1` | version, request identity, bounded UTF-8 text, bounded session conversation, policy/config identity, no user-selected endpoint |
| `trinite.orch.decision.v1` | allowable routes, rule/d1 source, optional candidate and **uncalibrated** score, selected route, abstain/fallback reason, policy identity |
| `trinite.orch.result.v1` | route, user-visible generated text, separately immutable formal operands/output/verification, `verified_exact`, `model_generated_unverified`, `blocked` or `failed` status |
| `trinite.orch.receipt.v1` | source and config identities, declared model/runtime/quantization and local endpoint identity, limited digests/timings, decision and outcome, redacted failures, replay scope |

Exact JSON key sets/canonicalization, identity schema, deadlines and server limits belong to the **implementation protocol frozen before execution**. Use strict duplicate-key and unexpected-field rejection; no invented confidence semantics. Model prose cannot mutate an exact-result field or determine `verified_exact`.

## Proposed controller state machine

1. **Reject** malformed JSON, wrong method/type, duplicate keys, invalid encoding, invalid session state and over-limit requests. No silent truncation, adaptation or backend fallback.
2. **Authorize allowed routes** from a static, reviewed policy and grammar, with explicit user approval for any side effect. Every route starts denied unless allowed; d1 never enlarges the allowed set.
3. **Route:** exact supported formal grammar uses formal verifier; ordinary allowed chat may go to teacher. For ambiguous supported classification, optional d1 returns a strictly parsed named choice among `chat`, `formal`, `clarify`, `unsupported`. Validate it against allowed routes. Missing, malformed, uncertain, timed-out or mismatched d1 deterministically takes rules-only safe fallback or `clarify`, not extra capability.
4. **Execute** one bounded approved route. No autonomous calls to third-party services, model-generated scripts/tools, dynamic tool registry or unapproved native model execution.
5. **Verify** all claimed exact arithmetic via independent first-party oracle; on disagreement report blocked/failed. Generated explanation is never a substitute for oracle proof or a scientific held-out result.
6. **Present** escape-only browser text plus route/evidence status; no model-supplied executable HTML, Markdown interpretation with raw HTML or hyperlink navigation.
7. **Retain** outcomes, blocked/failed/timeout/partial/unknown states and redacted metadata within an explicit retention policy; no auto-retry, no hidden data ingestion or publication.

Recorded controller **decision replay** can be deterministic for exactly specified inputs/policy even when **fresh model re-inference** differs by version, quantization, GPU, seeds, batch or runtime. A fixed sampling seed or temperature zero does not prove cross-platform bitwise model determinism.

## Versioned prototype safety envelope (PROPOSED, not implemented)

For the next PR, propose: loopback-only listener and strictly configured local inference endpoints; max 8 KiB per UTF-8 user message; max 16 KiB HTTP JSON request; max 16 in-memory session turns; max 64 KiB teacher output; end-to-end deadline <=120 s; concurrency cap must be declared and tested; automatic retries **zero**. The next PR must freeze real caps and prove enforcement. No wildcard CORS, unknown Host/Origin on mutating requests, remote listener, redirects, DNS/network endpoint selection by users, shell execution, unbounded subprocesses, implicit downloads, cloud fallback or client-side backend access.

Unknown/malicious model text, user prompt injections, adversarial HTML/JS, malformed decision records, unsafe formal expressions, unexpected resource exhaustion, interruptions and wrong-model responses are negative test cases. Retain meaningful failure evidence without storing private raw chats in Git or routine logs. Serve only static files owned by the application.

## External model and rights policy

Before **any authorized local run**, record exact model family, immutable artifact/weight hashes or repository revision, GGUF quantization, runtime build/version, endpoint contract and configured chat template; verify licenses, provenance, compatibility and hardware fit on the actual machine. Proposed model sources: [LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF), [d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF). Upstream LFM Open License v1.0 applies to weight assets; Trinite code remains MPL-2.0. A weight license does **not** grant automatic native student training-data admission or output-rights certification. No `trust_remote_code`, unrestricted pickle, hidden model download or external inference.

Teacher output and session transcripts are **runtime data**, not MTC-001 evidence, admitted native labels or historical training fixtures. Distillation requires a new versioned governance/data/rights/family-split/scientific gate. MTC-001's existing capture quarantine remains intact.

## Acceptance and version sequencing

**This design PR:** documentation changes only, under `docs/`; no root `AGENTS.md`, code, tests, scripts, configs, CI workflow, package lock, fixtures, preserved negative data or historical source/evidence changes. Check all internal links, consistency with existing TRI-I01–14, MTC priorities and Phase 5–9 gates. Docs approval is not execution authorization.

**Next separate PR — TRI-CHAT-001:** initial local text chat prototype with standard-library Python server, vanilla HTML/CSS/JS UI, deterministic rules, one narrow formal oracle, separately configured external local generative adapter and fault-injection tests using no real model calls in CI. Optional d1 adapter is non-blocking, preferably introduced after working rules-only chat. Owner-local true model smoke requires a **separate explicit execution approval** and recorded runtime identities.

**Later — TRI-DECIDE-001:** preregister d1-on vs rules-only and teacher-only comparison with source/weight pins, held-out families, routing confusion, false-tool-execution, abstention, exact-answer accuracy, p50/p95 latency, tokens, CPU/GPU memory, timeouts and failures. No superiority/acceleration claim without measured hardware and comparators.

**Longer term:** evaluate native Trinite as a small learned controller only after admissible task-routing data and separate learning protocol; native generalization remains an open scientific question. No GAN or multimodal addition is a prerequisite.

Affected historical contracts: **TRI-I01–I03, I05–I08, I10–I14** (preserved) and new **TRI-O01–O12** in [INVARIANTS](INVARIANTS.md#prospective-application-orchestration-invariants-tri-orch-001). This is a *plan*, not executed acceptance evidence.
