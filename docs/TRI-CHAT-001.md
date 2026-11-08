# TRI-CHAT-001 — separate next-PR local chat prototype

**Status:** PROPOSED only (9 November 2026), **not part of the TRI-ORCH-001 documentation PR**. Work may be proposed only after review of [TRI-ORCH-001](TRI-ORCH-001.md). Merging design documents does not authorize deployment, model acquisition, teacher calls or native training.

## Prototype target

A usable local text-only chat with a separately configured local generative teacher, deterministic permission/route controller and independently verified formal arithmetic. This demonstrates **application integration**, not native Trinite learning or deterministic neural language generation.

### Intended scope

1. Static vanilla HTML/CSS/JavaScript chat UI; escaped text, no bundler, Node, NPM, cloud assets, remote browser backend calls or executable model HTML.
2. Python 3.11+ standard-library local HTTP server on loopback; strict limits, JSON, content type, Host/Origin, timeouts, concurrency and error states; prefer `uv run` when appropriate but do not disturb frozen native hash-locked environments.
3. Rules-first controller; deny by default, only allowlisted narrow pure formal integer/GF(3) functions; no `eval`, arbitrary commands, shell/network tools or native training.
4. Local generative teacher adapter, explicit model/quantization/runtime/chat template identities, no implicit acquisition; errors and unavailability visible. External teacher prose always marked unchecked.
5. Independently checked exact-output typed field that teacher prose cannot change, with mismatch handling and versioned evidence.
6. **Optional, gated** non-generative d1 named-choice adapter; first chat must work without d1. If included, validate the pinned specialized `/v1/systemone` API with software doubles, handle abstain/error, do not allow privilege escalation.
7. Redacted request/decision/result receipts and strict separation from MTC-001 captures, datasets and historical producer source/evidence.

### Mandatory reviews and tests

| Gate | Acceptance evidence |
| --- | --- |
| Transport | Local binding, valid/invalid JSON, duplicate keys, UTF-8, content type, size limits, Host/Origin, endpoint validation and no cloud/redirect path |
| Routing | Rules-only deterministic replay, restricted d1 advice, abstention, invalid choice, absent/mismatched decision model, denied privileges and no adaptive retries |
| Formal | Oracle-backed exact arithmetic, unsupported expressions denied, generated prose unable to alter verified facts |
| UI | HTML/JS/XSS injection strings safely escaped, readable route/exact-vs-generated status, no browser calls to model endpoints |
| Operations | Enforced time/size/concurrency limits, cancellation/unknown/timeout/partial results, explicit config and privacy redaction |
| Adversarial | Hostile model messages and decision results, deliberately malformed requests, unsafe routing and hallucinated answers; no GAN training needed |
| Provenance | Exact source, config/model/backend identities on real runs; digest scope; recorded-event vs fresh neural replay distinguished |
| Isolation | No changes to native decoder/training/geometry/evidence, historical fixtures, MTC admission or frozen source closures |
| Claims | CI fake backend tests not counted as a real local chat result; owner-approved real smoke and measured hardware needed before claims |

The proposed finite default limits in [TRI-ORCH-001](TRI-ORCH-001.md) must be frozen and tested before implementation's real run, not treated as already enforced.

### Follow-on

**TRI-DECIDE-001 (later):** only after baseline works, evaluate optional d1 assistance against a deterministic rules baseline and teacher-only route on frozen test families. Report tool false positives, abstention, routing accuracy, cost, latency and real memory consumption. Separate scientific evaluation from usability smoke.

**Later research (not approved):** native tiny routing-model training/distillation may be explored only with independently admitted labels, rights, family isolation and reviewed protocols. GAN-based synthetic image/audio generation is optional exploratory data research, not needed for exact arithmetic or the chat milestone.

See [ROADMAP](ROADMAP.md), [application invariants](INVARIANTS.md#prospective-application-orchestration-invariants-tri-orch-001), [TECH-STACK](TECH-STACK.md), [MODULES](MODULES.md) and [AGENT-IBAE-GOVERNANCE](AGENT-IBAE-GOVERNANCE.md).
