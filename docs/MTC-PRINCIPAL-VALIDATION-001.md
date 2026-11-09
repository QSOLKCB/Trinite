# MTC-PRINCIPAL-VALIDATION-001 — Independent examiner audit + no-feedback control

**Status:** additive, owner-requested research proposal. No owner-local Qwen or LFM3 inference, model training, data admission, publication or cloud transfer is performed by the PR. The protocol is **engineering-exposed**, not a blinded or confirmatory capability benchmark.

## Motivation and evidence custody

The owner verified these source-bound historical results on 9–10 November 2026:

- LFM3 `single-002`, manifest `sha256:971a6c8633f09b6d5e2229c7ec81ebcee586355e3e501cba08adcc5f3eecca6e`: **2/4** correct on the TRAIN examples.
- First Qwen feedback `run-002`, manifest `sha256:33bc06af85e7fb491e5ca87f0cf9d5dc8d07bb73b9362bb061161284bed579d5`: LFM3 revised to **3/4** with four valid Principal critiques.
- Second successful Qwen feedback `round2-002`, manifest `sha256:4b7b334deacdb248f7608a041587c0fa61810d2094a04e40d5dbbf05a7115a20`: **3/4** retained, all four revisions scored, Qwen verdict **3/4** agreement with the exact oracle, one false positive. In both rounds Qwen labelled the incorrect claim `(8 + 15) mod 11 = 2` as correct. The exact result is 1.
- The earlier second-round `round2-001` archive also remains preserved with four HTTP 404 failures on the unsupported synthetic `sha256-...` model-name string; **not** a mathematical regression.

**Scientific question:** can Qwen3.5 27B distinguish correct from incorrect modular answers, and does one additional **no-feedback** LFM3 response replicate the performance observed under supervised prompting? This is a diagnostic, not a plan to admit Qwen-generated text as an arithmetic authority.

## Frozen evaluation design

`scripts/mtc_principal_validation_001.py` adds its own standalone protocol without changing any previous source-bound collector. Its planning stage verifies the historical `single-002` and `run-002` archives using their exact manifest IDs, requires the verified first Principal revised score of **3/4**, pins both installed Ollama tags and digests, and freezes a source/governance/test/workflow closure.

Exactly **12** local inference calls are scheduled, separated by lane:

1. **Examiner:** eight Qwen classifications; four correct and four deliberately incorrect claims from four first-party modular-arithmetic problems (including the known modulo-11 false positive). Each positive and negative claim for a problem shares its operands and modulus. This makes false-positive and false-negative verdicts visible and avoids declaring the Principal's `suggested_answer` as ground truth. The deterministic source code derives the oracle label from integer arithmetic. Only the task and supplied claim—not the expected answer or label—enter Qwen's prompt. Qwen returns strict `verdict`, `feedback`, `suggested_answer` JSON. Score TP, TN, FP, FN, uncertain and invalid separately. Eight examples are far too small for a general capability claim.
2. **No-feedback teacher:** four fresh LFM3 calls using the existing **original** TRAIN prompts, unmodified deterministic settings, and no Principal feedback in the context. Each is independently scored by the original integer oracle and compared descriptively against the **previously supervised 3/4** baseline. This is an *unpaired extra-sampling diagnostic*, not a randomized control sufficient to attribute causal improvement; the original first Principal revisions had different prompt contexts. No new development/locked-test prompts enter either training or a feedback loop.

The examiner runs **before** the no-feedback control. The outputs are never used as new labels for LFM3, never exposed to Trinite's native trainer, and never reclassified as independent evidence without the exact oracle checks.

All Ollama requests use the exact **installed tag names** (not synthetic digest names) with frozen digest checks before **and after** each call. Ollama tag identity is mutable and those checks are **not an atomic proof** of which weights produced a response. Each request uses direct `127.0.0.1` loopback only, no proxies/cloud fallback, the previously reviewed token/timeout and top-level `think:false`, `format:json` controls. Raw bounded responses, request SHA-256, digest checks, parsed classifications and failure records remain private.

## One-use execution authority and archive verification

- Plan is **read-only relative to prior evidence** and performs no inference. Freeze a new `trinite.mtc-principal-validation.plan.v1` record. Retain its request identity outside the plan directory.
- Capture requires a separately owner-created **exact-scope** JSON approval under a new private folder; an adjacent `O_EXCL` execution claim is written before inference. A failed/interrupted run consumes the approval, and neither automatic retry nor implicit resume exists. Keep failures as evidence.
- Capture stores raw responses with per-call fsynced JSONL records, then creates a private SHA-256 manifest only after all 12 calls are accounted for. HTTP failures, malformed feedback, truncated reasoning and uncertain verdicts remain distinct from incorrect arithmetic.
- Verification replays the frozen source closure, both previous manifest identities, every expected request hash, recorded response parsing and independent arithmetic, all private archive member hashes and the independently retained new manifest identity. It makes **zero** new model calls.
- All outputs remain `QUARANTINED_NOT_ADMITTED` and `training_admission=false`; no LFM3, Qwen or Trinite weights are modified. Data rights, no-feedback causal comparison and new independently sourced test sets require separate decisions.

## Owner-local commands (only after review and merge)

From a clean reviewed checkout with the local Ollama server deliberately configured for no-cloud inference and both models already installed, use a **new private path**. No original archive is edited:

~~~sh
cd ~/Trinite
git pull --ff-only
mkdir -m 700 -p "$HOME/trinite-principal/validation-001"

uv run --no-project python -I -B scripts/mtc_principal_validation_001.py plan \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --baseline-manifest-identity 'sha256:33bc06af85e7fb491e5ca87f0cf9d5dc8d07bb73b9362bb061161284bed579d5' \
  --out "$HOME/trinite-principal/validation-001/plan"
~~~

**Review and retain** the new printed `validation_request_identity` before any capture. The script exposes `authorization_value(plan, identity, approval_path, capture_path)` to construct the exact owner authorization. An independent owner confirmation should happen **in a terminal** before writing that file with `mtc.write_new()`; the approval should be in `$HOME/trinite-principal/validation-001/owner-approval.json` mode 0600, and must name exactly `validation-001/capture`. Do not confuse a copied template, GitHub merge, or permission to plan with the separate approval to use local GPU resources. Do not use Python `input()` in a heredoc; Bash can read an actual TTY.

After separately creating and reviewing that exact-scope approval:

~~~sh
uv run --no-project python -I -B scripts/mtc_principal_validation_001.py capture \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --plan "$HOME/trinite-principal/validation-001/plan/plan.json" \
  --request-identity 'sha256:ACTUAL_NEW_VALIDATION_REQUEST_IDENTITY' \
  --out "$HOME/trinite-principal/validation-001/capture" \
  --owner-approval "$HOME/trinite-principal/validation-001/owner-approval.json"

uv run --no-project python -I -B scripts/mtc_principal_validation_001.py verify \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --run "$HOME/trinite-principal/validation-001/capture" \
  --manifest-identity 'sha256:ACTUAL_NEW_VALIDATION_MANIFEST_IDENTITY'

uv run --no-project python -I -B -m json.tool \
  "$HOME/trinite-principal/validation-001/capture/summary.json"
~~~

Replace placeholders with **actual new** printed identities. Do not try to re-run a consumed approval, and do not use either of the old `round2` output directories as this protocol's output.

## Offline contract tests

Hosted GitHub CI uses a fake Ollama service on an ephemeral loopback port; it **does not contact real installed model weights** or the user's machine. The test service produces a deliberate Qwen **false positive** on the old modulo-11 example, correct classifications for the remaining examples, 3/4 previously supervised first revisions and 2/4 fresh unassisted teacher samples. This is a software-double test proving only that the implementation **correctly distinguishes** those outcomes.

Other tests cover identity tampering, frozen label drift, malformed reasoning-only replies, mandatory authorization, single-use claim, tag change during a call, no hidden teacher feedback, no overwrite of earlier evidence and manifest verification.

This evidence is neither causal proof of Qwen training value nor a benchmark of novel held-out tasks.
