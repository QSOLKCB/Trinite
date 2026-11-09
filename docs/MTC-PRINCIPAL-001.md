# MTC-PRINCIPAL-002 — structured Qwen → LFM3 supervision correction

**Status:** corrective PR for the owner-reported first hardware run, not another real model execution. The earlier Principal archive is complete, verified, private and immutable. The historical Principal v1 protocol is retained below for provenance; the following v2 policy supersedes it.

## Observed failure and current protocol

The owner reported four Qwen3.5 27B replies ending in `done_reason="length"` after 768 generated tokens of `message.thinking`, with empty `message.content`. There were zero valid critiques and zero LFM revisions. Since the LFM baseline also contained no final answers, the original recorded results do **not** measure teaching efficacy, mathematical ability or model accuracy.

**Principal plan and manifest schema v2** now freezes `response_policy={"think":false,"format":"json"}`, applied as **top-level** fields to Qwen critiques and LFM3 revisions. Qwen's feedback must still be valid strict JSON with `verdict`, `feedback` and `suggested_answer`; it is untrusted until an independently computed oracle is applied. A reply with `done_reason="length"` is invalid even if some JSON-like content is present. Blank, truncated or unparsable baseline answers do not trigger a Qwen call. Nothing is extracted from the thinking field as an answer.

The original maximum of four Principal reviews and four revisions, unchanged output caps and 300s deadline, source/model SHA-256 closure, permissions and training-admission gate still apply. Neither teacher nor Principal weights are updated, and no rights decision is made. Qwen3.5 or Ollama may ignore format flags even with thinking disabled: failures must be preserved rather than automatically retried or papered over.

## Fresh owner-local plan (after corrected baseline is verified)

Use a **new** verified LFM3 v3 archive under `single-002`, then freeze a **new** Principal v2 plan under `plan-002`:

~~~sh
cd ~/Trinite
uv run --no-project python -I -B scripts/mtc_principal_001.py plan \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --teacher-manifest-identity 'sha256:ACTUAL_NEW_LFM3_V3_MANIFEST' \
  --teacher-model 'lfm3:1.7b' \
  --principal-model 'qwen3.5:27b' \
  --out "$HOME/trinite-principal/plan-002"
~~~

Retain the actual printed request identity before explicitly authorizing a fresh capture under `run-002`, and separately verify its own NEW manifest. No historical run is reinterpreted, overwritten, promoted or replayed automatically.

---

## Historical MTC-PRINCIPAL-001 record (v1; retained for provenance)

# MTC-PRINCIPAL-001 — Qwen3.5 27B supervising one LFM3 teacher

**Status:** separate additive draft, stacked on MTC-LFM3-001 PR #147. No owner-local inference, GPU work, data admission or model training has been performed.

## Principal, teacher and authority

Qwen3.5 27B is the offline Principal: an advisory critic of LFM3 teacher answers. It can supply JSON feedback and suggest a corrected answer, but it does not control Trinite's independent arithmetic oracle, owner permissions, native model weights or admission policy.

This pilot does **feedback-at-inference-time**, NOT parameter training. Four already frozen TRAIN examples from one selected teacher in a verified MTC-LFM3-001 capture are reviewed by Qwen. If the review passes strict structure checks, the same LFM teacher gets one new inference with that feedback, and the independent oracle evaluates both original and revision. No development or locked-test item is fed back to a teacher.

The baseline archive, all four source attempts and both model digests are bound by independently retained SHA-256 identities before any new call. The previous MTC archive is verified again during the Principal capture and during final offline verification. The new Principal request binds all relevant source, workflow, governance, contract and test bytes, precise model tags, generation settings and a maximum of four Qwen reviews and four LFM revisions.

## Local resource and privacy limits

Qwen3.5 27B is listed by Ollama at approximately 17–20 GB, exceeding 16 GB of GPU VRAM before KV cache. Expect CPU/RAM offload and potentially slow runs on the 5060 Ti workstation. The call budget is deliberately eight inferences at most. Each request includes keep_alive=0 so Ollama may release the previous model before loading the next. Every response has a 64 KiB byte cap and an absolute 300-second deadline. No automatic retries or cloud/proxy calls.

Use exactly the installed tag, normally qwen3.5:27b, and an installed LFM3 tag found by ollama list. No model pull or automatic download is performed. A Principal response with no usable JSON, for instance because its reasoning exceeded the output budget, is retained as a failed review, not silently repaired by altering sampling or prompt text.

Output folders must be private and new, with owner-mode 0700 folders and 0600 files. Captures retain raw bounded Ollama API replies, exact request hashes, teacher revisions, Qwen verdicts, feedback and independent mathematical matches. The SHA-256 manifest requires an externally retained identity. All content is QUARANTINED_NOT_ADMITTED; no training dataset output is produced.

## Manual commands after independent review and both PR merges

From a clean reviewed checkout with locally running Ollama and the prior five-teacher capture independently verified:

~~~sh
mkdir -m 700 -p "$HOME/trinite-principal"

uv run --no-project python -I -B scripts/mtc_principal_001.py plan \
  --teacher-run "$HOME/trinite-mtc-lfm3/liquid-001/capture" \
  --teacher-manifest-identity 'sha256:PREVIOUS_MTC_MANIFEST' \
  --teacher-model 'EXACT_INSTALLED_LFM3_TAG' \
  --principal-model 'qwen3.5:27b' \
  --out "$HOME/trinite-principal/plan-001"

# Retain the printed principal_request_identity separately before authorizing:
uv run --no-project python -I -B scripts/mtc_principal_001.py capture \
  --plan "$HOME/trinite-principal/plan-001/plan.json" \
  --teacher-run "$HOME/trinite-mtc-lfm3/liquid-001/capture" \
  --request-identity 'sha256:PRINCIPAL_REQUEST_IDENTITY' \
  --out "$HOME/trinite-principal/run-001" \
  --authorize-principal

# Retain the printed principal_manifest_identity separately:
uv run --no-project python -I -B scripts/mtc_principal_001.py verify \
  --teacher-run "$HOME/trinite-mtc-lfm3/liquid-001/capture" \
  --run "$HOME/trinite-principal/run-001" \
  --manifest-identity 'sha256:PRINCIPAL_MANIFEST_IDENTITY'
~~~

Interrupted Principal runs retain partial evidence and are not automatically resumable: if a process was interrupted after issuing an HTTP call but before recording it, the execution state is unknown. Preserve the folder and create a separately frozen/authorized fresh attempt rather than silently repeating possible prior calls.

## Verification and scientific limits

Hosted PR CI uses fake Ollama services only. It checks strict Principal JSON, model tag/digest mismatches, complete request/response custody, feedback-to-revision routing, independently recomputed arithmetic, separate split policy and no unauthorized dataset promotion. No private response, model weight, browser session, native Trinite code or historical artifact is uploaded to GitHub.

TRI-I02/I03/I07/I08/I11/I13/I14 and TRI-O01/O02/O04/O05/O06/O08/O10/O11 remain authoritative. Source and manifest SHA-256 prove recorded identity within their scope, not producer authenticity, security certification or rights clearance. TRAIN-only tasks are engineering-exposed and not a new held-out benchmark. Independent admission/rights review and a controlled no-feedback comparison are prerequisites before claiming teaching efficacy, model fine-tuning or useful distillation.

The next gate is an MTC-002 source-specific training/admission proposal, only after independent review of verified owner-local Principal records. No automatic weight update or retraining occurs here.
