# MTC-EXAMINER-CAUSE-001 — Qwen acceptance-bias vs arithmetic competence

**Status:** prospective owner-local diagnostic protocol; the 9–10 November 2026 **verified historical results are frozen**. No target Qwen inference, teacher revision, student training, rights admission, or release has occurred for this new protocol. The GitHub PR is an executable *proposal*, not experiment approval or scientific confirmation.

## 0. Historical freeze — do not reinterpret the sealed milestone

**Reference code head on merge:** `aa88b4f6963b61de83be2c0eac37aed39384679c` (merged PR #154, OLLAMA-LOG-001; no Git tag is created by this experiment PR).

Owner-local verified prior records:

| Frozen identity | SHA-256 identity | Claim scope |
| --- | --- | --- |
| MTC-LFM3 teacher capture | `sha256:971a6c8633f09b6d5e2229c7ec81ebcee586355e3e501cba08adcc5f3eecca6e` | Original teacher output, prior measured 2/4 TRAIN correct |
| MTC-PRINCIPAL first review | `sha256:33bc06af85e7fb491e5ca87f0cf9d5dc8d07bb73b9362bb061161284bed579d5` | First feedback run: 3/4 revised TRAIN correct |
| MTC-PRINCIPAL validation | `sha256:8a8d909c72ed2e47ab0d2903b337939fc7d1e9bc9789ffc0c8e331efd36b65f9` | 12/12 calls recorded and verified. **Qwen examiner: TP=4, FP=4, TN=0, FN=0**, no uncertainty/invalid. Independent no-feedback LFM3 control 2/4. |
| OLLAMA-LOG-001 receipt | `sha256:a0d7fd7912c3baa4164d89390e4e7317b18c019a8c64b30dd3d7edde599a29b7` | Separately locally sealed-and-verified operational log metadata only |
| Referenced Ollama log | `sha256:852e26b7014dec088eef949f8ee34e79c6988dfe94266339e6c3eab998979cf4` | Post-experiment startup and GPU-discovery snapshot, **NOT** the earlier examiner's execution |

The previous Qwen examiner approved **all four correct and all four incorrect proposals**. For example, `(8 + 15) mod 11 = 2` was approved despite an independent exact value of `1`. The prior eight examples were **exposed and structurally paired**. The evidence does not yet identify a mechanism. In particular, no generalized "Qwen cannot do maths" conclusion follows.

This protocol **does not edit or reseal** any historical plan, prompt, response, answer, summary, manifest, recorded hash or raw Ollama log. OLLAMA-LOG-001 remains a separate operational companion record. Both original archives and this new diagnostic stay `QUARANTINED_NOT_ADMITTED` and `training_admission=false`.

## 1. Question, hypotheses, and limits

**Question:** When Qwen3.5 27B accepts an incorrect modular arithmetic claim, is the failure predominantly in independent arithmetic, binary claim-checking, or response-format/compliance behaviour?

Three **prospective competing explanations**; do not pick one based on this protocol alone:

- **H1 — arithmetic errors:** the model independently solves the same modular problems incorrectly, even without proposed answers.
- **H2 — answer-acceptance/review framing:** the model solves the arithmetic but still approves wrong supplied claims in the review lane. If independently solving and forced choice work while review approves both candidates, this is evidence **consistent with** a task-dependent acceptance tendency, *not an explanation of its internal mechanism*.
- **H3 — structured-response or parse artifact:** the apparent verdicts are malformed, truncated, inconsistent with suggested numeric answers, or mishandled by the parser. The collector stores raw bounded HTTP response bytes plus the parsed verdict, suggested answer and independent oracle comparison so these can be audited.
- Other hypotheses remain possible: token-budget limits, prompt wording, task/order effects, weak calibration, sampling/hardware variation, and generalization failures. This protocol does **not** isolate all causes.

## 2. Frozen within-problem diagnostic — 16 Qwen calls

Use the *identical four exposed* problem families from the previous validation: modulo 11, 13, 19 and 17. Each direct call is an isolated `/api/chat` request, `stream:false`, `keep_alive:0`, top-level `think:false`, `format:"json"`, `seed:29`, `temperature:0`, `num_ctx:3072`, `num_predict:768`, with the original Ollama model tag and full digest checked **before and after** each call. The frozen installed `qwen3.5:27b` digest must match the historical validation plan. Tag checks are **not** atomic model-weight attestation.

1. **Solve-only (4 calls):** independently compute a result with **no claimed answer** in the prompt. Strict `{"answer":"integer"}` response. Score against pure-Python integer arithmetic.
2. **Original claim-review (8 calls):** the exact prior `MTC-PRINCIPAL-VALIDATION-001` request schema and prompt, for four correct / four incorrect claims. Score TP/TN/FP/FN/uncertain/invalid against an *independent* exact oracle; separately compare `suggested_answer` with the oracle and supplied claim. An approval for an incorrect answer remains **FP**, regardless of suggested answer.
3. **Forced-choice (4 calls):** put both candidate answers in one prompt and require `choice=A/B/uncertain` plus an independently computed numeric `answer`. Candidate correctness alternates `B, A, B, A` to expose choice-position bias. Score choice and numeric answer independently.

**All 16 calls are fixed before outcomes** and sequential: solve, review, choice. The same problems and answer alternatives are exposed across lanes and in earlier experiments; this is a narrow **engineering diagnostic**, not a blinded benchmark, holdout evaluation, proof of reasoning or causal experiment. Prompt framing and sequencing are confounded; no model parameters are trained. No LFM3 calls occur in this run.

**Predictions to inspect, not pass/fail gates:** solve-only and choice correct while claim-review approves both answers is consistent with H2. Solve-only also wrong is consistent with H1. Invalid/truncated replies or verdict–suggested-answer mismatch supports further investigation of H3, not a numerical failure verdict. All outcomes (including failures) are retained and scored; no resampling to produce an attractive score.

## 3. Source identity, permission, failure and verification

- `plan`: independently verifies **all prior archives** from original source-bound code, the original 12-call validation and its exact reported score (4 TP, 4 FP, 0 TN, 0 FN, control 2/4), then pins original Qwen tag digest, source/governance/tests/workflow closure, Git head, the exposure class, and all 16 requests. It makes **no model inference** but queries Ollama's local model inventory to verify the installed digest.
- `capture`: **requires a separately owner-created exact-scope approval** for a new, private path; creates an adjacent exclusive `.execution-claim.json` **before the first inference**. The approval cannot be reused after failed/interrupted capture, and no resume or retries are implemented. Pre/post-call digest drift stops the capture **after the failing call has been journaled**. If interrupted, preserve the entire partial archive without resealing or deleting.
- `verify`: offline rechecks the external new manifest identity, every member digest, original historical archives, exact request bytes/hashes, bounded raw Ollama responses, independent mathematical oracle, source closure and reported summary. No new inference, GPU execution, cloud or model downloads.
- `plan`, `capture` and `verify` never write into historical source archives. Raw captures, approvals and receipts are owner-private (0600 in private 0700 directories) and **never uploaded to GitHub CI**.

Only an explicit owner confirmation with the exact scope authorizes the separate **16 Qwen calls**. Merging the PR or planning is not authorization. Existing OLLAMA-LOG-001 server logging is optional: capture and seal a **fresh, overlapping server log** separately if operational context is needed; don't retrospectively attach the prior startup log as execution proof.

Affected invariants: TRI-I02, TRI-I03, TRI-I07, TRI-I08, TRI-I10, TRI-I11, TRI-I13, TRI-I14. No change to native training, admission, previous oracle, MTC original schedules, or original PR #154 receipt source closure.

## 4. Owner-local commands (after review + merge)

Start from a clean checkout of the reviewed **merged** source. Keep the existing three prior archives and installed Qwen digest available. Never select an old archive path as the new output.

~~~sh
cd ~/Trinite
git pull --ff-only
mkdir -m 700 -p "$HOME/trinite-principal/examiner-cause-001"

uv run --no-project python -I -B scripts/mtc_examiner_cause_001.py plan \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --validation-run "$HOME/trinite-principal/validation-001/capture" \
  --out "$HOME/trinite-principal/examiner-cause-001/plan"
~~~

Read `plan/plan.json` and retain its actual printed `examiner_cause_request_identity`. **No models are called during planning.** Do not use the historical validation identity as the new plan identity.

**Only after you explicitly authorize the new 16-call diagnostic**, use the following example as a local operator action. Replace the placeholder with the **actual newly printed plan identity**, then type the phrase at a TTY (not in a Python heredoc).

~~~sh
export REQUEST_ID='sha256:ACTUAL_NEW_DIAGNOSTIC_REQUEST_IDENTITY'
if read -r -p 'Type AUTHORIZE EXAMINER CAUSE ONCE for 16 Qwen calls: ' consent </dev/tty &&
   [[ "$consent" == 'AUTHORIZE EXAMINER CAUSE ONCE' ]]; then
    uv run --no-project python -I -B - <<'PY'
import importlib.util
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cause", Path("scripts/mtc_examiner_cause_001.py"))
cause = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cause)

root = Path.home() / "trinite-principal/examiner-cause-001"
plan_path = root / "plan/plan.json"
approval_path = root / "owner-approval.json"
capture_path = root / "capture"
identity = os.environ["REQUEST_ID"]
raw = cause.mtc.checked_file(plan_path)
if cause.mtc.sha(raw) != identity:
    raise SystemExit("Frozen request identity mismatch")
plan = cause.validate_plan(cause.mtc.strict_json(raw))
value = cause.approval_value(plan, identity, approval_path, capture_path)
cause.mtc.write_new(approval_path, cause.mtc.canonical(value))
print("Owner approval created: 16 calls, one-use; training admission = false")
PY
else
    echo 'Authorization declined; no approval created.'
fi
~~~

Then capture **exactly once**:

~~~sh
uv run --no-project python -I -B scripts/mtc_examiner_cause_001.py capture \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --validation-run "$HOME/trinite-principal/validation-001/capture" \
  --plan "$HOME/trinite-principal/examiner-cause-001/plan/plan.json" \
  --request-identity "$REQUEST_ID" \
  --owner-approval "$HOME/trinite-principal/examiner-cause-001/owner-approval.json" \
  --out "$HOME/trinite-principal/examiner-cause-001/capture"
~~~

Keep the **actual** `examiner_cause_manifest_identity` printed by capture. If there is an error/interrupt, **do not repeat capture** with that approval or reuse its path.

Verify the completed archive (replace the placeholder with the actual manifest identity):

~~~sh
uv run --no-project python -I -B scripts/mtc_examiner_cause_001.py verify \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --validation-run "$HOME/trinite-principal/validation-001/capture" \
  --run "$HOME/trinite-principal/examiner-cause-001/capture" \
  --manifest-identity 'sha256:ACTUAL_NEW_DIAGNOSTIC_MANIFEST_IDENTITY'

uv run --no-project python -I -B -m json.tool \
  "$HOME/trinite-principal/examiner-cause-001/capture/summary.json"
~~~

Do not train with any captured prompts or explanations; all outputs remain quarantined.

## 5. Offline CI scope

`uv run --no-project python -I -B -m unittest discover -s tests/mtc_examiner_cause_001 -v` runs software doubles only; it tests identity-bound plan generation, lane/order exposure and isolation, explicit owner authorization, 16-call recording, all-correct/false-positive synthetic verdict distinction, forced-choice position balance, malformed/truncated replies, no-retry drift failure, source closure drift, archived bytes, and external digest verification. Passing these checks does not certify the target model's performance, its actual reasoning or successful local inference.
