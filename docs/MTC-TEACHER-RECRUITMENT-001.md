# MTC-TEACHER-RECRUITMENT-001 — Phi-4 and GLM-4.7 Flash recruitment

**Status:** additive, owner-local proposed experiment. Neither a reviewed/merged PR nor a generated plan authorizes new model execution. **No models were downloaded or run by this PR.** The results of the owner-local historic Qwen/LFM3 experiments are independent, immutable and **QUARANTINED_NOT_ADMITTED**.

## Selection and roles

The owner chose **Phi-4 Reasoning Plus first**, then a smaller-footprint **GLM-4.7 Flash** candidate, as alternatives after GPT-OSS underperformed and the Qwen3.5-27B examiner's observed failures. Candidates are **separately evaluated and approved**; results are never pooled and do not automatically appoint either model.

| Candidate | Exact Ollama tag | Release / quantization | Approximate model download | Proposed role |
| --- | --- | --- | --- | --- |
| Phi-4 Reasoning Plus | `phi4-reasoning:plus` | Microsoft 14.7B dense, Q4_K_M | 11 GB | Math teacher candidate |
| GLM-4.7 Flash community compact | `byczech/glm-4.7-flash-16G-UD-Q3_K_XL:30b` | Z.ai 30B total / 3B active MoE, **third-party** Q3_K_XL quantization | 14 GB | Coding teacher candidate, mathematical qualification |
| GLM-4.7 Flash official | `glm-4.7-flash:q4_K_M` | Official Ollama 30B / 3B active MoE Q4_K_M | 19 GB | Not included in this experiment: requires CPU offloading on a 16 GB GPU |

Ollama catalogue: `https://ollama.com/library/phi4-reasoning/tags`, `https://ollama.com/library/glm-4.7-flash/tags`, `https://ollama.com/byczech/glm-4.7-flash-16G-UD-Q3_K_XL/tags`. A third-party smaller quantization is **not** the same evidence as the official Q4 model; no assertion of identical numerical performance or publisher endorsements is implied. Model size is not peak GPU or RAM use. The local pilot is not proof of coding skill; **a separate code-generation/compile/test gate** is required before a coding supervisor can be appointed.

The user workstation is Ubuntu, Ryzen 9 5950X, RTX 5060 Ti 16 GB, ~30 GB system RAM. Always observe real local memory and output times. Do not alter the default inference machine, install new packages, silently pull a model, use a cloud API or engage an owner runner from this PR.

## Frozen Qwen failure evidence

The new planning gate re-verifies all recorded historical archives and their source closure:

- LFM3 baseline: `sha256:971a6c8633f09b6d5e2229c7ec81ebcee586355e3e501cba08adcc5f3eecca6e`.
- First Qwen Principal: `sha256:33bc06af85e7fb491e5ca87f0cf9d5dc8d07bb73b9362bb061161284bed579d5`.
- Qwen 12-call validation: `sha256:8a8d909c72ed2e47ab0d2903b337939fc7d1e9bc9789ffc0c8e331efd36b65f9`; verified 4/4 false approvals, 4/4 true approvals, zero true rejections; LFM3 no-feedback 2/4.
- **Latest Qwen examiner-cause 16-call audit**: `sha256:34e39cbc6db430a634c14a5d6ca941e7b50af7bd515684f470ad96908fd9d7cf`; verified solve-only **0/4**, review false positives **4/4**, true negatives **0/4**, forced choice **2/4**. The new plan refuses any changed historical score.
- Separately verified OLLAMA-LOG-001 operational logging: never substituted for raw API answers or mathematical proof. New operational logs, when captured and finalized, may be attached as distinct log receipts.

This is a role reassignment/recruitment **proposal**, not a model-deletion operation; do not uninstall Qwen or modify the historical archives.

## Prospective experiment

Each explicitly selected installed candidate gets its **own** plan, owner authorization, capture directory, immutable manifest and model digest. **Exactly 16 sequential independent API calls** per candidate on the four *exposed* previous modular problems: 4 solve without hints, 8 original balanced claim-review prompts, 4 counterbalanced A/B forced-choice comparisons. The first candidate does not see the second candidate's outputs. All evaluations are independently checked using integer arithmetic; Qwen verdicts never act as labels.

The recruitment protocol deliberately **does not send `think:false`**. It leaves Ollama's per-model thinking behavior at the model default, with `format:"json"`, `temperature:0`, `seed:29`, `num_ctx:4096`, `num_predict:2048`, `stream:false`, `keep_alive:0`. This is a different decoding policy from the historical Qwen validation and is **not a controlled same-settings numerical comparison**. A model may still emit empty, non-JSON or token-exhausted final answers; those are retained and scored invalid, not repaired. The original response content and any `message.thinking` remain in private bounded API-response envelopes. The archived `format` constraint is not an independent proof of correct structured output.

**Plan:** independently re-verifies original Qwen evidence and the latest examiner-cause archive, the original exact summary and the installed candidate's tag digest; records Git source commit + source/governance/test/workflow closure and exact request schedule **before outcomes**. Local inventory requests are not inference. Missing tag fails closed; no automatic downloads.

**Capture:** a separately owner-created exact-scope authorization is required for **one** model and **one** new private directory, consumed by `O_EXCL` execution claim **before** first inference. The controller saves each response, error, base64 payload, request SHA256 and post-call tag digest. The pre/post inventory comparison is *not* atomic model-weight attestation. If any call fails, it is recorded as a failed or invalid response; an identity drift stops and preserves partial evidence. No retry or resume with the consumed approval. If a pre-call check fails, the claim remains consumed even without an attempt row.

**Verify:** read-only, offline, recomputes all request hashes, raw bounded API response interpretations, independent oracle classifications, all member identities, source closure, one-use authorization/claim and prior evidence. Requires separately retained real new manifest SHA-256. No Ollama inference, no student training or data admission.

**Outputs:** `rights_status=QUARANTINED_NOT_ADMITTED`, `training_admission=false`, `automatic_promotion=false`. These 16 small reused prompts do not certify general math, code generation, scientific correctness or knowledge. Arithmetic truth remains the standard-library integer oracle. A math or coding teacher must separately pass held-out blinded tests and manual owner-governance review; no automatic promotion.

Affected invariants: TRI-I02/I03/I07/I08/I10/I11/I13/I14. No original experiment/fixture change, historical source-closure reseal, model-weight update, system package installation or PR merge.

## Owner-local steps — Phi-4 first, GLM later

Only **after independent review and merge**, use `uv run --no-project python -I -B` and separate directories. Local model download is an explicit **owner action**, not part of this workflow.

~~~sh
cd ~/Trinite
git pull --ff-only
ollama pull phi4-reasoning:plus
ollama list
mkdir -m 700 -p "$HOME/trinite-principal/recruitment-phi4-001"
~~~

Start Ollama in a separate private log-capture terminal as in OLLAMA-LOG-001; use `OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434`. **Do not seal a live log**.

With the locally installed candidate, freeze **only** the first plan:

~~~sh
uv run --no-project python -I -B scripts/mtc_teacher_recruitment_001.py plan \
  --candidate 'phi4-reasoning:plus' \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --validation-run "$HOME/trinite-principal/validation-001/capture" \
  --cause-run "$HOME/trinite-principal/examiner-cause-001/capture" \
  --out "$HOME/trinite-principal/recruitment-phi4-001/plan"
~~~

Read the plan and independently retain its **actual** newly printed `recruitment_request_identity=sha256:...`. No new GPU calls occur in planning. The old Qwen experiment manifest SHA is not a valid replacement for the new request identity.

### Exact-scope local authorization — separate, deliberate action

**Only if the owner decides to authorize the 16 Phi-4 calls after examining the frozen plan**, set `REQUEST_ID` to the exact value printed, then create authorization using an explicit TTY confirmation:

~~~sh
export REQUEST_ID='sha256:ACTUAL_PHI4_REQUEST_IDENTITY'

if read -r -p 'Type AUTHORIZE RECRUITMENT ONCE for 16 Phi-4 calls: ' consent </dev/tty &&
   [[ "$consent" == 'AUTHORIZE RECRUITMENT ONCE' ]]; then
  uv run --no-project python -I -B - <<'PY'
import importlib.util
import os
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "recruitment", Path("scripts/mtc_teacher_recruitment_001.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
root = Path.home() / "trinite-principal/recruitment-phi4-001"
plan_path = root / "plan/plan.json"
approval_path = root / "owner-approval.json"
capture_path = root / "capture"
identity = os.environ["REQUEST_ID"]
raw = r.mtc.checked_file(plan_path)
if r.mtc.sha(raw) != identity:
    raise SystemExit("Frozen request identity mismatch")
plan = r.validate_plan(r.mtc.strict_json(raw))
if plan["candidate"]["name"] != "phi4-reasoning:plus":
    raise SystemExit("Unexpected model")
r.mtc.write_new(approval_path, r.mtc.canonical(
    r.authorization_value(plan, identity, approval_path, capture_path)))
print("One-use Phi-4 authorization created; training_admission=0")
PY
else
  echo "Authorization declined; no new approval."
fi
~~~

Only after the approval exists, run **once**:

~~~sh
uv run --no-project python -I -B scripts/mtc_teacher_recruitment_001.py capture \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --validation-run "$HOME/trinite-principal/validation-001/capture" \
  --cause-run "$HOME/trinite-principal/examiner-cause-001/capture" \
  --plan "$HOME/trinite-principal/recruitment-phi4-001/plan/plan.json" \
  --request-identity "$REQUEST_ID" \
  --owner-approval "$HOME/trinite-principal/recruitment-phi4-001/owner-approval.json" \
  --out "$HOME/trinite-principal/recruitment-phi4-001/capture"
~~~

Capture prints a **new** `recruitment_manifest_identity=sha256:...`. Keep that exact identity externally and verify without rerunning inference:

~~~sh
uv run --no-project python -I -B scripts/mtc_teacher_recruitment_001.py verify \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --validation-run "$HOME/trinite-principal/validation-001/capture" \
  --cause-run "$HOME/trinite-principal/examiner-cause-001/capture" \
  --run "$HOME/trinite-principal/recruitment-phi4-001/capture" \
  --manifest-identity 'sha256:ACTUAL_NEW_PHI4_MANIFEST_IDENTITY'
uv run --no-project python -I -B -m json.tool \
  "$HOME/trinite-principal/recruitment-phi4-001/capture/summary.json"
~~~

No automatic promotion or student-weight update follows this verification.

### GLM-4.7 Flash — prospective later, separate run

Only once Phi-4 has been evaluated and independently verified should the owner decide whether the third-party GLM Q3 quantization is acceptable for a separate coding-oriented trial. Inspect its model card and licence, pull explicitly, and freeze a **new** plan under a **different** folder and exact model digest. No new model is installed by CI.

~~~sh
ollama pull 'byczech/glm-4.7-flash-16G-UD-Q3_K_XL:30b'
mkdir -m 700 -p "$HOME/trinite-principal/recruitment-glm-q3-001"
uv run --no-project python -I -B scripts/mtc_teacher_recruitment_001.py plan \
  --candidate 'byczech/glm-4.7-flash-16G-UD-Q3_K_XL:30b' \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --baseline-run "$HOME/trinite-principal/run-002" \
  --validation-run "$HOME/trinite-principal/validation-001/capture" \
  --cause-run "$HOME/trinite-principal/examiner-cause-001/capture" \
  --out "$HOME/trinite-principal/recruitment-glm-q3-001/plan"
~~~

The GLM plan needs its own **new, separately owner-authorized one-use 16-call capture** and independently verified manifest. **Never reuse the Phi approval or request digest.** The mathematical recruitment gate is not proof of GLM's coding ability; code-generated changes require a different reviewed suite with compilation, tests, security checks and provenance.

## Offline tests

Hosted CI runs only synthetic software doubles and controller logic: two candidate identity isolation; 16-call solve/review/forced-choice schedule; preservation of thinking/default settings; explicit authorization and no reused approval; failure/digest-drift preservation; invalid/truncated output; no implicit model admission; offline reconstruction, immutable source/historical archives and member hashes. **Passing CI does not mean Phi-4 or GLM was loaded or tested on owner hardware.**
