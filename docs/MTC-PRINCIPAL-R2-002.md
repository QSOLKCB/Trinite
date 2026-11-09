# MTC-PRINCIPAL-R2-002 — Ollama model-name compatibility correction

**Status:** additive corrective proposal following independently verified owner-local MTC-PRINCIPAL-R2-001 failure. This is a new script and a new source-bound evidence contract. It neither modifies nor reinterprets any prior run.

## Observed physical failure (9 November 2026)

The owner verified three independent source-bound archives:

- LFM3 v3 `single-002` (manifest `sha256:971a6c8633f09b6d5e2229c7ec81ebcee586355e3e501cba08adcc5f3eecca6e`): two TRAIN answers correct.
- Principal v2 `run-002` (manifest `sha256:33bc06af85e7fb491e5ca87f0cf9d5dc8d07bb73b9362bb061161284bed579d5`): Qwen supplied four valid critiques; LFM3 revised to **3/4**.
- Round 2 `round2-001/capture` (manifest `sha256:5b9c2a7af8427be870a4d05e54fe008da24a71bf2e9fe3260d473076c47c6a27`): archive verified; **four Qwen requests returned HTTP 404**, `model 'sha256-64ae5b1d…' not found`, leaving **zero critiques, zero revisions and four ungraded**.

The previous implementation generated a synthetic Ollama model request name `sha256-<digest>`. Ollama 0.40.15 on the owner's workstation **does not resolve that string as an installed model**. Its HTTP 404 errors were preserved as intended. The result is a **transport/model lookup failure**, not a decrease from 3/4 to 0/4 or a failed mathematics lesson.

## Corrected request protocol

`scripts/mtc_principal_round2_002.py` is an additive successor, with its own source-bound plan v3, one-use authorization v2, execution claim v2 and manifest v4. Its requests use the **real installed Ollama model names**: `qwen3.5:27b` and `lfm3:1.7b`, as frozen in the verified Principal's earlier plan. They are not synthesized from SHA-256 values.

The exact SHA-256 digest of each installed tag remains frozen as **identity evidence** and is checked against `/api/tags` before **and after** each local inference. The returned `model` field must also equal the requested installed name. Both digest checks are part of the new record protocol: the post-call observed digest is retained with the raw bounded HTTP outcome and the request hash. If post-call inventory is unavailable or the digest changed, this is a terminal, journalled failure, not an eligible answer or a retry.

**Limit:** Ollama's mutable tag API does not provide an atomic guarantee that a model tag could not change and be restored between checks. These are independent observed identity checks, **not** a cryptographic attestation of which weights produced a response. Preserve this distinction in any publication or research conclusion.

The original four TRAIN-only tasks, exact 3/4 frozen start, existing temperature/seeds, top-level `think:false` / `format:json`, response limits, loopback-only transport, sequential at most four Qwen critiques plus four LFM revisions, independent arithmetic oracle, no automatic retry and rights quarantine remain in place.

All previous source-bound artifacts are immutable. In particular, do **not** edit or rerun `round2-001/capture`, delete its one-use approval claim, or try to verify it against the new successor code. No learner weights are updated, and `training_admission=false` everywhere.

## Fresh owner-local workflow (after review and merge)

Use a **new** `round2-002` directory, not the failed `round2-001`, and use the **successful first Principal archive** `run-002` as the parent. This is the same second-round question starting from the verified 3/4, now with valid Ollama model names.

### 1. Plan (no inference)

~~~sh
cd ~/Trinite
git pull --ff-only
mkdir -m 700 -p "$HOME/trinite-principal/round2-002"

uv run --no-project python -I -B scripts/mtc_principal_round2_002.py plan \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --previous-run "$HOME/trinite-principal/run-002" \
  --previous-manifest-identity 'sha256:33bc06af85e7fb491e5ca87f0cf9d5dc8d07bb73b9362bb061161284bed579d5' \
  --out "$HOME/trinite-principal/round2-002/plan"
~~~

Retain the **new** `round2_request_identity` separately from the output folder. Do not use the earlier `sha256:55148e...` request identity.

### 2. Explicit one-use owner authorization (no inference)

The successor still needs the exact `owner-approval.json` record created by the **owner**, not a Boolean switch. It binds the new plan, source closure, absolute approval file path, absolute destination, both original manifest identities, exact frozen model names and digests, and eight-call limit. Use a **new** `round2-002/owner-approval.json` and `round2-002/capture` folder; never reuse old approval or its execution claim.

The module provides `authorization_value(plan, identity, approval_path, capture_path)` to construct the exact required record for review. To allow explicit confirmation, use Bash `read -r ... </dev/tty` before invoking a noninteractive `uv run --no-project python -I -B -` script which writes with `mtc.write_new` (mode 0600). Python `input()` inside a heredoc is not appropriate: standard input is already consumed by the program text. Never automatically create approval as a side effect of `plan` or a Github workflow.

### 3. Capture (only after a separate owner decision)

~~~sh
uv run --no-project python -I -B scripts/mtc_principal_round2_002.py capture \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --previous-run "$HOME/trinite-principal/run-002" \
  --plan "$HOME/trinite-principal/round2-002/plan/plan.json" \
  --request-identity 'sha256:ACTUAL_NEW_REQUEST_IDENTITY' \
  --out "$HOME/trinite-principal/round2-002/capture" \
  --owner-approval "$HOME/trinite-principal/round2-002/owner-approval.json"
~~~

The first capture attempt creates an adjacent exclusive one-use execution claim. An interruption consumes that approval; preserve the partial journal, never overwrite or retry without a new reviewed protocol and fresh authorization.

### 4. Verify and inspect (no inference)

~~~sh
uv run --no-project python -I -B scripts/mtc_principal_round2_002.py verify \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --previous-run "$HOME/trinite-principal/run-002" \
  --run "$HOME/trinite-principal/round2-002/capture" \
  --manifest-identity 'sha256:ACTUAL_NEW_ROUND2_MANIFEST'

uv run --no-project python -I -B -m json.tool \
  "$HOME/trinite-principal/round2-002/capture/summary.json"
~~~

All placeholders must be replaced by **actual newly recorded identities**. This remains a same-task, second-prompt experiment without a no-feedback control; 4/4 would not imply weight learning or generalization.

## Offline verification

The hosted fake-Ollama server deliberately **rejects the unsupported digest-looking model name with HTTP 404**, accepts the installed tags, tests post-call digest change detection, returns valid synthetic Qwen and LFM3 outputs, and verifies 3/4 → 4/4 as a software-double case. It also exercises archival and one-use authority. This is not real-model performance evidence.

No cloud runner gets local model weights, private responses or previous archives.
