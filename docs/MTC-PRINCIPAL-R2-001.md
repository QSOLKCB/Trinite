# MTC-PRINCIPAL-R2-001 — Second-round cumulative Qwen → LFM3 feedback

**Status:** proposed additive local experiment, not executed on owner hardware by CI. This is a **new script**, specifically to avoid changing the source closure needed to verify the historical MTC-LFM3 v3 and Principal v2 receipts.

## Why a new version

Owner-verified `single-002` baseline: 2/4 TRAIN arithmetic answers correct. Owner-verified `run-002` first Principal supervision: 3/4 revised answers correct, 4/4 valid Qwen critiques. The remaining error was `(8+15) modulo 11`: LFM3 answered 2, whereas the independent exact oracle computes **1**. Qwen incorrectly marked 2 "correct"; it is **not** the authority for mathematical correctness.

A repeat of the existing Principal v2 script would start at **2/4**, not improve from **3/4**. This experiment instead takes the **four recorded revised LFM3 responses from prior verified `run-002`** as its new baseline, and gives Qwen one fresh opportunity to critique those outputs. Every result is compared independently to the same arithmetic oracle. Both corrected-answer gains and regressions are reported.

## Design/limits

- The `plan` call re-verifies **both** existing source-bound archives, requiring exact externally retained Principal manifest identity, recursively bound original LFM3 manifest, four completed TRAIN revisions, exactly the observed 3/4 revised baseline, valid exact JSON and unchanged model identity. It freezes SHA-256 of each prior Principal row, exact revised answers, original correctness classifications, source/governance/workflow closure, local model digests and the existing `think:false`, `format:json` policy.
- Plan and capture destinations must be new directories outside both parent evidence archives; nested destinations are rejected before model inventory or inference so the verified parent membership remains unchanged.
- No held-out items are exposed to Qwen; only the original four TRAIN tasks are used. This is an **engineering-exposed repeat**, not a new independent generalization benchmark. Both models' weights are unchanged; new prompting does not constitute lasting student learning.
- `capture` requires a separate owner authorization and exact plan digest. At most **four Qwen critiques + four LFM3 revision calls** are issued, sequentially on loopback. Exactly the prior revised text becomes the next round's teacher answer. Existing Principal prompt rules, 300-second limit, 64 KiB response cap, explicit model digests and owner-private file permissions are inherited.
- Principal verdict accuracy is scored against **the independent oracle**. A Principal `correct` verdict for an incorrect answer is counted as a verdict disagreement, never used to replace truth.
- Each returned call envelope is fsynced immediately to a separate append-only call journal before parsing or starting a dependent stage. A failed, malformed or non-UTF-8 critique does not trigger an LFM3 revision. Unknown or interrupted outcomes are preserved, **not silently retried**; this collector has no resume action. Invalid or incomplete results remain quarantined with no promotion.
- The report separates corrected, retained, regressed, still incorrect and ungraded outcomes, and reports how many Qwen judgments agreed/disagreed with the oracle. An entirely failed evaluation is **not** an arithmetic regression.
- `verify` works without new inference; it recursively validates both parent manifests, exact source/file identities, the append-only call journal, all new Qwen and LFM request hashes, feedback and oracle classifications, the full new manifest, and independently computed summary.
- There is **no** dataset admission, student training, model weight change, auto-training or cloud upload. Source/manifest SHA-256 receipts identify recorded bytes but are not hardware execution attestations.

## Owner-local commands after code review and merge

Use a clean reviewed main checkout. Do not delete, overwrite, move or edit `single-002` or `run-002`.

**1. Freeze:** no inference.

~~~sh
cd ~/Trinite
git pull --ff-only
mkdir -m 700 -p "$HOME/trinite-principal/round2-001"
uv run --no-project python -I -B scripts/mtc_principal_round2_001.py plan \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --previous-run "$HOME/trinite-principal/run-002" \
  --previous-manifest-identity 'sha256:33bc06af85e7fb491e5ca87f0cf9d5dc8d07bb73b9362bb061161284bed579d5' \
  --out "$HOME/trinite-principal/round2-001/plan"
~~~

Keep the printed **round2_request_identity** separate from the run directory, verify the model identity and call budget, and authorize **only if desired**.

**2. Capture:** at most eight explicit loopback inferences, no training.

~~~sh
uv run --no-project python -I -B scripts/mtc_principal_round2_001.py capture \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --previous-run "$HOME/trinite-principal/run-002" \
  --plan "$HOME/trinite-principal/round2-001/plan/plan.json" \
  --request-identity 'sha256:ACTUAL_NEW_ROUND2_REQUEST' \
  --out "$HOME/trinite-principal/round2-001/capture" \
  --authorize-round2
~~~

Do **not** type the placeholder above as a literal digest. An interrupted run is retained as partial evidence and must not be resumed or overwritten without a separate reviewed recovery protocol.

**3. Verify and inspect:** no inference.

~~~sh
uv run --no-project python -I -B scripts/mtc_principal_round2_001.py verify \
  --teacher-run "$HOME/trinite-mtc-lfm3/single-002/capture" \
  --previous-run "$HOME/trinite-principal/run-002" \
  --run "$HOME/trinite-principal/round2-001/capture" \
  --manifest-identity 'sha256:ACTUAL_NEW_ROUND2_MANIFEST'
uv run --no-project python -I -B -m json.tool \
  "$HOME/trinite-principal/round2-001/capture/summary.json"
~~~

Preserve `round2_manifest_identity` separately. Even a result of 4/4 is a second-pass **same-task** gain, not evidence of generalization or attribution without a randomized no-feedback control and fresh held-out tasks.

## Validation

GitHub-hosted CI uses only a loopback software-double with deliberately wrong initial teacher answers, a wrong first-round verdict, a 3/4 prior revision score and a potential 4/4 second-round score. It checks immutability, model request shapes, parent source-bound identities, split isolation, no unauthorized calls, response failure controls and no training admission. This tests protocol correctness, not real model capability.
