# MTC-001 — Master Teacher Corpus: local multimodal capture pilot

**Status:** owner-requested new priority; implementation for review, **no live teacher calls, student training, data admission, release, or model capability result** from this PR. The existing diagnostic-first roadmap is deferred as work priority, not invalidated or weakened. The existing `TRI-I02/I03/I07/I08/I11/I12/I13/I14` requirements and historical frozen experiments remain unchanged.

## Why this is separate

The owner supplied a copy of an `ollama serve` terminal transcript from 8 November 2026 on RTX 5060 Ti. It demonstrates local model loads, GPU offload, `/api/chat` calls and timings for Qwen3-VL 2B, Qwen3-VL 4B Instruct, Qwen2.5-VL 3B and Moondream 1.8B. The upload includes a duplicated console session (96 `/api/chat` log entries, representing **48** calls in the repeated captured session, subject to complete log parsing). Neither prompts, images nor actual model answer bytes appear in that terminal output. It must not be called training data or a model benchmark.

This pilot creates the missing *application-side* collector, without changing native Trinite training. The two captured evidence surfaces are: (a) complete bounded JSON API response bytes including final content and any other returned fields, with the exact API request SHA-256; and (b) a bounded owner-supplied copy of the simultaneously captured `ollama serve` server log. The log is supplementary execution context, **not** sufficient to prove prompts/responses or actual GPU compute. The collector binds the exact script, normative MTC-001 protocol, both workflows, regression tests, root/scoped agent instructions, QSOL-IBAE governance, DATA, INVARIANTS, INSTRUCTIONS and current roadmap in a source/protocol/governance closure. Existing Trinite governance and DATA rules remain authoritative; this local closure is not a complete OS/runtime attestation. The collector does not pull models, enable a cloud endpoint, run student optimizer steps, or import a third-party corpus.

## Two distinct datasets, one master envelope

A small first **28-call** pilot intentionally compares separate learning modalities instead of multiplying arbitrary prompts:

| Lane | Models (examples, installed locally) | Public task families | Calls | Student eligibility |
| --- | --- | --- | ---: | --- |
| A — `formal` | `qwen3-vl:4b-instruct` or another pinned installed local text-capable model | modular addition (8), GF(3) qutrit SUM/SWAP/inverse (6), context-symbol XOR rules (8) | 22 | **Candidate only**; may qualify for a separately approved *formal-only* policy after review |
| B — `vision` | `qwen3-vl:2b` or `moondream:1.8b` | original, deterministic 3×3 occupancy images and exact counts (6) | 6 | **Not native training input**: current Trinite student has no vision encoder. A later explicitly reviewed text-projection/vision adapter would be required |

The teacher is asked for exactly `{"answer":"..."}`; the teacher's answer is **never** the ground truth. Original finite arithmetic/logic and rendered raster calculations provide an independently implemented second oracle path. All teacher raw bytes, including incorrect, malformed or verbose answers, stay in the private capture. An exact semantic oracle match is necessary for the `train-candidates.jsonl` convenience projection but **not sufficient** for data admission, useful learning, independence or legal rights.

`formal` and `vision` have separate teachers and separate task families while using identical evidence schema/capture controls. The other already installed models can be substituted only in a **fresh** request with their own real Ollama digest; don't claim the pilot compared all four models or merge different models' rows under one identity.

## Split, bias and leakage control

Whole structural families are assigned before any teacher response. `train` consists of moduli 5/7, GF(3) SUM and context rules 0/1, plus sparse visual grids. `development` uses modulus 11, GF(3) SWAP, context rule 2 and balanced grids. `locked_test` uses modulus 13, GF(3) inverse-SUM, context rule 3 and dense grids. This is **engineering-exposed and exploratory**, not a fresh generalization benchmark: both the generator code and its families are readable, and earlier Trinite maths/QEC families have historical exposure. No per-row random leakage, no validation/test prompts in train candidates, and no student tuning on teacher results. A later scientific experiment needs its own newly frozen, independent structural test families and a privileged evaluator not available to the learner.

The tiny 28-call first pass is an end-to-end smoke test, not a statistically representative dataset or performance comparison. A large Master Teacher Corpus requires a separate reviewed `MTC-002` version with expanded generated families, deduplication, independent oracles, rights decisions, budgets and diversity analysis; never turn up a mutable `--count` knob after seeing results.

## Collection sequence (owner-local; no hosted Ollama access)

Use `ollama serve` locally, listening on `127.0.0.1:11434`, and save its terminal stream to an owner-private log. Example in a **separate terminal**, only if an Ollama server is not already running:

```bash
(umask 077; mkdir -p "$HOME/trinite-mtc")
chmod 700 "$HOME/trinite-mtc"
(umask 077; OLLAMA_NO_CLOUD=1 ollama serve 2>&1 | tee "$HOME/trinite-mtc/ollama-serve.log")
```

A live `ollama serve` process might already be active; don't start a second. The printed `OLLAMA_NO_CLOUD` value in the old log was `false`, even though its launched `llama-server` command included `--offline`. For this new run explicitly disable cloud behavior at the server process and use **loopback only**. The local capture script rejects non-loopback and redirects and does not use proxy routing. The active model names must already be installed (`ollama list`); no implicit pull/download. Model digests are read from `/api/tags` and rechecked before model inference.

**Step 1 — freeze (no inference).** From a trusted, reviewed TRINITE checkout:

```bash
python3 -I -B scripts/master_teacher_corpus.py plan \
  --formal-model qwen3-vl:4b-instruct \
  --vision-model qwen3-vl:2b \
  --out "$HOME/trinite-mtc/plan-001"
```

Write down the printed `request_identity=sha256:...` separately before making teacher calls. `request.json` binds the exact collector/protocol/workflow/test and applicable governance closure, loopback endpoint, installed model digests, **all 28 public prompts**, immutable lane/split/family assignments and generation settings (temperature 0, seed 0, 64 generated tokens, 2,048-token context). A source or model revision mismatch blocks the capture; don't patch the old packet.

**Step 2 — explicitly authorize capture.** This is a **new, versioned pilot** only. It does not reuse the authorization of earlier local teacher experiments or GPU training protocols:

```bash
python3 -I -B scripts/master_teacher_corpus.py capture \
  --request "$HOME/trinite-mtc/plan-001/request.json" \
  --request-identity 'sha256:THE_FROZEN_REQUEST_DIGEST' \
  --out "$HOME/trinite-mtc/run-001" \
  --server-log "$HOME/trinite-mtc/ollama-serve.log" \
  --authorize-local-capture
```

Exactly **28** new `/api/chat` requests are scheduled, sequentially and with no retry or adaptive refill; invalid responses and runtime failures are retained. `--authorize-local-capture` is mandatory. The script has a hard 32-call ceiling and hard 64-KiB HTTP-response/8-MiB log bounds. An interrupted run leaves partial evidence **without** a complete manifest. Both freeze and capture require an existing destination parent and create their own output directory with mode `0700`; files are created with mode `0600` independently of the invoking shell's umask. The server log copied into a capture is also created owner-only. Ensure the separate original Ollama log is private as shown above. Do not re-run into the same destination or silently repair it; freeze a new request for new calls.

**Step 3 — offline verify and inspect.** Retain the printed manifest identity outside the run folder:

```bash
python3 -I -B scripts/master_teacher_corpus.py verify \
  --run "$HOME/trinite-mtc/run-001" \
  --manifest-identity 'sha256:THE_RETAINED_MANIFEST_DIGEST'
```

`verify` requires all exact members, confirms digest and strict regular-file membership, recomputes generation and a separate oracle implementation, checks every request digest and response byte capture, reconstructs the training-only projection and summary, and requires a separately retained manifest identity. Malformed HTTP JSON bodies and HTTP error bodies are preserved as bounded base64 response bytes with a recorded HTTP status and failure category. Over-limit bodies preserve the first 64 KiB and set `response_truncated=true` rather than pretending to retain the entire response. Socket failures without received bytes remain separately marked. It **does not** replay GPU execution or prove that the runtime log is genuine. A successful check means packet integrity and local numeric consistency, not student usefulness or rights admission.

## File boundaries

Each frozen run is an owner-private, non-uploaded directory:

- `request.json`: source/model/call/split freeze, with **public-only** task fields.
- `attempts.jsonl`: **bounded Ollama API response bytes (base64)** or an explicitly flagged truncated prefix, parsed final field where available, HTTP status, errors, request hash, clock timestamps, model digest and timing metadata, including rejected cases.
- `privileged-targets.jsonl`: full independent-oracle targets, kept separate from model prompts and future student inputs.
- `train-candidates.jsonl`: **only** exact-oracle-matched `train` candidates, marked `CANDIDATE_UNADMITTED`. No visual weights or external-prose training admission is implied.
- `summary.json`: counts grouped by lane, task and frozen split, including failed/invalid responses.
- `ollama-serve.log`: optional owner-provided raw server diagnostic snapshot; may contain local paths or environment details and is **never** published automatically.
- `manifest.json`: exact SHA-256 identity of every member; print and retain its own digest externally.

Raw captures can contain unknown copyrighted or personal content even when prompts are original. Raw teacher outputs, any reasoning traces, image bytes, logs, external model weights and downloaded assets **remain quarantined**. A future rights review must address the exact teacher model/version and its output terms, provenance, ownership and intended distribution. In particular, `docs/DATA.md`'s existing `teacher-origin-formal-policy.v1` only admits strictly parsed verified **numeric operand proposals** from its frozen gpt-oss teacher protocol; MTC-001 **does not** extend it to text answers, vision examples or free-form prose.

## GitHub Actions boundary

**Security correction for PR #141:** No GitHub Actions job checks out branch-selected code onto a persistent owner machine. A collaborator with repository write access can select a branch on `workflow_dispatch`; a checkbox or a branch check inside that same mutable workflow does not secure a self-hosted runner against this. Keep persistent owner machines out of unrestricted repository runner pools. A future Actions-triggered GPU job would require separate owner-reviewed, ephemeral runner isolation and repository-side trust controls beyond this source change.

`.github/workflows/master-teacher-corpus-contract.yml` runs offline contract tests on pull requests and manual dispatch on a hosted runner. It watches this pilot's source plus its normative governance files. `.github/workflows/master-teacher-corpus.yml` is a second **manual, hosted-only preflight**: it runs offline tests and CLI help checks, but intentionally cannot perform local freeze/capture/verify. Both have read-only GitHub permissions and no artifact upload.

For real local work, run the CLI in the reviewed exact checkout on the desktop **from your own terminal**, with the source/protocol hashes, stored log and explicitly retained request/manifest digests. No private prompts, model responses, images, log files, or teacher weights are automatically sent to GitHub. This workflow change does not certify a machine's trust, prevent malicious code on a separately configured runner, or authorise live inference or training.

## Review gates before any TRINITE training

1. Collect and verify an owner-local packet; inspect exact teacher revision/output conditions, classification errors, response schema rejection, duplication, model bias and failures.
2. Assess corpus/data/weight rights **separately**, including original first-party generated assets and teacher-output lineage. A model's software license is not a blanket output-rights certification.
3. Propose a *new*, source-specific `MTC-002` admissibility contract permitting only proved formal inputs/labels, if owner-approved. Quarantine raw outputs permanently; no automatic prompt→answer SFT.
4. Freeze a **new** Trinite native student training request and holdout groups, matched oracle-only baseline, compute/token exposure, random seeds, caps, full replay and unmodified CPU updater. Keep the visual lane as an external benchmark until a separate student vision adapter is reviewed.
5. Train only after explicit approval and report exact-answer+EOS, heldout learning, abstention, hallucination/shortcut controls, resource costs and negatives. A model calling Ollama for inference is not itself TRINITE training or student weight distillation.

**CI test (no Ollama):** `PYTHONPATH=src:scripts python3 -B -m unittest discover -s tests/master_teacher_corpus_v1 -v`.
