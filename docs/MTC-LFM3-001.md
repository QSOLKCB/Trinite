# MTC-LFM3-001 — five-model owner-local Ollama capture

**Status: review proposal; NO owner hardware inference has occurred.** This is a separate 30-call text-teacher experiment using five explicitly selected, installed LFM3 models and six frozen arithmetic/GF(3) prompts each. It does not change the historical MTC-001 28-call evidence or automatically add teacher outputs to Trinite training.

**Dependency:** This workflow PR is stacked on TRI-CHAT-DB-001 (#146), because its optional chat smoke exercises that PR's SQLite-enabled server flags. Merge #146 first. No Trinite training, model download, artifact upload, publication, or external inference is authorized by CI tests.

## Operations and capture contract

1. **Plan:** Read five exact installed text-capable LFM3 model tags and their SHA-256 digests from local Ollama /api/tags. Freeze 6 original formal prompts per model, exact generation settings, split/family assignments, source/governance/workflow/test closure, schedule and 30-call budget into a private plan.json. Print its request_identity SHA-256. No inference or pull.
2. **Capture:** A separate approved dispatch supplies that exact request identity. Send sequential local /api/chat requests with stream false, temperature 0, seed 17, context 2048, at most 96 output tokens, a 64 KiB response cap and 70-second transport deadline per request. Record exact request hashes, raw API response bytes, HTTP failures, installed digests and independent oracle comparisons. Bad results are retained; success is never presumed.
3. **Verify:** Without Ollama, recompute the frozen source closure and expected task identities, request hashes, parsed response/oracle classifications, all archive member SHA-256 values and separately retained manifest_identity. Require all 30 attempts. A passed verification confirms **recorded bytes and local arithmetic consistency**, not attested Ollama execution or student learning.
4. **Resume:** With explicit resume=true, continue only an incomplete run with an intact prefix of frozen attempts; do not rerun recorded rows. A partial JSON line or model/source drift must fail closed.

All outputs are **QUARANTINED_NOT_ADMITTED**; training_admission=0. Six formal tasks are exposed engineering checks, not a scientifically locked test of reasoning. The five models share possible training ancestry and are not independent truth oracles. Non-generative d1 and vision/audio/omni inference need separate versioned protocols and cannot be substituted into this text-only capture.

## Runner security prerequisites

The owner-local workflow targets an explicitly labeled self-hosted runner: self-hosted, Linux, X64, trinite-lfm3. **Do not use an unrestricted persistent desktop runner for arbitrary repository workflows.** Trinite MTC-001 documented the problem: branch-selected GitHub Actions code can execute on persistent hardware. The same-job main-branch guard is not a complete security boundary.

Before enabling this workflow, configure a GitHub **runner group restricted to this owner-approved workflow**, preferably with an ephemeral or isolated owner-local runner, and a protected environment named **trinite-owner-local** with required owner reviewers and main-only deployment branches. Protect main. Configure repository Actions variable TRINITE_OWNER_RUNNER_ARMED=yes *only once those protections work*. Configure environment variable TRINITE_OWNER_CHECKOUT as the absolute path of a manually reviewed, clean, exact main checkout.

The workflow never performs actions/checkout on the owner runner, never git pulls, and never uploads private artifacts. It refuses a dirty checkout and requires reviewed_sha to equal both github.sha and local Git HEAD. Neither that check nor a digest certifies the operating system or protects from malicious same-account processes. If you cannot configure the runner group and environment protection, use the fully local CLI rather than dispatching this workflow.

The runner must already have **uv, Ollama and Python**. The job sets UV_OFFLINE=1 and UV_PYTHON_DOWNLOADS=never. An already-running Ollama server is reused without being killed. If none is available at 127.0.0.1:11434, the job starts Ollama with OLLAMA_NO_CLOUD=1 and OLLAMA_HOST=127.0.0.1:11434 and stops *only its own* spawned process afterward. Owner must separately verify the settings of any existing Ollama service. No model pull occurs.

## GitHub Actions dispatch sequence

After merging #146 and this PR, manually prepare a clean checkout of the reviewed main commit on the owner runner. Navigate to Actions → **MTC-LFM3-001 owner-local Ollama capture (guarded)** → Run workflow on main. Supply a new run_id such as liquid-001, the reviewed_sha of that exact main commit, and type AUTHORIZE-LOCAL-MTC-LFM3.

**First dispatch:** choose operation=plan and enter five real installed, distinct Ollama model tags in the models field, separated by commas. Find their exact spellings with ollama list locally. Retain the printed request_identity outside the local run directory.

**Second dispatch:** choose operation=capture using the same run_id and exact request_identity. Explicitly authorize again. Optionally set initialize_db=true and chat_smoke=true to run both requested local Trinite commands. If an existing database is present it is validated rather than initialized again. Keep the printed manifest_identity separately; if capture is interrupted, preserve partial evidence and use resume only with the exact same source, models, schedule and approval.

**Third dispatch:** choose operation=verify, same run_id, exact manifest_identity; this does not perform more model calls.

The owner-local run directories are under $HOME/trinite-mtc-lfm3/RUN_ID/plan and /capture with mode 0700 and 0600 files. An Ollama process started by the workflow writes a private log in the same owner-private root. No raw prompts, private responses, logs, images or model weights are uploaded to GitHub.

## Optional SQLite initialization and chat smoke

The optional flags run the following exact command **only if** data/trinite.db does not exist. Existing databases undergo read-only schema validation first, with no overwrite:

~~~sh
uv run --no-project python -I -B scripts/trinite_chat_sqlite_v1.py init --enable-persistence
~~~

With chat_smoke=true, the workflow validates the existing DB, checks that 127.0.0.1:8765 is free, then starts:

~~~sh
uv run --no-project python -I -B scripts/run_trinite_chat_v1.py \
  --port 8765 \
  --enable-persistence \
  --db-path data/trinite.db
~~~

The launcher checks GET / on localhost, then stops the child process group with a bounded shutdown. It is intentionally **not a persistent chat daemon**, and it does not initiate a teacher call. To keep the chat interface running for interactive use, start it separately from your own terminal or an owner-managed service after review. GitHub Pages cannot host the local Python/SQLite backend.

## Offline tests and claim boundaries

~~~sh
python3 -I -B -m unittest discover -s tests/mtc_lfm3_001 -v
~~~

Hosted CI uses a fake loopback Ollama server, exercises freeze/capture/verify, five model digests, exact outputs, rights quarantine, interrupted/resumed captures, errors and tampering, plus the owner-launcher input validation. It never uses an RTX card, installed model weights or the owner runner.

Existing Trinite TRI-I02/I03/I07/I08/I11/I13/I14 and TRI-O01/O02/O04/O06/O08/O10/O11 remain in force: generated teacher text is not authority, independently verified exact labels are distinct from model prose, exposure is explicit, and admission/rights decisions require an independent new gate. Notion remains a curated knowledge index, not a raw capture repository.

**Restrictions:** This plan does not authenticate outputs by a hash alone, certify model-output rights, benchmark novel heldout generalization, secure a same-account attacker, or grant student training. A future MTC-002 admission contract must explicitly review dataset rights, model terms, split leakage and target use before any corpus rows are promoted.
