# OLLAMA-LOG-001 — Optional local Ollama server-log evidence receipt

**Status:** additive, offline-only operational evidence protocol. This is **not** a native Trinite training record, mathematical proof, replay attestation, or model-output source. It does not change historical MTC, Principal, SQLite, native learner or scientific fixtures. This proposal does **not** authorize live inference, cloud access, model downloads, historical resealing, or an automatic merge.

## Purpose and evidence classes

An owner-local Ollama server can be run with both stdout and stderr captured to a private file. This new, separate receipt associates that **finalized** log with an existing **separately identified** Trinite experiment manifest.

- **OBSERVED from log bytes:** SHA-256, byte count, line count, explicit UTC-offset timestamp interval, unzoned GIN HTTP-access interval (kept separate), version and listening address when found, cloud-enabled/disabled messages when found, and advertised GPU metadata when found.
- **OBSERVED from supplied experiment manifest bytes:** its SHA-256, size and schema. Require an *independently retained* expected SHA-256 and reject mismatches.
- **OPERATOR DECLARATION:** a log was stopped or rotated before sealing, and the selected experiment manifest is a related reference. Neither declaration is independently attested by this tool.
- **NOT CLAIMED:** that the log covers the entire server lifetime, that each request corresponds to that experiment, that a particular model produced an answer or actually ran on the named GPU, that CUDA timestamps attest computation, that server logs verify model-weight identity, or that Qwen/LFM3 output is correct. No math/classification inputs are admitted to training.

A SHA-256 receipt binds **bytes** and this source version. It cannot prove the source of those bytes or independently establish temporal association. Record correctness remains with the original source-bound MTC/examiner archive and independently calculated oracle.

The owner uploaded an example startup log reporting **Ollama 0.40.15**, `127.0.0.1:11434`, `Ollama cloud disabled: true`, and CUDA GPU discovery of an RTX 5060 Ti with `15.5 GiB` total and `14.6 GiB` available. The file also contained unzoned GIN request lines but **no previously completed Qwen examiner inference requests**; it began *after* the previous validated examination. Do **not** describe that startup log as execution evidence for the earlier experiment. The original failed 64-byte port-collision log has no successful-version/GPU facts, and a receipt must leave those fields empty.

## Scope and controls

- New self-contained Python **standard-library** tool: `scripts/ollama_log_receipt_001.py`; offline subcommands `seal` and `verify`, neither connects to Ollama, GPUs, GitHub, the network, or any cloud.
- Input log and experiment `manifest.json` must already be **owner-controlled single-link regular files** (mode 0600 or equivalently no group/other permissions), with private owner-controlled immediate parent directories (mode 0700), not symlinks/FIFOs or hardlinks. The tool uses `O_NOFOLLOW` and `O_NONBLOCK` for defensive reads.
- A log must be nonempty, at most **64 MiB**, with no line larger than **64 KiB**, and its final line must end with a newline. Changes to size, `mtime_ns` or `ctime_ns` during the read fail closed. This is not a lock against concurrent writers. Do not run `seal` while `ollama serve` is still writing: stop the server cleanly or rotate/close the log first. `--confirm-finalized` is a deliberate operator assertion, not a process-state sensor.
- A manifest input is at most **1 MiB**, whereas the serialized output receipt has its own **8 MiB maximum**. Both `seal` and `verify` enforce the exact same receipt limit; `seal` rejects a larger receipt **before creating a file**, even if the input log is below its separate 64 MiB limit. Repeated observed startup/GPU events can expand the structured receipt beyond 1 MiB. A log that would exceed the receipt limit requires a separately reviewed protocol or evidence-preserving session rotation; neither command silently truncates observations.
- A manifest must have a `trinite.*` schema and a nonempty membership digest map. This receipt **does not recursively verify** the manifest's underlying archive; use the archive's own exact-source `verify` first.
- All log material remains local and is **never copied into the receipt**. The new file stores only observed metadata and SHA-256 identities, not raw log lines, environment maps, prompts or responses. Metadata such as GPU device names remains private and is not uploaded to GitHub.
- Version and GPU values appear **only if matching literal server log events exist**. Multiple startup events are retained as separate observations. Explicit timezone-offset `time=` records are converted to UTC. GIN dates without offsets remain unzoned literals, *not* inferred to be Adelaide/UTC. Missing events yield null/empty fields; never fill them from local hardware context or the user's prior output.
- Receipt identity is a separately printed SHA-256 of canonical JSON. Verification requires that exact identity, reopens both inputs, reconstructs the metadata and requires source/test/docs/workflow/governance closure parity. No silent repair, overwrite, retroactive log modification, retry or resealing.
- The source-bound protocol is limited to current operational logging observations. A larger log needs a **new reviewed version or planned rotation**, not bypassing bounds. Same-UID local adversaries and kernel-level tampering are not excluded; file hashes and permission bits do not attest provenance.

This opt-in attachment is aligned with **TRI-I02/I03/I07/I08/I13/I14**, especially separation of teacher data from native training, scope-limited identity claims, failure retention, and versioned evidence semantics.

## Owner-local usage after PR review and merge

**First**, continue logging from a manually launched server if desired, but do not seal a live log. When a session is finished, stop the server in its original terminal with Ctrl+C, then inspect its final bytes; capture the final SHA-256 separately. Historical server output that was never logged cannot be reconstructed from the new file.

**Prepare a fresh private receipt directory:**

~~~sh
cd ~/Trinite
git pull --ff-only
mkdir -m 700 -p "$HOME/trinite-log-receipts/session-001"
~~~

**Seal:** this example selects the verified **prior validation** manifest as a *context-only association* for the later startup log. It **does not claim** the old validation ran during this log's observed interval. Use a new overlapping log from a subsequent experiment for meaningful execution context.

~~~sh
uv run --no-project python -I -B scripts/ollama_log_receipt_001.py seal \
  --log "$HOME/trinite-ollama-logs/ollama-serve-20261109T155349Z.log" \
  --experiment-manifest "$HOME/trinite-principal/validation-001/capture/manifest.json" \
  --experiment-identity 'sha256:8a8d909c72ed2e47ab0d2903b337939fc7d1e9bc9789ffc0c8e331efd36b65f9' \
  --session-label 'post-validation-startup-context' \
  --out "$HOME/trinite-log-receipts/session-001/receipt.json" \
  --confirm-finalized
~~~

Use that command **only once the specific log is closed**, with both input paths and the previously verified experiment manifest exactly as shown. The command prints `ollama_log_receipt_identity=sha256:...`; independently retain the *actual* printed digest. Existing output paths are refused rather than overwritten.

**Verify:** replace the example placeholder with the **actual newly printed receipt identity**, never with the older experiment-manifest identity.

~~~sh
uv run --no-project python -I -B scripts/ollama_log_receipt_001.py verify \
  --receipt "$HOME/trinite-log-receipts/session-001/receipt.json" \
  --receipt-identity 'sha256:ACTUAL_PRINTED_RECEIPT_IDENTITY' \
  --log "$HOME/trinite-ollama-logs/ollama-serve-20261109T155349Z.log" \
  --experiment-manifest "$HOME/trinite-principal/validation-001/capture/manifest.json"
~~~

A previously printed *mid-session snapshot* SHA-256 is not necessarily the final log hash, since more bytes can be appended. A log changed after sealing must make verification fail; preserve both receipt and observed failure and create a **separate versioned record** if genuinely needed.

## Tests and limitations

Run `uv run --no-project python -I -B -m unittest discover -s tests/ollama_log_receipt_001 -v`. Hosted CI runs synthetic text fixtures only, including a representative CUDA startup log, unzoned GIN lines, an explicit failed port-collision log, missing/invalid manifests, wrong external identities, changed bytes, a 25,000-line repeated-startup reproduction that produces a receipt above 1 MiB, consistent seal/verify limits, rejection of receipts exceeding 8 MiB, oversized/incomplete lines, symlinks/hardlinks, public file modes, source closure drift and no-clobber. No self-hosted runner, GPU benchmark, real-model inference, capture upload or automatic publication is involved.

**Governance:** this supplement is *OBSERVED* file data plus an explicitly marked *OPERATOR-DECLARED* association, never an *independently verified execution result*. It must not be silently used to resolve a failed or incomplete MTC model-output claim.
