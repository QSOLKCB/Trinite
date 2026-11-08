# TRI-CHAT-001 — local prototype run and evidence contract

**Status:** implementation proposal; acceptance tests use software doubles, not actual model weights. This work is additive to [TRI-ORCH-001](TRI-ORCH-001.md) and leaves native Trinite source, data, training, fixtures and negative evidence untouched.

## Scope

This prototype is a deterministic rules-first controller and a Python 3.11+ standard-library loopback HTTP/JSON server, with a vanilla HTML/CSS/JavaScript chat interface. It checks bounded integer and GF(3) expressions via an independent integer oracle. An external generative model is optional and **always returns unverified prose**, never authority or exact-oracle labels. No d1 decision adapter in this prototype: TRI-DECIDE-001 is a separately gated future comparison. No cloud fallback, runtime model downloads, Notion API, plugin execution, arbitrary shell, native training or remote listener.

## Offline tests

~~~sh
python3 -I -B -m unittest discover -s tests/trinite_chat_v1 -v
~~~

These source-bound software-double checks cover strict JSON (including excessive nesting), formal operations, oracle disagreement, deterministic policy, browser escaping and UTF-8 conversation history budget, loopback HTTP, Host/Origin and size/method rejection. Slow-trickle tests require independent absolute deadlines for inbound HTTP headers/body and outgoing teacher response headers/body. CI has no model calls. Green tests do not certify actual model compatibility, robustness, GPU performance, general learning, or deployment security.

## Run locally without model weights

~~~sh
python3 -I -B scripts/run_trinite_chat_v1.py --port 8765
~~~

Open http://127.0.0.1:8765/ on the same machine. Exact examples: 17 * 19 and gf3(2 + 2). Other chat is explicitly blocked as teacher_not_configured until a separately authorized local teacher is configured.

## Separately owner-approved local teacher

The compatible external local server must already be running and have actual separately verified model-file hash, runtime version, template and model ID. The CLI never acquires or launches a model. Below are *placeholders*, not observed or verified actual assets:

~~~sh
python3 -I -B scripts/run_trinite_chat_v1.py \
  --port 8765 \
  --model-port 8080 \
  --model-id ACTUAL_MODEL_ID \
  --model-file /absolute/path/to/verified-weights.gguf \
  --model-sha256 ACTUAL_64_HEX_SHA256 \
  --runtime-version ACTUAL_RUNTIME_BUILD \
  --chat-template ACTUAL_TEMPLATE_ID \
  --authorize-local-teacher
~~~

The adapter only calls the configured local loopback OpenAI-compatible chat-completions endpoint. No retries or cloud fallback. Invalid model ID, truncated response, timeout or unsafe output fails closed. The teacher response is classified as model_generated_unverified, while independently checked arithmetic is classified verified_exact.

Configured limits: HTTP JSON 16 KiB; message 8 KiB (each current/history entry); 16 history turns; teacher output 64 KiB; absolute inbound header deadline 2 seconds and body deadline 5 seconds; absolute teacher response deadline 30 seconds (including headers and slowly delivered body); 512 requested output tokens; one synchronous HTTP request at a time. The browser drops oversized assistant answers from replay history and prunes oldest complete pairs before each next request to stay within both 8-KiB entry and 16-KiB request budgets. These are application-level bounds, not proof of whole-system memory, latency or hardware containment. Never publish private chat transcripts or keys.

## Static GitHub Pages

The site/ directory is a public **static project explainer**, not a hosted chat/model. The separate Pages workflow builds and deploys from reviewed main only, even on manual dispatch. Its settings require the repository Pages Actions environment to be enabled before publishing; do not claim Pages is live just because the workflow exists. A public HTTPS Pages site cannot host the private Python model backend.

## Notion

Notion stores a **curated cloud knowledge index only**. The existing workspace now has a dedicated TRINITE — Project Knowledge & Decisions database under QSOLKCB — Project Overview. GitHub and immutable SHA-bound artifacts remain authoritative. Chat messages, raw teacher captures, private data and model outputs are not sent to Notion, and the prototype has no Notion dependency. Future retrieval requires a separate privacy/rights contract.

## Explicit exclusions

No native Trinite model reasoning or learning claim, d1 routing improvement, data admission, distillation, scientific generalization, native backend replacement, old fixture changes or release certification. TRI-O01–O12 remain prospective application invariants; actual test coverage is reported by source and environment.
