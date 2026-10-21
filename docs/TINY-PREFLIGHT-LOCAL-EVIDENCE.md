# Retained tiny-model local preflight, 21 October 2026

This is an **evidence-only negative/blocked preflight**, not execution of the
[64-call output comparison](TINY-OUTPUT.md). No scored slots or student training
ran. No producer, protocol, scorer, adapter, historical fixture, research gate or
roadmap is changed. Affected requirements: TRI-I01/I02/I03/I07/I08/I10/I12/I13/I14.
Native numerical and existing evidence compatibility are unchanged.

## Outcome and limits

- **Needle3:** zero model loads and zero calls. The frozen protocol requires
  `thinking=false`; the pinned official porting documentation describes a forced
  opening think token/internal reasoning, with no established disable control.
  Native greedy/temperature, confidence/suppression gates and penalty defaults,
  full-depth execution, and the exact model/engine pairing remain unverified.
  SDK 3.2.1, distribution 3.3.3 and model-config 3.1.0 are different identities;
  version mismatch alone is not proof of incompatibility. No unsupported option
  was silently ignored and no Needle result is fabricated.
- **Bonsai2-2B:** one non-target CPU-only smoke on the reported Ryzen 9 5950X,
  Haswell/AVX2 backend, one model load, no retries. Loading/invocation/capture
  succeeded; the entire 51-byte response failed the unchanged strict scorer:
  two `{"target": 1234}` objects separated by `</think>` and newlines, rather
  than one answer object. No substring rescue, cleanup, repair or retry is used.
  The SSE content, raw bytes, final-text bytes and result base64 agree exactly;
  the transport contains terminal `stop`, usage of 30 completion tokens and DONE.
- The actual request and runtime receipts retain `enable_thinking=false`,
  reasoning off/none and pure-content capture. The closing marker is generated
  content, not proof that the requested control was omitted or changed.
- Observed call time was approximately 0.773 seconds, load time 1.044 seconds.
  The 771,403,776-byte memory value is the maximum observed process-tree RSS sum,
  sampled about every 10 ms including load; shared pages may be double-counted.
  It is neither an exact peak nor device VRAM. This is not an RTX benchmark,
  model-quality result, matched comparison or causal parameter/precision claim.
- The proposed joint manifest passed structural/hash/limit checks locally, but
  `manifest_execution_ready=false`. Structural validation does not certify
  compatibility. No comparison request was frozen, packet sealed or scored call
  made. The later launch plan is an unexecuted proposal, not authorization.

The local checkout was reported clean before and after at main commit
`bb6758b69143f829dfdb4b44ece17170f81e9093`, tree
`ead80f22a3255d5669daeb7f407c57dc90aad4ac`. The eight retained checkout files were
independently compared to the GitHub tree. Checksums establish byte integrity,
not hardware attestation, honest execution, complete monitoring or independent
reproduction.

## Exact-byte archival scope

[commitment.json](../fixtures/tiny-preflight-local-20261021/commitment.json)
indexes the content-addressed [objects.zip](../fixtures/tiny-preflight-local-20261021/objects.zip).
It preserves 1,484 original logical member byte streams in 572 unique objects,
including raw transport, complete smoke attempt, prompts/configurations,
compatibility and rejection receipts, actual adapter/source snapshots, command
and test logs, failures, source identities, provenance and original SHA256SUMS.
The public ZIP is 1,416,318 bytes; original logical names map to immutable
`objects/<sha256>.bin` members. Treat all captured code as inert evidence.

The supplied original ZIP identity is
`sha256:2b06fcd8685fd022da26d19837517c6f5affc10c63e4ce4bcc203913b140f31d`
(4,685,694 bytes, 1,488 members). Every original member is accounted for by an
exact byte length/hash and retained object or explicit exclusion. Only the
7,960,079-byte Bonsai tokenizer metadata and three redundant Needle HTML pages
are excluded; corresponding metadata identities, source-text excerpts and
acquisition provenance remain. Excluded bytes cannot be reverified from this
subset, and the original ZIP cannot be reconstructed. The original SHA256SUMS
covers 1,487 members; the public commitment separately binds SHA256SUMS itself.
There are no new byte transformations in this publication. The supplied archive
already contains redacted derivatives and its own exclusions at
`reports/bundle-exclusions.json`; those are preserved, not retrospectively
resealed as originals. Model weights and executable runtime binaries were not
included in that supplied archive: their identities are records, not locally
reverified model/runtime payloads.

Privacy review found no obvious credential or signed-token markers. Task-local
paths and hardware details are retained to avoid silently changing byte-bound
receipts; this is not a general guarantee that logs can never contain private
information. No unrelated personal inventory is intentionally published.

### Upstream attribution

Needle source is retained from
[cactus-compute/needle b4fc11b](https://github.com/cactus-compute/needle/tree/b4fc11b04f1852888d375c8c6dfbaca44572d014),
under Apache-2.0. Its exact license is `assets/review/needle-license.txt` in the
object archive; this notice also applies to the seven-file nested
`assets/needle/sdk-source-snapshot.zip`, which has no internal license copy.
Needle model/tokenizer provenance and notices remain in
`assets/review/needle-model-README.md` and the acquisition records.

Prism runtime source is retained from
[PrismML-Eng/llama.cpp e3de715](https://github.com/PrismML-Eng/llama.cpp/tree/e3de715222b2667c54a267441159a5729f11155e),
under MIT, with exact copyright/license at `assets/review/prism-license.txt`.
The [Bonsai model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/tree/683f7be2629342a50987627b40881e54761a2430)
and Apache-2.0 notices are retained at `assets/review/prism-model-README.md`;
template/metadata are retained only as provenance evidence, not native training
admission. Retained upstream material is unmodified byte evidence, not relicensed
as Trinite code. Original license/copyright notices remain authoritative.

## Verification, gaps and dormant findings

Run with Python 3.11+ and no third-party dependencies:

```sh
python -I -S -B scripts/check_tiny_preflight_evidence.py
python -I -S -B -O scripts/check_tiny_preflight_evidence.py --self-test
python -B -m unittest discover -s tests -p test_tiny_preflight_evidence.py -v
```

The checker bounds archive/member expansion, rejects unsafe/duplicate members,
checks all retained hashes and original manifest coverage, reconstructs SSE
content and terminal state, cross-checks counts/configuration/receipt bindings,
and calls the **actual pinned repository scorer and manifest validator**. Only
the independently hashed repository `tiny_output_v1.py` and `contracts.py` are
compiled in an isolated no-site process from a private temporary source tree.
No uploaded adapter, model SDK, nested archive code, runtime, inference or
training is executed. This is not a new `check_assets` run: excluded binaries
and model weights are unavailable here. Negative tests exercise corruption,
truncated/invalid-terminal transport, changed output/counts/readiness/settings,
source mismatch, and environment/current-directory module injection.

Retained original tests report 13 comparison tests normally and under `-O`, 10
adapter tests normally and under `-O`, and 51 upstream tests passing. Root tests
passed 405 on the normal local Python 3.14 context; the sandbox Python 3.11 run
retained 8 failures and 5 errors from Phase 5 context/environment assumptions.
These are separate environments, not contradictory reruns silently replaced by
a green result. Adapter tests are synthetic and do not certify native settings.
The initial incorrect Prism backend directory and its failure logs are retained;
a separately recorded correction to the same CPU library preceded the only call.

There is an explicit bootstrap gap: initial discovery exited 2 before the logger;
its command is retained in `logs/bootstrap.json`, but stdout remains only in the
original tool transcript and stderr was suppressed. Do not claim every original
command stream is archived.

Static review also found dormant issues in captured, unexecuted paths: Needle
launch configuration identity/relative paths differ from `run_smoke`'s expected
SHA-256/resolved-path form; a future SSE adapter should require a proper terminal
reason before DONE, not DONE alone. The actual retained Bonsai stream does have
terminal `stop`. These are documented findings, **not fixes or permission to
alter the frozen protocol**. The retained historical adapter remains unchanged.
A future corrected, source-bound runtime/protocol review is required before any
scored launch. Existing learning, Phase 5, scaling and release gates stay blocked.
