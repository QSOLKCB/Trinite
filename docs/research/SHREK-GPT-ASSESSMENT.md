# shrek_GPT: source-bound lessons for Trinite

Reviewed 26 October 2026, read-only. No notebook, model, training, conversion or benchmark was executed. This is a source review supporting [the research update](UPDATE-2026-10-26.md) and [v1 roadmap](../ROADMAP-V1.md), not an enabled implementation or experiment.

## Source identity

- Repository: https://github.com/Amineharrabi/shrek_GPT
- Main commit inspected: `d78ce33b6da092d573907148aaad90d629f950ad`, committed 23 January 2026. This is an older experiment newly brought to our attention, not an October release.
- [Pinned README](https://github.com/Amineharrabi/shrek_GPT/blob/d78ce33b6da092d573907148aaad90d629f950ad/README.md), blob `6ce46a14403e1ed7d77d01b5c54b05286af7906b`.
- [Pinned notebook](https://github.com/Amineharrabi/shrek_GPT/blob/d78ce33b6da092d573907148aaad90d629f950ad/shrek_GPT.ipynb), blob `478b829568c8339e644003bc4447820eb2e2143e`.
- The inspected root contains only these two files. No repository licence, training-text files, released weights, lockfile or test suite was present. GitHub repository metadata reports `license: null`. Treat code/data/weight reuse rights as unestablished; use independently written designs and citations, not copied implementation or Shrek corpus.

## What the small-size claim actually supports

The retained notebook's final llama.cpp log reports 124.44 million parameters and a 63.66 MiB model tensor/file-size figure (4.29 bits per weight), with `ls -lh` reporting a 66M GGUF file. This is compressed GPT-2, not a 64-million-parameter model, not native ternary training, and not a demonstrated 64 MB total-memory deployment.

The same run reports 144.00 MiB of KV cache, 99.66 MiB compute buffer, and 0.19 MiB output buffer in addition to the 63.66 MiB mapped model buffer. These are logged allocations, not a measured peak-RSS total. Its inference context is 4096 despite the model's 1024-token training context; the runtime warns about this mismatch. The fine-tuning tokenizer itself truncates examples to 128 tokens.

The GGUF tensor inventory is mixed: 99 F32 tensors, 24 Q2_K, 24 Q3_K and one Q6_K. The NF4 reload requests FP16 compute and the logged KV cache is F16. Neither small storage nor a Q2_K label establishes integer-only execution or uniform 2-bit storage.

## Architecture and training evidence

Notebook cells 2 and 4 (zero-indexed; IDs `rXasHFsE4dX-` and `teYq39CY5i5N`) load pretrained `gpt2` through `GPT2LMHeadModel`, tokenize merged local text files and use an ordinary causal-LM Trainer. Settings include two epochs, per-device batch four, accumulation four, learning rate 3e-4 and FP16. Retained training output shows 602 steps and training-loss entries. There is no held-out evaluation dataset or pre/post-transformation quality comparison in the notebook.

The final export metadata describes GPT-2 with 12 layers, width 768, feed-forward width 3072, 12 attention heads and a 50,257-token vocabulary. It starts from pretrained GPT-2, not a small model learned from scratch; its capacity/training history is not a fair replacement for Trinite's native small-model lane.

Although the repository description calls this LoRA, the inspected notebook installs PEFT but does not construct an adapter, apply a LoRA configuration, or train one. The source supports ordinary GPT-2 fine-tuning, not an evidenced LoRA recipe.

## The important transformation-lineage caveat

Cell 5 (`KHxhHkNg5kEb`) reloads the saved fine-tuned model with NF4. Cell 6 (`k_9hsfxjCLJO`) then discards that object and reloads the original saved model, applies a random mask to modules selected by `isinstance(module, nn.Linear)`, and saves a separate `gpt2-shrek-chaos-sparse` directory. It supplies no mask seed or measured layer-wise coverage/sparsity inventory. Do not describe this as a demonstrated global 50% parameter reduction.

Cell 8 (`PT0GhxbIG2hk`) copies and converts `gpt2-shrek-chaos`, not `gpt2-shrek-chaos-sparse`, then applies GGUF Q2_K quantization. Thus the displayed source does not establish that the final small artifact contains the random sparsification, nor that its NF4 reload is an upstream stage in the final GGUF export. The README's list of techniques must not be mistaken for one verified serial pipeline.

## Performance and capability limits

The retained final output reports about 78 evaluation tokens/second for one short CPU run, with one inference thread and sampling temperature 1. This is author-retained output, not an independently repeated benchmark or a promise for Trent's Ryzen. The displayed response to `1+1 =` is not a correct arithmetic answer. A runnable, small file is not evidence of general reasoning quality.

Notebook installation commands use floating package upgrades, an unpinned PEFT Git install, and an unpinned llama.cpp clone. A retained runtime log identifies llama.cpp build 6373 / commit prefix `0fce7a12`, but the installation source does not freeze that build. Data inputs are absent, and no source-bound artifact hashes or full execution receipt are included. These gaps limit reproduction.

## Proposed roadmap lessons, independently implemented

1. **Measure deployment memory, not only weights.** For an approved CPU inference experiment, record artifact bytes, parameter count, tensor precisions, tokenizer bytes, context/batch settings, KV and scratch allocations, measured peak RSS, latency and throughput. Declare whether MB means decimal or MiB. Never turn mapped allocation sums into measured RSS.
2. **Make transformations auditable.** Bind input/output hashes and configuration for every fine-tuning, sparsity and quantization stage. Verify the exported artifact descends from the intended checkpoint, with round-trip/load and output-quality tests. This is a strong PROVENANCE-compatible lesson from the notebook's branch mismatch.
3. **Ablate one intervention at a time.** Preserve an unmodified baseline; separately test fine-tuning, quantization and any sparsity method before combinations. Use held-out task accuracy and generalization, not training loss or entertaining output, as acceptance criteria. Random weight destruction is an optional negative control, not an upgrade recommendation.
4. **Keep bounded-integer and low-bit claims separate.** A compressed external GPT-2 lane can be a prospective comparison only after model/data licensing and resource review. It must not be conflated with Trinite's native ternary weights, integer-only inference proposal or historical parameter cap.
5. **Pin and retain.** Freeze model revision, tokenizer, toolchain, seeds and corpus identities/rights; retain both failures and successful outputs. Explicitly validate context bounds rather than relying on a runtime warning.

These are proposed engineering/evaluation gates. No claim is made that adopting them improves Trinite's learned reasoning, and this review authorizes no new model run, capacity-cap change, donor import or runtime implementation.
