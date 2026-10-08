# CPU inference, export, and inspection

Status: Phase 2 implements CPU eager forward execution, named tensor/quantizer inspection, and bounded detached hidden-state capture. Generation, checkpoints/loading, packed export/import, and edge benchmarks remain planned.

## Reference execution

Start with explicit PyTorch CPU eager inference using the same tokenizer, causal mask, operations, quantizer, and checkpoint identity as evaluation. No CUDA import requirement, network fetch, remote code, or hidden fallback in the baseline lane. Future reference generation is greedy with declared BOS/EOS handling, max-new-token limit, context limit, and token tie rule (lowest token ID for equal logits).

The first lane recomputes the full context with no KV cache. Stop generation with an explicit length-limit outcome before exceeding 256 context tokens; never silently drop the prefix. Later caching or sliding-window behavior requires a contract and parity study. Capture mode is explicit and bounded.

Inspection APIs and resource bounds are documented in [PHASE-2.md](PHASE-2.md). The CLI reports model/tensor/code identities and selected capture metadata; it does not load a checkpoint or generate text.

## Export v0 candidate

Store ternary linear codes in a simple fixed 2-bit packing candidate: 00=0, 01=+1, 10=-1, 11=invalid. Flatten each matrix row-major, place four codes per byte from least-significant to most-significant pairs, and require zero padding pairs. Each matrix records shape, scalar float32 scale, code count, byte count, and hash; scalar/tensor byte order is little-endian. Embeddings and normalization tensors remain float32. The final format/schema version is frozen with the implementation and conformance fixtures.

Reject unknown format versions, invalid code 11, unexpected trailing bytes, nonzero padding, inconsistent counts/shapes, missing tensors, non-finite values/scales, or failed hashes before inference. Validate the full artifact inventory and tokenizer/config identity. Never infer that a file is valid solely from its suffix or aggregate hash.

Training exports its actual final forward codes/scales. Export-import tests compare code identity, scales, logits, greedy tokens, and selected hidden states against the reference under declared tolerances. Matching tokens alone is insufficient for geometry-backend equivalence.

## Honest resource accounting

For the first configuration, 1,179,648 linear codes need 294,912 bytes at fixed 2-bit packing. The 67,584 non-ternary parameters need 270,336 bytes as float32, plus 36 float32 matrix scales (144 bytes): **565,392 bytes** of these tensor payloads before metadata/alignment. This is arithmetic from the proposed shapes, not a measured artifact or peak-memory result. The theoretical log2(3) bits per code is not the fixed 2-bit format size.

Training latent tensors, gradients, optimizer states, activations, attention, checkpoints, temporary dequantization, Python/PyTorch libraries, and optional caches add memory. The initial reference implementation materializes floating-point matrices and does not achieve packed-resident inference merely because an export is small.

Benchmark export bytes, peak process RSS, load time, cold/warm prefill, decode rate, context/output lengths, threads, batch size, hardware/OS/backend, observer and inspection overhead, and output correctness. Energy claims require an actual measured energy procedure. Report repeat counts and variability.

The edge goal is CPU execution on ordinary laptops and capable SBCs. Qualify each supported device after measuring its RAM, architecture, available runtime, context, and latency. No “any laptop/SBC” guarantee. ARM and x86 results are separate evidence; packaging support alone is not a hardware performance result.

Adopt a native/bitnet.cpp backend only after architecture/export compatibility, output and capture parity, and a measured cost benefit. No optimized backend is currently selected or implemented.
