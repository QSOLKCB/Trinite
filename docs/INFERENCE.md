# CPU inference, export, and inspection

Status: Phase 2 implements CPU eager forward execution, named tensor/quantizer inspection, and bounded detached hidden-state capture. Phase 3 adds safe training checkpoints/loading. A bounded, in-memory ternary byte codec is implemented as preparation only. Generation, complete packed export/import, and edge benchmarks remain planned; Phase 6 is not complete.

## Reference execution

Start with explicit PyTorch CPU eager inference using the same tokenizer, causal mask, operations, quantizer, and checkpoint identity as evaluation. No CUDA import requirement, network fetch, remote code, or hidden fallback in the baseline lane. Future reference generation is greedy with declared BOS/EOS handling, max-new-token limit, context limit, and token tie rule (lowest token ID for equal logits).

The first lane recomputes the full context with no KV cache. Stop generation with an explicit length-limit outcome before exceeding 256 context tokens; never silently drop the prefix. Later caching or sliding-window behavior requires a contract and parity study. Capture mode is explicit and bounded.

Inspection APIs and resource bounds are documented in [PHASE-2.md](PHASE-2.md). The CLI reports model/tensor/code identities and selected capture metadata; it does not load a checkpoint or generate text.

## Export v0 candidate

Store ternary linear codes in a simple fixed 2-bit packing candidate: 00=0, 01=+1, 10=-1, 11=invalid. Flatten each matrix row-major, place four codes per byte from least-significant to most-significant pairs, and require zero padding pairs. Each matrix records shape, scalar float32 scale, code count, byte count, and hash; scalar/tensor byte order is little-endian. Embeddings and normalization tensors remain float32. The final format/schema version is frozen with the implementation and conformance fixtures.

Reject unknown format versions, invalid code 11, unexpected trailing bytes, nonzero padding, inconsistent counts/shapes, missing tensors, non-finite values/scales, or failed hashes before inference. Validate the full artifact inventory and tokenizer/config identity. Never infer that a file is valid solely from its suffix or aggregate hash.

Training exports its actual final forward codes/scales. Export-import tests compare code identity, scales, logits, greedy tokens, and selected hidden states against the reference under declared tolerances. Matching tokens alone is insufficient for geometry-backend equivalence.

### Implemented code-payload preparation

[`packed_ternary`](../src/trinite/packed_ternary/__init__.py) implements only the
code-to-byte primitive `trinite.ternary-2bit.v1`. This identifier names the
primitive's mapping and validation contract, not an export-bundle schema or a
selected model weight format. No model, quantizer, training, checkpoint reader,
historical source bytes or retained evidence is changed. Source receipts that
enumerate all first-party files will acquire a new identity; any such new run
requires its own request and evidence, never a rewritten historical packet.

- `pack_codes(codes)` accepts an exact built-in tuple of built-in integers from
  `{-1, 0, +1}` and returns immutable bytes. Callers supply logical row-major
  order; there is no padding between matrix rows. It does not flatten tensors,
  perform quantization, coerce values, or consume arbitrary iterables.
- `unpack_codes(payload, code_count)` accepts exact built-in bytes and a built-in
  integer count, returning an immutable tuple of integers. It rejects every
  reserved `11` pair, nonzero padding, missing/trailing bytes and invalid types.
  Booleans, numeric subclasses, floats, mutable buffers and tuple subclasses
  are rejected. Empty codes encode as empty bytes with count zero.
- Each call permits **0–1,048,576 codes**, inclusive, and exactly
  `ceil(code_count / 4)` bytes, at most **262,144 bytes**. The fixed count ceiling
  bounds CPU work and output allocation and covers every individual matrix in
  the first reference configuration. It is not a model-size or performance
  claim. The decoder checks the complete payload before allocating decoded
  output; inputs must already be materialized by the caller.
- There is no shape, count, scale, tensor inventory, format header, checksum,
  tokenizer/config identity or file I/O in the payload. The decoder validates
  byte length against the supplied count, but cannot authenticate that count:
  for example `00` hex can represent one through four zero codes. A future
  artifact importer must bind counts/shapes, hashes, scales and the entire
  inventory before inference. Valid code bytes alone do not certify an artifact.

Literal examples: `(0, +1, -1, 0)` encodes to `24` hex; flattening the two rows
`(0, +1, -1)` and `(+1, 0, -1)` encodes to `64 08` hex. The unused high pairs
in the final byte are zero. [`test_packed_ternary.py`](../tests/test_packed_ternary.py)
checks all 9,841 ternary sequences of lengths 0–8, all one/two-byte strings at
every matching count against an independent base-four oracle, hostile inputs,
longer round trips, real resource bounds and standard-library-only execution.

Affected requirements are TRI-I01, TRI-I04, TRI-I05, TRI-I12 and TRI-I14 at this
primitive boundary. Compatibility is additive: no floating-point or numerical
model semantics change. Full artifact rejection, final-forward code/scale
extraction, logit/token/capture parity, resource measurement, independent
replication and release gates remain unfulfilled by this preparation.

## Honest resource accounting

For the first configuration, 1,179,648 linear codes need 294,912 bytes at fixed 2-bit packing. The 67,584 non-ternary parameters need 270,336 bytes as float32, plus 36 float32 matrix scales (144 bytes): **565,392 bytes** of these tensor payloads before metadata/alignment. This is arithmetic from the proposed shapes, not a measured artifact or peak-memory result. The theoretical log2(3) bits per code is not the fixed 2-bit format size.

Training latent tensors, gradients, optimizer states, activations, attention, checkpoints, temporary dequantization, Python/PyTorch libraries, and optional caches add memory. The initial reference implementation materializes floating-point matrices and does not achieve packed-resident inference merely because an export is small.

Benchmark export bytes, peak process RSS, load time, cold/warm prefill, decode rate, context/output lengths, threads, batch size, hardware/OS/backend, observer and inspection overhead, and output correctness. Energy claims require an actual measured energy procedure. Report repeat counts and variability.

The edge goal is CPU execution on ordinary laptops and capable SBCs. Qualify each supported device after measuring its RAM, architecture, available runtime, context, and latency. No “any laptop/SBC” guarantee. ARM and x86 results are separate evidence; packaging support alone is not a hardware performance result.

Adopt a native/bitnet.cpp backend only after architecture/export compatibility, output and capture parity, and a measured cost benefit. No optimized backend is currently selected or implemented.
