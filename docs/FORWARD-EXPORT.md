# Immutable final-forward export candidate

Status: encode-only engineering increment. The native quantizer, model, updater,
checkpoint readers, training protocols and retained evidence are unchanged.
This is **not** a selected winning weight format, an importer, packed-resident
inference, text generation or completion of Phase 6.

Affected requirements: TRI-I01, TRI-I04, TRI-I05, TRI-I07, TRI-I08, TRI-I12 and
TRI-I14 at the export boundary. Compatibility is additive: the three new
production modules live under `src/trinite/forward_export/`; no historical flat
maths/retention source is rewritten. Enumerated whole-package source receipts
necessarily acquire a new identity and require separately bound evidence.

## Explicit API and trust boundary

In the complete [hash-locked CPU environment](GETTING_STARTED.md):

```python
from trinite.forward_export.source import read_source_snapshot
from trinite.forward_export.encoder import export_forward

# Explicit filesystem read of the fixed installed runtime source closure.
source = read_source_snapshot()
# In a source-bound workflow, retain and supply the independently expected
# identity; read_source_snapshot(expected_identity=...) checks installed bytes.
artifact = export_forward(
    model,
    source=source,
    expected_source_identity=source.content_identity,
)
# Exact immutable bytes: artifact.manifest, artifact.packed_codes,
# artifact.float_tensors. Persistence is the caller's explicit operation.
```

`read_source_snapshot` is the package's only filesystem operation. It reads the
fixed installed first-party runtime closure and CPU dependency lock with bounded
reads, rejecting missing, empty, symlinked-file and excessive members. It accepts
no caller-selected path and performs no write or network access. The package and
source API import with the standard library alone; the encoder explicitly needs
the pinned Torch/NumPy/safetensors dependencies. Wheel relocation is supported.

`SourceSnapshot` holds an exact immutable tuple of exact `(name, bytes)` tuples.
The encoder requires complete ordered membership, valid bounds and the expected
source identity. These are **caller-supplied captured input bytes**, not an
attestation of executed code: a caller can construct a snapshot, a matching hash
can be supplied by that same caller, and Python source can be modified after
capture or functions replaced in a process. Capture immediately before encoding
in a trusted unchanged process, and bind expected identities outside that
process when stronger custody is needed. This API does not authenticate a Git
commit, prove checkpoint origin, establish training history or certify that a
state is the final step of a research run. It accepts no checkpoint-origin hash.

Here **final-forward** means the native ternary quantizer's actual current codes
and scales for the supplied in-memory state, including latent edits after
initialization. The seed is only an initialization label. External checkpoint
loading and evidence verification remain the existing authorities; neither is
imported into the encoder. A final-checkpoint selection rule must be established
by its separately authorized producer and custody workflow.

The caller owns an idle model with no concurrent mutation. The encoder performs
no model forward, RNG operation, gradient update, mode change, file/network
operation, backend substitution or checkpoint/tensor deserialization. Ordinary interleaved state edits
are rejected by version/identity and inventory rechecks; this is not an atomic
snapshot or defense against a malicious concurrently modified Python process.

## Closed production profile

Only the exact native `Decoder` in the ternary lane is accepted. Config fields
are revalidated, with tensor shapes, counts and the full registered module tree
derived independently from that config. The native `Decoder.validate_state`
remains authoritative for numerical state validity; the export adds stricter
serialization-profile checks:

- Exact native block, linear, norm and ModuleList types; no extra parameters,
  buffers, modules, hooks, compiled or instance-overridden execution.
- Finite, contiguous, zero-offset CPU float32 Parameters with canonical storage;
  no shared parameter storage except the registered token/output embedding alias.
  Attribute resolution must reach each registered parameter/child exactly;
  instance-dictionary shadows are rejected, even if they currently point at the
  same object. This includes root embeddings/output, every linear/norm weight
  and nested child module.
- Unchanged config, native lane, normalization epsilon and head dimensions;
  no CPU autocast or non-little-endian environment.
- The existing `quantize` supplies int8 codes and a finite, positive scalar
  float32 scale. Codes use the existing `packed_ternary.pack_codes` primitive.
  Neither the quantizer nor packing is reimplemented.
- Embeddings and normalization remain float32; each matrix scale is float32 in
  the safetensors member and recorded as an exact canonical float-hex string.
  Zero matrices retain the native positive scale floor and all-zero codes.

The profile deliberately rejects some layouts/hooks that a general PyTorch
module can support. It neither silently fixes them nor mutates the model.

## Candidate byte contract

`trinite.final-forward-ternary.candidate.v1` is a new engineering-candidate
bundle identifier. It is distinct from the primitive `trinite.ternary-2bit.v1`
and from the unchanged native `trinite.absmean-ste.v0` quantizer.

The returned frozen dataclass contains only three immutable byte members:

1. `manifest`: canonical Trinite JSON v1, no floating JSON numbers. It binds the
   complete config and tokenizer semantics/source, current latent-state inventory
   and identity, source-byte receipt and identity, registered modules, explicit
   token/output alias, tensor shapes/counts/hashes, ternary occupancy, matrix
   offsets/counts/byte lengths, exact scale values/hashes, complete named float
   inventory and both payload member identities. Environment versions, byte
   order, thread count and deterministic-algorithm setting accompany the result.
2. `packed_codes`: concatenated per-matrix row-major code payloads in canonical
   native inventory order. Each matrix starts on its own byte boundary; its last
   byte has zero padding pairs. No padding is inserted between matrix rows.
3. `float_tensors`: the existing reputable `safetensors.torch.save` serializer
   stores `float.<parameter-name>` and scalar `scale.<matrix-weight-name>` tensors.
   Tied output storage is not duplicated. No pickle, optimizer, RNG state,
   training data or latent ternary matrix is serialized in this payload.

Manifest identity is obtained by hashing its complete immutable bytes externally;
it is intentionally not a self-referential field. The dataclass constructor is
not an artifact validator. Only the encoder creates a validated export; this
increment has no API that accepts arbitrary artifact bytes for inference.

### Resource preflight

Before tensor finite scans, cloning, quantizing or packing, config-derived counts
must satisfy the existing 5,000,000-parameter ceiling and **every individual
matrix must have at most 1,048,576 codes**, the merged codec ceiling. A valid
native model with a larger individual matrix is rejected, even if it otherwise
fits a model budget.

Callers may lower `max_payload_bytes` (hard limit 32 MiB) and
`max_manifest_bytes` (hard limit 512 KiB). The conservative preflight allowances
are packed-code bytes plus raw float/scale bytes plus `8 + 1024 × number of
unique parameters-as-tensors` for safetensors framing, and
`16384 + 2048 × unique tensor count + 512 × module count` for manifest metadata.
Exceeding either allowance rejects before quantization. Actual serialized sizes
are checked again before returning. These allowances can reject an artifact
whose eventual bytes would fit a smaller budget; they are deliberately
conservative. They do not bound peak RSS, Python object overhead or native
quantizer intermediates.

Source capture separately limits each file to 128 KiB and total captured bytes
to 1 MiB. Returned source metadata binds hashes/sizes, not copies of source files.
All member counts and metadata length are bounded. There is no streaming input,
implicit iterable consumption, arbitrary path or unbounded tensor allocation.

## Local conformance and actual byte sizes

The targeted numerical suite checks tiny and reference models at seeds 0 and 1.
**Test-only** reconstruction uses safetensors' authoritative `load` and the
existing `unpack_codes`, copies effective matrices and floating exceptions into
the same native dense Decoder, and compares against the native ternary Decoder.
Logits and selected embedding/every-block/final captures are bit-exact, including
right padding and a two-item batch. Two prompts per case have exact logits and
identical greedy trajectories for at most six new tokens, stopping on EOS.
This trusted-output test oracle is intentionally not a strict importer and does
not ship in the package. There is no production generation claim.

Other tests cover repeatable complete export bytes; odd matrix dimensions and
padding; full membership/alias/hash/count consistency; post-initialization
latent edits; zero, tiny and extreme finite inputs; nonfinite parameters/scales;
wrong lanes/shapes/layouts/dtypes; extra/shared tensors; every registered
parameter/child attribute-shadow boundary; hooks and execution
replacement; stale config/source; malformed source snapshots; pre-allocation
bounds; mutation detection; and unchanged parameters, gradients, RNGs, modes and
thread count. Standard-library AST/import checks enforce dependency direction.
A real built-wheel test removes the build tree, imports only the extracted wheel,
checks relocation, rejects a changed installed-source identity and verifies that
no training or checkpoint module was imported. It installs no dependencies.

Measured on Linux x86_64, CPython 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5,
safetensors 0.6.2, one Torch thread and deterministic algorithms:

| Bytes | Tiny: 6,752 parameters | Reference: 1,247,232 parameters |
| --- | ---: | ---: |
| Packed ternary codes | 512 | 294,912 |
| Float embeddings/norms plus matrix scales | 18,840 | 270,480 |
| safetensors framing/header | 944 | 4,488 |
| Serialized payload total, excluding manifest | **20,296** | **569,880** |
| Canonical manifest, measured separately | 12,259 | 45,876 |

The manifest sizes above are for seed 0 in this local environment; source byte
lengths, environment labels and state/occupancy metadata can change metadata
length. The reference raw tensor/code arithmetic remains 565,392 bytes before
safetensors framing. These are actual serialized lengths, not packed-resident
runtime memory, RSS, speed, energy or hardware-benefit measurements.

Run this increment's checks explicitly:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_forward_export_boundaries.py' -v
PYTHONPATH=src python -m unittest discover -s tests/forward_export -p 'test_encoder.py' -v
PYTHONPATH=src python -m unittest discover -s tests/forward_export -p 'test_installed_wheel.py' -v
PYTHONPATH=src python -m unittest discover -s tests/model -v
```

The separate [final-forward export workflow](../.github/workflows/forward-export.yml)
is the mandatory new CPU lane on Python 3.11/3.12/3.13, with hash-locked
dependencies. It runs the standard-library boundary tests, numerical export
suite and real-wheel smoke. The historical 261-case aggregate runner and its
pinned suite inventory remain unchanged. A workflow definition is not evidence
that hosted CI has executed; current hosted results and existing-suite results
must be reported separately for the reviewed commit.

## Remaining gates

A future separately reviewed **strict importer** must validate untrusted bytes,
including canonical metadata, exact complete inventory, versions, member hashes,
shape/count/offset/scale consistency, reserved code pairs, padding, duplicate or
extra members, malformed safetensors headers and resource bounds before tensor
allocation or inference. The encode-only increment cannot certify this rejection
boundary. Reuse the existing safe tensor authority without introducing a second
reader or an inference-to-training dependency.

Packed-resident execution, offline loading/generation, durable external custody,
independent-host replication and measured CPU resources remain separate work.
Weight-format selection, original learning and fresh-family adequacy, geometry-on
Phase 5, scaling and release readiness remain blocked by their own gates.
