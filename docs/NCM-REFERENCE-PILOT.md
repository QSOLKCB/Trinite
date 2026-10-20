# External NCM reference feasibility pilot v1

Status: corrective loader contract for review. The first authorized local attempt
failed during generator state validation, before inference. **No model inference,
solve-rate or latency result has been obtained.** CPU tests use
explicit software doubles, never fabricated GPU evidence. This separate external
pretrained lane does not replace Trinite, train anything, change geometry loss,
select a quantizer, or clear any native learning/release gate.

## Scope, sources and rights

This implements the actual [Next-Concept Modeling paper](https://arxiv.org/abs/2610.14162v1)
[official code](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203)
API, pinned at `c70274273f489abc73832b2b98584a8483b6b203`.
The [manifest](../scripts/ncm_reference_v1/upstream-pin.json) records SHA-256 for
all ten selectively vendored source/license files, the exact easy-Sudoku YAML,
the checkpoint model card, and both weights. Vendored sources are unchanged MIT
code, with the original copyright/license retained under `vendor/LICENSE`.
New wrapper code retains Trinite's MPL-2.0. These are separate software licenses.

The [checkpoint release](https://huggingface.co/hazyresearch/ncm-checkpoints/tree/d342ce8aaea58830eef3b8ad473664673e41f692/sudoku/easy)
is revision `d342ce8aaea58830eef3b8ad473664673e41f692`; its model card declares MIT.
The 102,560,203-byte generator and 59,816,209-byte codec are not vendored or
implicitly downloaded. SHA-256 commitments come from the release's LFS metadata;
local bytes must match before restricted loading. Their original training-data
lineage/exposure is external and is not Trinite-admitted native training data.

The upstream MIT-labelled dataset is **excluded**, because permissive copyright
licensing does not satisfy Trinite's no-copyright-data policy. Only static pickle
opcode inspection established the API format; it was never unpickled or used
as pilot input. The exact inspected source revision/hash is recorded in the
[protocol](../scripts/ncm_reference_v1/protocol.json). No upstream examples are copied.
Four commissioned formal boards are constructed arithmetically, with one missing
cell per row and independent missing-digit/row/column/box verification. The
source-specific admission record binds origin, procedures, rights assessment,
review date and limitations. These are four correlated visible variants of one
family, not a held-out Sudoku benchmark, difficulty reproduction or accuracy study.

## Actual upstream boundary

Prompt IDs are exactly `[11] + 89 row-major board tokens + [11]`: blank 0,
row separator 10, BOS 11. Generation receives **clues only**, calls
`net.generate(None, prefix, temperature=0, top_k=0, top_p=1, cfg_scale=0,
use_kv_cache=False)`, and produces 322 codes across twelve levels. The wrapper
uses `vq.idx_to_zhats(..., return_zlevels=True)` and the final codec decoder,
then raw argmax over all 12 tokens. No clue overwrite, grammar repair, solver,
retry, answer conditioning or constrained decoding is applied.

Scoring separately reports exact output shape/separators, all rows/columns/boxes,
clue preservation, solved status, reference agreement and matching cells. A valid
but clue-breaking grid is not solved. Malformed outputs never get partial-length
zip scoring. Gold codec reconstruction is a separate diagnostic **after** all
clue-only generation and is not a solve-rate result.

Upstream's unrestricted `weights_only=False` helper is not included or invoked.
Both local checkpoints are rehashed from the exact bytes passed to
`torch.load(..., weights_only=True, map_location='cpu')`, followed by bounded
tensor/state-dict and strict model key/shape/dtype checks. Unsupported checkpoint
objects fail; there is no unsafe fallback or custom global allowlist. Safe loading
still does not make arbitrary checkpoints trustworthy: only pinned files are allowed.
The original finite-only validator rejected a nonfinite generator tensor after
restricted deserialization. That log did not name the tensor, so the exact rejected
key is not established. No compatibility or successful model load is claimed.

### Corrective loader contract v2

The pinned generator registers `attn_mask_for_masking` as a persistent, fixed
attention buffer, not a learned parameter. Its intended forbidden-attention entries
are negative infinity. Rejecting every nonfinite floating state tensor therefore
conflicts with the pinned model format even if all learned weights are valid.

The [versioned loader contract](../scripts/ncm_reference_v1/loader-contract-v2.json)
requires an explicit generator/tokenizer role, model schema and fixed-buffer
mapping before reading or deserializing checkpoint bytes. Generator calls require
the complete fixed pair; tokenizer calls forbid that pair and all exemptions.
It then requires exact checkpoint keys, CPU plain strided tensors, bounded sizes
and exact fresh-model shapes/dtypes before loading. Only the complete generator fixed pair
`attn_mask_for_masking` and `lvl_1L` receives fixed-value validation. Both must equal
values independently generated from the pinned configuration, and those values
must also equal the freshly constructed model buffers. The mask has shape
`[1,1,413,413]`, 55,169 zeros and 115,400 negative infinities; **every position**
must match, not merely counts. Level IDs have shape `[1,322]` and dtype int64: the pinned upstream
`torch.full((l,), i)` receives an integer enumerate index, not a floating fill
value. A separate real-PyTorch CPU test checks the exact expression, independently
of the standard-library tensor-double suite. No buffer is
silently removed, replaced or repaired. All remaining floating tensors must be
finite; complex tensors are rejected. NaNs, positive infinities and misplaced
negative infinities fail. Named shape/dtype/nonfinite-count diagnostics are emitted
before value rejection during existing CPU state loading, not a separate GPU run.

The first failed archive remains negative evidence, not zero solves. Preserve its
original producer/request and external ZIP digest. This correction does not edit
historical fixtures, the experiment protocol, upstream code/config/weight pins,
budgets, scorer or numerical generation settings. Its changed source closure and
versioned loader contract require a **new frozen request** after review, merge and
green CI, plus separate local-run authorization. Verify old evidence only with its
original source-bound producer; never reseal it using the new loader. New CPU tests
use explicit tensor doubles and do not establish actual-checkpoint compatibility.

All executable upstream imports are hash-bound; foreign `models`/`utils` namespace
collisions and upstream bytecode caches reject. The launcher uses a private empty
bytecode-cache prefix before importing its wrapper. No remote model-code loading
or network operation occurs during a run. Upstream optional pretrained context
branches remain disabled by the exact, hash-checked Sudoku config.

## Frozen budget and measurements

Only one visible RTX 5060 Ti, 16 GB class (`sm_120`), is admitted. Require at least
15 GiB total and 6 GiB currently free, native sm_120 support and exact declared
Python/Torch dependencies. No CPU/other-GPU fallback. Fixed float32, no TF32,
deterministic algorithms, seed 0, batch 1, four smoke items, one cold generation,
one warmup, four warm generations and four separate gold-codec diagnostics.
The parent kills the fresh worker at 180 seconds, including imports/loading.
The PyTorch allocator is capped at 4 GiB and peak allocated/reserved memory is
checked separately, including model construction. This is **not** a whole-device
VRAM cap; driver/context/other-process memory is outside it. Budgets are
conservative limits, not measured guarantees of fit or completion.

CUDA synchronization surrounds preparation, generator, codec/output transfer and
total timings. Cold model loading, first inference, warmup and warm calls remain
separate; every per-item raw output and code is retained. The generator includes
prefix embedding and multiscale generation; the codec covers latent reconstruction,
decoder and CPU output transfer. Full codec encode/decode diagnostics are charged
separately. Python/platform, driver, CUDA/cuDNN, build, GPU, exact direct dependency
versions and complete installed-package inventory are recorded. No speed comparison
against Trinite, paper reproduction or universal determinism is claimed.

## CPU verification, then local RTX execution

The review/merge/green cycle comes first. Do not run the GPU commands before it.
CPU contracts require no third-party dependencies:

```sh
PYTHONPATH=scripts python -B -m unittest discover -s tests/ncm_reference_v1 -v
```

After the user authorizes the local run on the reviewed, merged, green commit,
create a separate Python 3.11 environment, never modify the canonical CPU lock:

```sh
python3.11 -m venv .venv-ncm
. .venv-ncm/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r scripts/ncm_reference_v1/requirements-gpu.txt
python -m pip check
```

These explicit official-package installs execute third-party software and require
normal user authorization. They are version-pinned, not a new hash-locked canonical
CPU environment. NVIDIA's compatible driver and `nvidia-smi` must already work;
the wrapper does not change drivers or system settings. Only use the two exact
`.pt` files from the linked immutable checkpoint folder, saved as `generator.pt`
and `tokenizer.pt` in a new local directory. No dataset is needed. Check both:

```sh
python -B scripts/run_ncm_reference.py check-assets --checkpoints /path/to/weights
request_sha=$(python -B scripts/run_ncm_reference.py freeze --output /new/request.json)
python -B scripts/run_ncm_reference.py run --request /new/request.json \
  --request-sha256 "$request_sha" --checkpoints /path/to/weights --output /new/run.zip
python -B scripts/run_ncm_reference.py verify --archive /new/run.zip --expected-sha256 PRINTED_ZIP_SHA256
python -B scripts/run_ncm_reference.py replay --request /new/request.json \
  --request-sha256 "$request_sha" --checkpoints /path/to/weights \
  --expected /new/run.zip --expected-sha256 PRINTED_ZIP_SHA256 --output /new/replay.zip
```

Keep the printed request and archive identities independently. Every destination
must be new. Atomic hard-link publication never replaces files/symlinks; unsupported
filesystems fail rather than overwrite. Complete attempt directories and per-call
progress records survive failure, timeout, interruption and final publication errors.
An abruptly killed parent may leave only the attempt directory. Atomic visibility
is not a directory-fsync/crash-durability guarantee.

The bounded ZIP includes complete source closure, request/inputs/protocol, raw result,
timings, logs, progress, status and member inventory. Offline verification checks an
externally supplied ZIP digest, complete source/request equality and recomputed exact
metrics, not merely stored success labels. It is structural/semantic verification,
**not hardware attestation**. Exact replay actually reruns inference and compares
all result/environment bytes, excluding observed latency/peak-memory sidecars. A
mismatch retains both archives and a false-match receipt; no tolerance or success
fallback. This is inference replay, never optimizer replay. Archive or replay
success does not establish learning adequacy.

Affected requirements: TRI-I01/I07/I08/I12 (inspectable pinned optional boundary),
TRI-I02/I03/I11 (external pretrained disclosure/formal input admission), and
TRI-I10/I13/I14 (bounded claims/failures/versioning). Native CPU baselines,
installed-wheel API, historical protocols/fixtures and all earlier negative
results are unchanged. Future native NCM training or broader evaluations require
separate reviewed protocols and authorization.


The optional CPU-only upstream dtype regression requires an existing PyTorch 2.8
CPU environment (no model imports, checkpoints or CUDA execution):

```sh
python -B -m unittest discover -s tests/ncm_reference_cpu -v
```

This check is separate from standard-library CI contracts; neither test lane proves
released-checkpoint loading or GPU inference compatibility.
