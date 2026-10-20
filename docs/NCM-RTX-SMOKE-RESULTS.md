# External NCM RTX smoke checkpoint, 2026-10-20

**Local feasibility established, not general Sudoku ability.** The first attempt
failed before inference. After the strictly scoped [PR #52 loader correction](https://github.com/QSOLKCB/Trinite/pull/52),
the separately authorized pilot and replay each produced four valid clue-preserving
Sudoku grids with exact target agreement. Their complete recorded result bytes
match, including latent IDs and declared environment. This is producer evidence,
not hardware attestation or a native Trinite improvement.

## Results and limits

| Measurement | First attempt | Corrective pilot | Replay |
| --- | --- | --- | --- |
| Worker state | Failed before inference | Completed | Completed |
| Warm clue-only boards, valid / clues preserved / solved / exact target | Unmeasured | 4/4 for each | 4/4 for each |
| Separate gold-codec target reconstructions | Not run | 4/4 | 4/4 |
| Cold model load | Unmeasured | 782.71 ms | 798.01 ms |
| First inference | Unmeasured | 458.54 ms | 398.76 ms |
| Warmup | Unmeasured | 64.83 ms | 62.16 ms |
| Four warm totals, range | Unmeasured | 62.46–63.97 ms | 61.54–62.06 ms |
| Four warm totals, mean | Unmeasured | 63.32 ms | 61.81 ms |
| PyTorch peak allocated / reserved | Unmeasured | 244.59 / 268.00 MiB | 244.59 / 268.00 MiB |

These are four exposed, correlated digit-shift variants from one formal family,
with 72 clues and one missing cell per row. Cold and warmup calls repeat board 0,
not extra independent samples. Gold-codec reconstruction receives the target and
is not counted as clue-only solving. Outputs were not repaired, overwritten with
clues or selected through retries. Four samples do not support a reliable latency
comparison, generalization, upstream-paper reproduction, or a speedup over Trinite.

The declared runtime is Python 3.11.15, PyTorch 2.8.0+cu128 and one RTX 5060 Ti
16 GB-class device with capability `[12, 0]`. These are retained producer records,
not independent hardware attestation. Timing and allocator-memory sidecars are
not required to match replay; the entire `result.json` is byte-identical:
`32fa519bbbeeb83b721f18de559b0bed4e81ed818f1a725b0cbd83e35cb11199`.
Memory measures the PyTorch allocator, **not whole-device VRAM**. The 180-second
worker budget and 4 GiB allocator limit remain unchanged.

## Failure retained, not rewritten

The first attempt's generic `ValueError: nonfinite checkpoint tensor` occurred
while loading the generator, before any inference output. It is **not zero solves**.
The original log did not identify the rejected tensor. Its failure is not
retroactively reclassified as success or proof that the checkpoint was corrupt.

In the successful retest, 225 named floating-tensor diagnostics identify the sole
nonfinite tensor as fixed `attn_mask_for_masking`: float32 `[1,1,413,413]`, zero
NaNs, zero positive infinities and 115,400 negative infinities. All other 224
floating-tensor diagnostic records report zero nonfinite values. Counts alone
would not suffice: the frozen loader requires every fixed-mask value and position,
complete metadata/schema, the exact int64 level buffer, and finite values for
all non-exempt floating tensors. Restricted `weights_only=True` loading and all
asset pins remain intact. Checkpoint bytes are absent, so the archive verifier
does not deserialize weights or independently prove their contents.

## Public archive and honest derivation

The new [public fixture](../fixtures/ncm-rtx-smoke-v1/manifest.json) retains:

- `run.zip` and `replay.zip` byte-for-byte, including complete 22-file producer
  snapshots, frozen requests, every raw output/latent code, timings, all eleven
  progress records per attempt, worker diagnostics, status and original inventory.
- A distinct `failed-run.public-derived.zip`: all 21 original source files,
  request and failed status are exact copies. The original inventory is retained
  unchanged as `original-inventory.json`; `worker.redacted.log` replaces exactly
  five absolute private checkout prefixes with `[REDACTED_CHECKOUT]/`. All other
  traceback bytes remain unchanged. Every original/derived member name, size,
  hash and transformation is recorded in the canonical public manifest.
- Exact copied offline verification receipts, successful exit-time records and
  the original replay-equality receipt. The failed receipt says failed; a
  verifier exit of zero is not a successful model run.

The public manifest SHA-256 is:
`acb310148fdae63081ac603630fd6d58dd49a79a5766114c0b08c822506dc584`.

The original uploads remain private, unchanged and read-only. Their outer hashes
are `6ade722cd398e3255c6d50537ce8c9d864c36ac03ecc38e4dbd02bdbe8c9c4c0`
(first attempt) and `0c03026722197c3a7d059b23e414e1bbd3dd2d4ceb434c9f391287a58180802f`
(retest). The public fixture intentionally omits outer shell/setup/checkout logs,
workstation paths, redundant attempts and collection snapshots. No checkpoint or
upstream dataset bytes are included. Source/license snapshots remain unchanged.

The failed public ZIP has a **new digest and different format**. It is not the
original failed archive, cannot use the original archive's verifier receipt as
its own, and is not a reseal. Public readers cannot reconstruct/re-hash omitted
outer uploads or the original private traceback. Original-to-derived mapping and
outer hashes are provenance assertions checked against the private originals at
creation, not cryptographic proof supplied by the public subset alone.

| Producer | Commit | Git tree | Frozen request SHA-256 |
| --- | --- | --- | --- |
| Failed attempt | `7a9d2a446cd5bbafd74a6f43f8b5004bf5c2c2d3` | `ade155dc3683b90fbdec2789bcd54ce3fcafba9f` | `34f0df74d21797441b466de382053baf0b0eba2df0575360916499f5ebd2aabc` |
| Pilot and replay | `2eb6b3d19a543880e135409ec1f5f0ebec51c35d` | `32d7b164f0141132715cd02b37a5d7bae7e99afe` | `764af970c917dee53df7adb07a07781c5fe4919f8fb692dfc5b5478c4e5753c9` |

Original nested ZIP digests are retained in the manifest; successful archives are
also pinned independently by the verifier. The two frozen requests differ because
the reviewed source closure changed, not because inputs or numerical settings changed.

## Offline verification

From the matching checkout, with only Python's standard library:

```sh
PYTHONPATH=scripts python -B scripts/verify_ncm_rtx_archive.py
PYTHONPATH=scripts python -B -m unittest discover -s tests/ncm_archive_review -v
```

The verifier checks the external manifest commitment, bounded safe ZIP structure,
source snapshots and Git blob identities, frozen numeric inputs, original or
derived member maps, raw metrics with the unchanged authoritative scorer,
complete progress/timing records, failed/completed states and exact recorded
replay equality. It never extracts or executes uploaded code, imports Torch,
downloads/loads weights or runs inference. Public-byte verification is read-only.
Its output explicitly states that private original handoffs were not reverified
and no new inference or hardware attestation occurred.

Affected requirements: TRI-I07/I08 (source/input and replay scope), TRI-I10
(bounded measurements), TRI-I13 (failure retention), TRI-I14 (additive archive
compatibility). No runtime, loader, protocol, input generation, weight pin,
training rule, earlier fixture, native gate or release status changes. Further
experiments require a separate frozen protocol and authorization.
