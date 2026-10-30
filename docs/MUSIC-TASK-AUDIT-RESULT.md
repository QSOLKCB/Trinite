# Preserved task audit and limited CUDA availability observations

## Outcome and scope

This preserves the owner's PR102 **model-free task audit**, plus a separate
limited availability diagnostic. It is not another mixed/negative training
experiment, a repaired CUDA environment, or a replacement for the earlier
successful [CPU/GPU comparison](MUSIC-PLACEMENT-RESULT.md).

The producer is merge `7dbe25f7c26f15836de584fb9a0690119d62b28c`, tree
`5f43900f391d5167fefc5edceb581984deafd23d`. Four original JSON/exit files and the
handoff checksum manifest are preserved byte-for-byte in the
[public subset](../fixtures/music-task-audit-result-v1/README.md). New notes and
provenance are explicitly DERIVED; historical producers and evidence stay intact.

## What the offline audit establishes

The retained report joins **29 historical journals, 3,520 output occurrences and
88 labeled sections**, with five preservation receipts and 240 regenerated row
identities. These are correlated renderings, reused seeds and repeated outputs,
not thousands of independent subjects. The underlying exposed development set
still has only two structural parents.

Training rest positions are `{2,6}` and tie starts `{0,4}`. Development uses rests
`{0,7}` and tie starts `{1,5}`. The placement swap crossed existing templates
with training duration families, but did not expand aggregate role-specific
position support. Scalar onset overlap across meters does not establish the same
position/duration context. The original report separately preserves absolute,
bar-relative and rational-time contexts and actual prompt/target-mask checks.

All **16 canonical development sections** retain rest-only accuracy **0/10**:
nine answers `8` and one `6` (parent c, 4/4), followed by EOS, versus ten gold
answers `7`. The audit includes same-gold tie-only contrasts rather than treating
all equal-label variants as equivalent performance. Placement accounting remains
80 slots, 70 unique prompts, 30 changed-input pairs and ten identical-input pairs.
The finite 42-placement enumeration is a support inventory, not a new curriculum.

These are reproducible descriptive facts about the task and retained outputs.
They do not prove why a neural model failed, identify its internal shortcut,
show general musical/counting ability or constitute fresh confirmation. No new
training, checkpoint selection, model inference or intervention is performed.

## What the CUDA receipt does—and does not—say

The supplied metadata reports Python 3.12.13 in a virtual environment and Torch
`2.8.0+cu128`. The optional Torch child reported compiled CUDA `12.8`, unavailable
CUDA, zero visible devices, observed stderr and child exit zero. The actual stderr
is private; its presence alone does not identify a cause.

The fixed NVIDIA utility was **not executed**. Its status is `unsafe_executable`,
with no process return code. The frozen trust gate can reject ownership,
write-permission/path checks, invalid executable type or some filesystem errors.
The receipt does not say which branch failed. This label is not evidence of
malware, a driver fault or broken hardware, and is not a failed NVIDIA utility
execution. It must not be silently changed to `tool_unavailable` or exit 9.

Both retained launcher exits are zero: they emitted reports. That does not mean
CUDA is available, a training workload is compatible, or a GPU trial is admitted.
Installed version metadata and canonical source hashes do not certify installed
wheel bytes, hardware, permissions or an invocation's namespace. Root cause
remains unknown. No new timing or CPU/GPU performance comparison is supplied.

The owner-derived handoff says source acquisition received approval, while the
diagnostic used restricted execution without escalation; it reports reuse of the
failed preparation's lexical interpreter path and one audit plus one explicit
CUDA-query invocation after a successful source clone. It also retains an earlier
clone failure (exit 128). Those are **attributed owner assertions**, not facts
independently proven by these reports. Exact historical host/container/namespace
equivalence is not asserted. No permission-mode history is inferred from the JSON.

## Read-only preservation check

```sh
PYTHONDONTWRITEBYTECODE=1 python -I -B scripts/check_music_task_audit_result.py
```

The new standard-library checker authenticates original bytes and the frozen
implementation before project imports. It calls the unchanged PR102 offline audit
checker and compares the **complete report**, not only headline counts. It checks
the CUDA report's canonical identity, three source hashes, fixed retained schema,
statuses and false claim flags without importing or executing that diagnostic.
No NVIDIA command, CUDA API, model, checkpoint or tensor backend is called.

This establishes byte integrity, source binding, reproduced public-data analysis
and receipt consistency. It is not a new availability observation, numerical
replay, hardware/timing attestation, proof of an exact historical interpreter or
namespace, proof of single-attempt history, or full original custody. Source
hashes alone do not attest that a reported invocation actually occurred.

`source-SHA256SUMS` is unchanged and includes two omitted owner-derived narrative
files. The public subset does not pretend those entries are locally present.
Raw stderr, environments, operational logs and source-verification records remain
private. The new provenance file labels their reported context and omissions.

## Compatibility and conformance

This additive preservation supports TRI-I07, I08, I10, I13 and I14. It changes no
historical source or fixture. Normal and optimized tests cover byte/semantic
resealing, source/import substitution, unsafe paths, membership/size bounds and
claim escalation. Historical preservation and required CPU conformance are
reported separately. Actual CUDA diagnostics and target model studies remain
unrun here; no repair, permission escalation, retry or training is authorized by
this preservation PR.
