# Local Ryzen batch-layout evidence, 2026-10-20

This evidence-only addition retains a local CPU run of merged PR #54 at commit
`f4e393456f0d708af0221a8de4767438c700710a`, tree
`7b346f04b5cbb6f2b84f90af0e6215bcd68b8caa`. It changes no runtime, protocol,
training procedure, previous fixture, or historical source closure.

## Result and limits

The supplied Ryzen 9 5950X evidence records Python 3.12.13, PyTorch 2.8.0+cpu,
NumPy 2.3.5, deterministic float32 CPU execution with one intra-op thread.
Across four frozen endpoints, eight batch layouts, and 42 prompts, the recorded
report conforms: 10,920 vector comparisons, zero violating coordinates, maximum
absolute difference 0.0000011920928955078125. The accounting is 2,500 physical
forward calls, 6,720 row evaluations, and 6,720 active decisions.

The local run and same-environment replay both completed with exit zero. Their
complete computed reports are byte-identical; four-cell and 32-layout streams
also match exactly. Fifty focused tests are recorded passing, not a full local
repository test run. Offline review independently checked all 104 original
manifest entries, all 149 source identities against the merged Git tree, the
recorded numerical comparisons, streams, statuses, and replay receipt. It did
not execute the model again or attest the hardware.

This supports only batch-layout conformance and exact local CPU replay. It does
not establish better model quality, GPU execution, a speedup, hardware-wide
bitwise equivalence, or a new training result. Recorded timings are observations,
not a controlled performance comparison. The previously documented independent
adversarial review gap remains open. This PR does not repeat those experiments.

## Retention, compression, and privacy

`fixtures/batch-composition-ryzen-20261020/commitment.json` maps every one of the
105 supplied transfer members to a content-addressed public object. Equal
objects are stored once; large objects use XZ, following the existing archive
fixture pattern. Each compressed object is bounded to 64 MiB for both decoding
memory and decoded size. The report and all numerical requests/streams retain
exact original bytes. No model weights, environments, or new producer ZIP are
added.

Private home paths, temporary-directory prefixes, and the hostname in the kernel
log are replaced by explicit
placeholders in derived log/metadata/helper copies. Each member records its
original identity and size, public identity and size, and ordered replacement
rule/count. Rules retain the original token's SHA-256, not the private text.
These derived copies are not claimed to match original byte identities. The
original 104-entry file manifest is retained unchanged; `inventory.json` covers
all 105 members including the manifest. The local orchestration helper is inert
historical text, not an executable entry point for this fixture.

The supplied tar.xz transfer identity is
`sha256:02b43fe86e11c32d3fd66ab1bf6e041f34af836f2eead5ff66b47a27a9b562fd`.
The original larger ZIP was not supplied, so neither its contents nor equivalence
to this transfer have been independently verified. The earlier sandbox-blocked
setup is referenced only by historical metadata/digests, not uploaded or treated
as a numerical failure. The public fixture is a derived, fully inventoried
representation of the supplied tar members, not a byte-identical republishing
of that private archive.

The independently inspected numerical identities are:
- Request: `sha256:aa4313375461b6cebfe2b73b4f20d71ccca1b9a38e0bc15ab676fe1238a2b335`.
- Report: `sha256:92b7ad27e9e10f980ab28899028f02d0f21167b92df854102cd554e78f2e2f1a`.

## Offline check

From the repository root:

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests -p test_batch_composition_ryzen_evidence.py -v
```

This checks bounded object decoding, exact inventory and hashes, the distinction
between original and derived bytes, unchanged approved source identities,
actual formal-data admission, existing report validation, and exact retained
replay streams. It needs only the standard library and never runs inference.
The new archive-only workflow runs it on Python 3.11, 3.12, and 3.13. The original
[batch-composition contract](BATCH-COMPOSITION.md) remains authoritative.
