# Local Needle compatibility v2: retained negative result

One owner-run CPU attempt on 2026-10-21 exercised the reviewed
[non-target compatibility v2 probe](TINY-COMPATIBILITY-V2.md), based on main
`8823a92cdaf28139a852f896af457b3b0a013535`, tree
`29942d2ff759cb4cefcc512308932f12d933d361`. This evidence-only addition changes
no producer, scorer, protocol, budgets, historical fixtures or native semantics.
Affected requirements: TRI-I01/I02/I03/I05/I07/I08/I10/I11/I12/I13/I14.

## What happened

The retained session reports Linux x86-64, Python 3.14.4 and an AMD Ryzen 9
5950X. Model loading completed; the worker emitted ready, call-start and result
events, exited zero, and capture completed. The single probe published a packet.
Exit zero means packet publication, **not a successful model answer**.

The exact 450-byte SDK serialization reports `success=false`,
`error_code="truncated"`, `error="tool call truncated: token budget exhausted"`,
`function_calls=[]` and `confidence=0.0558`. It is faithful SDK-return
serialization, not raw native output or token-level traces. No retry, budget
increase, response repair, second invocation or Bonsai execution is included.

The unchanged v1 strict scorer first rejects unknown envelope fields
`peak_ram_mb` and `reason`. That is not the sole defect: reported failure,
truncation, low confidence and the absence of any answer remain independently
incompatible with acceptance. No valid call was removed by SDK grounding in
this observed empty-call result; the retained SDK path returns early for an
empty call list. Neither the SDK nor scorer is modified to admit this result.
Attempt acceptance, correctness and target readiness are all **false**.

The reported call duration is 0.18700096099928487 seconds, excluding loading.
The serialization also carries native-reported decode throughput 519.4 tokens/s,
prefill throughput 1379.5 tokens/s and `peak_ram_mb=103.5`. These are unvalidated
opaque metadata, not independently measured performance. Official generated-token
and memory measurements remain null with reasons. There is no throughput,
quality, general compatibility, matched-decoding or equal-compute conclusion.

## Independent byte commitments

- Supplied original ZIP: 3,773,058 bytes, 802 members; SHA-256
  `ee4a48806be2307af22ab2d0cdeb75f29e6dd8bd12125943d9c504f0e83fabe0`.
- Frozen request: SHA-256
  `71d51bae151c124b0de3ba6dd2e1efab704bc61a2a8285e22b84ac3a17121716`.
- Published packet: SHA-256
  `3a4384848c8d2ef99d44051662134be364fac3f7b3d46c8704972f3a90627081`.
- [Fixture commitment](../fixtures/needle-compatibility-local-20261021/commitment.json):
  SHA-256 `2e38d7a9153c7a95ce205c48d2927bce2ad184861b55a7aad6af9a4d00b3c2c3`.

The content-addressed fixture retains all 802 original member byte strings in
713 unique compressed objects. Every original member has its original path,
size, SHA-256 and retained object path. There are no new omitted members or
redactions. The objects ZIP is a different container: it does **not** reproduce
the original ZIP's container bytes or claim that its digest is reconstructible.
The original member checksum manifest is preserved and checked. `.gitattributes`
marks the object ZIP binary so checkout cannot normalize captured byte strings.

The original archive already omitted model weights, native libraries, Git
administrative data and environments. Its exact omission ledger, staged-to-
original mappings, source inventories, SDK source snapshot, license and prior
acquisition provenance remain included. Asset identities do not replace missing
binary bytes or establish complete binary closure. Source-before/source-after
receipts report all 1,255 tracked files unchanged; they are retained local
observations, not independent execution or hardware attestation.

The bootstrap ledger preserves a real limitation: preliminary discovery was not
captured by the later command logger, and one initial directory enumeration was
truncated in the tool return. The complete later read-only enumeration is labeled
as a reread, not the original stream. Required tests, request freeze, native probe
and packet verification have retained complete command/exit/stdout/stderr records.
Final ZIP construction/checksum closing receipts were left beside the original
archive to avoid self-reference; their absence inside this ZIP is not concealed.

## Read-only verification and tests

The new isolated checker hashes its fixture and all five pinned authoritative
repository sources before loading the offline verifier/scorer. It validates the
request, raw serialization, protocol stream, staged source/asset mappings and
published packet against retained receipts, and invokes the actual v2 verifier
and unchanged strict scorer. It never imports retained Needle SDK/adapter code,
loads a model/native library, launches a probe or executes uploaded helpers.

```sh
python -I -S -B scripts/check_needle_compatibility_evidence.py
python -B -m unittest discover -s tests -p test_needle_compatibility_evidence.py -v
python -B -O -m unittest discover -s tests -p test_needle_compatibility_evidence.py -v
python -B -m unittest discover -s tests -p 'test_tiny*' -v
python -B -O -m unittest discover -s tests -p 'test_tiny*' -v
PYTHONPATH=src python -B -m unittest discover -s tests -v
PYTHONPATH=src python -B -m unittest discover -s tests/upstream -v
```

The original local logs report 49 tiny tests normally, the same 49 under `-O`,
441 root tests and 51 upstream tests passing. These suites overlap; the counts
must not be summed as distinct tests. New CI performs offline archive consistency
and regression checks only. `verified=true` means read-only source/capture
consistency, **not inference replay, independent execution proof, hardware
attestation, supported-pair certification, scored readiness or native learning**.
All learning, scaling, Phase 5 and release gates remain unchanged.

## Attribution and scope

Retained Needle SDK and source/header evidence originate from
[cactus-compute/needle b4fc11b04f1852888d375c8c6dfbaca44572d014](https://github.com/cactus-compute/needle/tree/b4fc11b04f1852888d375c8c6dfbaca44572d014).
The model metadata and embedded RAW tokenizer tail originate from
[Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb](https://huggingface.co/Cactus-Compute/needle3/tree/d6c7d6a2e9655505e0f8e23339356dd1705900eb).
The tokenizer tail is not a standalone `tokenizer.model`. The original
Apache-2.0 license is retained at archive member
`reports/provenance/prior/assets/review/needle-license.txt`; original source
notices, the SDK project metadata, model README and license inventories remain
byte-exact. The repository checkout's MPL-2.0 license and its special upstream
notices remain intact. This archive is engineering evidence, not admitted native
training data or a model/runtime distribution.
