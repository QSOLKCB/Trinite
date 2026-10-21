# Local teacher-input pilot: execution and exact replay, no demonstrated advantage

One owner-supplied local experiment at merged PR #61, commit
`60a1797124a8390c1183a130846a82e06dea94ab` (tree
`63febaf4046f64b67bff78e97748810668a58d3f`), ran on 2026-10-21.
This archive adds evidence only. It does not change the frozen
[teacher protocol](TEACHER-INTAKE.md), existing numerical sources, historical
fixtures or admission policy. It is not a new teacher or student run.

## What the retained records show

- Exactly 32 teacher conversations produced 1,523 generated tokens. Strict intake
  admitted 19 unique numeric proposals and retained 13 duplicate-family rejections.
  Only repository-rendered inputs and independently computed labels entered the
  student; raw teacher reasoning and prose stayed quarantined.
- The separate, visible four-fact training diagnostic scored 4/4 after 256 updates.
  This is a memorization check, not a generalization result.
- All four dense CPU comparison cells completed 256 updates, 1,024 scheduled
  visits and 3,072 answer/EOS targets each. All initial exact counts were zero.
- One full student replay produced exactly the original numerical report bytes,
  including training histories and model/optimizer identities. No new teacher
  call was used for replay. The reports do not serialize tensor checkpoints.

| Seed | Arm | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| 0 | teacher_selected | 9/19 | 7/66 | 7/91 |
| 0 | oracle_only | 8/19 | 7/66 | 6/91 |
| 1 | teacher_selected | 10/19 | 7/66 | 7/91 |
| 1 | oracle_only | 9/19 | 8/66 | 7/91 |

Every residue 00–12 occurs exactly seven times in the 91-row modulus-13 test.
Always returning any one valid residue and EOS therefore scores **7/91**.
Teacher-selected test accuracy does not exceed this simple constant-answer
baseline. Predictions are not all constant, but neither seed establishes useful
held-out arithmetic or a teacher advantage. This small exploratory comparison
also does not establish that teachers are generally ineffective or inferior.
The arms do not match all operand constraints; six training families overlap.
No new Phase 5, release, ternary, speed or reasoning claim follows.

## Runtime and preserved preparation failures

The recorded teacher is `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`. Acquisition records bind original
safetensor shards and tokenizer; local conversion repacked MXFP4 experts into
GGUF. The actual runtime was an **Unsloth llama.cpp fork/prebuilt**, revision
`f6b9ea74368fb3cbde9f70ef4cee5c82237d48e1`, not an independently verified official
upstream build. RTX 5060 Ti logs report 25/25 layers offloaded and a CUDA model
buffer of 13,141.29 MiB. A retained process snapshot reports 13,574 MiB; this is
not a peak-memory benchmark or hardware attestation. Student execution used the
unchanged Python 3.12 CPU lock on the Ryzen 9 5950X.

Sandbox network/GPU checks and setup/test failures remain in the evidence. The
first teacher supervisor failed on a missing `_colorize` dependency while
constructing argparse, before launching a model. The original frozen request,
adapter and failure log remain. A complete support package and v2 adapter were
reviewed and a fresh request frozen before any generation. The two requests
change only runtime-manifest adapter/support paths and corresponding bytes;
slots, prompts, protocol and repository source bindings are unchanged. There
were two supervisor invocations but one actual generation worker invocation,
with no repeated teacher call. Archived approval receipts are local assertions,
not independently authenticated approval transcripts.

## Archive scope, rights and integrity

The compact fixture is
[`fixtures/teacher-pilot-local-20261021`](../fixtures/teacher-pilot-local-20261021).
Its canonical commitment maps 689 original logical files plus the unchanged
repository verification-loader source and a sourced model-license notice to 548
deduplicated byte objects in a 1,205,969-byte ZIP. The original uploaded ZIP had 853 members (852 checksummed
files and `SHA256SUMS`), 11,041,296 bytes and identity:

`sha256:354829e6f7b6a0312397d8b40e43b61ead198f269e6596d0d8053adb7cbc4045`

The commitment accounts for every original member as retained or explicitly
excluded, with its exact size and SHA-256. All reports, requests, raw teacher
captures/events/receipts, student attempt directories, partial records,
publication journals, setup failures, original/v2 adapter source, support-file
inventories, acquisition records and complete 126-file bound repository source
closure are retained byte-for-byte from the uploaded ZIP. Original user-side
privacy transformations are preserved separately in `bundle-transformations.json`;
this archival step performs no additional transformation of retained members.

The 164 exclusions are the large tokenizer payload, compiled helper binaries,
Python support archives (including an incomplete setup archive) and bulk upstream
runtime/converter dependency source. Their exact identities, source indices and
exclusion reasons remain, together with runtime build metadata, licenses,
conversion/adapter source and acquisition provenance. This is **not a complete
inference-runtime distribution or a byte-identical copy of the original ZIP**.
The public fixture cannot reconstruct or independently byte-verify the excluded
assets; it can inspect their recorded commitments only. Original assets and full
ZIP remain with the owner. Model weights were never in the uploaded ZIP.

Repository source remains MPL-2.0. Retained runtime/converter material keeps its
upstream LICENSE and THIRD_PARTY_LICENSES notices in the logical archive. Model
metadata/chat templates retain their source attribution and the pinned model's
[Apache-2.0 license](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/LICENSE), added verbatim as `attribution/gpt-oss-20b-LICENSE.txt` from that pinned model revision. The acquisition metadata lists no separate NOTICE file. Teacher-origin numeric inputs are admitted only
under the existing scoped policy; this archive does not widen output rights or
relicense external material. Local paths are execution context, not download URLs.

## Read-only offline check

From a trusted checkout, with Python 3.11 or newer:

```sh
python -I -S -B scripts/check_teacher_pilot_evidence.py
python -I -S -O -B scripts/check_teacher_pilot_evidence.py
PYTHONPATH=src:scripts:. python -B -m unittest discover -s tests -p 'test_teacher_pilot_evidence.py' -v
```

The standalone check requires isolated/no-site startup before standard-library
imports and uses the retained, identity-checked source loader before importing
project validators. It verifies archive bytes, source/request/intake bindings,
exact labels, raw prediction scores, training budgets, retained attempt receipts
and original/replay byte equality. It does not execute the captured adapter,
load weights, invoke GPU inference or rerun numerical training. Ordinary unit
tests invoke that same isolated checker rather than importing unchecked project
validators. Byte-preserving fixture attributes prevent Git newline conversion.

Integrity and conformance of retained records are not independent execution,
official-origin, reproducible-build or hardware attestation. A fresh optimizer
replay was recorded locally; the offline check verifies those records and does
not itself repeat the optimizer. Affected requirements: TRI-I02/I03/I05/I07/I08/
I11/I12/I13/I14. Historical sources, protocols and negative evidence are unchanged.
