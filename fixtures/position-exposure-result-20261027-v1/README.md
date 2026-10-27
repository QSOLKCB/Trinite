# Positional-exposure negative result: partial public subset

The owner-run PR79 twelve-cell result at producer
[`e68666973ead5bc774673570550731eb05835dc3`](https://github.com/QSOLKCB/Trinite/commit/e68666973ead5bc774673570550731eb05835dc3)
is retained here without altering its frozen protocol or historical evidence.
Read the [outcome and verification limits](../../docs/POSITION-EXPOSURE-RESULT.md)
and [pre-outcome protocol](../../docs/POSITION-EXPOSURE-V1.md).

**229 original files are byte-identical:** matrix plus 19 files in every one of
twelve cells: completion, report, description, five training score records and
raw journals, and three development score records and raw journals. These cover
10,464 prediction records (3,072 at the final checkpoint), not independent trials.

`source-bindings.json` is a new derivative private-audit receipt, not the
original request. It binds all 256 producer source commitments and admitted
input identities. Its pinned identity is
`sha256:96094083da3baea1157887af26e38366617427f1264982bde636795dfee5504b`.
This README is explanatory, not an original execution record.

Original request identity:
`sha256:cba9f27710cdc85016527ad412d8d80db0655ccfe0f6f7515f9ee0814fb8ede9`.
Original matrix identity:
`sha256:199bd917fece38e34190e103f11aad641d07ccf9005fadf3b6631a61c4dad78e`.
Private delivery v2 archive SHA-256:
`63646cc27eeaa5b90d15170ad1f2a42de731f0a3de4f34128318956d1ed2142a`.

Original request/environment, checkpoints, host logs, resource captures and the
full private handover are absent. References to omitted artifacts stay unchanged.
The receipt cannot reconstruct the omitted request. This is partial public
custody, not closed custody, numerical replay or host attestation.

```sh
PYTHONPATH=src:scripts python -B scripts/check_position_exposure_result.py
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/position_exposure_result_v1 -v
```

The read-only checker validates selected original bytes and recomputes retained
metrics and all four frozen contrasts, including exposure accounting. It compares
fixed-arm predictions/model identities with the earlier bound public study.
It does not load tensors, construct models, train or generate; private checkpoint
byte-parity observations are not independently verified by this public subset.
Do not amend or reseal originals to match changed source. All contrasts fail;
no training increase, curriculum change, research execution or release is implied.
