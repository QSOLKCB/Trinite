# Position-study mixed/negative result: partial public subset

This directory preserves the owner-run PR77 position-coupling result from
26 October 2026 UTC at producer commit
[`050f8cc109453bb3e6d0de62b2c0dd2e93841ed3`](https://github.com/QSOLKCB/Trinite/commit/050f8cc109453bb3e6d0de62b2c0dd2e93841ed3).
Read the [outcome and limits](../../docs/POSITION-STUDY-RESULT.md) and unchanged
[frozen protocol](../../docs/POSITION-STUDY-V1.md).

## Original and derivative records

**115 original files are retained byte-for-byte**, without redaction or
resealing: `matrix.json` and 19 files for each control/candidate seed 0/1/2:
completion, report, description, five training score JSON records with their
five raw JSONL journals, and three development score JSON records with their
three raw JSONL journals. These contain 5,232 predictions across milestones
and strata, including 1,536 final-checkpoint predictions. Repeated observations
across checkpoints, seeds, arms and renderings are not independent examples.

`source-bindings.json` is a **new derivative private-audit receipt**, not an
original execution artifact or a replacement for the request. It identifies
all 239 original producer source commitments and admitted input identities.
Its independently pinned SHA-256 is
`69bc803e9ce63d9a3e0a0ce8d3aa5c5bed39f309f9f54f681d27432681c29962`.
This README is explanatory, not an original execution record.

- Original request identity: `sha256:bdf6d3bfc7e05467f324e2794cfec70791b082ac08fad384b00fbcfcd6c09623`.
- Original matrix identity: `sha256:bcfee16fbbdddff24442027d22e32fe321dd0294479e40e78f44f0a929b5a762`.
- Original private archive SHA-256: `0eb62c73754e392399394625cf581243046fb046160556ba36167090bb754a70`.

The original request/environment, checkpoints, resource captures, host logs and
full private handover are intentionally absent. Original matrix/completion
references to omitted artifacts remain unchanged. **This is partial public
custody, not closed custody or complete checkpoint verification.** The
source-only receipt cannot reconstruct the omitted request/environment.

## Read-only verification

From a checkout with the pinned CPU dependencies:

```sh
PYTHONPATH=src:scripts python -B scripts/check_position_result.py
PYTHONPATH=src:scripts python -B -m unittest discover \
  -s tests/position_result_v1 -v
```

The checker validates original selected bytes and source bindings and reuses
authoritative admission, score validation and descriptive reporting to
recompute retained output metrics. Torch imports do not mean model execution:
no model is constructed, no checkpoint tensors are loaded and no training or
generation is launched. It does not provide numerical replay, independent-host
replication or hardware attestation. Do not amend original artifacts to match a
changed source tree or silently treat the derivative receipt as an original
request. Future changes require separate versioned evidence.

All six cells fitted training; the frozen all-seed/length transfer criteria
failed. The small in-range gains remain development observations, not a selected
winning seed, learned arithmetic guarantee, capacity limit or general rejection
of position coupling. No learning, fresh-confirmation, Phase 5 or release gate is
unlocked. Existing historical source closures and fixtures remain unchanged.
