# Non-certifying fixed-fit forward comparison

This is a separately authorized engineering diagnostic, not a replacement for
any native run or its guards. See the [tool contract](../../FORWARD-CHECKPOINT-COMPARISON.md).
The [machine report](diagnostic.json) binds the executed diagnostic source,
numerical source closure, fixed input specification, both original checkpoints,
original metadata and observed host. Its SHA-256 is
`5722cf8d2d0d5b1cddab5124806386fe28c76f60dcf4b7e21b00ea215eee8750`.

## Observation

On this one explicitly configured CPU diagnostic host, the prior checkpoint
produced **8/8** exact answer+EOS and the current checkpoint **7/8**, on the same
eight already-exposed public training fit rows. No heldout input was executed and
no heldout label was generated. This is descriptive training-fit behavior, not a
new learning/generalization result or fit certificate.

The only differing greedy row was `c0631`, expected `1`:

| Checkpoint | Raw greedy answer+EOS | First answer-token margin: `1` minus strongest competitor `-` |
| --- | --- | --- |
| Prior | `[49, 257]` (`1`, EOS) | `+0.10430073738098145` |
| Current | `[45, 49, 257]` (`-1`, EOS) | `-0.44473910331726074` |

Both final checkpoints contain 24,768 unique float32 model elements across
11 tensors. All 11 tensors differ; 19,679 elements have different bit patterns.
Maximum absolute difference is `0.003052853047847748`. This counts differing
float32 elements, not individual flipped bits. Teacher-forced later positions
use gold prefixes and are not free-generation continuation diagnostics.

The difference persists when both fixed checkpoints use the same foreign-host
forward implementation. This does **not** identify the training divergence step,
attribute its cause, establish nondeterminism, reproduce optimizer updates, or
provide a formal floating-point error bound. CPU/runtime metadata still differ
from the original environments and are reported, not overwritten. Native
foreign-environment rejection remains intact.

## Execution and preservation

Executed the documented command with the pinned prior payload from PR #122 and
external PR #124 payload at `e44d11b6863f6dc920169b231c55f80ebb10b7d0`.
The current payload remains external to this branch; no evidence commit was
cherry-picked. Only checkpoint tensors and metadata were read from the archives;
no contained source, native binary or heldout material was executed. Input payload
hashes were checked before and after, and all model tensors remained unchanged.

Explicit settings: CPU float32, one intra-op and one inter-op thread,
deterministic algorithms, highest float32 matmul precision, MKLDNN disabled;
120 CPU-second and 4 GiB address-space launcher bounds. These are diagnostic
settings, not a forged native runtime fingerprint. No optimizer was constructed,
no backward/update/resume/feasibility/freeze was run, and no main attempt or
research/release gate is authorized by this output.

Normal and optimized engineering tests cover an authentic native zero-step safe
snapshot, exact native forward-logit parity, forbidden training/heldout paths,
strict source imports, duplicate/nonfinite JSON and tensor headers, shape/dtype/
finite/closed tensor inventories, source/input tampering, archive/path/resource
bounds, unchanged tensors and truthful element-difference accounting. CI is
engineering-only and does not repeat this historical comparison.
