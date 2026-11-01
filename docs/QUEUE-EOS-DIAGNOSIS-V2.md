# Queue EOS collapse: diagnosis and preparation

Preparation only. No new training, optimizer step, scientific freeze or held-out
model diagnostic was run. The [v1 negative result](QUEUE-LEARNING-RESULT.md),
[protocol](QUEUE-LEARNING-V1.md), producer and every historical fixture remain
unchanged. TOMBSTONE-derived learning is outside this queue-only change.

## Verified baseline and evidence

Current main was fetched at `41628375b6e417cab09f686a85d7df92b33926b5`, the merge
of PR #113. All 72 push workflows completed successfully, including queue
engineering; the [CI receipt](evidence/queue-eos-diagnostics-v2/baseline-ci.json)
identifies their exact head and URLs. There are no repository SKILL.md files or
scoped AGENTS.md files in the affected paths. Root instructions and affected
queue, model, tokenizer, training and checkpoint contracts were inspected.

The original archive SHA-256 is
`c404bd28b39cc6b2359c0964ef68ee207468037c865039babe647a643292d958`.
All 9,201 extracted file hashes were checked against the published SHA256SUMS
before reading training checkpoints. No evidence was resealed or overwritten.
The unchanged v1 loader rejected this host's environment, as intended: original
EPYC/AVX512, Python 3.12.14, Linux/glibc 6.18.57/2.41 versus current
Ryzen/AVX2, Python 3.12.13, Linux/glibc 7.0.0/2.43; interop threads also differ.

The new v2 **foreign-environment forensic inspection** checks published file
anchors, exact v1 source closure, plan/data/initialization identities and uses
the authoritative closed safetensors reader. It explicitly records both
environments. It does not change v1 restoration, authorize resume, establish
exact numerical replay or attest the original updates. AdamW.step is patched
to fail, all measurements use no_grad, and model identities must stay equal.
The [raw training report](evidence/queue-eos-diagnostics-v2/training-forward.json)
retains all initial/final outputs and strict parsing status for 256 unique train
rows per seed. No validation/test examples enter these diagnostics.

## Findings

| Boundary inspected | Evidence and conclusion |
| --- | --- |
| Answer boundary and shift | ASCII prefix length is a byte offset. BOS occupies position 0. The first scored answer token is at 1+len(prefix); native logits at len(prefix), the final `=`, predict it. Synthetic unequal-length answers and gradient support test the actual target_loss. No off-by-one defect found. |
| Masks and normalization | BOS/prompt/PAD targets are zero; decimal bytes and one EOS per row are one. Native weights are loss[:,1:], targets ids[:,1:], logits from ids[:,:-1]. Mean divides by scored targets, not prompt length or batch size. Synthetic tests check PAD/prompt zero gradient and unequal lengths. |
| Supervision counts | One unique pass has 444 answer-byte targets and 256 EOS targets: 700 total. Two scheduled passes give 888+512=1,400 targets and 512 visits. EOS is 36.57% of supervision; answer supervision is present. |
| Token IDs and serialization | Bytes 0..255, BOS 256, EOS 257, PAD 258; decimal zero is byte 48, never EOS. Synthetic 0/7/42/256 round trips preserve canonical digits and EOS. No serialization defect found. |
| Causal attention | Lower triangular attention excludes future keys; right padding is masked. Changing answer/future tokens leaves prefix logits equal within 1e-6 absolute tolerance. Full teacher-forced boundary logits agree with prompt-only first-generation logits at that tolerance. |
| Checkpoint loading | Published checkpoint/data hashes match. Safe reader validates closed inventories, shape/dtype/finiteness, tied parameters, source/plan/data/init identities and recorded progress. V1 rejects the foreign environment; the separately named v2 inspection records this limitation. No wrong-checkpoint evidence found. |
| Optimizer | Native code zeroes gradients, backpropagates, clips, calls single-tensor AdamW once then advances counters. Each unique parameter appears once. Retained moments have step 64, histories have 64 entries, cursor 512, target count 1,400; every named model tensor differs from initialization. This supports actual optimization but is not optimizer replay. |
| Stopping and parsing | Greedy starts with BOS+prompt, takes unrestricted argmax, appends EOS and stops. Cap remains five tokens. EOS-only [257] is malformed empty output. Synthetic tests cover immediate EOS, digit+EOS, cap exhaustion, leading zeroes, invalid bytes/specials and out-of-range output. No stopping/parsing defect found. |

On this host, both final checkpoints emit `[257]` on **256/256 train rows**.
Teacher-forced answer accuracy is **0/444** and EOS accuracy **256/256** for
each seed; first-answer accuracy is 0/256. Initial models emit no EOS-only
answers and also have zero exact train answers.

| Seed | Initial mean train NLL | Final mean train NLL | Final answer-byte NLL | Final EOS NLL |
| --- | --- | --- | --- | --- |
| 17 | 5.5510 | 2.5767 | 2.9273 | 1.9687 |
| 29 | 5.5793 | 2.5382 | 2.9377 | 1.8454 |

The retained original final mean losses agree to the displayed precision.
Exact float.hex sums, target denominators, original losses and parameter changes
are in the report. These are foreign-host forward observations, not exact replay.
The model reduced loss while failing to learn an answer boundary that generates
digits. An aggregate loss alone concealed that failure. The demonstrated gap is
**diagnostic visibility**, not a demonstrated numerical implementation defect.
No updater, model, codec, parser, decoder or loss correction is justified here.

There are 125 distinct public prefixes among 256 train rows, and **all 125 have
conflicting targets** because the hidden suffix can change. Full deterministic
exact fit is therefore impossible. This is partial observability by design,
not label corruption. It cannot explain EOS-only generation: every first answer
target is a digit. The separate eight-prefix sanity check takes one fixed label
per prefix to ask whether this model/procedure can fit a tiny unambiguous pool.

## Unresolved hypotheses

Insufficient optimization at 64 updates, competition between EOS and decimal
classes, weak boundary/position conditioning and limited useful input learning
remain hypotheses. Lower NLL, finite gradients, changed tensors and valid custody
do not distinguish them or prove useful forecasting. More training may still
produce EOS-only or a constant numeric answer. No EOS reweighting, suppression,
forced numeric decoding or postprocessing is proposed. Any such change would
require a separately versioned experimental treatment.

The [v2 proposal](QUEUE-LEARNING-V2.md) changes only the optimization horizon
in the main training treatment, adds an independent train-only fit gate, and
replaces exposed evaluation families prospectively. Fresh-family construction
is a new generalization setting, so it cannot isolate a causal gain over v1's
exposed held-out scores.

Affected: TRI-I01–I03/I05/I07/I08/I11–I14. Installed API and historical numerical
semantics are unchanged. No learning, release or later research gate is advanced.
