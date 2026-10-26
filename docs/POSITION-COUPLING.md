# CPU position coupling preparation

Status: **checkout-only engineering conformance**, not a training experiment or
an arithmetic improvement. This increment implements explicit position addresses,
a strict input-derived arithmetic slot map and a source-bound synthetic check.
It does **not** implement a new research `freeze/run/verify` producer, checkpoint
import/resume, learned LUT, packed inference or scientific target execution.

Run and assess the already implemented [balanced-coverage comparison](BALANCED-ADDITION.md)
first. Its outcome was unrun at the immutable v0.5.0 checkpoint. Do not infer a
result from this code, replace that experiment, or reinterpret historical
[negative results](UNSEEN-NEGATIVE-RESULT.md). A later reviewed study must freeze
the remaining scientific choices below before target outcomes.

## Implemented boundary

The new `scripts/position_coupling_v1/` package is intentionally outside the
installed native API. `PositionDecoder` inherits native dense initialization and
state/input validation and reuses the existing attention/feed-forward blocks,
normalization, tied output and float32 operations. A small versioned forward
orchestration replaces only the embedding position lookup with an explicit
CPU int64 tensor of the exact `[batch, sequence]` input shape. Every address,
including masked positions, must be within the existing embedding inventory.
No implicit broadcasting, wrapping, clipping, backend fallback or GPU is allowed.

The model identity is `trinite.explicit-position-decoder.engineering.v1`, not
`trinite.decoder.v0`. Identical parameter bytes do **not** identify identical
behaviour: explicit position IDs and their schema/mode are additional inputs.
The receipt binds them separately. A future checkpoint protocol must also bind
these semantics; an old checkpoint's tensor identity alone is not admission to
this lane. There is no new checkpoint reader in this increment. Because `position_ids` is
mandatory, this is not a drop-in replacement for native `target_loss` or greedy
scoring. A future source-bound training adapter must supply those IDs explicitly;
no such trainer/scorer integration is implemented here.

Historical `src/trinite/model.py`, tokenizer, numerical updater/scorer, checkpoints,
source maps and fixtures stay byte-for-byte unchanged. This avoids changing old
producer identities. No optimizer or scorer is duplicated. Equal ordinary
sequence addresses reproduce native logits, captures and parameter gradients
exactly for the same input shapes in the pinned CPU scope. Comparisons that
change batch/prefix shapes use the existing model conformance bounds, absolute
2e-6 and relative 1e-5, rather than claiming cross-shape bit identity.

Causal attention always follows actual sequence order, **never** position IDs.
Repeated or descending addresses cannot reveal future tokens. Right padding is
still masked in attention, states, logits and capture, even if its token or
position IDs differ within their legal bounds. Capture selections are sequence
indices, not place-value addresses.

## Arithmetic slot schema

`trinite.arithmetic-position-slots.v1` accepts a complete prompt containing two
already zero-padded unsigned decimal operands of equal width D, with D in 1–4:
`47+58=` or `12+03=`. Signs, spaces, alternate carriers, partial prompts and
unequal widths are rejected. This is a deliberately narrow engineering syntax,
not automatic admission of a new training corpus.

`prefix_positions(prompt, emitted_count, mode=..., context_length=64, offset=0)`
returns positions for the **actual input tokens**, BOS + prompt + emitted slots.
It receives no answer values, sum, carry states or gold output length. The prompt
alone fixes D+1 answer-digit slots plus one EOS slot; emitted_count is bounded by
that D+2 total. The complete input must fit both the context and address bounds.

| Actual token/slot | Coupled position ID |
| --- | --- |
| BOS, `+`, `=` | 0 |
| Operand digits from most to least significant | offset+D down to offset+1 |
| Emitted answer slots 0 through D | offset+1 through offset+D+1 |
| Final emitted slot D+1, reserved for EOS | 0 |
| Right padding appended by the caller | Any legal ID; conformance uses 0 |

Sequence mode instead uses `offset + actual_sequence_index`, including BOS and
EOS. Nonnegative offsets are explicit and bounds-checked. This increment does
not sample offsets or claim that later evaluation embedding rows have been
trained. The synthetic receipt fixes offset zero. A future study must decide
and audit the complete shared offset/exposure policy; even exposed embedding
rows do not prove learned transfer to unseen significance or length.

For example, the coupled addresses for `BOS 47+58=501 EOS` at offset zero are
`0 2 1 0 2 1 0 1 2 3 0`. `501` is the reversed representation of 105.
For `12+03=`, the three-digit padded answer 015 is emitted as `510`.
In next-token training, construct positions for actual tokens and then slice
input IDs, attention masks **and position IDs** together with `[:-1]`; the loss
uses shifted target IDs/masks `[1:]`. Do not assign the next predicted digit's
address to the preceding token. In particular, `=` retains position zero even
though its logits predict the units digit.

Generated token values do not affect the slot map. A wrong digit, arbitrary
byte or misplaced special token receives the same slot address as any other
value at that index; bad model output is an outcome, not repaired input. A
future greedy scorer stops on the actual EOS token, retains malformed/unfinished
outputs and never forces a correct digit or EOS. The final reserved slot maps
to zero even if the model emits something other than EOS there.

`decode_answer` accepts exactly D+1 ASCII digits followed by EOS, reverses them
and strips leading zeroes for ordinary numeric display. Invalid sequences return
`None`. It neither computes the expected sum nor certifies arithmetic correctness,
and does not fill missing digits, repair output, retry or call a solver. Exact
representation accuracy including EOS must remain separately reportable.

## Local engineering check

From a clean checkout of the reviewed commit and the unchanged hash-locked CPU
environment in [GETTING_STARTED](GETTING_STARTED.md):

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/position_coupling_v1 -v
python -B scripts/check_position_coupling.py --output /new/path/position-conformance.json
```

The output must be outside the repository tree (resolved paths are checked).
The output parent directory must already exist; the file must not exist. The
launcher creates it exclusively and refuses overwrites. A computation failure
returns nonzero and records a failed report; retain that report and process logs.
A terminated process or failed filesystem write can leave a partial file, not a
valid receipt. This is not a hardened concurrent-filesystem custody protocol.

The CLI sets one CPU intra-op/inter-op thread and strict deterministic algorithms
for its own process. It requires the canonical Linux x86_64 CPU dependency lock,
uses fresh dense initialization seeds 0/1/2 and the existing 86,528-parameter
configuration (context 64, two blocks, width 64, four heads, FF width 128).
Two visible synthetic examples exercise same-shape native forward/capture/full
gradient equality, coupled causal isolation, masked-position isolation, model
immutability and caller RNG preservation. No optimizer step is performed.

The report contains actual source hashes (including the new contract, launcher,
tests and workflow), architecture, position schema/modes, exact token/position
arrays, tensor identities, environment and every completed seed on success.
Source identities are checked again after computation. Retain the exact Git SHA
and an independently held SHA-256 of the report when handing it over. A report
is a scoped self-reported engineering observation, not a signature, independent
host attestation, closed PROVENANCE custody, complete checkpoint replay or a
learning result. The numerical hashes alone do not certify the scientific
protocol. No research mode or target-training argument is exposed.

## Prospective arithmetic experiment

After the #74 evidence is reviewed, consider a separately versioned CPU study:

- Two fresh dense arms share exactly the same padded operands and fixed-width
  reversed answers. The control uses ordinary positions; the candidate uses
  significance-coupled positions. This isolates position assignment **conditional
  on the common new format**. Comparisons with old #74 results additionally
  change representation/data and cannot isolate position coupling.
- Pair seeds 0/1/2, effective initial tensors, architecture, row order, optimizer,
  input/target exposure, updates, decoding and final-checkpoint rule. A 4,000-update
  budget is a candidate, not a frozen or executed protocol. Padding changes token
  budgets; do not reuse #74 totals. Keep geometry zero, with no KV, quantizer,
  teacher, capacity, trace-objective or radix intervention bundled.
- Freeze exact train/development family inventories before outcomes. Keep both
  operand orders, alternate carriers and padded renderings of a fact together.
  Define carries, zeroes, repeated digits and long carry chains explicitly.
  Historical 0–12 facts remain exposed development, not fresh confirmation.
- Train a bounded mix of one-/two-digit problems. Measure new combinations
  within trained lengths separately from three-/four-digit length shift. The
  current slot map can represent those lengths but does not demonstrate transfer.
- Define all offset bounds and evaluation offsets before outcomes; record which
  embedding rows and significance relationships actually received training.
- Predeclare numerical success margins, tolerated seen-fit regression, minimum
  per-seed criteria and total time/memory/storage limits. Retain every seed,
  failure and timeout; do not select only successful restarts.
- Report raw greedy representation+EOS exactness, ordinary decoded arithmetic
  correctness, malformed/unfinished cases, carry/length strata, seen fit, input
  sensitivity and CPU wall time/peak RSS. Correlated carriers are not independent
  trials. Include constant/majority/lookup controls and a labelled non-neural
  exact oracle. No answer repair or evaluator-assisted inference.
- Reuse authoritative admission, update, scorer and safe tensor operations where
  compatible; add source-bound request, checkpoint/resume and complete replay
  contracts for the new semantics. Validate shifted labels/masks and prompt-only
  generation before target execution. Select a useful recipe using development,
  then freeze genuinely fresh confirmation families without peeking at outcomes.

Those family counts, budgets, success thresholds and full producer do not yet
exist in this increment. Engineering CI does not close these research gates.

## Separate learned LUT direction

The owner's Elite-inspired idea remains a separate compression experiment:
fixed coordinate+seed hashing chooses a learned shared floating-point table
entry. Gradients from all aliased connections accumulate into that entry.
Begin with modest feed-forward sharing and an untied control initialized from
identical expanded effective weights. Reusing a seed with differently shaped
parameters does not by itself match initialization.

Only a later intervention quantizes the table to ternary values. Count all
learned entries, scales, embeddings and exceptions; occupancy affects whether
scale reduction over table entries equals reduction over expanded weights.
Generating a full float matrix for `F.linear` is not evidence of reduced working
memory or packed acceleration. Measure CPU hash/gather overhead and peak memory
before claiming a speedup. Compression does not automatically improve arithmetic
transfer. No LUT code is added here; GPU comparisons remain optional later work,
not a requirement of the native CPU-first path.

## Research basis and claim limits

- [Position Coupling, Cho et al., NeurIPS 2024, v2](https://arxiv.org/abs/2405.20671v2):
  shared digit significance positions, reversed/padded answers and training
  offsets motivate this test. Its main Appendix C setting uses width 512,
  1,000,000 examples and 50,000 updates, materially different from this small
  diagnostic. This is a Trinite adaptation, not a paper reproduction.
- [Teaching Arithmetic to Small Transformers, Lee et al., 2023, v1](https://arxiv.org/abs/2307.03381v1):
  format/sampling sensitivity and continuing unseen-length failures motivate
  matched controls; reversal alone does not prove generalisation.
- [Compressing Neural Networks with the Hashing Trick, Chen et al., 2015, v1](https://arxiv.org/abs/1504.04788v1):
  learned shared parameters with fixed hashed addresses motivate the separate
  LUT study, not an arbitrary trained model compressed into a seed for free.
- [Neural GPUs Learn Algorithms, Kaiser and Sutskever, 2016, v3](https://arxiv.org/abs/1511.08228v3)
  remains background only: a different recurrent architecture and substantial
  729-instance searches, not the implementation guide for this increment.

These are cited ideas, not imported paper text, training data, model weights or
third-party runtime code. This increment addresses TRI-I01/I02/I05/I07/I08/I12/I14
engineering boundaries and preserves TRI-I11/I13. It changes no historical
learning, format-selection, geometry, scaling or release-readiness gate.
