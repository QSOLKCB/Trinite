# Procedural dataset and training admission v1

Status: bounded engineering implementation for the three external formal modules.
**No model training, optimizer, model selection, scientific pilot or learning gain.**
The installed native CLI, numerical updater, geometry (loss remains zero), queue
v1/v2 and all historical evidence remain unchanged. Supports TRI-I03, I07–I08,
I10–I14. Admission is a source-specific checkout-only API, not a generic license
for synthetic data or automatic authorization to execute training.

## Deterministic generator and grouped inventory

`scripts/procedural_dataset_v1/cases.py` generates requests on demand from an
integer seed 0..65535. Version 1 has nine whole structural families: factor
counts 2/3/4 for ternary evidence; original parent byte lengths 4/6/8 for
TOMBSTONE; square context/symbol/meaning dimensions 2/3/4 for context relations.
Within each task those sizes belong respectively to train/validation/test.
All seeds, mask interventions, index/byte renamings, outcome counterfactuals,
and parent mutations inherit their original family's split. Canonical duplicate
variants are removed deterministically by the generator before inventory.
Admission independently regenerates the complete inventory, checks canonical
public duplicates and query-independent public-prefix collisions across families,
and retains seed, family, variant and split only in privileged provenance.

These are coarse dimension/length structural partitions, **not proof of unseen
reasoning structure**. All generators and conformance cases are public and
engineering-exposed; even the splits named validation/test are not fresh research
evidence. Seeds can repeat semantic structure. Dimensions/lengths expose split
membership and can correlate with difficulty; TOMBSTONE length/error precedence
also supplies legitimate easy cues. No random-row generalization claim follows.
A future experiment needs frozen fresh family inventories, balanced controls,
matched exposure and a separate prediction-commitment/evaluation controller.
Never select generators, seeds, thresholds or models using these held-out labels
and rebrand the resulting comparisons as unseen confirmation.

## Public carrier and deterministic task targets

The compact carrier is canonical JSON arrays tagged `E1`, `T1`, or `C1`, with
fixed positional fields. Encode/decode preserves the entire approved public
schema, including all operands, masks, active state and entry ordering; schemas
and fixed input-kind literals are losslessly restored. Unknown fields, wrong
kinds/ranges, duplicate IDs, noncanonical missing trits, empty possibility masks,
noncanonical JSON, or overlength input are rejected, never repaired/truncated.
Ternary entry IDs are public operands, not generator provenance IDs.

- **E1:** weights and each entry's ID, active flag, trits and observation mask.
  Target rows contain status, exact unreduced support S (null when undefined),
  denominator D, total weight W, observed count and factor count. This preserves
  S/D and D/W exactly, including inactive/abstaining distinctions. It is arithmetic
  reconstruction, not a calibrated probability or predictive validation.
- **T1:** before, removed, after and proposed byte offset. Target is the native
  semantic label `valid-splice`, `argument`, `length` or `mismatch`; never a receipt
  hash. Every supplied operand is public, so the classification is determined.
  Valid supplied repeated-span witnesses do not establish unique historical deletion.
- **C1:** dimensions, full total rule table, possibility masks and public query.
  Target is `1`, `-1` or `0` for support in all worlds, contradiction in all worlds,
  or mixed worlds. Unknown context can still entail/refute; inconsistent masks
  are invalid rather than relation zero.

The model prefix is carrier plus `\n=`. Native `ByteTokenizer` measures actual
UTF-8 bytes plus BOS/EOS under the unchanged 256-token context. Admission reserves
96/12/2 answer bytes for E1/T1/C1 without evaluating held-out labels; training
answers receive exact byte/token/loss-mask measurement. Loss starts only after
the separator and includes EOS. Inference `public_only()` returns prefix IDs
without EOS, and accepts only a public observation and module name—not a target,
seed, family, split, receipt, source path or rights record. The public carrier
contains facts sufficient to solve the problem, not a precomputed answer.

## Source-specific admission and custody

The admission record identifies original commissioned formal numeric facts,
minimal notation, authorship, rights basis, exact scope, generation and verification,
review date and limitations. No book text, images, personal chat, corpus, teacher
answers or pretrained weights enter any artifact. Generator software licenses
are distinct from the data rights assessment; this is not legal certification.
Jung's bibliographic inspiration in earlier docs imports no copyrighted content.

`prepare` creates a fresh owner-only directory outside the checkout, preserving
partial failures with the failing request, module/operation and bounded process
evidence. Its files are `attempt.json` (seed/protocol/source/binary pins retained before execution),
`public.json` (observations only), `requests.json`
(privileged generation inventory), `train-targets.json` (train-only answers and
native token/mask arrays), `admission.json`, and source/binary/protocol-bound
`manifest.json`. Retain the printed SHA-256 independently; `verify` requires it.
No caller-controlled relative file paths are followed. Extra/missing artifacts,
symlinks/hardlinks, duplicate JSON keys, unknown fields, altered source/build
identities, counts, labels, masks, provenance or hashes fail closed. Recomputing
self-reported hashes cannot pass full independent regeneration/native oracle replay.
Verification is read-only; it never overwrites historical evidence or caches success.

The train filter precedes native privileged oracle calls. Validation/test targets
are neither produced nor exported by admission. Public observations are checked
against native `observe` for TOMBSTONE/context; ternary has only `score`, so its
fully public request is strictly validated directly. Every training target uses
actual module execution and an independent Python formal oracle, with full
TOMBSTONE receipt reconstruction and typed native-code checks. Test conformance
separately exercises actual oracle outcomes for all three module splits and retains
a source-bound aggregate in `fixtures/procedural-dataset-v1/conformance.json`.
This receipt is regenerated using actual binaries, not trusted as cached success.
`training_rows()` re-verifies the whole source-bound bundle before returning only
train target/token rows. It does not import a model, execute updates or enable a
training CLI. Future training must consume only those validated rows under its
own frozen, reviewed protocol and explicit run approval.

JSON/file separation is not OS isolation. The trusted operator must expose only
public prefixes to an inference process and keep the entire private bundle and
its paths inaccessible. This layer is not a prediction commitment/scoring gate.
A malicious writer with checkout/build access remains outside its threat model.

## Offline execution and verification

Use Python 3.11+, Rust 1.90.0, C99 compiler, ar/pkg-config and OpenSSL development
headers already installed. No implicit downloads or runtime network calls occur.

```sh
python3 scripts/run_procedural_dataset_v1.py build
python3 scripts/run_procedural_dataset_v1.py prepare --seed 17 --out /tmp/procedural-v1-new
# Keep printed hash independently; destination must not already exist.
python3 scripts/run_procedural_dataset_v1.py verify --bundle /tmp/procedural-v1-new --manifest-sha256 HASH
PYTHONPATH=src:scripts python3 -m unittest discover -s tests/procedural_dataset_v1 -v
PYTHONPATH=src:scripts python3 tests/procedural_dataset_v1/conformance.py --verify-fixture
```

Fixed build paths and operation allowlists forbid caller-selected executables.
The unchanged bounded transport closes stdin, uses sanitized environment,
kills/reaps process groups on a five-second deadline and caps buffered combined
stdout/stderr at 16,384 bytes. Source and binary identities are rehashed before
and after every run; pinned TOMBSTONE upstream/build identities are also checked.
Build records are disposable local bindings, not remote compiler attestations.
Compiler/system libraries/hardware and check-to-execution races remain trust
boundaries. Malformed requests, timeout, provider/ABI/crypto failure and unknown
native statuses reject the attempt; none become semantic negative labels.

Tests cover native outcomes, regeneration, compact roundtrips, duplicate/path and
source/label tampering, byte budgets, train/evaluation custody and failure retention.
Engineering timings/token counts are not equal-compute or context-capacity results.
Later fresh frozen comparison must measure matched full-information baselines,
serialization/module/preprocessing/inference costs, memory, failures and negative
transfer before claiming context offload or competence gains. Parameter count,
more examples and external computation alone establish none of those claims.
