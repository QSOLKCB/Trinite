# Qutrit conformance module v1

Status: additive engineering implementation, 2 November 2026. This is the second
approved increment after [OMI](OMI-MODULE-V1.md), before the separately scoped
bounded controller. No training, optimizer update, model change, QEC rebuild,
hardware action or deployment is included. The existing
[qutrit decoder and noisy-syndrome adapter](QUTRIT-MODULES-V1.md) remain unchanged.

## Pinned mathematical source and limits

This independently implemented local two-qutrit adapter follows the SUM/SWAP
conventions verified in [QEC's exact conformance tests](https://github.com/QSOLKCB/QEC/blob/27bbc798c4e38138ca5b0fff70b15c3271a1917d/tests/test_qutrit_css_decoupling_conformance.py),
at release [v177.2](https://github.com/QSOLKCB/QEC/releases/tag/v177.2).
QEC's upstream reference is [hmx21/css-decoupling](https://github.com/hmx21/css-decoupling/tree/a48048c1097f4d1b37a55eca9a1796a823f0c124)
(MIT). Attribution and immutable source identities are retained; no donor code,
prose corpus, remote dependency, or full toric certificate is imported.
Our new code and reused repository JSON parser retain [MPL-2.0](../LICENSE).
Generated operands and answers are commissioned finite arithmetic facts, not
admitted copyrighted text or pretrained teacher outputs. This narrow source
admission does not relax [DATA](DATA.md).

QEC's full 18-qutrit/17-gate certificates remain external references. This module
verifies only bounded local gate conjugation, not their general theorem, Lean-to-
Python correspondence, a decoder threshold, quantum hardware, or model learning.
It imports none of QEC's IRC/security authority operations.

## Exact public problem

A word is exactly four ASCII digits in `012`, ordered **x0,x1,z0,z1**; residue 2
means -1 modulo 3. Global Pauli phase is omitted. These are explicitly public
mathematical operands, **not hidden physical errors given to a syndrome decoder**.
A circuit is one of `S1`, `S2`, `W`, `S1W`, `WS1`, `S2W`, `WS2`, read left to right.

- `Sw`: weighted SUM control 0, target 1: x1 += w*x0, z0 -= w*z1 modulo 3.
- `W`: swap both x and z coordinates.
- `apply`: return the four-trit forward conjugation.
- `inverse`: reverse gate order and negate each SUM weight; return four trits.
- `check`: compare the supplied four-trit candidate to the forward answer;
  return `1` for equality, `0` for inequality. Candidate is empty for other tasks.

A wrong-sign/order/weight construction is not automatically negative: if its
output happens to agree (for example the zero vector), the correct label is `1`.
Malformed input, unsupported circuits, stale bindings, timeouts and process
failures are errors, **never a semantic `0` label**.

Exactly five request fields: `schema`, `word`, `circuit`, `task`, `candidate`.
Schema is `trinite.qutrit-conformance-request.v1`. Explicit CLI operations:
`observe` emits those public operands under
`trinite.qutrit-conformance-observation.v1`; `evaluate-privileged` emits only
`schema: trinite.qutrit-conformance-outcome.v1` and `answer`.
No privileged default exists. The bounded parser rejects duplicate/unknown fields,
null, float, escaped/non-ASCII strings, wrong types, trailing input and oversized
requests. CLI errors exit 2, stderr diagnostics, no success JSON. Output-I/O
failure may leave partial bytes; callers must require a successful process and
complete validated JSON. The Python boundary uses the existing bounded process
runner, strict schemas, source/binary rehashing and a disposable local build receipt.
This is not OS isolation or hostile-host attestation; source/binary TOCTOU and
operator access remain trust boundaries.

## Compact carrier and token accounting

Example public carrier: `["Q1","1001","S1","apply",""]` followed by newline `=`.
Parsing must losslessly reproduce the canonical public observation. No answer,
family, split, seed, oracle, source path or provenance is added to the prompt.
A check candidate is deliberately supplied public problem data, not a privileged
answer leak. `public_only()` does not invoke the oracle or native runner.

The actual ByteTokenizer counts ASCII bytes plus BOS, answer and EOS. Prompt cap
is 120 bytes; four answer bytes plus EOS reserve at most 126 complete tokens,
within the existing 256 context. Check answers use one byte plus EOS. No silent
truncation, vocabulary change or dense text certificate is used. `ledger()`
reports prompt/answer/supervision counts separately and full-prefix forward work
`sum(P+k for k in range(G))`, not a claim of KV-cache reuse. The future controller
must meter total calls, emitted tokens and repeated-prefix work separately.

## Engineering admission and exposure

Structural families are **SUM** (both weights; train), **SWAP** (validation), and
**mixed** (both orders and weights; test). Every operand, candidate and apply/
inverse/check view of a structure stays together. `admit_train()` rejects other
families before any privileged invocation, checks native observations and answers
against the independent oracle, masks the entire prompt/BOS, and records source
hashes, provenance and `training_authorized: false` outside the public carrier.

All three families and all operands are intentionally exposed in conformance
tests. These splits test boundary enforcement, **not fresh scientific heldouts**;
shared algebra and only three families limit any generalization interpretation.
Withholding the entire SWAP primitive confounds unseen operator semantics with
compositional generalization. A future training protocol should consider teaching
both primitives before holding out suitable composition families; this engineering
partition is not a ready-to-run scientific split.
A future study requires its own preregistered exposure assessment, controls,
source freeze, budget, holdouts and explicit approval. No native updater is called.

## Reproduce without training

```sh
python3 scripts/run_qutrit_conformance_v1.py build
(cd external/qutrit-conformance-v1 && cargo test --offline --locked &&
 cargo fmt --check && cargo clippy --offline --locked --all-targets -- -D warnings)
PYTHONPATH=src:scripts python3 -B -m unittest discover -s tests/qutrit_conformance_v1 -v
python3 scripts/run_qutrit_conformance_v1.py inspect --word 1001 --circuit S1
```

Requires official Rust 1.90.0; dependencies are only the local crate. Both Rust
and Python independent oracles conjugate nine computational-basis kets, rather
than reuse the production Pauli update equations. Exhaustive 81-word, seven-
circuit checks cover forward/inverse, commutation preservation, valid and invalid
candidate comparisons, and convention-negative examples. Additional tests enforce
actual CLI rejection, public noninterference, family-before-privilege admission,
real tokenizer counts/masks, source binding and infrastructure-failure separation.

Affected TRI-I01/I03/I05/I07/I08/I11/I12/I13/I14. Historical fixtures, stopped runs,
8/8 versus 7/8 outcomes, PR125 diagnostics and numerical semantics are untouched.
No demonstrated held-out learning gain follows from engineering conformance.
