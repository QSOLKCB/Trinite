# Context-symbol relation v1

Status: original synthetic external engineering module, not a trained capability.
Rust 1.90.0, standard library only; no native imports, RNG, network, floats,
model loading, optimizer or data admission changes. Supports TRI-I03, I04,
I07–I08 and I10–I14. Historical queue-v2 and all producer evidence stay unchanged.

## Exact finite semantics

The request schema is `trinite.context-symbol-request.v1`. Required keys only:
`contexts`, `symbols`, `meanings` (integers 1..8); `table` (exactly contexts
rows, each exactly symbols integer meaning indices 0..meanings-1);
`context_mask`, `symbol_mask` (boolean vectors of corresponding dimension);
and `query` (one meaning index). The dense table is a total, single-valued map
from context × symbol to mutually exclusive meaning. Every cell is explicit.
No corpus-derived symbolism or psychological truth is encoded.

Masks are **possibility sets**, not per-field evidence-present bits: true means
that candidate remains compatible with visible evidence. A singleton identifies
the observed value; all true means wholly unknown; subsets express partial
knowledge. All false is inconsistent and rejected, never mapped to zero.
The compatible worlds are exactly the Cartesian product of admitted contexts
and symbols. There is no hidden actual context, symbol, meaning or answer.

For the proposition "this symbol in this context has query meaning", return:
- +1 when every compatible world supports it;
- -1 when every compatible world contradicts it;
- 0 when some support and some contradict it.

Thus missing context does **not** automatically mean zero. If all candidate
contexts assign the queried meaning to the candidate symbol, the result is +1.
With only one meaning, every valid request supports its only possible query.
Unknown input, mixed entailment, invalid input and learner abstention are
different concepts. This is a ternary logical relation, not a weighted evidence
score, probability, confidence calibration or model-weight precision claim.

## Separate public and privileged operations

Run in `external/context-symbol-relation-v1`:
```sh
cargo build --offline --locked --release
target/release/trinite-context-symbol-relation-v1 observe < example-request.json
target/release/trinite-context-symbol-relation-v1 evaluate-privileged < example-request.json
```

`observe` validates and serializes the supplied public facts only, using schema
`trinite.context-symbol-observation.v1`. It does not call the relation evaluator.
It contains no computed relation/label, target answer, family ID, seed, hidden
state or evaluator path. The query is the public proposition to answer, not its
truth value. Public rules deliberately contain sufficient information to derive
truth; that is the formal reasoning task, not answer leakage.

`evaluate-privileged` emits only schema
`trinite.context-symbol-outcome.v1` and `relation`. A future controller must
isolate this operation/output from learner input, including file paths and
metadata, and commit predictions before evaluator disclosure. CLI separation
and JSON alone are not a security sandbox. This module does not implement that
controller, native admission or training.

Example rule table `[[0,1],[1,1]]`: symbol 0 with query 0 supports in context 0,
contradicts in context 1, and is mixed with both contexts possible. Symbol 1
contradicts query 0 even when context is unknown. Tests retain these same-symbol
changed-context counterfactuals and verify joint index-renaming invariance.

## Bounds and failure policy

Input is at most 16,384 bytes, JSON nesting at most 8, dimensions at most 8
and table at most 64 cells. Evaluation visits at most 64 cells. Parser allocation
is bounded by input bytes and depth. CLI reads at most MAX_BYTES+1; invalid
requests exit 2 with stderr and no stdout. There is no repair or truncation.
The caller must close stdin and enforce a process deadline: the byte cap does
not time out a stalled input pipe. Non-UTF8 command arguments are rejected.
Only unescaped ASCII strings, canonical signed i64 decimal integers, booleans,
arrays and objects are admitted. Reject null, float/exponent/leading-zero forms,
negative zero, overflow, escapes, non-ASCII, duplicate/unknown/missing fields,
trailing input, malformed dimensions and out-of-range indices. Zero indices
are valid; masks cannot be encoded as numeric 0/1. Direct Rust APIs revalidate.
Output has fixed field order and newline; request key order/whitespace does not
change its bytes. No runtime cost or acceleration claim follows from these bounds.

## Verification and rights

```sh
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --release
cargo build --offline --locked --release
python3 tests/oracle.py target/release/trinite-context-symbol-relation-v1
```

The independent Python oracle constructs truth sets, exhausts all 1..2 context,
symbol and meaning dimensions, every total table, nonempty possibility mask and
query, and checks maximum dimensions, counterfactuals, renaming invariance,
canonical observations and malformed requests. It is not an independent host
replication or learning result. All examples are original small synthetic tables.
Public tests are training-exposed engineering fixtures, never fresh unseen data.

Jung and collaborators, *Man and His Symbols* (1964), is bibliographic inspiration
only; no book text, OCR, images, quotations or personal chat is imported.
Context sensitivity is not a validation of psychological theories. Background:
[Bowman et al. (2015)](https://aclanthology.org/D15-1075/) on entailment and
[Lake and Baroni (2018)](https://proceedings.mlr.press/v80/lake18a.html) on systematic
generalization motivate evaluation questions, not proof this module benefits
Trinite. See the [prospective module roadmap](EXTERNAL-ENVIRONMENTS-ROADMAP.md).
