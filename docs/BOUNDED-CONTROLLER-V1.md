# Bounded external controller v1

Engineering only, after the separate OMI and qutrit-conformance modules. No
training, checkpoint loading, latent recurrence, architecture changes, held-out
scores or learned reasoning are implemented or claimed. Historical runs and
frozen bridges are unchanged. TRI-I01/I05/I07/I10/I11/I12/I13/I14 apply.

`bounded_controller_v1.run(freeze(request), next_token)` accepts a trusted
synchronous callback mapping an immutable tuple of public byte-token IDs to one
next token. It sees only BOS, the admitted compact public carrier and its own
current answer prefix. Prior step output becomes the next public word; traces,
source receipts, original answer, scores and corrections never enter its context.
No teacher or exact oracle is called in the execution path. `evaluate(result)`
is a separate post-execution arithmetic evaluator, not a model callback or
receipt/custody verifier; a forged result is not authenticated by this scorer.

The callback is an interface, not an implemented neural checkpoint adapter.
ConstantPolicy is explicitly an input-insensitive engineering control. A future
native adapter must verify checkpoint/source/runtime identities separately and
report actual forward counts and memory. This in-process API is not a sandbox:
trusted callbacks must terminate and must not mutate global state, files or read
privileged information. Injected callback code/configuration are not covered by the controller source
closure; receipts explicitly mark them unbound. The CLI records its constant
configuration and its ConstantPolicy implementation is in that closure.
No hard wall-time or OS memory enforcement is claimed.
A hanging/untrusted callback requires a separately reviewed subprocess adapter.

## Request and bounds

Exact keys: `schema: trinite.bounded-episode.v1`, `module: omi|qutrit`,
`observation` (existing module schema), `mode: direct|stepwise`, `budget`
with exact integer keys `calls` (1..8), `tokens` (1..40), `prefix_positions`
(1..4920). Unknown fields, booleans, stale source inventories and malformed
observations fail closed. Rust observe admission checks binary/source receipts
on every call. Source closure includes both module closures and tokenizer,
contracts, package initializer, gate, launcher and this contract.

Direct OMI forward/backward k and stepwise k one-step prompts ask the same final
question for k=1..7. They have equal *ceilings*, not equal realized compute.
Stepwise consumes its own predicted words, never oracle-corrected intermediates.
Repeated words stop when further requested steps remain. Completion takes
precedence at the exact finite horizon: a legitimate fixed point at k=1 or
period-two return at k=2 is not artificially marked as failure. Early cycles
produce a stopped receipt, not a success. Direct period is one decimal byte plus EOS;
all other OMI answers are four uppercase hex bytes plus EOS. Qutrit is one-shot
apply/inverse/check over its declared circuit vector; check has one byte plus
EOS, other outputs four GF(3) digits plus EOS. It is not physical QEC decoding.

Before dispatch, reserve worst-case answer work using the actual ByteTokenizer
and module ledger. BOS counts, EOS counts; no truncation or transcript growth.
120 ASCII prompt bytes + BOS + four answer bytes + EOS =126 context tokens.
Five full-prefix operations cost121+122+123+124+125=615 positions; eight calls
reserve40 operations and4920 positions. A one-byte answer uses two operations,
243 positions at the maximum prompt size. Actual carriers are smaller. A bound
that cannot accommodate the complete next answer stops before admission/policy.
Missing EOS, wrong alphabet/length, malformed token, repeated state, exhausted
budget or completed horizon stops without retry/repair. No ninth call is possible.

Receipts retain public prompts/raw valid tokens and stop status externally.
`full_prefix_operations` and `processed_prefix_positions` count policy invocations
and their exact prefix shapes, **not FLOPs, neural forwards or speed claims**.
Policy, native-runner and total seconds are separately measured wall times;
peak memory and actual neural-forward count are null (not measured). Timing is
not deterministic replay identity. Callback exceptions stop with a policy_error receipt (exception class only),
without retry. Failed invocations count as operations/positions, not emitted
tokens; invalid responses likewise are not accepted tokens. These receipts are not scientific evidence or
an authorization to train. Fresh exact scientific freeze and owner approval
remain required; engineering-exposed inputs cannot become blind holdouts.

## Local reproduction

Use Rust1.90 and build both local source-bound runners first:

```sh
python3 scripts/run_omi_module_v1.py build
python3 scripts/run_qutrit_conformance_v1.py build
PYTHONPATH=src:scripts python3 -B -m unittest discover -s tests/bounded_controller_v1 -v
PYTHONPATH=src:scripts python3 -B -O -m unittest discover -s tests/bounded_controller_v1 -v
```

`python3 scripts/run_bounded_controller_v1.py --constant 0000` reads one bounded
JSON request on stdin and emits an engineering result, without automatic oracle
scoring. It has no arbitrary executable, training or checkpoint option.
