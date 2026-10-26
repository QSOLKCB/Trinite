# Native learning diagnostic workflow v1

Status: an additive, checkout-only engineering workflow for the diagnostic-first
[roadmap](ROADMAP-V1.md#milestone-1-one-canonical-native-diagnostic-workflow).
The full target learning run has **not** been executed or reported by this
implementation. Passing mechanics tests is not a new learning result.

## Scope and compatibility

`scripts/run_native_diagnostic.py` composes the existing admitted arithmetic
corpus, dense `BudgetPlan`, authoritative `training.run_steps` updater,
`training.target_loss`, safe budget checkpoint reader/writer and PROVENANCE
observer. Its versioned helpers live in `scripts/native_diagnostic_v1/`.
It neither replaces the installed API nor edits historical numerical producers,
protocols, fixtures or evidence. The immutable v0.4.0 baseline stays unchanged.

The fixed target is one seed from `0`, `1`, or `2`, with the historical dense
arithmetic `extended` plan: 86,528 parameters, context 64, two blocks, width 64,
four heads, feed-forward width 128, batch eight, constant learning rate 0.001,
and 4,096 updates. It retains checkpoints/diagnostics at steps 0, 1,024, 2,048,
3,072 and 4,096. The underlying plan limits scored update targets to 524,288.
The initial model is randomly initialized, not loaded from an external teacher.

Only exact admitted **training** rows enter updates and diagnostics. The inherited
updater's validation-named diagnostics deliberately alias training rows; they
are not held-out evaluation. No model selection or automatic follow-up run is
performed. Geometry loss is zero. Ternary comparisons, capacity increases,
RSH intake, teacher calls and a native KV-cache backend remain separate work.

Affected requirements are TRI-I01–I03, I05–I08 and I10–I14: inspectable native
inputs, CPU operation, observer isolation, source/data identity, scoped replay,
honest claims, split isolation, explicit boundaries and retained failures.
TRI-I04 and TRI-I09 semantics are unchanged; this dense-only increment introduces
neither a quantized kernel nor a geometry intervention.

## What the diagnostics distinguish

- Mechanics preflight independently checks shifted answer/EOS targets and
  token-weighted loss on different answer lengths, causal-prefix isolation,
  masked padding and tied embeddings. It exercises three tiny native updates
  and split safe-checkpoint continuation against uninterrupted updates, including
  optimizer/history identity and observation/RNG isolation. This is an engineering
  control, not the 4,096-update target experiment or a quality threshold.
- Teacher-forced observations record each scored token's loss, probability,
  rank and predicted token, including first-answer and EOS behavior. These use
  gold prefixes and must not be presented as free generation accuracy.
- Raw deterministic generation receives the **prompt only**, with a five-token
  cap. Complete answer, answer-plus-EOS, malformed output, unfinished output,
  premature EOS and length distribution remain distinct. There is no repair,
  constrained-output rescue or hidden retry.
- Training-derived majority and lookup outputs and the independently evaluated
  formal oracle are explicitly **non-neural controls**. Different-answer pairs
  expose identical-output behavior; admitted surface carriers compare prompts
  for the same formal input. Correlated pairs are not independent trials, and
  differing outputs alone do not establish reasoning.
- Observed timing and worker RSS are resource descriptions, not replay-equality
  requirements. Native KV state is `unsupported`; its payload is unavailable,
  not a measured zero. See the separately proposed
  [compute/KV harness](COMPUTE-KV-HARNESS-PROPOSAL.md).

## Commands and authorization boundary

Use a clean checkout of the reviewed producer commit and the hash-locked CPU
Python environment in [GETTING_STARTED.md](GETTING_STARTED.md). The launcher
requires this checkout's `src/`, deterministic CPU execution and one thread;
it does not download dependencies or models. Run commands from the repository
root. The output parent directory must already exist.

```sh
export PYTHONPATH=src
python scripts/run_native_diagnostic.py preflight
mkdir -p reports/native-diagnostic-v1
python scripts/run_native_diagnostic.py freeze \
  --seed 0 --request reports/native-diagnostic-v1/request-seed0.json
```

`freeze` creates a new request and prints its `sha256:…` identity. Retain that
identity independently of the request directory. The request binds protocol,
plan, seed, source closure, locked numerical environment and admitted corpus.
Existing request files are not overwritten; changing source or environment
requires a new request, never a patched historical hash.

The following command is an **execution recipe, not permission to run it**.
A full target run needs its own reviewed authorization after the producer and
protocol are published. `--authorize-run` acknowledges that prior approval; it
does not grant it. Do not invoke the internal `_worker` action directly.

```sh
python scripts/run_native_diagnostic.py run \
  --request reports/native-diagnostic-v1/request-seed0.json \
  --request-identity 'sha256:REPLACE_WITH_INDEPENDENT_REQUEST_IDENTITY' \
  --output reports/native-diagnostic-v1/attempt-seed0 \
  --authorize-run
```

Every attempt uses a fresh output directory. The controller prints an attempt
record, with a completion identity only on successful completion. Retain that
`sha256:…` independently as well. Preserve the entire directory after success,
failure, interruption or timeout; do not replace it with a successful retry.
A different seed requires its own request and separately scoped run.

For a completed attempt, both commands below are read-only and require the
original request identity and externally retained completion identity:

```sh
python scripts/run_native_diagnostic.py audit \
  --request reports/native-diagnostic-v1/request-seed0.json \
  --request-identity 'sha256:REPLACE_WITH_INDEPENDENT_REQUEST_IDENTITY' \
  --output reports/native-diagnostic-v1/attempt-seed0 \
  --expected-identity 'sha256:REPLACE_WITH_INDEPENDENT_COMPLETION_IDENTITY'

python scripts/run_native_diagnostic.py summarize \
  --request reports/native-diagnostic-v1/request-seed0.json \
  --request-identity 'sha256:REPLACE_WITH_INDEPENDENT_REQUEST_IDENTITY' \
  --output reports/native-diagnostic-v1/attempt-seed0 \
  --expected-identity 'sha256:REPLACE_WITH_INDEPENDENT_COMPLETION_IDENTITY'
```

## Retention, resource limits and verification claims

The request fixes a 900-second worker deadline and a 480-second cumulative
update-time allowance, checked across milestone segments rather than treating
each `run_steps` call as a fresh study budget. The controller's worker deadline
covers the subprocess; parent-side setup/audit is not included in that figure.
The update allowance is checked at update boundaries and is not a hard CPU-time
quota. Context, training-target, output-token and source-defined artifact bounds
remain enforced independently. A limit failure is not a completed result.

Individual workflow artifacts are limited to 16 MiB; retained payload writes
are capped at 32 MiB and the worker log at 1 MiB. The controller terminates a
worker whose log exceeds its bound. The PROVENANCE packet duplicates payload
bytes and adds manifests, so these limits are **not** a total-directory quota.
Do not describe observed RSS or artifact limits as a hard process-memory cap.
Whole-worker lifetime RSS includes imports and preflight; it is not isolated
model, scratch or KV memory. No speed or memory improvement is established here.

Retained files include the request, preflight, per-milestone diagnostic JSON,
safetensors and checkpoint metadata, report, resource observations, PROVENANCE
packet, verification and completion records. The controller retains its worker
log and attempt outcome; failures can also leave a failure record and partial
files. Missing completion is not repairable by inventing a completion receipt.

`audit` checks external commitments, source/environment/input binding, upstream
custody verification, safe checkpoint structure, milestone/model consistency,
exact admitted rows and recomputable summary/control fields. `summarize` audits
first rather than trusting an unchecked summary. Safe checkpoint loading uses
the existing reader, which reconstructs seeded initial state and its initial
loss; these commands are not computation-free. Neither reruns target training
or regenerates diagnostic predictions. Retained probabilities and ranks
are producer observations, not independently replayed inference. Full numerical
replay and independent-host reproduction require separate evidence.

## Verification and remaining gates

With the pinned CPU environment installed:

```sh
PYTHONPATH=src:scripts python -m unittest discover -s tests/native_diagnostic_v1 -v
```

Also run the mandatory existing suites in [AGENTS.md](../AGENTS.md) and the
[operating instructions](INSTRUCTIONS.md); a focused suite is not a full project
pass. Report actual counts, environment and failures with the PR/run rather than
turning these instructions into a claim of successful execution.

This is the first dense-arithmetic slice of Milestone 1, not closure of every
roadmap measurement. Prospective target evidence, useful-task quality, further
balanced-learning/counterfactual studies, ternary diagnostics, complete resource
accounting and independent replay remain separate gates. All reports keep
`learning_adequate`, `phase5_ready` and `release_ready` false. Mechanics success
must not be used to relax a failed scientific gate or launch larger experiments.
