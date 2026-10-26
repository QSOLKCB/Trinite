# Frozen addition token diagnostic v1

Status: separately versioned observational implementation and engineering controls.
**No target diagnostic result is claimed.** This lane cannot train, tune, repair,
choose a checkpoint, or turn a failed raw answer into a success. Target execution
requires reviewed green exact-commit CI and separate owner local-run authorization.
The [original addition producer](ADDITION-DEVELOPMENT.md) and all its source-bound
files remain byte-identical. Its historical implementation-status statement is not
rewritten when later local outcomes become available.

## Question and fixed scope

After the matched-coverage development run, where does the frozen model assign
probability to the correct next answer token, relative to EOS? First-token checks
alone cannot distinguish answers such as `11` and `13`. This workflow observes
**every answer byte and the final EOS**, with one-based ranks, probabilities,
argmax, cross-entropy, and an authoritative native target-loss cross-check.

The only admitted models are all six final **step 4000** checkpoints from one
completed, externally committed addition-development-v1 matrix: control and
candidate for seeds 0, 1 and 2. All 36 familiar-answer, 124 novel-answer and 138
range/length development rows are required in each cell. No selection, missing
cell exclusion, retry, training example substitution, new data, or retraining is
permitted. These are already research-exposed, correlated development families,
not new confirmation trials or independent samples.

For each target, the diagnostic reads the **correct preceding answer tokens**.
It is conditional teacher forcing, not prompt-only generation. Logits at sequence
position `target_position - 1` predict that target. EOS at a gold answer position
is a conditional premature-termination tendency; EOS at the final target is the
correct end marker. Neither is a replay of the raw generation trajectory. Ranks
sort descending logits, resolving ties by lowest token ID. Probabilities use the
canonical float32 backend; cross-entropy is stable even when softmax underflows.
All answer positions are retained, with separate first/later/final-EOS summaries.

Original prompt-only greedy predictions and their exact summaries are copied
**byte-for-byte** alongside the new measurements. The unchanged authoritative
scorer checks retained original score consistency. These scores are not rerun,
repaired, or replaced by gold-prefix accuracy. Report both separately: a strong
conditional second digit does not fix an incorrect first digit. This diagnostic
may localize an error pattern, but cannot establish a causal internal mechanism.

## Import and evidence boundaries

Freeze requires independent external identities for the original request, original
matrix, and **all six final models**. Do not derive a supposedly independent expected
identity from the artifact being checked. The source-bound model-only importer
validates original source/environment/admission, all five checkpoint milestones,
closed provenance and named artifact index, exact scheduled visits and history
continuity, safe model and AdamW slot structure, and retained original scores.
It uses the authoritative safetensors reader and copies only model parameters;
it never constructs an optimizer, performs an update, or runs a forward pass.
Structural import is not numerical prediction or optimizer replay.

The original producer's 189 source commitments are preserved, including its bound
documentation. The new request additionally binds every new runtime module, fixed
launcher and this contract, and checks actual import origins. The original exact
CPU environment must still match. A changed interpreter, backend or original source
must be rejected, not patched into the old request. Preserve the mismatch receipt.
An old producer checkout is never executed from an uploaded bundle.

Use a quiescent local input and checkout. Per-cell inventories reject observed
links, special files, replacement and mutations, and bracket import/inference.
They are not a race-free filesystem snapshot against a hostile concurrent writer.
Input, request and output must be disjoint; new outputs are exclusive, never
in-place. The six-cell producer is larger than the legacy single-packet bound,
so its matrix root and each cell are checked separately without weakening that
bound. Original producer bytes and old negative-result fixtures remain untouched.

## Resources, failures and claims

The canonical lane is dense CPU float32 with the hash-locked dependencies, one
thread and deterministic algorithms. No GPU, download, teacher, solver fallback,
KV implementation or backend substitution is added. Model/gradient/RNG/mode
identities are checked before and after observations. There are exactly two
forwards per row (diagnostic plus authoritative target loss), at most 596 per
cell and 3576 for a completed six-cell run; this is not a latency benchmark.

Each fixed child worker has a 900-second whole-worker deadline and a 1 MiB log
cap. Six sequential workers are at most 5400 seconds, plus controller/import and
audit overhead, not a whole-command time guarantee. Each artifact is at most
16 MiB and retained writes at most 32 MiB per cell. Provenance duplication and
input imports are not a total-directory or hard RSS cap. Sampled process peak RSS
includes imports. No optimizer object is constructed or updated. Retained AdamW slot tensors are
loaded only for structural validation.

Progress is retained in chunks of at most 32 rows, separately by cell and stratum;
an interruption may lose the current incomplete chunk, never silently certify it.
All six outcomes (completed, failed, timed out, interrupted or not attempted) stay
in the matrix, and partial files/logs stay on disk. Outer controller/source/input
or filesystem failures may leave no final matrix; retain those partial files and
stderr too. Do not discard a negative result or reseal a failure as completion.

Audit/summarize are read-only source/input/custody/structure and retained-metric
consistency checks, **not fresh inference replay**. A retained probability is not
independently remeasured by audit. All learning adequacy, Phase 5, release and
fresh-confirmation flags remain false; no threshold automatically changes them.
Local environment evidence is private until reviewed; never publish an original
private bundle merely because its mathematical inputs are public.

## Later owner-authorized local workflow

Keep independently retained identities and logs outside the checkout. Use new
request/output paths. `models.json` is a canonical JSON object mapping exactly
`control-0`, `candidate-0`, `control-1`, `candidate-1`, `control-2`, `candidate-2`
to their independently retained `sha256:...` final-model identities. The producer
path is the original **attempt** directory containing `matrix.json` and six cells,
not the enclosing handover archive directory.

```sh
PYTHONPATH=src:scripts python -B scripts/run_frozen_token_diagnostic.py preflight
PYTHONPATH=src:scripts python -B scripts/run_frozen_token_diagnostic.py freeze \
  --producer /ABSOLUTE/ORIGINAL/attempt \
  --producer-request-identity sha256:ORIGINAL_REQUEST \
  --producer-matrix-identity sha256:ORIGINAL_MATRIX \
  --model-identities /ABSOLUTE/INDEPENDENT/models.json \
  --request /ABSOLUTE/NEW/request.json
# Retain the printed new request identity independently, before inference.
PYTHONPATH=src:scripts python -B scripts/run_frozen_token_diagnostic.py run \
  --producer /ABSOLUTE/ORIGINAL/attempt \
  --request /ABSOLUTE/NEW/request.json --request-identity sha256:NEW_REQUEST \
  --output /ABSOLUTE/NEW/diagnostic-attempt --authorize-run
# Independently retain the printed diagnostic matrix identity, including failures.
PYTHONPATH=src:scripts python -B scripts/run_frozen_token_diagnostic.py audit \
  --producer /ABSOLUTE/ORIGINAL/attempt \
  --request /ABSOLUTE/NEW/request.json --request-identity sha256:NEW_REQUEST \
  --output /ABSOLUTE/NEW/diagnostic-attempt --expected-identity sha256:DIAGNOSTIC_MATRIX
# summarize accepts the same arguments and returns the same verified per-cell summaries.
```

Do not invoke `_worker` directly, supply a replacement expected identity after a
mismatch, or combine original scores from different producers. Freeze itself
imports model tensors for safe structure validation but performs no inference.
Preflight is data-only; neither command authorizes the subsequent target run.

## Compatibility and engineering controls

Affected invariants: TRI-I01–I03, I05–I08, I10–I14. This new checkout-only lane does
not change the installed API, native arithmetic, scorer, admission, safe tensor
reader, historical source closure or learning gates. Toy conformance models and
short engineering schedules are explicitly not target six-model results.

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/frozen_token_diagnostic_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/addition_development_v1 -v
PYTHONPATH=src python scripts/check_research_cpu.py
```

The new suite covers all-position shifts and tie rules, immutability, true custody
and model-only import tripwires, external identity/context rejection, chunk and
summary consistency, filesystem/failure boundaries and explicit authorization.
Historical CPU aggregate coverage does not subsume this checkout-only suite.
