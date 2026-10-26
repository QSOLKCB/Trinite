# Frozen-checkpoint unseen evaluation v1

Status: additive, checkout-only engineering preparation for a **later local
inference run**. This implementation does not evaluate the owner's trained
checkpoint, retrain it, or report a new generalization result.

## Question and exposure boundary

Given the preselected final step-4,096 dense arithmetic checkpoint from
[native diagnostic v1](NATIVE-DIAGNOSTIC.md), how does its prompt-only output
compare with its teacher-forced predictions on rows absent from that checkpoint's
training schedule?

The fixed evaluation set is the existing admitted arithmetic corpus's **64
validation rows and 64 test rows**, reported separately. Their families are
disjoint from the 264 training rows and from each other. Each held-out split
contains **one semantic family**, compared with eight training families; its
64 correlated task/sign/carrier rows are not 64 independent trials. No iid
confidence interval is reported. All rows are retained;
there is no easy-item selection, label-dependent filtering, replacement or retry.
The checkpoint's inherited validation-named training diagnostics used training
rows, not these held-out rows.

These are **historical, publicly exposed research splits**, used in earlier
Trinite studies. “Unseen” means absent from this checkpoint's declared native
training schedule, not globally unknown or untouched by adaptive research.
This is exploratory diagnosis, **not fresh confirmatory evaluation**. Reusing
these outcomes to change a recipe, then reporting them as fresh confirmation,
is not permitted. New family-level confirmation needs its own later protocol
and exposure review; merely changing row IDs or resplitting public examples is
not enough.

The final checkpoint is selected before outcomes, not by comparing evaluation
scores across milestones or seeds. This workflow introduces no pass threshold,
checkpoint election, training updates, optimizer construction, geometry loss,
ternary comparison, teacher call, model download or new KV-cache backend.
`learning_adequate`, `phase5_ready` and `release_ready` remain false regardless of
observed accuracy.

## Compatibility and identities

`scripts/run_unseen_evaluation.py` and `scripts/unseen_evaluation_v1/` compose
existing native admission, tokenizer, loss, prompt-only greedy generation,
safe tensor reading and custody operations. The original numerical source,
`native_diagnostic_v1` producer, `AGENTS.md`, dependency lock, historical
protocols and evidence are unchanged. This is not an installed inference API.

The original producer request, externally retained completion identity and
externally retained final model identity bind the input packet. Its final
checkpoint metadata and safetensors are checked against those commitments,
the original source/environment context and scheduled training history. Only
model parameters enter the inference decoder; retained optimizer tensors are
structure-checked, not restored into an optimizer or advanced.

A separate evaluation request binds the evaluator source closure, protocol,
input packet identities, checkpoint identity, complete evaluation rows and
numerical environment **before** any target inference. Both source identities
remain explicit: new evaluator identity does not replace the old producer
identity. Nothing rewrites the original packet or substitutes historical hashes
into a changed checkout.

Run in the original compatible, hash-locked CPU environment. Foreign source or
environment mismatches are intentional failures: retain them, do not patch the
fingerprint, impersonate the producer, or fall back to another numerical backend.
A changed evaluator requires a new request. Requests and result destinations
are exclusive and may not overwrite prior attempts. Input/audit traversal is
bounded to 1,024 members, 16 MiB per file and 128 MiB of retained file sizes
(including custody copies). Root/member identities are checked before and after
verification. Keep local packets quiescent: these checks provide hash-bound
read-only integrity, not adversarial race-free filesystem containment.

Affected requirements are TRI-I01–I03, I05–I08 and I10–I14: inspectable admitted
inputs, CPU isolation, source/evidence binding, scoped reproducibility, honest
claims, family separation, explicit boundaries and retained failures. Dense
model numerics, quantization and geometry semantics remain unchanged.

## Measurements and controls

- **Raw generation:** prompt only, deterministic native greedy decoding, context
  64 and at most five output tokens. Retain token IDs, answer correctness,
  complete answer-plus-EOS correctness, malformed output, unfinished output,
  premature EOS, stop reason and output lengths. `raw_malformed` concerns byte
  encoding/termination only, not validity of arithmetic-answer syntax.
  No answer-conditioned prefix,
  constrained repair, oracle fallback, retries or best-of selection is allowed.
- **Teacher forcing:** gold-prefix token observations, including loss, rank,
  probability, first-answer and EOS behavior. These are not free-generation
  accuracy. Token-weighted loss uses the authoritative native loss operation.
- **Non-neural controls:** majority and lookup derive only from training rows;
  an absent training lookup remains a miss, never an answer supplied from the
  evaluation set. Constant output and an independently checked exact formal
  oracle are separately labelled; neither is a model prediction or rescue.
  The ex-post best-constant ceiling is descriptive and explicitly gold-informed,
  not a training-derived comparator. Beating these weak controls is no adequacy gate.
- **Reporting:** retain every row and scored target, split-level, task-level and
  family-level counts, different-answer input-sensitivity comparisons and
  same-semantics carrier comparisons. Related carriers/pairs are correlated,
  not independent trials. Different outputs alone do not prove reasoning.
- **Resources:** wall time runs from after initial request validation through
  measurements. RSS is process high-water at that sample, including imports and
  completed validation; later finalization is excluded. It is not an isolated
  model/scratch measurement or the final whole-worker peak. Native reusable KV cache is unsupported, not a measured zero-byte
  cache. Equal output-token caps do not establish equal compute.

## Commands and authorization

Use a clean checkout of the reviewed implementation and the original compatible
CPU environment described in [GETTING_STARTED.md](GETTING_STARTED.md). Run from
the repository root. `--producer` names the original completed diagnostic
attempt directory containing `request.json`, `completion.json`, checkpoint
artifacts and its custody packet, **not** an archive or source checkout.
Keep it outside the new output directory and do not publish private host paths
or the full returned packet as a repository fixture.

```sh
export PYTHONPATH=src
mkdir -p reports/unseen-evaluation-v1
python scripts/run_unseen_evaluation.py freeze \
  --producer /path/to/original/diagnostic-attempt \
  --producer-request-identity 'sha256:REPLACE_WITH_ORIGINAL_REQUEST_IDENTITY' \
  --producer-completion-identity 'sha256:REPLACE_WITH_ORIGINAL_COMPLETION_IDENTITY' \
  --model-identity 'sha256:REPLACE_WITH_FINAL_MODEL_IDENTITY' \
  --request reports/unseen-evaluation-v1/request.json
```

Retain the printed evaluation-request identity independently. The following is
an **execution recipe, not authorization to execute the target**. Publish and
review the protocol first; obtain the separately scoped local-run approval.
`--authorize-evaluation` acknowledges that approval and does not grant it.
Do not invoke the internal worker directly.

```sh
python scripts/run_unseen_evaluation.py evaluate \
  --producer /path/to/original/diagnostic-attempt \
  --request reports/unseen-evaluation-v1/request.json \
  --request-identity 'sha256:REPLACE_WITH_EVALUATION_REQUEST_IDENTITY' \
  --output reports/unseen-evaluation-v1/attempt \
  --authorize-evaluation
```

Keep the entire output directory and printed completion identity independently.
Each completed row and its teacher-forced targets is retained immediately; a
mid-run failure preserves this prefix without claiming a complete result.
Preserve incomplete, failed, interrupted and timed-out attempts; do not discard
negative results or silently retry. The fixed worker has a 900-second outer
wall deadline; controller setup and audit are outside that subprocess deadline.
This is not a hard CPU-time or process-memory quota. Artifact and log byte bounds
are separately enforced; custody duplication is not a total-directory quota.

After a completed attempt, both commands below are read-only:

```sh
python scripts/run_unseen_evaluation.py audit \
  --producer /path/to/original/diagnostic-attempt \
  --request reports/unseen-evaluation-v1/request.json \
  --request-identity 'sha256:REPLACE_WITH_EVALUATION_REQUEST_IDENTITY' \
  --output reports/unseen-evaluation-v1/attempt \
  --expected-identity 'sha256:REPLACE_WITH_EVALUATION_COMPLETION_IDENTITY'

python scripts/run_unseen_evaluation.py summarize \
  --producer /path/to/original/diagnostic-attempt \
  --request reports/unseen-evaluation-v1/request.json \
  --request-identity 'sha256:REPLACE_WITH_EVALUATION_REQUEST_IDENTITY' \
  --output reports/unseen-evaluation-v1/attempt \
  --expected-identity 'sha256:REPLACE_WITH_EVALUATION_COMPLETION_IDENTITY'
```

`audit` validates external commitments, original producer and new evaluator
context, admitted rows, safe checkpoint structure, custody and recomputable
report/control consistency. `summarize` audits before displaying results.
Neither command regenerates target predictions nor replays optimizer updates.
Retained probabilities/ranks/losses are producer observations: structural
validation is not independently replayed inference, a proof of numerical
correctness, hardware attestation or independent-host replication.

## Engineering verification and remaining work

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/unseen_evaluation_v1 -v
```

The dedicated CI runs engineering controls on Python 3.11, 3.12 and 3.13 with
the hash-locked CPU dependencies. Synthetic/tiny test evidence is not the real
checkpoint's held-out score. Run the existing mandatory suites in
[AGENTS.md](../AGENTS.md) and [INSTRUCTIONS.md](INSTRUCTIONS.md) as well; report
actual pass/fail/unrun status rather than treating a focused suite as a full
project pass.

The authorized local target inference, its retained results, independent
numerical replay, genuinely fresh confirmation and any subsequent intervention
remain separate work. No outcome here automatically advances Phase 5, format
selection, capacity, scaling or release readiness.
