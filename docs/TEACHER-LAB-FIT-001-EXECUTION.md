# TT-003-FIT-001 execution implementation

PROPOSED implementation of the unchanged [FIT design](TEACHER-LAB-FIT-001.md).
No target run or result is supplied by this document. The design JSON and
selector remain unchanged. Historical TT-003 producers and evidence are not
modified or resealed. Governance, orchestration, native execution and evidence
verification remain distinct; this is procedural enforcement, not IBAE native
seals, owner authentication or machine attestation.

## Exact source and admission

Use `scripts/run_teacher_fit.py` from a clean checkout of the exact reviewed
implementation commit. The source closure inventories all tracked source,
scripts, tests, documentation, configuration, workflows and root files, including
all applicable AGENTS files. Each working file is compared to its committed Git
blob as well as SHA-256, including when Git's assume-unchanged bit is set.
Untracked imports/decision files, symlinks and dirty source fail closed. The
checkout, Python startup and local filesystem must already be trusted; this
check is not an adversarial sandbox.

Prepare a fresh private TT-002 packet with the existing `teacher_dataset.py
prepare` and retain its printed manifest anchor. FIT freeze uses the original
full TT-002 cold verifier. Worker ingress subsequently reads only the anchored
manifest, exact rights record and TRAIN bytes; it independently checks all 972
TRAIN row identities, prompts, loss masks and algebra/ket oracles before applying
the unchanged selector. No validation/test labels enter model construction,
updates, diagnostics or selection. Full dataset admission is a separate
preflight process, not a claim that the dataset verifier never sees those files.

The native `RunConfig.order` marker remains its required historical enum.
The actual order is the explicitly frozen seed-independent FIT visit ledger;
`run_steps` consumes that ordered list without reordering. All optimizer defaults
and the entire native RunConfig serialization are bound and checked. The model,
single-tensor AdamW, target loss, gradient clipping, evaluation and closed
safetensors reader/writer are reused unchanged. Geometry loss remains zero.

## Review, freeze and owner approval

First run the normal and optimized FIT contracts, existing TT-002/TT-003
contracts and the synthetic native canary. Publish the implementation PR and
obtain review and passing relevant CI on its exact head. No automatic merge is
required or permitted. The externally retained canonical review record has:

- `schema`: `trinite.teacher-fit-001.v1.review`
- `commit`: the full exact reviewed SHA
- `reviewed`: true; `review_url`: the actual GitHub review evidence URL
- `checks`: records with `name`, `head_sha`, `conclusion` and `url`, all exact-head
  successes; include labels `FIT contracts`, `TT-002 contracts`, `TT-003 contracts`
  mapped to their actual workflow URLs.

This record transcribes observed external review/CI, not model-invented review.
The validator checks binding and completeness, not GitHub authentication. Missing
review or failed/pending checks blocks freeze. It is not a training approval.

Use the existing pinned Linux x86_64 CPU FP32 interpreter, without dependency
substitution. `environment --out ABSOLUTE_NEW_FILE` records native environment,
interpreter, stdlib and virtualenv implementation file hashes; retain its printed
identity independently. This preflight imports the native runtime to inspect its
configuration, but allocates no model and performs no updates. The target worker
imports model code only after separate source/request/owner-approval validation.

Create one fresh private experiment parent outside the checkout. Then:

```sh
PINNED_PYTHON scripts/run_teacher_fit.py freeze \
  --dataset DATASET --dataset-identity sha256:DATASET_ANCHOR \
  --experiment ABSOLUTE_EXPERIMENT_DIRECTORY \
  --review REVIEW_JSON --review-identity sha256:REVIEW_ANCHOR \
  --environment ENVIRONMENT_JSON --environment-identity sha256:ENVIRONMENT_ANCHOR
```

Freeze creates three read-only `requests/seed-{17,29,43}.json` files and a separate
`request-anchors.json`. Copy the three request anchors to an independently held
location. The request binds source, actual environment, model/optimizer,
rights, every panel row and all 1,024 visits, baselines, evidence and each fixed
absolute output path. POSIX read-only mode is an accident guard, not cryptographic
immutability or owner authentication.

**Stop here for three separate owner-local approvals.** There is no command
that writes an approval. The owner supplies canonical JSON (sorted keys, compact
separators, LF termination), with separate retained SHA-256, matching the exact
validation shape in `spec.approval_expected`: schema
`trinite.teacher-fit-001.v1.owner-approval`, owner `QSOLKCB`, scope
`one-owner-local-native-cpu-fit-attempt`, exact request/source/seed/dataset/schedule/
environment/output identities, the exact `limits` object and
`training_authorized=true`. The agent must not manufacture these records.
These are explicit local declarations, not cryptographic signatures.
After receiving them, retain a separate canonical approval-set JSON keyed by
`17`, `29`, `43`, each containing `path`, `identity` and `request_identity` for
that owner's actual record. Retain its SHA-256 separately. This is an inventory
of existing approvals, not a fourth approval or an approval-writing mechanism.

## Single-use bounded execution

Only after all three matching approvals have been supplied, dispatch in order:

```sh
PINNED_PYTHON scripts/run_teacher_fit.py run \
  --request REQUEST_JSON --identity sha256:REQUEST_ANCHOR --dataset DATASET \
  --approval OWNER_APPROVAL_JSON --approval-identity sha256:APPROVAL_ANCHOR \
  --approval-set THREE_APPROVALS_JSON --approval-set-identity sha256:SET_ANCHOR
```

Each frozen absolute `seed-N` output directory is atomically created once. Its
continued existence consumes that request's attempt even if startup fails.
Never delete it to retry. The next seed requires a sealed, resolved preceding
process outcome. CI target dispatch is prohibited. No GPU path exists.

The independent supervisor begins its 900-second wall deadline before admission,
launches one process group, and kills/reaps it on timeout or interruption. It
records the original child return code separately from its own exit status;
Linux process-group inspection also rejects a reaped leader with surviving
descendants. An unresolved termination blocks the next run. There is no automatic retry.
An 8-GiB address-space ceiling, 16-MiB per-file ceiling and 128-MiB retained run
inventory ceiling are additional declared operational bounds. These are not
performance claims or RSS estimates. Logs and failed/partial directories remain.

Inside the worker, one cumulative budget sums elapsed native updater invocations;
each invocation performs exactly the next single native update. Before/after
checks enforce 600 seconds across all 256 calls, without resetting the allowance.
A single update is cooperative and may finish after the remaining native time;
that outcome is recorded as failed, never promoted to completion. Checkpoint
observation time is outside this cumulative native counter but inside the
independent 900-second whole-process deadline. Native evaluation inside
`run_steps` remains part of native elapsed time.

Each update is journaled; fixed checkpoints 0/64/128/192/256 preserve complete
model and AdamW tensors, progress, actual identities, resource use and diagnostics.
Both gold and raw greedy prefixes retain all 259 FP32 logits in little-endian
binary, original prefixes, predictions and binary64 hexadecimal metrics. Positions
are derived from this experiment: 252 gold positions per checkpoint, plus actual
1–5 greedy positions for each of 72 rows. No historical 13,689-position inventory
is reused. Diagnostic streams are flushed as observations arrive so partial
output remains inspectable. Checkpoint completion requires unchanged model,
optimizer, progress, model mode, Python RNG and Torch RNG. No sampling or training
RNG is used by the observer.

## Verification and private handoff

`verify --run RUN --manifest-identity sha256:RUN_ANCHOR --request-identity
sha256:REQUEST_ANCHOR --dataset DATASET --out NEW_RECEIPT` is separate recorded
verification. It recomputes position inventories, greedy outputs, every saved
probability, rank, margin, NLL and aggregate, update/target counts and all file
identities. Native training losses are observations retained in the update
journal; checkpoint loss is cross-checked against its native evaluation record.
This does not claim recomputation of historical optimizer updates from logits.
Raw tensor custody is separate from semantic restoration.
For interrupted unsealed diagnostics, every complete retained metric line is
recomputed against its prefix and FP32 vector. Torn JSON tails and unmatched
partial vector bytes remain retained without invented numerical claims.

`replay` accepts the same arguments and performs closed native model/AdamW
restoration plus exact-byte forward replay at every retained complete checkpoint,
including native evaluation losses. A 900-second cooperative replay deadline
bounds this read-only process; the owner may apply a separate OS supervisor.
No optimizer updates occur. Replay receipts must state their scope, and any
failed comparison must be retained. A successful restore/forward check is not
complete optimizer replay. This workflow reports `optimizer_replayed=false`.
Same-host replay is not independent-host reproducibility.

After all three attempts, `synthesize --experiment EXPERIMENT --dataset DATASET
--request-anchors ANCHORS_JSON --out NEW_PRIVATE_DIRECTORY` requires the full
three-attempt inventory and re-verifies recorded evidence. It produces
`TT-003-FIT-001-TRIPLE-SYNTHESIS.md`, complete numerical synthesis JSON, all-attempt
combined ZIP, external archive member inventory, SHA-256 sidecar and read-back
archive verification receipt. Original attempts, data, approvals and complete
source closure remain private. Optional separately produced per-seed native
replay receipts are retained alongside the recorded verification.

A failed/incomplete attempt cannot become a completed run. A retained 72/72
endpoint observation is still reported if a later bookkeeping failure occurs;
the observation never erases the failed process status. No seed is omitted or
chosen as preferred, and unavailable checkpoints remain unavailable. Unknown
consumption after a hard kill is not invented from the last completed checkpoint.
The complete attempt journal and original failure/status records are authoritative
observations with explicitly limited completeness. All scores describe exposed
TRAIN fit; no held-out generalization, causal identification, distillation,
TT-004, release, publication or model scaling follows.

Affected invariants: TRI-I01/I02/I03/I05/I06/I07/I08/I10/I11/I12/I13/I14.
The one-step synthetic canary is engineering conformance, not target-run evidence.
Correctness outranks speed. Preserve what happened.
