# Retention evidence assurance v2

Trent requested this increment after PR #20, using the supplied independent
review of `main` at `9dbc9cdc4ef08a00a9bd6ffeb82ed540ae25dcce`.
The delivered review ZIP has SHA-256
`95a608a508b5db98aa0cf02b8f3d50d37cf6132e3196f168d08e7720be9ad048`;
all 200 manifest members were checked. The review found no blocking schedule or
final-scoring defect. Its four recommendations concern assurance, integration
coverage, strict criteria and repeated evidence operations. This increment
implements those recommendations under TRI-I07, I08, I11, I12, I13 and I14.

## Compatibility and specification

The v1 [retention implementation and evidence](MATHS-RETENTION.md) remain intact.
`src/trinite/retention_v2/` is a separate package so new Python/JSON files do not
enter v1's flat-file source scan. The v1 source receipt retains all 81 original
names and bytes. Model, quantizer, tokenizer, corpus, families, native updater,
AdamW, safe tensor reader and original producer snapshots are unchanged.

The new machine protocol is `retention_v2/protocol.json`, identity
`sha256:d4160712380a354fd0ac0272a1b576bbad74696e74190b457b23b4b86e5760e6`.
Research retains the two seeds and pooled/rehearsal profiles, 4,096 updates,
eight-example batches, 86,528-parameter dense model, LR 0.001 and zero geometry
loss. State construction delegates the exact existing materialized schedules;
the updater's historical `validation` slot is adapted through an explicit unique
training diagnostic view. No reused test or validation rows enter that view.

The separately identified `smoke` profile uses a one-block width-16 model,
four updates, diagnostics every two updates and generation cap two. Its scoring
view takes the first semantic identity per task/split and both carriers from the
fully admitted corpus. It exercises real admission, observation, safe loading,
inference and replay. It establishes software conformance only; it does not
weaken the research plan or supply capability evidence.

Each independent report/verification invocation admits fresh corpus bytes and
creates immutable parsed examples for that invocation. Report assembly reuses
that object for scoring, task coverage, carrier/class metrics and exposure.
There is no cache of admission, source audit, custody or verifier success.

## Exact criteria

Training criteria remain 9/10 per task, using integer cross multiplication.
Expected task names and exact positive counts derive from the admitted scoring
view. Missing/extra tasks, boolean/zero/negative/out-of-range counts and wrong
aggregate totals fail before a criterion can pass. Thus 60/66 passes and 59/66
fails. The stricter helper reproduces all eight published corpus/cell results.
The supplied advisory helper informed this validation; no review/model prose
enters the training corpus.

## Verification levels and commitments

An experiment request (`trinite.maths-retention-request.v2`) binds the full
protocol, explicit research/smoke profile, runtime source closure, fixed launchers,
root agent instructions, exact environment and corpus/admission identities.
An independent verification request (`trinite.retention-verification-request.v2`)
binds that experiment identity, environment identity, every expected cell output,
custody receipt/manifest and the root experiment request/worker errors. Its
path and identity are supplied separately from the run. Production creates it
exclusively after the fixed workers finish and before verification. Readers reject
changed bytes against those externally held commitments, even if custody was
coherently reconstructed. Missing cells can be committed as incomplete, but
cannot receive a successful verification receipt.

| Receipt check | What true establishes |
| --- | --- |
| `closed_custody` | Fresh upstream closed-bundle verification, one bounded index, exact names/input identities and root copies |
| `request_and_source_binding` | Current input, protocol, source, profile, environment and checkpoint context match |
| `safe_tensor_structure` | Closed named model/AdamW inventory, shapes, CPU dtype, finite values and slot counters pass the unchanged reader |
| `scheduled_visits_and_targets` | Every example identity, step/cursor and target count matches the frozen schedule |
| `final_predictions_and_loss` | Fresh safe-state model scoring regenerates all predictions, scores, exposure, criteria and final loss |
| `complete_optimizer_replay` | Fresh native initialization/updating regenerates exact final model/AdamW bytes and all metadata/history/diagnostic/step bytes |

`trinite.retention-verification.v2` receipts list the exact compared artifacts,
experiment and verification request identities, requested tier, individual
booleans, passed/failed outcome, error, replay environment identity (null without
completed replay), and `learning_adequacy=false`. A completed summary validates
that each cell actually established the requested tier. The cell's historical
filename `verification.json` contains the unchanged **upstream custody receipt**;
the new scope-aware receipts live in the v2 summary and verifier output.

The `inference` tier leaves optimizer replay false. Finite, coherently resealed
historical losses/moments under newly declared external commitments may pass
this explicitly limited tier. Original commitments still reject any changed
artifact. The `replay` tier recomputes the full update trajectory from its native
seed and rejects those numerical changes. Custody integrity, final inference and
complete update replay are distinct assurances. None grants learning adequacy.

Foreign environment fingerprints are rejected, without rewriting an old request.
The supplied review reports a separate Ryzen/AVX2 replication and exact local
replay, with counts differing from the original EPYC/AVX512 producer. That is
review evidence, not a claim of newly reproduced foreign-host model bytes here.
Any new full run has its own request and environment identity.

## Fixed controller and commands

Use the CPU lock in [GETTING_STARTED](GETTING_STARTED.md). Research is the default
profile; `--run-profile smoke` is valid only when freezing a distinct request.
The CLI defaults to full replay; API callers must select their desired tier.

```sh
PYTHONPATH=src python scripts/run_maths_retention_v2.py \
  --freeze-request /tmp/retention-v2-request.json
PYTHONPATH=src python scripts/run_maths_retention_v2.py \
  --request /tmp/retention-v2-request.json --request-identity sha256:EXPERIMENT \
  --output /tmp/retention-v2-run \
  --verification-request /tmp/retention-v2-verification.json --tier replay
# Retain the emitted verification request identity independently.
PYTHONPATH=src python scripts/run_maths_retention_v2.py --verify-only --tier replay \
  --request /tmp/retention-v2-request.json --request-identity sha256:EXPERIMENT \
  --output /tmp/retention-v2-run \
  --verification-request /tmp/retention-v2-verification.json \
  --verification-request-identity sha256:VERIFICATION > /tmp/retention-v2-audit.json
```

Workers use a fixed script/interpreter, closed stages/cells, no shell and a
1,200-second deadline. Request files must be disjoint and outside the output;
overwrite and symlink ancestors fail. Every fixed cell is attempted after timeout
or spawn failure. Worker errors, logs, available computational artifacts, failed
comparison receipts and all four records are retained. If summary cannot complete,
a conservative controller failure inventory records four unverified cells.
Pairing failure suppresses contrasts while preserving the cells. Read-only
verification writes no run artifacts; failed receipts are emitted to the caller's
external output/log. Existing same-tier summaries must match fresh regeneration.

## Scientific scope and validation status

This is an assurance/maintainability increment over the unchanged joint-training
protocol. Production diagnostics remain reused, exploratory development data;
carriers/projections are correlated. Class-specific unique/singular solve counts,
majority baselines, per-task counts, retention and actual exposure remain visible.
There is no ratio sweep, selected winner, sequential-forgetting arm or new accuracy
hypothesis. A fresh family-level confirmation set and a separately frozen scientific
question must precede those later comparisons. Geometry, format, scaling, original
fold/composed learning adequacy and release stay blocked.

Integration tests execute real admitted smoke cells and upstream custody, root
copy and coherently resealed prediction rejection, the limited/full assurance
boundary, contextual mismatch, read-only CLI parity and observer filesystem
failure. Controller-only process fakes cover complete inventories, pairing,
spawn/timeout failures and fixed subprocess boundaries. The required CPU runner
includes `tests/retention_v2`; standard-library discovery checks criteria,
dependencies and unchanged v1 source closure. Completed checks and measured
assurance runs will be recorded after execution; no future success is assumed.
