# Non-displacing auxiliary engineering implementation

Status: additive implementation for review. The [original proposal](PROCEDURAL-QEC-NONDISPLACING-PROPOSAL.md), its planning receipt, historical protocols, negative studies, native updater and existing learning thresholds remain unchanged. **No scientific protocol is frozen and no target research cell is authorized or executed by this increment.**

## Implemented boundary

`procedural_qec_auxiliary/inputs.py` executes the independently byte-pinned historical audit against the actual clean producer `93522bf8f18fe38970ceafdd2ace792f8922c382`. It regenerates and admits the historical dataset with the original code. It separately generates and admits current data in an isolated source-only snapshot. Complete row schemas and every field except top-level `generator_revision` and `example_identity` must match bijectively. Both identities, original source/data commitments, all mappings and historical primary order are retained. A caller-provided pass flag or two functions over current rows cannot substitute for this boundary.

`scheduler.py` reconstructs each original eight-row final-only batch, then inserts a distinct eight-row update. Every rank uses the original historical identity. The procedural cadence is fold prefix, GF(3) prefix, fold trace, GF(3) trace with independent cursors advancing by eight only when selected. The extra-final control has one independent final cursor per kind. Removing the additional updates recovers the original primary batches in exact slot order. Full 4,096-round enumeration is data-only and performs no model calls.

`spec.py` contains a reviewable candidate description and a separate fixed engineering profile: eight paired rounds, sixteen native updates, batch eight, context 64, one block, width 16, two heads and feed-forward width 32, seeds zero and one. Research request creation and state construction fail closed. There is no authorization flag, arbitrary profile file or command-line override that enables the proposed research matrix.

`experiment.py` composes the unchanged native Decoder, `target_loss`, AdamW, `run_steps`, authoritative safe tensor reader, original prompt-only scorer and closed PROVENANCE verifier. Every encoded row is padded to 64, giving `[8, 63]` shifted native model inputs. Target-normalized primary and extra losses are separate optimizer updates. No gradient accumulation, new coefficient, nine-row batch, numerical kernel or hidden optimizer is introduced.

New source-bound engineering request, checkpoint, report, verification-request and summary schemas bind the full current source/launcher closure, candidate description, actual CPU environment, original/current input identities, mapping receipt and complete schedule. Checkpoints retain model and AdamW tensors, native-step progress, target counters, loss/gradient histories and unique training-only diagnostics. Mid-pair and completed-pair restarts use the same native step counter; no primary update is silently repeated.

The four-cell runner retains failures and every cell. Verification freshly checks original admitted inputs, external commitments, actual closed custody, safe tensor/progress structure, scheduled visits/targets, final predictions/loss and complete seed-started model/AdamW/history replay. The archive adds another full native replay before preserving all run bytes, requests, mapping and exact source custody. Read-only transport verification is explicitly separate from numerical replay.

The dormant candidate uses the same production composition with distinct research schemas. Its data-only plan constructor validates 8,192 native updates and the proposed model bounds without constructing a model. Its shared diagnostic selector enumerates every unique admitted training row: 1,468 final examples and 1,204 each of prefix and trace. It excludes held-out rows and rejects repeated or missing diagnostic denominators. Candidate metadata has an explicit 8 MiB serialization bound for all 8,192 eight-row histories; the engineering bound stays 1 MiB. This byte ceiling is not a storage allocation. Model-facing request, state, report, verification and summary entry points still reject research before model work or artifact creation.

Pure report and aggregate helpers implement the unchanged six absolute task thresholds and report treatment-minus-control differences separately. Both treatment seeds must meet every absolute threshold; positive fold contrast requires a strictly positive difference in both seeds. Neither condition silently substitutes for the other. Failed cells, missing replay checks and mismatched paired primary sequences cannot produce an evaluated aggregate gate. No checkpoint or arm is selected, and no training result establishes held-out adequacy.

The dormant research transport checker validates the full unique diagnostic inventory, native progress/history, gate/contrast arithmetic and required replay receipts under a distinct archive schema. Synthetic packets exercise these branches only as structural tests. Public archive construction, verification and restoration remain engineering-only; no synthetic packet is represented as numerical replay or experimental evidence.

## Engineering checks and interpretation

The dedicated workflow runs real historical reconstruction, malformed mapping/order/cursor tests, exact planning counts, target/mask checks, native padded forward/loss/gradient conformance, odd/even native-step interruption and replay, checkpoint/source tampering, four native engineering cells, separate read-only replay and archive-time replay. The native padding comparison uses `rtol=1e-5, atol=1e-6`; this is an engineering tolerance, not a historical bitwise-trajectory claim. Same-environment complete checkpoint replay requires exact bytes.

All model scoring in the engineering profile uses a small unique subset of admitted training rows. Scheduled repetition never enlarges diagnostic denominators. Empty auxiliary groups have no accuracy estimate. Engineering reports always set `training_gate_evaluated`, learning adequacy, Phase 5, format and release eligibility false. Their tiny-model values must never be interpreted as outcomes of the proposed full experiment.

The full proposed recipe has 32,768 training updates and 849,862 targets over four cells; preserving all three complete replay passes entails 131,072 native updates before diagnostics. Data-only enumeration checks these counts, historical primary identities and the unchanged plan bounds. It supplies no wall-time, RSS, storage or learning estimate.

## Local conformance commands

Use a clean separate checkout of the exact original historical producer. It must have no generated bytecode or untracked source files. Install the existing hash-locked CPU requirements into a separate environment following [getting started](GETTING_STARTED.md).

```sh
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRINITE_HISTORICAL_SOURCE=/absolute/path/to/clean-93522bf-checkout
python -m unittest discover -s tests -p test_auxiliary_spec.py -v
python -m unittest discover -s tests -p test_procedural_qec_auxiliary.py -v
python -m unittest discover -s tests/procedural_qec_auxiliary_data -v
python -m unittest discover -s tests -p test_procedural_auxiliary_archive.py -v
python -m unittest discover -s tests/procedural_qec_auxiliary -v
```

`run_procedural_qec_auxiliary.py --prepare-request PATH --run-profile engineering --historical-source PATH` creates only an engineering request. The printed independent request identity is required for `--run`; the independently retained verification-request identity is additionally required for `--verify-only`. Every output directory/request is new and exclusive. The command rejects `--run-profile research` before importing the numerical backend.

These source-bound production, historical admission and numerical runner operations are checkout-only, as they require the fixed launcher/evidence and a separate historical Git source. Package installation is not represented as a standalone research runtime. The immutable historical protocols already shipped in the wheel retain their existing isolated-wheel tests.

## Remaining scientific and operational gate

Review and approve the hypothesis, extra-final control, full-row historical mapping, fixed cadence/padding, native normalization, final-checkpoint rule and positive-fold-in-both-seeds contrast. Then publish a separately identified scientific protocol before any target outcomes. Green engineering CI and a code review do not freeze that protocol or authorize research execution.

The smallest later execution request must name the exact reviewed source/protocol and approved CPU host/environment; authorize the two arms × two seeds × 4,096 paired rounds, all three full replay passes and diagnostics; and provide total wall-time, memory/storage and retained-artifact allowances plus failure/timeout policy. The existing 960-second per-invocation ceiling is only a software bound. It is neither a total allocation nor evidence that a full cell and its diagnostics fit. Exceeding a bound requires a separate reviewed change, never a silent increase or repeated restart until success.

The implemented engineering orchestration, checkpoint/replay and archive path can be exercised safely now. Dormant full-profile planning, unique diagnostics, candidate report/contrast assembly and archive validation are covered by data-only and synthetic tests; they require no target research run. Activating a target experiment requires a separate reviewed source change replacing the disabled candidate boundary with the approved immutable scientific protocol and resource envelope. No merge, tag, release or remote research dispatch follows from this PR.

Affected requirements: TRI-I01/I02/I03/I05/I06/I07/I08/I11/I12/I13/I14. TRI-I09/I10 remain claim boundaries. Original folded/composed learning adequacy, fresh family confirmation, independent-host replication, format selection, Phase 5, scaling and release remain blocked.
