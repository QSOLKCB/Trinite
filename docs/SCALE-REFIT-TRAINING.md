# Fixed-code scale-refit training pilot v1

Status: implementation and prospective protocol; target outcomes not yet measured.
This is the separately requested integration of [fixed-code scale refitting](SCALE-REFIT.md)
into a bounded model/training lane. It is not the learned-scale parameterization
or four-update conformance fixture of [PR #48](LEARNED-TERNARY-SCALES.md).

## Compatibility and identities

The checkout-only `scripts/scale_refit_training_v1` package wraps the unchanged
native decoder's six attention/feed-forward matrices per block. Model ID:
`trinite.fixed-code-refit-decoder.pilot.v1`. Candidate quantizer ID remains
`trinite.ternary-fixed-code-scale-refit.v1`, because it invokes that exact existing
implementation rather than inventing another numerical kernel. The paired
`absmean` arm invokes `trinite.absmean-ste.v0`. The inherited ModelConfig records
historical architecture defaults; the wrapper's explicit model/arm/quantizer
inventory is authoritative for this lane. Native CLI/checkpoint loaders do not
accept this model ID. Historical native, refit, learned-scale, PCGrad, protocols,
source closures and fixtures are unchanged.

Every forward recomputes the authentic baseline codes at the current latent W,
then changes only the scalar when the existing exact integer SSE guard proves
strict improvement of actual float32 products. Codes and sparsity are preserved
at the same W; the two trained trajectories may diverge, including their codes.
Detached refitted scales introduce zero parameters. The surrogate mask remains
`abs(W / baseline_scale) < 1`, not the mask of the new scale. There is no scale
gradient, scale optimizer slot or learned-scale projection. Bounds are checked
before decoder allocation: no quantized matrix may exceed 65,536 entries.
All original CPU float32, finite-input, explicit forward and mask contracts apply.

The native `run_steps`, AdamW constructor/validator, admission, loss and greedy
generation functions are reused. No second updater, tokenizer, admission kernel,
tensor reader or numerical quantizer is introduced. Embeddings/output tie,
normalization and activations remain floating point. F.linear still uses
floating-point products; the integer certification pass adds overhead.

## Prospective matched pilot

[protocol.json](../scripts/scale_refit_training_v1/protocol.json) freezes:

- Arms: absmean and fixed-code refit, each seeds 0 and 1, no discarded cells.
- Same 6,752-parameter decoder, context 32, one block, width 16, two heads,
  FF width 32; same latent initialization and parameter inventory.
- Existing admitted formal-v1 fixture: 36 train examples and six validation
  examples from disjoint modulus families. Test examples never enter updates,
  quality scoring or selection. Admission necessarily audits original bytes,
  including their split integrity; that is not model evaluation of test labels.
- Native hashed row order and batch four, 60 AdamW updates, 480 answer/EOS target
  tokens per cell; LR .003, decay .01, clip one and native AdamW settings.
- Validation loss at steps 20/40/60, final step selected without tuning, early
  stopping, best-seed selection or changing the prescribed horizon.
- Final train/validation target cross entropy and prompt-only greedy exact
  complete answer+EOS counts, fixed two-token cap. All individual predictions
  are retained. No teacher-forced accuracy is substituted for generation.
- Same-current-weight baseline/refit reconstruction SSE and fixed-input F.linear
  output SSE are recorded separately per matrix; lower weight SSE does not
  promise lower output error, held-out loss or accuracy.

This six-example, one-validation-family visible fixture was used extensively
for earlier engineering. Results are an exploratory training smoke/pilot, not
fresh held-out generalization, robust uncertainty estimates or general reasoning.
Two seeds cannot establish a general performance advantage. The separately
requested pilot authorizes this small validation comparison only; historical
research, learning adequacy, geometry, scale, format-selection and release gates
remain unchanged. There is no external dataset, pretrained input, paid compute,
geometry objective or learned-scale/PCGrad intervention.

CPU is pinned Linux x86_64 Torch 2.8.0+cpu, one thread, strict determinism and
hash-locked dependencies. The updater checks a 120-second invocation budget
between steps; this is not a hard whole-process deadline or memory cap. Dedicated
CI imposes a 20-minute job deadline. Model/attention and artifact bounds remain
explicit. Each checkpoint is at most 2 MiB and the four-cell report at most 16 MiB;
partial completed-cell records can additionally occupy up to 16 MiB. Wall time
and whole-process peak RSS are logged separately, including replay overhead.
The matrix has 240 unique updates. Four prefix-resume checks per cell each
reconstruct/complete 60 updates, bringing one report to 1,200 executed updates;
full report verification repeats another 1,200. Equal unique tokens/updates are
not equal compute, and no speed claim is made.

## Checkpoints and evidence

`trinite.scale-refit-training-checkpoint.v1` is canonical bounded JSON containing
lossless float.hex model/AdamW values, exact config and model inventory, full
training and validation history, cursor/token count, initialization identity,
input identities, source closure and exact environment. Initialization uses the
native private seeded generator; updates and greedy generation make no RNG draws.
Caller global RNG state is neither consumed nor restored.

Resume first validates the externally supplied byte identity and bounded step,
reconstructs the complete prefix from its declared arm/seed, then compares every
checkpoint byte before returning usable state. It cannot load arbitrary tensors
or executable serialization. The report verifies resume from 0/1/30/60 to the
same final state, for every arm/seed. This is bounded full-prefix replay-to-resume,
not constant-time loading, native checkpoint migration or cross-host replay.

The run request freezes actual source/input/environment identities before target
outcomes. A report is created exclusively; verification recomputes all four cells,
resume checks and final quality and compares exact bytes without editing the
original. Completed cells are retained in an exclusive sidecar even if a later
cell fails. Failures/tracebacks and resource logs remain review evidence; an
incomplete run does not become a completed matrix. These bindings are software
replay, not an independent PROVENANCE custody attestation.

```sh
export PYTHONPATH=src:scripts
python -m unittest discover -s tests/scale_refit_training_v1 -v
python scripts/check_scale_refit_training_cpu.py
request_id=$(python scripts/run_scale_refit_training.py freeze --request /new/request.json)
python scripts/run_scale_refit_training.py run --request /new/request.json \
  --request-identity "$request_id" --output /new/report.json
# Keep the printed report identity independently; verification never overwrites.
python scripts/run_scale_refit_training.py verify --request /new/request.json \
  --request-identity "$request_id" --output /new/report.json \
  --expected-identity sha256:YOUR_RETAINED_DIGEST
```

All numerical evidence must distinguish source/environment replay from quality.
Negative results and ties are retained. The historical aggregate inventory is
preserved; the new aggregate appends only this suite. Required native, upstream,
source-custody and previous engineering CI remain enabled.

Affected requirements: TRI-I01/I02/I03/I04/I05/I06/I07/I08/I10/I11/I12/I13/I14.
TRI-I09 remains a closed geometry/causal-claim boundary. No merge, release, model
format selection or broader research activation is part of this increment.
