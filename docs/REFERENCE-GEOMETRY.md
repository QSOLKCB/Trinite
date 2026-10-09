# External model internal-geometry comparison

This separate descriptive study runs the retained arithmetic seed-0 dense
Trinite checkpoint beside Qwen3-0.6B stock and with a frozen SYSTEM-only
Modelfile-derived profile. It implements internal activations, rather than
output embeddings. The [protocol](REFERENCE-GEOMETRY-PROTOCOL.md) was committed
before outcomes at `5ea3c7dea35f98ef276113c7c57dd26c9b31429c`.

The external model is pinned to revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Its marketed name is 0.6B; actual
parameter counts are measured from loaded tensors. Safetensors, configuration,
tokenizer, model card and Apache-2.0 license bytes are checked against the
official revision. The permissive license is not evidence about training-data
exposure, which remains unknown. All external assets, including the 1.5 GB
weight file, remain in the explicitly acquired local directory. Bundles retain
their size and digest manifest, not their complete bytes. No remote Python is
loaded. The original native checkpoint reader and producer source closure are
preserved; a different producer cannot silently load the checkpoint.

## Reproduce

Use a source checkout. Install the original lock and separate backend extension:

```bash
python -m pip install --require-hashes --no-deps -r requirements.lock -r requirements.reference.lock
python -m pip check
python scripts/acquire_reference_model.py /path/to/model
```

The acquisition command is explicitly networked. Everything below is offline.
Restore the existing [scalar archive](SCALAR-LEARNING.md) and verify its transport
receipt first. Its `producer` and `run` directories contain the exact numerical
source and checkpoint used by this study. Then freeze inputs and execute:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
python scripts/run_reference_geometry.py /path/to/study --freeze \
  --model /path/to/model --producer /path/to/scalar/producer \
  --native-run /path/to/scalar/run
python scripts/run_reference_geometry.py /path/to/study
python scripts/run_reference_geometry.py /path/to/study --verify-only
python scripts/run_reference_geometry.py /path/to/study --replay
```

Freeze creates a canonical request and the exact profile:

```text
FROM ./model
SYSTEM Return only the final exact answer. Do not include explanations or markup.
```

`./model` denotes the explicitly bound local asset directory in the request's
transport locations. This narrowly supported profile is interpreted by the
instrumented Transformers adapter. It is not an actual Ollama execution and
cannot interpret arbitrary Modelfile directives. Stock and profile use the same
weights, original chat template, CPU float32 eager attention and greedy
32-token cap. Both use `enable_thinking=False`. Native byte tokens and Qwen BPE
tokens have different costs; no matched-compute claim is made.

The sixteen probes are visible native training examples in two paired families,
four operations and two carriers. Retained admission/source IDs and prompts
make exposure inspectable. Labels are used only for descriptive exact-string
agreement. Prefix states come from the last user-content token before an answer
is supplied. Full rendered inputs, token IDs, offset mappings and selected
positions are retained; prefix tokenization is repeated at each endpoint.
First-block and final-normalized layers are preselected. Captures and generation
are separate passes; parameters, buffers and caller RNG must remain unchanged.

## Evidence and limitations

Each condition has its own closed PROVENANCE bundle, capture and fresh-worker
replay receipt. A fourth bundle binds all condition manifests and the derived
analysis. Read-only verification checks retained receipt bytes, membership,
float32 reconstruction, input alignment and recomputed results. `--replay`
additionally executes the three fresh workers against the locally available
input assets; the retained receipt alone is not a new replay. The observer runs
after computational workers exit and does not execute model code.

The independent Fraction oracle centers exact float32 coordinate values before
constructing Grams. It checks CKA squared and every squared-distance matrix
entry against the primary float64 kernel. Both share parsing and identity
infrastructure; this is not independent-host replication or mathematical proof
of a model mechanism. Degenerate geometry is unavailable, not fabricated zero.
Widths can differ: CKA compares sample geometry, never arbitrary cross-model
coordinate cosine. Per-model distance matrices, paired-carrier distances and
prefix path lengths preserve each model's own space.

Two families are a small correlated diagnostic set. There are no confidence
intervals, significance tests, causal conclusions, model rankings or claims of
generalization. Changes in stock versus profile can reflect the extra context
and changed positions. They do not isolate a universal system-prompt mechanism.
The blocked scalar fold-sum gate, held-out scoring and Phase 5 stay unchanged.

## Conformance

Root standard-library tests check the independent oracle, width mismatch,
invariances, degenerate cases, byte tampering, retained receipt deletion and
closed Modelfile rejection. `tests/reference` requires the pinned external
backend and compares selected native/Qwen states to direct forwards, checking
immutability and chat alignment. Tiny random Qwen models in CI are simulations;
routine CI does not download pretrained model weights or claim empirical parity.
The reference CI runs this suite on Python 3.11, 3.12 and 3.13 separately from the
existing CPU research suites.
