# Getting started

Foundation commands require Python 3.11 or newer and no runtime/test packages. Phase 3 adds optional CPU model/training commands and standard-library evidence verification. Serving remains future work.

From the repository root, run without installing anything:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m trinite validate-config configs/reference.json
PYTHONPATH=src python3 -m trinite tokenize 'é=1' --answer-start 3 --pad-to 8
PYTHONPATH=src python3 -m trinite audit-fixture fixtures/formal-v1
```

answer-start is a UTF-8 byte offset, not a character offset. BOS/EOS are included in the context limit. Token loss masks mark prediction targets before the future trainer shifts them. PAD/BOS never contribute to loss; EOS does. Generated invalid UTF-8 byte sequences remain available through decode_bytes; text display has an explicit strict/replacement/escaping policy.

Create a fixture in a **new** output directory:

```bash
PYTHONPATH=src python3 -m trinite generate-fixture /tmp/trinite-formal-v1
PYTHONPATH=src python3 -m trinite audit-fixture /tmp/trinite-formal-v1
cmp fixtures/formal-v1/dataset.json /tmp/trinite-formal-v1/dataset.json
cmp fixtures/formal-v1/manifest.json /tmp/trinite-formal-v1/manifest.json
```

The output directory must not already exist; generation never overwrites existing files. IO failures can leave an incomplete directory, which the audit rejects. Reads are bounded to 256 KiB for the dataset and 16 KiB for the manifest. Audit stable directories: this Phase 1 reader does not provide a hostile-concurrent-filesystem snapshot guarantee. Unknown fields, changed schema/source receipts, invalid examples, modified splits/masks, extra/missing files, symlinks, and noncanonical artifact JSON are rejected.

The default seed is 0. --seed changes family assignment deterministically and produces new content identities. Tests and CI compare the seed-0 artifacts byte for byte instead of replacing their expectations.

For optional package installation, use a virtual environment and the pinned build backend:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'setuptools==84.0.0'
python -m pip install --no-build-isolation --no-deps -e .
trinite validate-config configs/reference.json
python -m unittest discover -s tests -v
```

Backend acquisition needs network access or a pre-provisioned wheel. The runtime and source-tree workflow remain offline. The foundation itself has no runtime/test dependencies; requirements.lock is the optional CPU model/training acquisition lane. The package version 0.0.0 denotes unreleased foundation code; this PR creates no tag or release.

CPU CI runs the standard-library suite, config validation, frozen fixture audit, and exact replay under Python 3.11/3.12/3.13. Actions are pinned to full commit identities; hosted jobs have read-only repository access and a five-minute limit. The separate Phase 2 model jobs acquire hash-locked CPU packages before their offline conformance steps and have a ten-minute limit. All jobs use hosted runners with read-only access, without secrets or self-hosted execution.

## CPU reference model

The frozen acquisition lane is Linux x86_64 with CPython 3.11, 3.12 or 3.13. Create a separate environment and install the full CPU lock (about 200 MB of wheel downloads; no GPU packages):

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --no-deps -r requirements.lock
python -m pip check
PYTHONPATH=src python -m unittest discover -s tests/model -v
PYTHONPATH=src python -m trinite inspect-model --lane dense
PYTHONPATH=src python -m trinite inspect-model --lane ternary --text '(1+1)%2=0' --capture-layer final --capture-position 0
```

Acquisition needs the official CPU index or a pre-provisioned wheel cache; every selected package/version/hash is pinned, including transitive dependencies. This lock's setuptools is Torch runtime support; optional isolated project builds retain their separately pinned backend. Source-tree model execution does not need a build. To use an offline wheelhouse, supply --no-index --find-links=/path/to/wheels with the same lock. The model suite is separate from foundation discovery and must be run explicitly; it never silently skips missing Torch.

inspect-model builds an untrained reference model. --seed selects a private initialization generator; matching configs/seeds create matching dense/ternary latent tensors without consuming the global Torch RNG. --text runs a forward pass, and paired --capture-layer/--capture-position options select detached states. Supported layers are embedding, block.0 through block.5 for the default model, and final. Positions include BOS/EOS. The CLI returns named tensor/code identities, hex scales, environment and capture metadata; it does not generate text or load a checkpoint. See [PHASE-2.md](PHASE-2.md) for resource bounds and evidence limits.

For arrays through the explicit Python API:

```python
import torch
from trinite.model import CaptureSpec, Decoder
from trinite.tokenizer import ByteTokenizer

encoding = ByteTokenizer().encode('1+1=2')
model = Decoder(lane='ternary', seed=0)
ids = torch.tensor([encoding.input_ids], dtype=torch.int64)
mask = torch.tensor([encoding.attention_mask], dtype=torch.bool)
with torch.no_grad():
    output = model(ids, mask, capture=CaptureSpec(('final',), (0, 1)))
print(output.logits.shape, output.captures['final'].shape)
```

## Tiny native training and safe resume

After installing the complete CPU lock, run the separate required training and upstream suites explicitly:

```bash
PYTHONPATH=src python -m unittest discover -s tests/upstream -v
PYTHONPATH=src python -m unittest discover -s tests/training -v
PYTHONPATH=src python -m trinite train /tmp/trinite-dense --config configs/tiny-training.json --dataset fixtures/formal-v1 --lane dense
PYTHONPATH=src python -m trinite train /tmp/trinite-ternary --config configs/tiny-training.json --dataset fixtures/formal-v1 --lane ternary
PYTHONPATH=src python -m trinite verify-observation /tmp/trinite-ternary/provenance
```

Each run uses a new output directory. The default observer is mandatory and pinned; --observer off records explicit unintegrated computation. A successful pause also returns zero; it has compute_outcome=paused. Resume requires the same frozen plan, source, inputs and environment:

```bash
PYTHONPATH=src python -m trinite train /tmp/trinite-paused --config configs/tiny-training.json --dataset fixtures/formal-v1 --stop-after 23
PYTHONPATH=src python -m trinite train /tmp/trinite-resumed --config configs/tiny-training.json --dataset fixtures/formal-v1 --resume /tmp/trinite-paused/checkpoint
cmp /tmp/trinite-ternary/checkpoint/tensors.safetensors /tmp/trinite-resumed/checkpoint/tensors.safetensors
cmp /tmp/trinite-ternary/checkpoint/metadata.json /tmp/trinite-resumed/checkpoint/metadata.json
```

--stop-after is an absolute completed-step number, not an additional-step count. There is no mid-update checkpoint. --source-revision accepts a caller-declared full commit SHA; otherwise the revision is marked unreported. Exact installed procedure files are independently hashed in every checkpoint/context; dirty Git state is explicitly not inspected by the API. Run context/report timestamps and observer overhead differ between invocations; result/history/inventory and final checkpoint bytes replay exactly within the demonstrated scope. See [TRAINING.md](TRAINING.md) for the complete protocol, validation bounds and eligibility distinction.

verify-observation checks a stable frozen bundle with the actual upstream verifier and returns its complete report. It rejects open/missing evidence even if upstream integrity alone passes. It verifies byte integrity and references, not training correctness or custody. A required observation failure returns nonzero while retaining completed computation and a false eligibility flag. report.json includes outcome, errors, complete verification, wall/observer time and Linux process-lifetime peak RSS.

Manual tiny training becomes available after the workflow merges to main. Enter the exact plan and dataset identities printed/recorded by the procedure; [PHASE-3.md](PHASE-3.md) lists the frozen values. The two-lane workflow uses hosted CPU, new temporary directories and 14-day review artifacts. It does not archive a release or acquire the future curriculum.

## Existing checkouts with newline conversion

Source and fixture identities use raw bytes. The LF attributes protect fresh checkouts, but Git can retain CRLF in unchanged files when upgrading an existing checkout. This correction changes all three byte-bound source blobs and both retained fixture blobs so the transition from the initial Phase 1 revision refreshes them. The regression suite exercises a pre-attributes CRLF checkout and a normal two-tree upgrade, including the exact original contracts.py and tokenizer.py blobs.

The source receipt rejects CRLF explicitly and points here instead of hashing converted bytes. If an older working tree still reports this error, first commit or otherwise preserve local edits and confirm that the affected files have no staged or unstaged changes with git status. Then refresh just the tracked byte-bound files from the current index:

```bash
git checkout-index --force -- src/trinite/contracts.py src/trinite/data.py src/trinite/tokenizer.py fixtures/formal-v1/dataset.json fixtures/formal-v1/manifest.json
PYTHONPATH=src python3 -m trinite audit-fixture fixtures/formal-v1
```

The refresh command replaces those working files with indexed content using the current attributes; do not run it over unsaved edits. git add --renormalize alone changes the index and does not guarantee that stale working-tree bytes are refreshed. Keep the retained fixtures intact: regenerating them to match a converted checkout would replace the evidence rather than repair the checkout.

The comparison dependency also pins the exact bytes of
`src/trinite/QEC-LICENSE.txt`. Its LF attribute protects fresh checkouts. If an
existing checkout retained CRLF after the attribute was introduced and QEC pin
verification fails, preserve local edits and confirm this file is clean before
refreshing it from the current index:

```bash
git checkout-index --force -- src/trinite/QEC-LICENSE.txt
PYTHONPATH=src python -c 'from trinite.qec_source import verify_pin; print(verify_pin())'
```

Do not edit the pin or normalize the license in the verifier to conceal a
checkout-byte mismatch.

## Detached geometry observation

Use newly trained final checkpoints from the current source tree, the same process environment and the exact tiny training plan above. Different workflow workers may have different strict environment fingerprints; the manual geometry workflow trains both lanes and measures them on one worker. Each output below must be a new directory.

```bash
PYTHONPATH=src python -m unittest discover -s tests/geometry -v
PYTHONPATH=src python -m trinite freeze-observation /tmp/trinite-geometry-request --dataset fixtures/formal-v1 --training-config configs/tiny-training.json --dense-checkpoint /tmp/trinite-dense/checkpoint --ternary-checkpoint /tmp/trinite-ternary/checkpoint
```

Freeze prints request_file and request_identity. Copy the exact returned identity into REQUEST_ID, then measure that unchanged request:

```bash
REQUEST_ID='sha256:<paste the returned 64 hex digits>'
PYTHONPATH=src python -m trinite measure-observation /tmp/trinite-geometry --request /tmp/trinite-geometry-request/request.json --request-identity "$REQUEST_ID" --dataset fixtures/formal-v1 --training-config configs/tiny-training.json --dense-checkpoint /tmp/trinite-dense/checkpoint --ternary-checkpoint /tmp/trinite-ternary/checkpoint
PYTHONPATH=src python -m trinite verify-observation /tmp/trinite-geometry/provenance
```

Source/input/environment/request changes reject; an observation never updates a model or optimizer. The mandatory pinned observer is the default. --observer off retains explicit unintegrated computation with false evidence eligibility. Required observer failure returns nonzero while preserving completed computational artifacts. Read [GEOMETRY.md](GEOMETRY.md) for schemas, bounds and the frozen control limitation, and [PHASE-4.md](PHASE-4.md) for descriptive results.

The manual geometry workflow uses the frozen plan/dataset identities from PHASE-3, trains both lanes on hosted CPU, freezes the request, measures, verifies and uploads 14-day review artifacts. Download them before expiry when retaining a run; this is not a durable release archive or an edge-device benchmark.

## Combined CPU conformance and performance characterization

CI runs the same required model/training/geometry cases in one process to amortize imports:

```bash
PYTHONPATH=src python scripts/check_cpu.py
```

Standalone suite commands above remain supported. Foundation and pinned upstream conformance remain separate standard-library runs. Dependency-consuming workflows cache downloads, install fresh with required hashes, run pip check and execute every current test on all three Python versions. [OPTIMIZATION.md](OPTIMIZATION.md) defines the exact reuse gates and opt-in benchmark commands.

## Three-lane research comparison

[COMPARISON.md](COMPARISON.md) gives frozen dense/ternary/four-state and exact QEC task matrix commands. Run comparison conformance after acquiring the CPU lock:

```bash
PYTHONPATH=src python -m unittest discover -s tests/comparison -v
PYTHONPATH=src python -m trinite inspect-model --lane four-state --text "1+1=2"
```

