# Getting started

Phase 1 requires Python 3.11 or newer and no runtime/test packages. It does not implement a model, training, or serving.

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

Backend acquisition needs network access or a pre-provisioned wheel. The runtime and source-tree workflow remain offline. requirements.lock has no runtime/test dependencies. The package version 0.0.0 denotes unreleased foundation code; this PR creates no tag or release.

CPU CI runs the standard-library suite, config validation, frozen fixture audit, and exact replay under Python 3.11/3.12/3.13. Actions are pinned to full commit identities; hosted jobs have read-only repository access and a five-minute limit. CI installs no model packages and uses no privileged/self-hosted runner.

## Existing checkouts with newline conversion

Source and fixture identities use raw bytes. The LF attributes protect fresh checkouts, but Git can retain CRLF in unchanged files when upgrading an existing checkout. This correction changes all three byte-bound source blobs and both retained fixture blobs so the transition from the initial Phase 1 revision refreshes them. The regression suite exercises a pre-attributes CRLF checkout and a normal two-tree upgrade, including the exact original contracts.py and tokenizer.py blobs.

The source receipt rejects CRLF explicitly and points here instead of hashing converted bytes. If an older working tree still reports this error, first commit or otherwise preserve local edits and confirm that the affected files have no staged or unstaged changes with git status. Then refresh just the tracked byte-bound files from the current index:

```bash
git checkout-index --force -- src/trinite/contracts.py src/trinite/data.py src/trinite/tokenizer.py fixtures/formal-v1/dataset.json fixtures/formal-v1/manifest.json
PYTHONPATH=src python3 -m trinite audit-fixture fixtures/formal-v1
```

The refresh command replaces those working files with indexed content using the current attributes; do not run it over unsaved edits. git add --renormalize alone changes the index and does not guarantee that stale working-tree bytes are refreshed. Keep the retained fixtures intact: regenerating them to match a converted checkout would replace the evidence rather than repair the checkout.
