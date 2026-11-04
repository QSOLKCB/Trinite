# TT-002 — QEC-derived formal teacher dataset

**Scope:** additive, source-specific offline curriculum with independent mathematical verification, structured training/validation/test partitions and distinct data/weight rights receipts. No model training, model changes, privileged tool execution, GPU use, release promotion, or scientific generalization claim.

This original first-party curriculum uses GF(3) qutrit SUM/SWAP conventions from QSOLKCB/QEC v177.2 at commit 27bbc798c4e38138ca5b0fff70b15c3271a1917d. The existing Trinite qutrit-conformance adapter provides an independently implemented nine-basis-state conjugation oracle, prompt/carrier contracts and native byte-token ledger. The original generator in this increment separately computes exact exponent transforms, inverse transformations and check labels. They must agree for every row.

The external hmx21/css-decoupling reference is attributed at commit a48048c1097f4d1b37a55eca9a1796a823f0c124 (MIT). Its source, documentation, copyrighted prose, model outputs and any pretrained weights are **not** training data or vendored by this change. Generator code uses Trinite's MPL-2.0 licence; the numeric generated dataset and hypothetical future checkpoint each have separately recorded rights and do not automatically inherit MPL.

## Dataset inventory and split boundaries

Seven circuits x 81 public four-residue words x four task views = **2,268** exactly regenerated examples. Each circuit includes apply, inverse, candidate correct and deliberately wrong candidate check examples. Wrong candidates have one canonical nonzero residue shift. Every algebraic output is checked against the independent ket-permutation oracle and verified for inverse roundtrip. Original minimal ASCII prompts use the already implemented Q1 carrier, with no oracle labels, family metadata or source file paths in the public model prompt. Token counts and supervised answer+EOS boundaries are checked against the unchanged ByteTokenizer.

| Partition | Complete circuit families | Rows |
| --- | --- | ---: |
| Train | S1, S2, W (both SUM weights plus SWAP primitive) | 972 |
| Development validation | S1W, WS2 | 648 |
| Development test | S2W, WS1 | 648 |

All operands, task variants and transformations of each whole circuit family stay together. This fixes a problem with using the old external adapter's engineering split as a research split: that one withheld the SWAP primitive, confounding primitive familiarity with composition. There are no identical public prompts across these partitions, but algebraically equivalent outputs can occur across families.

**Scientific boundary:** all seven circuits and their task views already appear in exposed engineering tests. These validation/test labels are **not secret or fresh scientific evaluation**. This dataset makes no held-out reasoning or transfer claim. TT-003 needs fresh exposure-controlled tasks and independent prediction commitment before any confirmatory claim.

## Artifacts and data/rights separation

The offline preparation produces a new owner-only directory with eight files: plan.json, dataset-rights.json, weights-rights.json, train.jsonl, validation-public.jsonl, test-public.jsonl, privileged-targets.jsonl and manifest.json.

Training items contain their own ground-truth answers and token-budget metadata. Withheld public rows contain ONLY schema, item ID, observation and public prompt; privileged-targets.jsonl stores evaluation answers separately. These files are not an OS sandbox, so a future privileged evaluator/controller must enforce actual isolation, not merely trust a file naming convention.

The dataset-rights record admits only original generated finite arithmetic facts, commissioned by the owner-scoped TT-002 request; it documents origin, reviewer scope, date, source and donor commit identities, provenance, data policy, exposures and explicit exclusions. **Admitted formal-only does not authorize a training run**. The dataset distribution licence is undecided, separately from the code's MPL-2.0. The weights-rights record declares weights NOT_CREATED and their license UNDECIDED.

A source-bound manifest hashes every packet member and the original source/configuration closure, including normative TT-002 documentation and imported Python code. The manifest digest is emitted by prepare and must be independently retained outside the frozen packet. Verification rejects extra/missing/malformed/symlink/hardlinked files, mismatched source or rights, and tampered or incorrect answers. It independently regenerates the **entire** inventory and all oracle comparisons, rather than trusting the packet's self-reported verification result. No historical TT-001 source, native model, optimizer, previous QEC experiment, frozen fixture or negative evidence is modified. Staged publication cleans up on failed write or fsync.

## Commands and CI

Run offline with Python 3.11 or newer and a trusted checkout:

~~~sh
python3 scripts/teacher_dataset.py inspect
python3 scripts/teacher_dataset.py prepare --out /tmp/teacher-gf3-fresh
# Retain the manifest_identity from the prepare output separately:
python3 scripts/teacher_dataset.py verify --packet /tmp/teacher-gf3-fresh --identity sha256:RETAINED
PYTHONPATH=src:scripts python3 -B -m unittest discover -s tests/teacher_dataset_v1 -v
~~~

The read-only GitHub Actions lane executes normal and optimized Python tests plus a complete fresh prepare/verify; it has no Rust build, download, CI cache, training or GPU access. Existing dedicated Rust conformance tests separately exercise the physical Rust binary/implementation. This dataset only reuses the independently specified arithmetic and Python oracle, not a runtime claim about compiled Rust code.

## TT-003 prerequisites

Train-only ingestion must be separately reviewed and owner-authorized with a frozen model, exact available data, controls, matched budgets, independent heldout exposure, and robust evaluator isolation. For RTX 50-class GPU training, the prospective fail-closed CUDA 12.8+ / native sm_120 preflight documented in the original Teacher Lab roadmap remains mandatory. No native model, context limit, CUDA backend or architecture is changed by TT-002.

Affected boundaries: TRI-I01/I02/I03/I05/I07/I08/I11/I12/I13/I14. All historical protocols and release gates remain intact.
