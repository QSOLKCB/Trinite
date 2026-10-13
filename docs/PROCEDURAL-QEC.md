# QEC-guided exact procedural supervision

**PR #34 — exploratory implementation, no completed full research outcome.** This detour does not change historical models, frozen optimizer behavior, earlier protocols, geometry loss or immutable v0.1.0/v0.2.0 tags.

## Research question

Can independently verified intermediate computations improve original complete final-answer-plus-EOS accuracy, rather than only enable the model to mimic traces?

The first protocol (v1) was committed at 8a6116251e1d270dc39623dadfa58c18911a0be6 before implementation. Early **CI admission tests**, before any trained result, exposed a genuine prompt collision between arithmetic subtraction and fold sums. Therefore v1 is retained **unexecuted**, and a corrected [protocol v2](../src/trinite/procedural_qec/protocol-v2.json) was frozen in commit 0697d5944d48517bb0ab04afc211af7a9826cd2f. Its SHA-256 is 3c44c40c10fd63026cc7f8822f9b1a0842737897e33a91afef75a6c28d70d1bb. Old v1 outcomes have not been silently relabelled; no full model training run has yet been published.

## Mathematical source and training data

QEC direct donor: the existing byte-pinned, attributed v170.1.0 qutrit GF(3) add function, checked by an independent integer running-sum modulo-three oracle. Enumerate triples over {0,1,2} and render only minimal symbolic tasks. These 27 triples and two carrier variants are **train-only** mathematical facts, not unexposed confirmation.

Fold transfer: use the existing rights-admitted scalar fold SUM facts, and **only their inherited training families**. Verify the original labels independently with the scalar oracle. Generate each running accumulator from zero; render a bounded halfway prefix task and a complete trace per original carrier. Every derived row retains its full parent operands, parent example identity, semantic identity, family, source lineage, carrier and split. The original fold final prompt/answer is byte-identical, and all renderings (including BOS and EOS) fit 64 byte tokens.

Arithmetic retention uses the **same admitted numeric facts and answer targets**, but adds an explicit R/ prefix to every arithmetic prompt. This is mandatory to prevent contradictory original S: subtraction and S: fold-sum labels. Consequently arithmetic results in PR #34 are **namespaced** and not directly comparable to previous unprefixed arithmetic outputs. Original fold final-answer outputs remain unchanged.

| Task | Prompt | Exact answer |
| --- | --- | --- |
| Fold final | S:2,-1,2= | 3 |
| Fold prefix | P:2,-1,2@1= | 2 |
| Fold full trace | T:2,-1,2= | 0,2,1,3 |
| GF(3) final | G:2,1,2= | 2 |
| GF(3) prefix | H:2,1,2@2= | 0 |
| GF(3) full trace | J:2,1,2= | 0,2,0,2 |
| Namespaced subtraction | R/S:2,1= | 1 |

Distinct prefix, trace and final namespaces prevent target contradictions. Prefix/trace labels are exact, not teacher-generated. This corpus is visible development material. Other transformations of known families are not fresh held-out evidence.

## Frozen two-arm schedule

Each eight-example update consumes exactly **four namespaced arithmetic-final, two original fold-sum-final and one GF(3)-final examples** in both arms. The eighth example is an additional fold/GF(3) final answer in the **final-only** control arm, or an alternating fold/GF(3) prefix/full trace in the **procedural** arm.

Two fixed seeds (0 and 1), identical initial model within a seed, CPU-only dense 86,528-parameter reference research model, context 64, 4,096 steps, native AdamW and learning rate 0.001. Final checkpoint only; no early stopping or post-hoc winner selection. A separate width-16/eight-update smoke profile is software integration only.

Arms match **updates, shared rehearsal visits, initializer and scoring view**, not total final-answer visits, output token exposure or compute. The signed exposure report records per-task/mode visit counts, scored targets and input tokens. No claim of a causally isolated trace effect follows from this comparison.

Primary metric: original greedy complete final answers **plus EOS**, on all unique admitted training rows, with separate counts for arithmetic add/subtract/multiply/divide, fold-sum and QEC GF(3). Secondary metrics: prefix and full-trace accuracy from free-running decoding with **no teacher-forced intermediate states supplied during evaluation**. An exact trace alone does not satisfy the original final-answer gate.

A research-training-only candidate passes only if every original final task reaches at least 90% in both seeds **and** every worker, all custody checks and complete native optimizer replay succeed. Training acceptance cannot unlock fresh generalization, the distinct original fold/composed-output or arithmetic gates, ternary-weight format selection, Phase 5, scaling or release readiness.

## Execution and replay

Use the hash-locked CPU environment from GETTING_STARTED.md (Linux x86_64, Python 3.11–3.13, Torch 2.8.0+cpu). The CLI explicitly sets single-thread and deterministic Torch flags before the environment guard.

    export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

    # Freeze from the published producer, before inspecting any learning results.
    python scripts/run_procedural_qec.py --freeze-request /tmp/procedural-request.json --run-profile research

    # Use exact request_identity returned by the freeze; output and external
    # verification-request paths must be distinct and newly created.
    python scripts/run_procedural_qec.py --run \
      --request /tmp/procedural-request.json --request-identity sha256:REQUEST \
      --verification-request /tmp/procedural-verification.json \
      --output /tmp/procedural-run

    # Independently retain verification_request_identity from the preceding
    # output; this verifies all cell custody, original answers and EVERY update.
    python scripts/run_procedural_qec.py --verify-only \
      --request /tmp/procedural-request.json --request-identity sha256:REQUEST \
      --verification-request /tmp/procedural-verification.json \
      --verification-request-identity sha256:VERIFICATION \
      --output /tmp/procedural-run

The verifier is read-only. It rechecks external exact-byte commitments, closed PROVENANCE indexes, model and AdamW inventory, complete update histories, original answers and final free-running prefixes/traces, then replays all native updates. A matching final checkpoint without replay is insufficient. Failure/incomplete cells retain errors and block aggregate acceptance.

For integration-only smoke, freeze a distinct request with --run-profile smoke, and use a new output directory. Tests:

    python -m unittest discover -s tests -p test_procedural_qec.py -v
    python -m unittest discover -s tests/procedural_qec -v

The separate procedural-qec GitHub Actions workflow uses the pinned CPU environment; existing 261 research cases and their shard partition are unchanged. The complete 4,096-update research matrix and external evidence archive are not published by this PR. The old task-balance archival closure remains a separate obligation.

## License, admission and scope

No new QEC source code or external donor prose is imported. The existing historically attributed QEC v170.1.0 pin carries its original terms; current QEC's repository-wide MPL-2.0 policy does not erase historical grants. Generated numeric/finite-field facts have an independent narrow rights-admission statement, not admission by software license alone. Not a legal opinion.

Required invariant coverage: TRI-I02/I03/I05/I06/I07/I08/I11/I12/I13/I14. No new quantizer, STE, geometry objective, adaptive quota controller, teacher distillation or quantum-computing claim is in scope. If the model still fails, capacity, representation and compositional learning remain competing explanations.
