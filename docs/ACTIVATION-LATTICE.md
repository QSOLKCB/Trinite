# Activation-to-lattice observation — exploratory v1

**Status:** separately frozen descriptive research producer. No geometry loss, changed
weights, new teaching data or Phase 5 intervention. Builds on the merged
[exact Menger-style 3×3×3 scaffold](LATTICE-SCAFFOLD.md), not on a visual claim.

## Question and prior exposure

Do **already trained native Trinite internal prompt states** have local
nearest-neighbour relationships that a fixed, three-axis map preserves on a
20-node Menger graph more often than cardinality- and edge-matched controls?

The sole input is the complete previously published
`fixtures/reference-geometry-v1/Trinite-reference-geometry-v1.zip.part000`
(SHA-256 `666e8f4d5b49da60aada4a26acbfa6001a5a8b8b62447c1ddfd3a5613cfd8721`).
The packet contains the *native trained arithmetic* condition, exact original
reference request, detached float32 first-block/final states, model inventory
and previously retained provenance. The new producer checks the **entire exact
archive digest**, original request identity, and applies the original capture
auditor to the selected native capture. This does not rerun the original training
or independently recreate its complete upstream provenance history.

The 16 native probes are **training-exposed and correlated**; therefore this
retrospective measurement is exploratory regardless of the family-separated
normalization. No new native training, model selection, held-out model scoring,
neural-map fitting, causal inference or generalization claim is made.

## Frozen recipe

Protocol: `src/trinite/lattice_observation/protocol.json`, identity
`sha256:5c92baba23f0c2fa41a61d65ea5e74d088262608907127b2318351dc0cd1f59b`.
The protocol was committed ahead of derived outcomes.

1. Require the original source-bound request, sixteen distinct original probes,
   both existing layers `block.0` and `final`, the last native prompt-prefix
   anchor before the supplied target, and exact 64-dimensional float32 payloads.
2. For each layer, project onto **three fixed, seeded ±1 signed axes**
   (seed 29; signs SHA-256 addressed by axis and coordinate). No trainable
   weights or dimension selection.
3. Fit independent min/max affine normalizations using **only rows 0–7**, from
   the first family. Evaluate **only rows 8–15**, from the different family.
   Clip out-of-range evaluation axes to [0,2] and report the clipping count.
   Degenerate fitted axes map to 1 and are explicitly reported. This is
   normalization-only calibration, not held-out model evaluation.
4. Four preselected conditions use exactly 20 anchors and the same number of
   undirected, unweighted graph edges:
   - `menger`: depth-one retained 20/27 voxels and genuine face adjacency.
   - `nonrecursive`: first 20 of the full 3×3×3 coordinate grid, with the
     lowest-distance edges matching the Menger edge budget.
   - `shuffled-topology`: exactly the Menger anchors with a fixed
     node-label permutation applied to the Menger graph edges; degree
     multiset and edge budget are preserved.
   - `random-anchors`: fixed-hash 20-of-27 anchor selection with the same
     distance-based edge budget as the nonrecursive control.
5. Map each evaluation vector to its nearest three-dimensional anchor
   (lowest-index tie break), then find its **nearest raw 64D evaluation
   neighbour**. Report exact shortest-path hops on each candidate graph,
   sum of hops (lower is better descriptively), same-or-adjacent count (higher
   is better), anchor collisions, occupancy and disconnected pair counts.
   Disconnected pairs score 21 hops. Repeated queries and shared training
   exposure are not treated as statistically independent.

All control arms share the **same signed projection and normalization**.
No hyperparameter, anchor count, depth, control seed, label permutation or
metric is chosen after looking at outputs. A comparison of these controls
does not isolate the causal mechanism of learning; graph topology and
placement are jointly varied in two controls.

## Run with Python standard library only

The existing review packet is the only required input. No download, model
reload, GPU, Torch, teacher data or external service is used.

```sh
export PYTHONPATH=src
python scripts/run_lattice_observation.py --freeze-request /tmp/activation-lattice-request.json
# Replace REQUEST with the printed SHA-256 from the previous command.
python scripts/run_lattice_observation.py --run \
  --request /tmp/activation-lattice-request.json \
  --request-identity sha256:REQUEST \
  --output /tmp/activation-lattice-analysis.json
# Replace ANALYSIS with the external result identity emitted by --run.
python scripts/run_lattice_observation.py --verify-only \
  --request /tmp/activation-lattice-request.json \
  --request-identity sha256:REQUEST \
  --output /tmp/activation-lattice-analysis.json \
  --artifact-identity sha256:ANALYSIS
python -m unittest discover -s tests -p 'test_lattice_observation.py' -v
```

Generation and freeze use **exclusive creation**, never overwrite or reseal
historical bytes. Verification is read-only, checks independently supplied
request and result hashes, checks the producer/source identities, audits the
historic native capture again and recalculates all results. All input reads are
bounded to 2 MiB; symlink paths and noncanonical JSON are rejected. Retain
the two identities outside the working directory for reproducibility.

## Scientific limits and next step

A better score for the recursive graph is not proof of an internally
Menger-shaped neural representation, understanding or improved reasoning.
The *model was not trained to align with this lattice*. This is a
small retrospective nearest-neighbour/quantization diagnostic: no
uncertainty estimation, prospective independence or calibrated hypothesis
test is claimed. Negative/equal/degenerate outcomes are legitimate results.

For a prospective confirmatory study, freeze a **new** request before new
activations are generated, reserve unseen semantic families, independently
replicate, compare matched projections and controls with uncertainty, and
retain native checkpoint custody plus exact source/optimizer history as
applicable. RES=RAG/CSNP remains a distinct proposed metric adapter.

**All earlier gates remain blocked:** task-balanced full research matrix,
original fold/composed learning adequacy, fresh generalization, format
selection, Phase 5 geometry intervention, scaling and release. Affected
TRI-I01/I02/I03/I05/I09/I11/I12/I13/I14.
