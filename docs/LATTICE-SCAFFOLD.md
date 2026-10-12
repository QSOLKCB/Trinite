# Exact recursive 3×3×3 lattice scaffold (v1)

This separately versioned **analytic reference** implements the proposed
recursive 3×3×3 Sierpiński-style cubic geometry, with no neural activation
projection, training, geometry loss or evidence of reasoning benefit.

## Frozen construction

- Protocol: `src/trinite/lattice-scaffold-protocol.json`
- Identity: `sha256:3fb56ed2e0ce54872b8e8e63e5e22d044913ff891c742379ba07ae1b3b21f4f0`
- Geometry: **Menger-style 20/27 retention**. Children have digits
  `(x,y,z)` from `0..2`; retain a child if *at most one digit is 1*.
  All 20 retained child offsets use lexicographic x/y/z order.
- Depth: exact integer 0–3, with occupied cell counts 1, 20, 400, 8,000.
  Depth 0 is the root cube. No adaptive depth or unbounded recursion.
- Coordinates: integer zero-origin leaf cube address. A parent at address
  `(a,b,c)` has a child at `(3a+x,3b+y,3c+z)`. Side length at depth d
  is `3**d` discrete cube positions per axis. Each level is emitted in
  **parent-major, then lexicographic-child order** (not globally sorted).
- Parent/child identity: every node has a stable index in its own level;
  nonroot nodes carry `parent_index` and `child_rank` (0–19).
  Level `occupancy_identity` is SHA-256 of canonical ordered coordinate
  arrays. Identity changes if the occupancy/order changes.
- Graph: undirected **six-neighbour face** adjacency of leaf cubes only,
  represented as sorted index pairs `(i,j)` with `i < j`. No
  diagonal/corner/long-range edges. `edge_identity` hashes the exact
  canonical edge list. Integer degree histograms and level counts are
  retained for independent checking.

The full source receipt binds the existing Trinite canonical JSON/identity
contract, the new construction, its canonical protocol and the fixed launcher.
Outputs are canonical JSON with UTF-8, sorted keys, compact separators and
one terminal LF, with SHA-256 identities. There is no random seed, floating
point arithmetic, tensor, scheduler, optimizer, pretrained corpus or GPU.

## Generate and verify

Run from the repository root (Python 3.11+, **no Torch required**):

```sh
PYTHONPATH=src python scripts/generate_lattice_scaffold.py --depth 2 --output /tmp/trinite-lattice-d2.json
PYTHONPATH=src python scripts/generate_lattice_scaffold.py --verify /tmp/trinite-lattice-d2.json
PYTHONPATH=src python -m unittest discover -s tests -p 'test_lattice_scaffold.py' -v
```

Generation exclusively creates the output and refuses overwrites. Verification
reads at most 2 MiB and rejects larger/unbounded input before JSON parsing. Verification
is read-only and reconstructs the whole expected canonical artifact from its
depth and **current exact source/protocol bytes**; a mutated edge, parent,
count, source receipt, claim flag or noncanonical JSON fails. CLI symlink paths
are rejected. A changed producer must use a newly versioned protocol and
fresh artifact: do not patch an old identity or reseal an experiment.

The independent structural suite covers occupancy, child addresses,
cardinalities, brute-force face-graph comparisons at depths 0–2, degree
conservation at depth 3, mutation rejection and CLI boundaries.

## Research boundaries and next gate

This is only an **address/adjacency candidate** for future geometry observation.
It does not attach model activations to anchors, compare geometric losses,
improve maths retention, or enable Phase 5.

A separately reviewed **source-bound neural observation** must freeze selected
model/layer/token, capture provenance, normalization, dimensionality reduction
or training-only fitting, split isolation, occupancy-to-feature correspondence,
and measured units before collecting outcomes. Compare the recursive hierarchy
against **capacity- and cardinality-matched** nonrecursive anchors, shuffled
ancestry and fixed-seed random anchors, with predeclared metrics and negative
results retained. No visually pleasing geometry is evidence of learned structure.

The prior task-balanced maths 8,192-update matrix still requires an independent
full research run and replay/archive; existing learning gates and geometry-on
intervention remain blocked. Affected: TRI-I01/I02/I05/I09/I11/I12/I13/I14.
