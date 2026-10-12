# MaleCNS v1.0 — third-party source data (CNR-001)

The three files in this directory are an **unaltered upstream selection**, not
synthetic biology or newly created training data.

- Biological source: MaleCNS v1.0, Janelia FlyEM / Google Research /
  University of Cambridge; neuPrint dataset `male-cns:v1.0`.
- Paper: Berg, S. et al., *Sexual dimorphism in the complete Drosophila male
  central nervous system connectome*, *Cell* (2026),
  DOI: https://doi.org/10.1016/j.cell.2026.08.015
- Project: https://male-cns.janelia.org/
- Copyright/data licence: **CC BY 4.0**; retain attribution.
- Intermediate distributor: https://github.com/us/connectorch
- Exact distributor commit: `c2bb1e838ee56ab67bb4d620895e5d63f43a274c`.
- Original paths: `src/connectorch/datasets/data/malecns_sample.ct/`.
- `edges.parquet` Git blob SHA-1: `261a6b06a18ac1133639e8af7a72f28d9b6e21b9`.
- `nodes.parquet` Git blob SHA-1: `6df627cff126789637f1c4e68301b5ecf2ca5bd9`.
- `metadata.json` Git blob SHA-1: `124459573f18a4c3fbfd97de057b99a181b547fe`.

The original 100-neuron, 3,630-directed-connection sample was obtained from
the published traced-only MaleCNS v1.0 connections with a minimum of five
synapses. It was **not randomly sampled**: the distributor greedily grew a dense
subgraph from the highest-degree 4,000 bodies. This creates strong sampling
bias, and this sample is not a representative entire-brain connectome.

CNR-001 uses this as a fixed *structural reference only*: it collapses direction
for the primary 20-node topology comparison and retains provenance/selection
identities. No biological data is used to update Trinite weights, no trained
biological simulator is claimed, and the upstream library's MIT code licence
does not replace the biological data's CC BY 4.0 obligations.

For the reproduction script and strict resource limits see
[docs/CNR-001.md](../../docs/CNR-001.md).
