// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Trent Slade / QSOL-IMC.
// Test driver only; the three imported upstream modules remain byte-identical.
import { selectEtq101Basis, buildRootAdjacency, graphDegreePotential } from './donor/src/etq-model.mjs';
import { buildEventDocument, buildLiftedGraph, buildSiteRegistry } from './donor/src/etq-v3-core.mjs';
import { canonicalSerialize, canonicalObjectSha256, sha256Bytes, utf8 } from './donor/src/etq-v3-canonical.mjs';

const basis = selectEtq101Basis();
const adjacency = buildRootAdjacency(basis);
const document = buildEventDocument();
const graph = buildLiftedGraph();
const registry = buildSiteRegistry();
const plainHash = value => sha256Bytes(utf8(canonicalSerialize(value)));
const commitments = {
  preservedV2BasisFixtureSha256: plainHash(basis),
  preservedV2AdjacencyFixtureSha256: plainHash(adjacency),
  preservedV2DegreePotentialNumeratorsFixtureSha256: plainHash(graphDegreePotential(adjacency).numerators),
  siteRegistrySha256: canonicalObjectSha256('ETQ/V3/SITE-REGISTRY/v1', registry),
  eventsSha256: canonicalObjectSha256('ETQ/V3/EVENTS/v1', document.events),
  liftedEdgesSha256: canonicalObjectSha256('ETQ/V3/LIFTED-EDGES/v1', graph.edges),
  eventDocumentSha256: canonicalObjectSha256('ETQ/V3/EVENT-DOCUMENT/v1', document),
};
process.stdout.write(canonicalSerialize({
  commitments, eventDocumentCanonical: canonicalSerialize(document), graph, registry,
}));
