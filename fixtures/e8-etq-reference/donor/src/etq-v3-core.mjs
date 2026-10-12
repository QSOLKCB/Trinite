// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Trent Slade / QSOL-IMC.

/**
 * ETQ-303 v3 exact finite event protocol.
 *
 * This module contains only exact finite combinatorics and integer/rational
 * metadata. It defines no hertz, waveform, wall-clock time, eigensolver,
 * stochastic process, physical measurement, or rendered audio.
 */

import {
  MODEL_ID as BASE_MODEL_ID,
  MODEL_VERSION as BASE_MODEL_VERSION,
  OUROBOROS_DIMENSION,
  QUTRIT_DIMENSION,
  SCL_STENCIL,
  SELECTED_FIXED_ROOTS,
  SELECTED_TRIALITY_ORBITS,
  TERNARY_REGISTER_LANES,
  buildRootAdjacency,
  buildTernaryMidiCodebook,
  graphDegreePotential,
  selectEtq101Basis,
} from "./etq-model.mjs";

export const MODEL_ID = "ETQ-303";
export const MODEL_VERSION = "3.0.0";
export const PROFILE_ID = "full-qutrit-event-protocol-v1";
export const EVENT_SCHEMA_ID = "qsol.etq-303.events/v1";
export const SITE_REGISTRY_SCHEMA_ID = "qsol.etq-303.site-registry/v1";
export const LIFTED_GRAPH_SCHEMA_ID = "qsol.etq-303.cartesian-graph/v1";

export const BASE_SITE_COUNT = OUROBOROS_DIMENSION;
export const FIBRE_DIMENSION = QUTRIT_DIMENSION;
export const EVENT_COUNT = BASE_SITE_COUNT * FIBRE_DIMENSION;

if (BASE_SITE_COUNT !== 101 || FIBRE_DIMENSION !== 3 || EVENT_COUNT !== 303) {
  throw new Error("ETQ-303 requires the preserved 101 x 3 construction");
}

export const BASE_RELEASE = Object.freeze({
  modelId: BASE_MODEL_ID,
  modelVersion: BASE_MODEL_VERSION,
  gitTag: "v2.0.0",
  gitCommit: "8c24d58ca76abbac77c427a4f63ca434570c82b3",
  versionDoi: "10.5281/zenodo.21432511",
  basisSha256: "97cfd1f087745422fd66d3640c7b86c3209593c4b53741018c08a5e9cdb15f6f",
  adjacencySha256: "29ae0af5b1090c9de30f1efc25789060fb1791eb175d2afcd6888847f7fe6324",
  degreePotentialNumeratorsSha256:
    "517b9b86e07a1788cabc446be93e573d21719af3c1e0eeaedae00b81689c4468",
  contractPayloadSha256:
    "d17e4e104bae41e649f1589467fb8122eb42fc9be4298110f4c21753580b3eab",
});

/** Exact Gaussian-unit exponents for exp(-i*pi*d_q/2) = i^g_q. */
export const FIBRE_PHASE_GAUSSIAN_EXPONENTS = Object.freeze(
  SCL_STENCIL.map((curvature) => modulo(-curvature, 4)),
);
export const GAUSSIAN_UNIT_SYMBOLS = Object.freeze(["1", "i", "-1", "-i"]);
export const FIBRE_PHASE_GAUSSIAN_SYMBOLS = Object.freeze(
  FIBRE_PHASE_GAUSSIAN_EXPONENTS.map(
    (exponent) => GAUSSIAN_UNIT_SYMBOLS[exponent],
  ),
);

export function modulo(value, modulus) {
  if (!Number.isSafeInteger(value) || !Number.isSafeInteger(modulus) || modulus <= 0) {
    throw new RangeError("modulo requires safe integers and a positive modulus");
  }
  return ((value % modulus) + modulus) % modulus;
}

export function gcd(left, right) {
  if (!Number.isSafeInteger(left) || !Number.isSafeInteger(right)) {
    throw new TypeError("gcd requires safe integers");
  }
  let a = Math.abs(left);
  let b = Math.abs(right);
  while (b !== 0) [a, b] = [b, a % b];
  return a;
}

export function lcm(left, right) {
  if (left === 0 || right === 0) return 0;
  return Math.abs((left / gcd(left, right)) * right);
}

function requireIndex(value, maximumExclusive, name) {
  if (!Number.isSafeInteger(value) || value < 0 || value >= maximumExclusive) {
    throw new RangeError(`${name} must be an integer in [0, ${maximumExclusive - 1}]`);
  }
  return value;
}

export function tensorIndex(siteIndex, fibreLabel) {
  requireIndex(siteIndex, BASE_SITE_COUNT, "siteIndex");
  requireIndex(fibreLabel, FIBRE_DIMENSION, "fibreLabel");
  return FIBRE_DIMENSION * siteIndex + fibreLabel;
}

export function tensorAddress(index) {
  requireIndex(index, EVENT_COUNT, "tensorIndex");
  return {
    siteIndex: Math.floor(index / FIBRE_DIMENSION),
    fibreLabel: index % FIBRE_DIMENSION,
  };
}

/** Support permutation underlying R_101 tensor F_3. */
export function exactMonomialStep(siteIndex, fibreLabel) {
  requireIndex(siteIndex, BASE_SITE_COUNT, "siteIndex");
  requireIndex(fibreLabel, FIBRE_DIMENSION, "fibreLabel");
  const phaseGaussianExponent = FIBRE_PHASE_GAUSSIAN_EXPONENTS[fibreLabel];
  const targetSiteIndex = (siteIndex + 1) % BASE_SITE_COUNT;
  const targetFibreLabel = (fibreLabel + 1) % FIBRE_DIMENSION;
  return {
    sourceTensorIndex: tensorIndex(siteIndex, fibreLabel),
    targetSiteIndex,
    targetFibreLabel,
    targetTensorIndex: tensorIndex(targetSiteIndex, targetFibreLabel),
    phaseGaussianExponent,
    phaseSymbol: GAUSSIAN_UNIT_SYMBOLS[phaseGaussianExponent],
  };
}

/** Exhaustive exact power on a basis state; no floating arithmetic is used. */
export function exactMonomialPower(siteIndex, fibreLabel, power) {
  requireIndex(siteIndex, BASE_SITE_COUNT, "siteIndex");
  requireIndex(fibreLabel, FIBRE_DIMENSION, "fibreLabel");
  if (!Number.isSafeInteger(power) || power < 0) {
    throw new RangeError("power must be a nonnegative safe integer");
  }
  let currentSite = siteIndex;
  let currentFibre = fibreLabel;
  let accumulatedPhaseGaussianExponent = 0;
  for (let step = 0; step < power; step += 1) {
    const transition = exactMonomialStep(currentSite, currentFibre);
    accumulatedPhaseGaussianExponent = modulo(
      accumulatedPhaseGaussianExponent + transition.phaseGaussianExponent,
      4,
    );
    currentSite = transition.targetSiteIndex;
    currentFibre = transition.targetFibreLabel;
  }
  return {
    siteIndex: currentSite,
    fibreLabel: currentFibre,
    tensorIndex: tensorIndex(currentSite, currentFibre),
    accumulatedPhaseGaussianExponent,
    accumulatedPhaseSymbol:
      GAUSSIAN_UNIT_SYMBOLS[accumulatedPhaseGaussianExponent],
  };
}

/** Event n is the nth state reached from |0,0> by the support permutation. */
export function eventAddress(sequenceIndex) {
  requireIndex(sequenceIndex, EVENT_COUNT, "sequenceIndex");
  const siteIndex = sequenceIndex % BASE_SITE_COUNT;
  const fibreLabel = sequenceIndex % FIBRE_DIMENSION;
  return {
    siteIndex,
    fibreLabel,
    tensorIndex: tensorIndex(siteIndex, fibreLabel),
  };
}

/** Chinese-remainder inverse of eventAddress. */
export function eventIndexFromAddress(siteIndex, fibreLabel) {
  requireIndex(siteIndex, BASE_SITE_COUNT, "siteIndex");
  requireIndex(fibreLabel, FIBRE_DIMENSION, "fibreLabel");
  const k = modulo(2 * (fibreLabel - modulo(siteIndex, 3)), 3);
  const sequenceIndex = siteIndex + BASE_SITE_COUNT * k;
  if (sequenceIndex >= EVENT_COUNT) throw new Error("CRT inverse escaped domain");
  return sequenceIndex;
}

export function buildSiteRegistry() {
  const basis = selectEtq101Basis();
  const adjacency = buildRootAdjacency(basis);
  const degreePotential = graphDegreePotential(adjacency);
  const midiCodebook = buildTernaryMidiCodebook();
  if (
    basis.length !== BASE_SITE_COUNT ||
    adjacency.length !== BASE_SITE_COUNT ||
    degreePotential.degrees.length !== BASE_SITE_COUNT ||
    midiCodebook.length !== BASE_SITE_COUNT
  ) {
    throw new Error("preserved ETQ-101 fixtures have inconsistent dimensions");
  }

  const sites = basis.map((rootDoubled, siteIndex) => {
    const isFixed = siteIndex < SELECTED_FIXED_ROOTS;
    const orbitIndex = isFixed
      ? null
      : Math.floor((siteIndex - SELECTED_FIXED_ROOTS) / FIBRE_DIMENSION);
    const internalTrialityLabel = isFixed
      ? null
      : (siteIndex - SELECTED_FIXED_ROOTS) % FIBRE_DIMENSION;
    const code = midiCodebook[siteIndex];
    if (code.basisIndex !== siteIndex) throw new Error("v2 codebook order mismatch");
    return {
      siteIndex,
      rootDoubled: [...rootDoubled],
      baseStateType: isFixed ? "fixed-singlet" : "triality-orbit-state",
      fixedIndex: isFixed ? siteIndex : null,
      orbitIndex,
      internalTrialityLabel,
      internalTrialityLane:
        internalTrialityLabel === null
          ? null
          : TERNARY_REGISTER_LANES[internalTrialityLabel],
      baseDegree: degreePotential.degrees[siteIndex],
      liftedDegree: degreePotential.degrees[siteIndex] + 2,
      degreePotentialNumerator: degreePotential.numerators[siteIndex],
      degreePotentialDenominator: degreePotential.normalizationDenominator,
      v2MidiNote: code.midiNote,
    };
  });

  return {
    schema: SITE_REGISTRY_SCHEMA_ID,
    modelId: MODEL_ID,
    modelVersion: MODEL_VERSION,
    baseModel: `${BASE_MODEL_ID}@${BASE_MODEL_VERSION}`,
    siteCount: BASE_SITE_COUNT,
    sites,
  };
}

function compareEdges(left, right) {
  return (
    left.source - right.source ||
    left.target - right.target ||
    left.kind.localeCompare(right.kind)
  );
}

/** Exact Cartesian-product edge list G_101 square C_3. */
export function buildLiftedGraph() {
  const basis = selectEtq101Basis();
  const adjacency = buildRootAdjacency(basis);
  const edges = [];

  for (let fibreLabel = 0; fibreLabel < FIBRE_DIMENSION; fibreLabel += 1) {
    for (let left = 0; left < BASE_SITE_COUNT; left += 1) {
      for (let right = left + 1; right < BASE_SITE_COUNT; right += 1) {
        if (adjacency[left][right] === 1) {
          edges.push({
            source: tensorIndex(left, fibreLabel),
            target: tensorIndex(right, fibreLabel),
            kind: "base-e8-edge",
          });
        }
      }
    }
  }

  const fibreEdges = [[0, 1], [0, 2], [1, 2]];
  for (let siteIndex = 0; siteIndex < BASE_SITE_COUNT; siteIndex += 1) {
    for (const [leftFibre, rightFibre] of fibreEdges) {
      edges.push({
        source: tensorIndex(siteIndex, leftFibre),
        target: tensorIndex(siteIndex, rightFibre),
        kind: "fibre-c3-edge",
      });
    }
  }
  edges.sort(compareEdges);

  const edgeKeys = new Set();
  const neighbours = Array.from({ length: EVENT_COUNT }, () => []);
  for (const edge of edges) {
    if (edge.source >= edge.target) {
      throw new Error("lifted graph edges must use source < target");
    }
    const key = `${edge.source},${edge.target}`;
    if (edgeKeys.has(key)) throw new Error("duplicate lifted graph edge");
    edgeKeys.add(key);
    neighbours[edge.source].push(edge.target);
    neighbours[edge.target].push(edge.source);
  }
  const degrees = neighbours.map((row) => row.length);
  const visited = new Set([0]);
  const queue = [0];
  for (let head = 0; head < queue.length; head += 1) {
    const current = queue[head];
    for (const neighbour of neighbours[current]) {
      if (!visited.has(neighbour)) {
        visited.add(neighbour);
        queue.push(neighbour);
      }
    }
  }
  const baseEdgeCount = adjacency.flat().reduce((sum, value) => sum + value, 0) / 2;

  return {
    schema: LIFTED_GRAPH_SCHEMA_ID,
    modelId: MODEL_ID,
    modelVersion: MODEL_VERSION,
    construction: "cartesian-product-G101-square-C3",
    vertexCount: EVENT_COUNT,
    edgeCount: edges.length,
    baseLayerCount: FIBRE_DIMENSION,
    baseEdgesPerLayer: baseEdgeCount,
    fibreEdgesPerSite: fibreEdges.length,
    minimumDegree: Math.min(...degrees),
    maximumDegree: Math.max(...degrees),
    connected: visited.size === EVENT_COUNT,
    degrees,
    edges,
  };
}

export function buildCanonicalEvents() {
  const events = [];
  for (let sequenceIndex = 0; sequenceIndex < EVENT_COUNT; sequenceIndex += 1) {
    const address = eventAddress(sequenceIndex);
    const transition = exactMonomialStep(address.siteIndex, address.fibreLabel);
    const nextSequenceIndex = (sequenceIndex + 1) % EVENT_COUNT;
    const expectedNextAddress = eventAddress(nextSequenceIndex);
    if (transition.targetTensorIndex !== expectedNextAddress.tensorIndex) {
      throw new Error("event traversal and monomial support disagree");
    }
    events.push({
      sequenceIndex,
      tensorIndex: address.tensorIndex,
      siteIndex: address.siteIndex,
      fibreLabel: address.fibreLabel,
      transitionPhaseGaussianExponent: transition.phaseGaussianExponent,
      transitionPhaseSymbol: transition.phaseSymbol,
      nextSequenceIndex,
      nextTensorIndex: transition.targetTensorIndex,
    });
  }
  return events;
}

export function buildEventDocument() {
  const siteRegistry = buildSiteRegistry();
  const liftedGraph = buildLiftedGraph();
  const events = buildCanonicalEvents();
  return {
    schema: EVENT_SCHEMA_ID,
    modelId: MODEL_ID,
    modelVersion: MODEL_VERSION,
    profile: PROFILE_ID,
    lineage: { ...BASE_RELEASE },
    stateSpace: {
      expression: "H_303=H_101 tensor C^3",
      basisSemantics: "101-root-indexed-sites-times-independent-ternary-fibre",
      tensorBasisOrder: "tensorIndex=3*siteIndex+fibreLabel",
      dimensions: { baseSites: BASE_SITE_COUNT, fibre: FIBRE_DIMENSION, total: EVENT_COUNT },
    },
    exactStep: {
      expression: "F_303=R_101 tensor (X_3 exp(-i*pi*D_3/2))",
      supportAction: "(j,a)->(j+1 mod 101,a+1 mod 3)",
      phaseConvention: "exp(-i*pi*d_a/2)=i^g_a",
      sclStencil: [...SCL_STENCIL],
      gaussianExponents: [...FIBRE_PHASE_GAUSSIAN_EXPONENTS],
      gaussianPhaseSymbols: [...FIBRE_PHASE_GAUSSIAN_SYMBOLS],
      gaussianUnitSymbolLookup: [...GAUSSIAN_UNIT_SYMBOLS],
      supportCycleLength: EVENT_COUNT,
      operatorOrder: EVENT_COUNT,
    },
    graphContext: {
      construction: liftedGraph.construction,
      vertexCount: liftedGraph.vertexCount,
      edgeCount: liftedGraph.edgeCount,
      minimumDegree: liftedGraph.minimumDegree,
      maximumDegree: liftedGraph.maximumDegree,
      connected: liftedGraph.connected,
      eventTraversalIsGraphWalk: false,
    },
    siteRegistry: siteRegistry.sites,
    events,
    claimBoundary:
      "An exact 303-state tensor extension and deterministic event traversal of the preserved ETQ-101 v2 selected basis. The 303 states are root-indexed tensor states, not 303 distinct E8 roots. Receiver encodings are deterministic conventions and no acoustic, physical, statistical, or perceptual claim is made.",
  };
}

/** Exhaustive finite checks used by tests and the verifier. */
export function exactInvariantSummary() {
  const sites = buildSiteRegistry().sites;
  const graph = buildLiftedGraph();
  const events = buildCanonicalEvents();
  const visitedTensorIndices = new Set(events.map((event) => event.tensorIndex));
  const visitedAddresses = new Set(events.map((event) => `${event.siteIndex},${event.fibreLabel}`));
  const phaseAfterThree = exactMonomialPower(0, 0, 3);
  const phaseAfter303 = exactMonomialPower(0, 0, EVENT_COUNT);

  return {
    dimensions: { baseSites: BASE_SITE_COUNT, fibre: FIBRE_DIMENSION, total: EVENT_COUNT },
    arithmetic: {
      gcd101And3: gcd(BASE_SITE_COUNT, FIBRE_DIMENSION),
      lcm101And3: lcm(BASE_SITE_COUNT, FIBRE_DIMENSION),
    },
    eventTraversal: {
      entries: events.length,
      uniqueTensorIndices: visitedTensorIndices.size,
      uniqueAddresses: visitedAddresses.size,
      closesAfter303: events.at(-1).nextTensorIndex === events[0].tensorIndex,
    },
    monomialStep: {
      phaseExponents: [...FIBRE_PHASE_GAUSSIAN_EXPONENTS],
      phaseSymbols: [...FIBRE_PHASE_GAUSSIAN_SYMBOLS],
      threeStepAccumulatedPhaseGaussianExponent: phaseAfterThree.accumulatedPhaseGaussianExponent,
      after303: phaseAfter303,
    },
    graph: {
      vertices: graph.vertexCount,
      edges: graph.edgeCount,
      minimumDegree: Math.min(...sites.map((site) => site.liftedDegree)),
      maximumDegree: Math.max(...sites.map((site) => site.liftedDegree)),
      connected: graph.connected,
    },
  };
}
