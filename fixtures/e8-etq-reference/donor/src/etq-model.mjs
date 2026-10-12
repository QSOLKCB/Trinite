// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Trent Slade / QSOL-IMC.

/**
 * ETQ-101 reference construction.
 *
 * The implementation uses doubled integer E8-root coordinates. If r is a
 * conventional root, this module stores 2r, so every norm squared is exactly
 * eight and the adjacency condition r_a . r_b = 1 is the integer test
 * (2r_a) . (2r_b) = 4.
 *
 * No external packages are required. The module is deterministic and is safe
 * to import in either a modern browser or Node.js.
 */

export const MODEL_ID = "ETQ-101";
export const MODEL_VERSION = "2.0.0";
export const QUTRIT_DIMENSION = 3;
export const SELECTED_FIXED_ROOTS = 2;
export const SELECTED_TRIALITY_ORBITS = 33;
export const QUTRIT_SUBSPACE_DIMENSION =
  SELECTED_TRIALITY_ORBITS * QUTRIT_DIMENSION;
export const OUROBOROS_DIMENSION =
  SELECTED_FIXED_ROOTS + QUTRIT_SUBSPACE_DIMENSION;
export const PHASE_THETA_RAD = Math.PI / 2;
// Authored v2 step denominator. Its current numerical coincidence with the
// basis size times qutrit order does not make it E8- or selector-derived.
export const DIMENSIONLESS_STEP_DENOMINATOR = 303;
export const DIMENSIONLESS_STEP_DELTA =
  (2 * Math.PI) / DIMENSIONLESS_STEP_DENOMINATOR;
export const CANONICAL_GENERATOR_WEIGHTS = Object.freeze({
  e8RootGraph: 11 / 20,
  qutritHopping: 1 / 4,
  graphDegreePotential: 1 / 5,
  sclStatic: 0,
  qutritNumber: 0,
  ouroborosRing: 0,
});
export const MIDI_NOTE_MINIMUM = 0;
export const MIDI_NOTE_MAXIMUM = 127;
export const MIDI_NOTE_COUNT = MIDI_NOTE_MAXIMUM - MIDI_NOTE_MINIMUM + 1;
export const TERNARY_REGISTER_LANES = Object.freeze([
  "low",
  "mid",
  "high",
]);
export const QUTRIT_ROOT_OF_UNITY = Object.freeze({
  re: -0.5,
  im: Math.sqrt(3) / 2,
});
export const SCL_STENCIL = Object.freeze([1, -2, 1]);

function midiRange(minimum, maximum) {
  return Object.freeze([minimum, maximum]);
}

function createTernaryMidiLayout() {
  if (SELECTED_FIXED_ROOTS !== 2) {
    throw new RangeError("the ternary MIDI layout requires two fixed bookends");
  }
  if (TERNARY_REGISTER_LANES.length !== QUTRIT_DIMENSION) {
    throw new RangeError("register-lane count must equal qutrit dimension");
  }
  if (OUROBOROS_DIMENSION > MIDI_NOTE_COUNT) {
    throw new RangeError("the ETQ basis does not fit in the MIDI note domain");
  }

  const unusedNoteCount = MIDI_NOTE_COUNT - OUROBOROS_DIMENSION;
  const windowMinimum =
    MIDI_NOTE_MINIMUM + Math.floor(unusedNoteCount / 2);
  const windowMaximum = windowMinimum + OUROBOROS_DIMENSION - 1;
  const firstQutritNote = windowMinimum + 1;
  const lastQutritNote = windowMaximum - 1;
  const laneStride = SELECTED_TRIALITY_ORBITS;
  const centeredQutritMidpoint = (QUTRIT_DIMENSION - 1) / 2;
  const lanes = Object.freeze(
    TERNARY_REGISTER_LANES.map((name, qutritLabel) => {
      const minimum = firstQutritNote + laneStride * qutritLabel;
      return Object.freeze({
        qutritLabel,
        centeredQutritValue: qutritLabel - centeredQutritMidpoint,
        sclCurvature: SCL_STENCIL[qutritLabel],
        name,
        midiNoteRange: midiRange(minimum, minimum + laneStride - 1),
      });
    }),
  );
  const fixedSingletNotes = midiRange(windowMinimum, windowMaximum);

  return Object.freeze({
    noteCount: MIDI_NOTE_COUNT,
    stateCount: OUROBOROS_DIMENSION,
    qutritStateCount: QUTRIT_SUBSPACE_DIMENSION,
    laneStride,
    trialityShift: laneStride,
    windowMinimum,
    windowMaximum,
    firstQutritNote,
    lastQutritNote,
    midiNoteDomain: midiRange(MIDI_NOTE_MINIMUM, MIDI_NOTE_MAXIMUM),
    occupiedNoteRange: midiRange(windowMinimum, windowMaximum),
    qutritNoteRange: midiRange(firstQutritNote, lastQutritNote),
    fixedSingletNotes,
    lanes,
    basisMap:
      `fixed0->${windowMinimum};fixed1->${windowMaximum};` +
      `(m,q)->${firstQutritNote}+${laneStride}*q+m`,
    inverseMap:
      `${windowMinimum}->fixed0;${windowMaximum}->fixed1;otherwise ` +
      `q=floor((note-${firstQutritNote})/${laneStride}),` +
      `m=(note-${firstQutritNote})%${laneStride}`,
  });
}

/** Single derived source of truth for the canonical ternary MIDI window. */
export const TERNARY_MIDI_LAYOUT = createTernaryMidiLayout();

// Twice the D4 triality matrix. Applying this matrix and dividing by two is
// exact on the doubled-coordinate E8 roots used by this module.
export const D4_TRIALITY_NUMERATOR = Object.freeze([
  Object.freeze([1, 1, 1, 1]),
  Object.freeze([1, 1, -1, -1]),
  Object.freeze([1, -1, 1, -1]),
  Object.freeze([-1, 1, 1, -1]),
]);

export function lexicographicCompare(left, right) {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    if (left[index] !== right[index]) {
      return left[index] - right[index];
    }
  }
  return left.length - right.length;
}

export function vectorKey(vector) {
  return vector.join(",");
}

export function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * right[index];
  }
  return total;
}

function populationCount(integer) {
  let value = integer;
  let count = 0;
  while (value !== 0) {
    count += value & 1;
    value >>>= 1;
  }
  return count;
}

/** Generate the standard 240 E8 roots in doubled integer coordinates. */
export function generateE8Roots() {
  const roots = [];

  // 112 roots of the form (+/-2, +/-2, 0, ..., 0), representing
  // conventional roots (+/-1, +/-1, 0, ..., 0).
  for (let first = 0; first < 8; first += 1) {
    for (let second = first + 1; second < 8; second += 1) {
      for (const firstSign of [-2, 2]) {
        for (const secondSign of [-2, 2]) {
          const root = Array(8).fill(0);
          root[first] = firstSign;
          root[second] = secondSign;
          roots.push(root);
        }
      }
    }
  }

  // 128 roots (+/-1, ..., +/-1), representing half-integer roots, with an
  // even number of negative entries.
  for (let mask = 0; mask < 256; mask += 1) {
    if (populationCount(mask) % 2 !== 0) {
      continue;
    }
    const root = [];
    for (let coordinate = 0; coordinate < 8; coordinate += 1) {
      root.push((mask & (1 << coordinate)) === 0 ? 1 : -1);
    }
    roots.push(root);
  }

  roots.sort(lexicographicCompare);
  return roots;
}

function applyTrialityBlock(block) {
  return D4_TRIALITY_NUMERATOR.map((row) => {
    const numerator = dot(row, block);
    if (numerator % 2 !== 0) {
      throw new Error("Triality left the doubled-coordinate E8 lattice");
    }
    return numerator / 2;
  });
}

/** Apply the embedded D4 x D4 order-three triality to an E8 root. */
export function applyTriality(root) {
  if (!Array.isArray(root) || root.length !== 8) {
    throw new TypeError("An E8 root must contain exactly eight coordinates");
  }
  return [
    ...applyTrialityBlock(root.slice(0, 4)),
    ...applyTrialityBlock(root.slice(4, 8)),
  ];
}

export function trialityOrbit(root) {
  const first = [...root];
  const second = applyTriality(first);
  const third = applyTriality(second);
  const closure = applyTriality(third);
  if (vectorKey(closure) !== vectorKey(first)) {
    throw new Error("Embedded triality did not close after three applications");
  }
  return vectorKey(first) === vectorKey(second)
    ? [first]
    : [first, second, third];
}

/**
 * Partition the 240 roots into the 12 fixed roots and 76 three-cycles of the
 * embedded triality. Three-cycle representatives are lexicographically least.
 */
export function classifyTrialityOrbits(roots = generateE8Roots()) {
  const rootKeys = new Set(roots.map(vectorKey));
  const seen = new Set();
  const fixed = [];
  const triples = [];

  for (const root of [...roots].sort(lexicographicCompare)) {
    if (seen.has(vectorKey(root))) {
      continue;
    }
    const rawOrbit = trialityOrbit(root);
    for (const member of rawOrbit) {
      const key = vectorKey(member);
      if (!rootKeys.has(key)) {
        throw new Error("Embedded triality did not preserve the E8 root set");
      }
      seen.add(key);
    }

    if (rawOrbit.length === 1) {
      fixed.push(rawOrbit[0]);
      continue;
    }

    const representative = [...rawOrbit].sort(lexicographicCompare)[0];
    triples.push([
      representative,
      applyTriality(representative),
      applyTriality(applyTriality(representative)),
    ]);
  }

  fixed.sort(lexicographicCompare);
  triples.sort((left, right) => lexicographicCompare(left[0], right[0]));
  return { fixed, triples };
}

/**
 * Canonical ETQ-101 ordered root labels: two fixed roots followed by 33
 * complete triality orbits, each ordered (r, tau r, tau^2 r). The returned
 * roots label an orthonormal basis of ell^2(S_101); they are not a vector
 * basis of R^8.
 */
export function selectEtq101Basis(roots = generateE8Roots()) {
  const { fixed, triples } = classifyTrialityOrbits(roots);
  if (fixed.length !== 12 || triples.length !== 76) {
    throw new Error(
      `Unexpected triality decomposition ${fixed.length} + 3*${triples.length}`,
    );
  }

  const basis = fixed.slice(0, SELECTED_FIXED_ROOTS).map((root) => [...root]);
  for (const orbit of triples.slice(0, SELECTED_TRIALITY_ORBITS)) {
    for (const root of orbit) {
      basis.push([...root]);
    }
  }

  if (basis.length !== OUROBOROS_DIMENSION) {
    throw new Error(`Canonical selector produced ${basis.length} basis states`);
  }
  return basis;
}

/** Return p such that U_tau |j> = |p[j]>. */
export function trialityPermutation() {
  const permutation = Array.from(
    { length: OUROBOROS_DIMENSION },
    (_, index) => index,
  );
  for (let orbit = 0; orbit < SELECTED_TRIALITY_ORBITS; orbit += 1) {
    const offset = SELECTED_FIXED_ROOTS + QUTRIT_DIMENSION * orbit;
    for (let label = 0; label < QUTRIT_DIMENSION; label += 1) {
      permutation[offset + label] =
        offset + ((label + 1) % QUTRIT_DIMENSION);
    }
  }
  return permutation;
}

/**
 * Apply an integer power of a finite permutation to one index.
 *
 * The orbit length is derived from the validated permutation and selected
 * index, so this helper is correct for fixed points and cycles of any length.
 */
export function applyPermutationIndex(permutation, index, power = 1) {
  if (!Array.isArray(permutation) || permutation.length === 0) {
    throw new TypeError("permutation must be a non-empty array");
  }
  if (!Number.isSafeInteger(index) || index < 0 || index >= permutation.length) {
    throw new RangeError(`index ${index} is outside the permutation domain`);
  }
  if (!Number.isSafeInteger(power)) {
    throw new TypeError("power must be a safe integer");
  }

  const targets = new Set();
  for (const target of permutation) {
    if (
      !Number.isSafeInteger(target) ||
      target < 0 ||
      target >= permutation.length
    ) {
      throw new RangeError("permutation targets must stay within its domain");
    }
    if (targets.has(target)) {
      throw new RangeError("permutation must be a bijection");
    }
    targets.add(target);
  }

  let orbitLength = 1;
  let orbitIndex = permutation[index];
  while (orbitIndex !== index) {
    orbitIndex = permutation[orbitIndex];
    orbitLength += 1;
    if (orbitLength > permutation.length) {
      throw new RangeError("permutation orbit did not close");
    }
  }

  const normalizedPower = ((power % orbitLength) + orbitLength) % orbitLength;
  let result = index;
  for (let step = 0; step < normalizedPower; step += 1) {
    result = permutation[result];
  }
  return result;
}

/** E8 root-polytope adjacency restricted to the canonical 101 roots. */
export function buildRootAdjacency(basis = selectEtq101Basis()) {
  const adjacency = Array.from(
    { length: basis.length },
    () => Array(basis.length).fill(0),
  );
  for (let left = 0; left < basis.length; left += 1) {
    for (let right = left + 1; right < basis.length; right += 1) {
      if (dot(basis[left], basis[right]) === 4) {
        adjacency[left][right] = 1;
        adjacency[right][left] = 1;
      }
    }
  }
  return adjacency;
}

export function graphDegrees(adjacency) {
  if (!Array.isArray(adjacency) || adjacency.length === 0) {
    throw new TypeError("adjacency must be a non-empty matrix");
  }
  const dimension = adjacency.length;
  return adjacency.map((row) => {
    if (!Array.isArray(row) || row.length !== dimension) {
      throw new RangeError("adjacency must be square");
    }
    return row.reduce((sum, value) => sum + value, 0);
  });
}

export function buildGraphLaplacian(adjacency) {
  const degrees = graphDegrees(adjacency);
  return adjacency.map((row, rowIndex) => {
    return row.map((value, columnIndex) =>
      rowIndex === columnIndex ? degrees[rowIndex] : -value,
    );
  });
}

export function graphSummary(adjacency) {
  const degrees = graphDegrees(adjacency);
  const edgeCount = degrees.reduce((sum, degree) => sum + degree, 0) / 2;

  const visited = new Set([0]);
  const queue = [0];
  while (queue.length > 0) {
    const current = queue.shift();
    for (let neighbour = 0; neighbour < adjacency.length; neighbour += 1) {
      if (adjacency[current][neighbour] === 1 && !visited.has(neighbour)) {
        visited.add(neighbour);
        queue.push(neighbour);
      }
    }
  }

  return {
    vertices: adjacency.length,
    edges: edgeCount,
    minimumDegree: Math.min(...degrees),
    maximumDegree: Math.max(...degrees),
    connected: visited.size === adjacency.length,
  };
}

/**
 * Exact centered degree potential for the selected root graph.
 *
 * For graph dimension n, degree sum S, and degrees d_j, the returned integer
 * numerators are n*d_j-S. Dividing by their maximum absolute value produces a
 * trace-zero diagonal with operator norm one. The integer receipt is the
 * canonical identity; `diagonal` is its binary64 realization.
 */
export function graphDegreePotential(
  adjacency = buildRootAdjacency(selectEtq101Basis()),
) {
  if (!Array.isArray(adjacency) || adjacency.length !== OUROBOROS_DIMENSION) {
    throw new RangeError(
      `adjacency must contain ${OUROBOROS_DIMENSION} rows`,
    );
  }

  const dimension = adjacency.length;
  const degrees = graphDegrees(adjacency);
  const degreeSum = degrees.reduce((sum, value) => sum + value, 0);
  const numerators = degrees.map((degree) => dimension * degree - degreeSum);
  const denominator = Math.max(...numerators.map((value) => Math.abs(value)));
  if (denominator === 0) {
    throw new RangeError("degree potential is undefined for a regular graph");
  }

  const degreeCounts = new Map();
  for (const degree of degrees) {
    degreeCounts.set(degree, (degreeCounts.get(degree) ?? 0) + 1);
  }
  const traceNumerator = numerators.reduce((sum, value) => sum + value, 0);
  const maximumAbsoluteNumerator = Math.max(
    ...numerators.map((value) => Math.abs(value)),
  );

  return {
    degrees,
    degreeSum,
    meanDegree: { numerator: degreeSum, denominator: dimension },
    numerators,
    normalizationDenominator: denominator,
    traceNumerator,
    maximumAbsoluteNumerator,
    degreeDistribution: [...degreeCounts.entries()].sort(
      ([left], [right]) => left - right,
    ),
    expression:
      `V_degree[j,j]=(${dimension}*degree_j-${degreeSum})/${denominator}`,
    diagonal: numerators.map((value) => value / denominator),
  };
}

/** Normalized qutrit-cycle Laplacian, zero on the two fixed singlets. */
export function qutritHoppingLaplacian() {
  const matrix = Array.from({ length: OUROBOROS_DIMENSION }, () =>
    Array(OUROBOROS_DIMENSION).fill(0),
  );
  for (let orbitIndex = 0; orbitIndex < SELECTED_TRIALITY_ORBITS; orbitIndex += 1) {
    const start = SELECTED_FIXED_ROOTS + QUTRIT_DIMENSION * orbitIndex;
    for (let row = 0; row < QUTRIT_DIMENSION; row += 1) {
      for (let column = 0; column < QUTRIT_DIMENSION; column += 1) {
        matrix[start + row][start + column] =
          row === column
            ? (QUTRIT_DIMENSION - 1) / QUTRIT_DIMENSION
            : -1 / QUTRIT_DIMENSION;
      }
    }
  }
  return matrix;
}

/**
 * Canonical dimensionless v2 generator.
 *
 * K=(11/20)*L_E/(2*d_max)+(1/4)*L_Q+(1/5)*V_degree.
 * The function constructs only K; it does not assign a physical clock or
 * implement the matrix exponential used by the specified Floquet profile.
 */
export function canonicalGeneratorMatrix(
  adjacency = buildRootAdjacency(selectEtq101Basis()),
) {
  const potential = graphDegreePotential(adjacency);
  const graphLaplacian = buildGraphLaplacian(adjacency);
  const maximumDegree = Math.max(...potential.degrees);
  const qutritLaplacian = qutritHoppingLaplacian();
  const weights = CANONICAL_GENERATOR_WEIGHTS;

  return graphLaplacian.map((row, rowIndex) =>
    row.map((value, columnIndex) => {
      const graphTerm =
        weights.e8RootGraph * value / (2 * maximumDegree);
      const qutritTerm =
        weights.qutritHopping * qutritLaplacian[rowIndex][columnIndex];
      const potentialTerm =
        rowIndex === columnIndex
          ? weights.graphDegreePotential * potential.diagonal[rowIndex]
          : 0;
      return graphTerm + qutritTerm + potentialTerm;
    }),
  );
}

/** diag(0,0, 1,-2,1, ..., 1,-2,1). */
export function curvatureDiagonal() {
  const diagonal = Array(SELECTED_FIXED_ROOTS).fill(0);
  for (let orbit = 0; orbit < SELECTED_TRIALITY_ORBITS; orbit += 1) {
    diagonal.push(...SCL_STENCIL);
  }
  return diagonal;
}

/** diag(0,0, 0,1,2, ..., 0,1,2). */
export function qutritNumberDiagonal() {
  const diagonal = Array(SELECTED_FIXED_ROOTS).fill(0);
  const labels = Array.from(
    { length: QUTRIT_DIMENSION },
    (_, qutritLabel) => qutritLabel,
  );
  for (let orbit = 0; orbit < SELECTED_TRIALITY_ORBITS; orbit += 1) {
    diagonal.push(...labels);
  }
  return diagonal;
}

/** diag(0,0, -1,0,1, ..., -1,0,1). */
export function centeredQutritNumberDiagonal() {
  const diagonal = Array(SELECTED_FIXED_ROOTS).fill(0);
  const midpoint = (QUTRIT_DIMENSION - 1) / 2;
  const centeredLabels = Array.from(
    { length: QUTRIT_DIMENSION },
    (_, qutritLabel) => qutritLabel - midpoint,
  );
  for (let orbit = 0; orbit < SELECTED_TRIALITY_ORBITS; orbit += 1) {
    diagonal.push(...centeredLabels);
  }
  return diagonal;
}

/** Return the MIDI note derived for one selected orbit/qutrit state. */
export function midiNoteForTernaryState(orbitIndex, qutritLabel) {
  if (
    !Number.isSafeInteger(orbitIndex) ||
    orbitIndex < 0 ||
    orbitIndex >= SELECTED_TRIALITY_ORBITS
  ) {
    throw new RangeError("orbitIndex lies outside the canonical selector");
  }
  if (
    !Number.isSafeInteger(qutritLabel) ||
    qutritLabel < 0 ||
    qutritLabel >= QUTRIT_DIMENSION
  ) {
    throw new RangeError("qutritLabel lies outside the qutrit domain");
  }
  return (
    TERNARY_MIDI_LAYOUT.firstQutritNote +
    TERNARY_MIDI_LAYOUT.laneStride * qutritLabel +
    orbitIndex
  );
}

/**
 * Canonical symbolic MIDI codebook for the selected ETQ decomposition.
 *
 * The basis-sized window is maximally centered in MIDI's 7-bit note domain,
 * with the lower start selected when the two possible margins differ by one.
 * The fixed singlets are codebook bookends. Each qutrit label receives one
 * complete register lane; orbit index supplies the offset within it. This is
 * a symbolic identity map and declares no acoustic frequency.
 */
export function buildTernaryMidiCodebook() {
  const [lowFixedNote, highFixedNote] =
    TERNARY_MIDI_LAYOUT.fixedSingletNotes;
  const entries = [
    {
      basisIndex: 0,
      stateType: "fixed-singlet",
      fixedIndex: 0,
      orbitIndex: null,
      qutritLabel: null,
      lane: "fixed-low-bookend",
      midiNote: lowFixedNote,
    },
    {
      basisIndex: 1,
      stateType: "fixed-singlet",
      fixedIndex: 1,
      orbitIndex: null,
      qutritLabel: null,
      lane: "fixed-high-bookend",
      midiNote: highFixedNote,
    },
  ];

  for (let orbitIndex = 0; orbitIndex < SELECTED_TRIALITY_ORBITS; orbitIndex += 1) {
    for (let qutritLabel = 0; qutritLabel < QUTRIT_DIMENSION; qutritLabel += 1) {
      entries.push({
        basisIndex:
          SELECTED_FIXED_ROOTS + QUTRIT_DIMENSION * orbitIndex + qutritLabel,
        stateType: "qutrit-orbit-state",
        fixedIndex: null,
        orbitIndex,
        qutritLabel,
        lane: TERNARY_REGISTER_LANES[qutritLabel],
        midiNote: midiNoteForTernaryState(orbitIndex, qutritLabel),
      });
    }
  }

  entries.sort((left, right) => left.basisIndex - right.basisIndex);
  return entries.map((entry) => Object.freeze(entry));
}

/** Invert a note in the canonical ternary codebook, or return null. */
export function basisIndexFromMidiNote(note) {
  if (!Number.isSafeInteger(note)) {
    throw new TypeError("MIDI note must be a safe integer");
  }
  const [lowFixedNote, highFixedNote] =
    TERNARY_MIDI_LAYOUT.fixedSingletNotes;
  if (note === lowFixedNote) {
    return 0;
  }
  if (note === highFixedNote) {
    return 1;
  }

  const offset = note - TERNARY_MIDI_LAYOUT.firstQutritNote;
  if (offset < 0 || offset >= TERNARY_MIDI_LAYOUT.qutritStateCount) {
    return null;
  }
  const qutritLabel = Math.floor(offset / TERNARY_MIDI_LAYOUT.laneStride);
  const orbitIndex = offset % TERNARY_MIDI_LAYOUT.laneStride;
  return SELECTED_FIXED_ROOTS + QUTRIT_DIMENSION * orbitIndex + qutritLabel;
}

/** Diagonal of U_tau^power D U_tau^-power. */
export function permuteDiagonal(
  diagonal,
  permutation = trialityPermutation(),
  power = 1,
) {
  const result = Array(diagonal.length).fill(0);
  for (let index = 0; index < diagonal.length; index += 1) {
    result[applyPermutationIndex(permutation, index, power)] = diagonal[index];
  }
  return result;
}

export function phaseKickDiagonal(theta = PHASE_THETA_RAD) {
  return curvatureDiagonal().map((value) => {
    const phase = -theta * value;
    return { re: Math.cos(phase), im: Math.sin(phase) };
  });
}

/**
 * Pure, normalized ETQ-101 ternary/SCL state.
 *
 * Fixed singlets have zero phase. A qutrit state (m,q) carries the qutrit
 * character 2*pi*q/3 followed by the declared SCL phase kick -theta*d_q.
 * "Ouroboros" remains a declared model recipe name, not a standard state
 * class.
 */
export function ternaryOuroborosState({ theta = PHASE_THETA_RAD } = {}) {
  const curvature = curvatureDiagonal();
  const normalization = 1 / Math.sqrt(OUROBOROS_DIMENSION);
  return curvature.map((value, index) => {
    const ternaryPhase =
      index < SELECTED_FIXED_ROOTS
        ? 0
        : (2 * Math.PI * ((index - SELECTED_FIXED_ROOTS) % QUTRIT_DIMENSION)) /
          QUTRIT_DIMENSION;
    const phase = ternaryPhase - theta * value;
    return {
      re: normalization * Math.cos(phase),
      im: normalization * Math.sin(phase),
    };
  });
}

export function complex(re = 0, im = 0) {
  return { re, im };
}

export function complexAdd(left, right) {
  return complex(left.re + right.re, left.im + right.im);
}

export function complexMultiply(left, right) {
  return complex(
    left.re * right.re - left.im * right.im,
    left.re * right.im + left.im * right.re,
  );
}

export function complexScale(value, scalar) {
  return complex(value.re * scalar, value.im * scalar);
}

function complexMatrixShape(matrix, label) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new TypeError(`${label} must be a non-empty matrix`);
  }
  if (!Array.isArray(matrix[0]) || matrix[0].length === 0) {
    throw new TypeError(`${label} must contain non-empty rows`);
  }

  const columns = matrix[0].length;
  for (const row of matrix) {
    if (!Array.isArray(row) || row.length !== columns) {
      throw new RangeError(`${label} must be rectangular`);
    }
  }
  return { rows: matrix.length, columns };
}

export function complexMatrixMultiply(left, right) {
  const leftShape = complexMatrixShape(left, "left matrix");
  const rightShape = complexMatrixShape(right, "right matrix");
  if (leftShape.columns !== rightShape.rows) {
    throw new RangeError(
      `incompatible matrix dimensions ${leftShape.rows}x${leftShape.columns} ` +
        `and ${rightShape.rows}x${rightShape.columns}`,
    );
  }

  const rows = leftShape.rows;
  const columns = rightShape.columns;
  const shared = leftShape.columns;
  const output = Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => complex()),
  );
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      let value = complex();
      for (let index = 0; index < shared; index += 1) {
        value = complexAdd(
          value,
          complexMultiply(left[row][index], right[index][column]),
        );
      }
      output[row][column] = value;
    }
  }
  return output;
}

export function complexIdentity(dimension) {
  return Array.from({ length: dimension }, (_, row) =>
    Array.from({ length: dimension }, (_, column) =>
      complex(row === column ? 1 : 0, 0),
    ),
  );
}

export function complexMatrixPower(matrix, exponent) {
  let result = complexIdentity(matrix.length);
  let factor = matrix;
  let remaining = exponent;
  while (remaining > 0) {
    if (remaining % 2 === 1) {
      result = complexMatrixMultiply(result, factor);
    }
    remaining = Math.floor(remaining / 2);
    if (remaining > 0) {
      factor = complexMatrixMultiply(factor, factor);
    }
  }
  return result;
}

export function qutritOperators() {
  const zero = () => complex(0, 0);
  const X = Array.from({ length: QUTRIT_DIMENSION }, () =>
    Array.from({ length: QUTRIT_DIMENSION }, zero),
  );
  const Z = Array.from({ length: QUTRIT_DIMENSION }, () =>
    Array.from({ length: QUTRIT_DIMENSION }, zero),
  );

  for (let source = 0; source < QUTRIT_DIMENSION; source += 1) {
    X[(source + 1) % QUTRIT_DIMENSION][source] = complex(1, 0);
    const phase = (2 * Math.PI * source) / QUTRIT_DIMENSION;
    Z[source][source] = complex(Math.cos(phase), Math.sin(phase));
  }

  const N = Array.from({ length: QUTRIT_DIMENSION }, (_, row) =>
    Array.from({ length: QUTRIT_DIMENSION }, (_, column) =>
      complex(row === column ? row : 0),
    ),
  );
  const D = Array.from({ length: QUTRIT_DIMENSION }, (_, row) =>
    Array.from({ length: QUTRIT_DIMENSION }, (_, column) =>
      complex(row === column ? SCL_STENCIL[row] : 0),
    ),
  );
  return { X, Z, N, D };
}

/** Local F_3 = X exp(-i theta D), which has exact order three. */
export function twistedQutritStep(theta = PHASE_THETA_RAD) {
  const { X } = qutritOperators();
  if (SCL_STENCIL.length !== QUTRIT_DIMENSION) {
    throw new Error("SCL_STENCIL must define exactly one value per qutrit state");
  }
  const phase = Array.from({ length: QUTRIT_DIMENSION }, (_, row) =>
    Array.from({ length: QUTRIT_DIMENSION }, (_, column) => {
      if (row !== column) {
        return complex();
      }
      const angle = -theta * SCL_STENCIL[row];
      return complex(Math.cos(angle), Math.sin(angle));
    }),
  );
  return complexMatrixMultiply(X, phase);
}

export function maxComplexMatrixDifference(left, right) {
  let maximum = 0;
  for (let row = 0; row < left.length; row += 1) {
    for (let column = 0; column < left[row].length; column += 1) {
      maximum = Math.max(
        maximum,
        Math.hypot(
          left[row][column].re - right[row][column].re,
          left[row][column].im - right[row][column].im,
        ),
      );
    }
  }
  return maximum;
}

export function canonicalModelSummary() {
  const roots = generateE8Roots();
  const { fixed, triples } = classifyTrialityOrbits(roots);
  const basis = selectEtq101Basis(roots);
  const adjacency = buildRootAdjacency(basis);
  const degreePotential = graphDegreePotential(adjacency);
  const midiCodebook = buildTernaryMidiCodebook();
  return {
    modelId: MODEL_ID,
    modelVersion: MODEL_VERSION,
    dimensions: {
      total: OUROBOROS_DIMENSION,
      fixedSinglets: SELECTED_FIXED_ROOTS,
      qutritOrbits: SELECTED_TRIALITY_ORBITS,
      qutritSubspace: QUTRIT_SUBSPACE_DIMENSION,
    },
    constants: {
      phaseThetaRad: PHASE_THETA_RAD,
      qutritOrder: QUTRIT_DIMENSION,
      midiNoteDomain: [...TERNARY_MIDI_LAYOUT.midiNoteDomain],
    },
    dynamics: {
      weights: { ...CANONICAL_GENERATOR_WEIGHTS },
      dimensionlessStepDelta: DIMENSIONLESS_STEP_DELTA,
    },
    e8: {
      rootCount: roots.length,
      doubledRootNormSquared: dot(roots[0], roots[0]),
      trialityFixedRoots: fixed.length,
      trialityThreeCycles: triples.length,
    },
    selectedGraph: graphSummary(adjacency),
    degreePotential: {
      degreeSum: degreePotential.degreeSum,
      meanDegree: degreePotential.meanDegree,
      normalizationDenominator: degreePotential.normalizationDenominator,
      traceNumerator: degreePotential.traceNumerator,
      maximumAbsoluteNumerator: degreePotential.maximumAbsoluteNumerator,
      degreeDistribution: degreePotential.degreeDistribution,
    },
    midiCodebook: {
      entries: midiCodebook.length,
      uniqueNotes: new Set(
        midiCodebook.map((entry) => entry.midiNote),
      ).size,
      occupiedNoteRange: [
        Math.min(...midiCodebook.map((entry) => entry.midiNote)),
        Math.max(...midiCodebook.map((entry) => entry.midiNote)),
      ],
      laneSizes: TERNARY_REGISTER_LANES.map(
        (lane) => midiCodebook.filter((entry) => entry.lane === lane).length,
      ),
    },
  };
}
