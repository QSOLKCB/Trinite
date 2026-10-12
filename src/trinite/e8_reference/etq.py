# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Trent Slade / QSOL-IMC.
"""Exact ETQ-101/303 reference map, compatible with pinned SONIFICATION.

New Python implementation of finite indexing/phase/graph formulas at
27cfea99a3bab445c74b1aba5c38993fe931f568. No neural mapping or dynamics solver.
"""
from types import MappingProxyType

from ..contracts import identity, integer, json_bytes
from . import root_orbits

SONIFICATION_COMMIT = '27cfea99a3bab445c74b1aba5c38993fe931f568'
SCL_STENCIL = (1, -2, 1)
PHASE_EXPONENTS = tuple(-d % 4 for d in SCL_STENCIL)
GAUSSIAN_SYMBOLS = ('1', 'i', '-1', '-i')
BASE_RELEASE = MappingProxyType({
    'modelId': 'ETQ-101', 'modelVersion': '2.0.0', 'gitTag': 'v2.0.0',
    'gitCommit': '8c24d58ca76abbac77c427a4f63ca434570c82b3',
    'versionDoi': '10.5281/zenodo.21432511',
    'basisSha256': '97cfd1f087745422fd66d3640c7b86c3209593c4b53741018c08a5e9cdb15f6f',
    'adjacencySha256': '29ae0af5b1090c9de30f1efc25789060fb1791eb175d2afcd6888847f7fe6324',
    'degreePotentialNumeratorsSha256': '517b9b86e07a1788cabc446be93e573d21719af3c1e0eeaedae00b81689c4468',
    'contractPayloadSha256': 'd17e4e104bae41e649f1589467fb8122eb42fc9be4298110f4c21753580b3eab',
})


def selected_basis():
    """Canonical 2 singlets + first 33 complete triality triples; 101 labels."""
    orbits = root_orbits()
    fixed = [o[0] for o in orbits if len(o) == 1][:2]
    triples = [o for o in orbits if len(o) == 3][:33]
    return [[int(x) for x in root] for root in fixed + [r for o in triples for r in o]]


def base_adjacency():
    basis = selected_basis()
    return [[int(i != j and sum(x * y for x, y in zip(a, b)) == 4)
             for j, b in enumerate(basis)] for i, a in enumerate(basis)]


def internal_step(site_index):
    """F_Q=X exp(-i*pi*D/2) on internal triples; singlets have phase one."""
    j = integer(site_index, 'site index', 0, 100)
    if j < 2:
        return j, 0
    m, q = divmod(j - 2, 3)
    return 2 + 3 * m + (q + 1) % 3, PHASE_EXPONENTS[q]


def tensor_index(site_index, fibre_label):
    return 3 * integer(site_index, 'site index', 0, 100) + integer(fibre_label, 'fibre label', 0, 2)


def tensor_address(index):
    return divmod(integer(index, 'tensor index', 0, 302), 3)


def event_address(sequence_index):
    n = integer(sequence_index, 'sequence index', 0, 302)
    return n % 101, n % 3


def event_index(site_index, fibre_label):
    tensor_index(site_index, fibre_label)
    return site_index + 101 * (2 * (fibre_label - site_index % 3) % 3)


def monomial_power(site_index, fibre_label, power):
    """R_101 tensor F_3 on a basis address; exact Gaussian exponent, O(1)."""
    tensor_index(site_index, fibre_label)
    integer(power, 'power', 0, 2**53 - 1)
    # Every complete fibre cycle contributes 3+2+3 = 0 mod 4.
    phase = sum(PHASE_EXPONENTS[(fibre_label + k) % 3] for k in range(power % 3)) % 4
    return (site_index + power) % 101, (fibre_label + power) % 3, phase


def site_registry():
    basis, adjacency = selected_basis(), base_adjacency()
    degrees = [sum(row) for row in adjacency]
    degree_sum = sum(degrees)
    numerators = [101 * d - degree_sum for d in degrees]
    denominator = max(map(abs, numerators))
    sites = []
    for j, root in enumerate(basis):
        m, q = (None, None) if j < 2 else divmod(j - 2, 3)
        sites.append({
            'siteIndex': j, 'rootDoubled': root,
            'baseStateType': 'fixed-singlet' if j < 2 else 'triality-orbit-state',
            'fixedIndex': j if j < 2 else None, 'orbitIndex': m,
            'internalTrialityLabel': q,
            'internalTrialityLane': None if q is None else ('low', 'mid', 'high')[q],
            'baseDegree': degrees[j], 'liftedDegree': degrees[j] + 2,
            'degreePotentialNumerator': numerators[j],
            'degreePotentialDenominator': denominator,
            'v2MidiNote': (13, 113)[j] if j < 2 else 14 + 33 * q + m,
        })
    return sites


def lifted_graph():
    adjacency, edges = base_adjacency(), []
    for a in range(3):
        for left in range(101):
            for right in range(left + 1, 101):
                if adjacency[left][right]:
                    edges.append({'source': tensor_index(left, a),
                                  'target': tensor_index(right, a), 'kind': 'base-e8-edge'})
    for j in range(101):
        for a, b in ((0, 1), (0, 2), (1, 2)):
            edges.append({'source': tensor_index(j, a), 'target': tensor_index(j, b),
                          'kind': 'fibre-c3-edge'})
    edges.sort(key=lambda e: (e['source'], e['target'], e['kind']))
    neighbours = [set() for _ in range(303)]
    for edge in edges:
        neighbours[edge['source']].add(edge['target'])
        neighbours[edge['target']].add(edge['source'])
    seen, pending = {0}, [0]
    while pending:
        for other in neighbours[pending.pop()] - seen:
            seen.add(other)
            pending.append(other)
    degrees = [len(row) for row in neighbours]
    return {
        'schema': 'qsol.etq-303.cartesian-graph/v1', 'modelId': 'ETQ-303',
        'modelVersion': '3.0.0', 'construction': 'cartesian-product-G101-square-C3',
        'vertexCount': 303, 'edgeCount': len(edges), 'baseLayerCount': 3,
        'baseEdgesPerLayer': sum(map(sum, adjacency)) // 2, 'fibreEdgesPerSite': 3,
        'minimumDegree': min(degrees), 'maximumDegree': max(degrees),
        'connected': len(seen) == 303, 'degrees': degrees, 'edges': edges,
    }


def canonical_events():
    events = []
    for n in range(303):
        j, a = event_address(n)
        target_j, target_a, phase = monomial_power(j, a, 1)
        events.append({
            'sequenceIndex': n, 'tensorIndex': tensor_index(j, a),
            'siteIndex': j, 'fibreLabel': a,
            'transitionPhaseGaussianExponent': phase,
            'transitionPhaseSymbol': GAUSSIAN_SYMBOLS[phase],
            'nextSequenceIndex': (n + 1) % 303,
            'nextTensorIndex': tensor_index(target_j, target_a),
        })
    return events


def event_document():
    """Compatible finite ETQ event document, not the donor implementation receipt."""
    graph = lifted_graph()
    return {
        'schema': 'qsol.etq-303.events/v1', 'modelId': 'ETQ-303', 'modelVersion': '3.0.0',
        'profile': 'full-qutrit-event-protocol-v1', 'lineage': dict(BASE_RELEASE),
        'stateSpace': {
            'expression': 'H_303=H_101 tensor C^3',
            'basisSemantics': '101-root-indexed-sites-times-independent-ternary-fibre',
            'tensorBasisOrder': 'tensorIndex=3*siteIndex+fibreLabel',
            'dimensions': {'baseSites': 101, 'fibre': 3, 'total': 303},
        },
        'exactStep': {
            'expression': 'F_303=R_101 tensor (X_3 exp(-i*pi*D_3/2))',
            'supportAction': '(j,a)->(j+1 mod 101,a+1 mod 3)',
            'phaseConvention': 'exp(-i*pi*d_a/2)=i^g_a',
            'sclStencil': list(SCL_STENCIL), 'gaussianExponents': list(PHASE_EXPONENTS),
            'gaussianPhaseSymbols': [GAUSSIAN_SYMBOLS[g] for g in PHASE_EXPONENTS],
            'gaussianUnitSymbolLookup': list(GAUSSIAN_SYMBOLS),
            'supportCycleLength': 303, 'operatorOrder': 303,
        },
        'graphContext': {k: graph[k] for k in ('construction', 'vertexCount', 'edgeCount',
                                              'minimumDegree', 'maximumDegree', 'connected')}
                        | {'eventTraversalIsGraphWalk': False},
        'siteRegistry': site_registry(), 'events': canonical_events(),
        'claimBoundary': 'An exact 303-state tensor extension and deterministic event traversal '
                         'of the preserved ETQ-101 v2 selected basis. The 303 states are root-indexed '
                         'tensor states, not 303 distinct E8 roots. Receiver encodings are deterministic '
                         'conventions and no acoustic, physical, statistical, or perceptual claim is made.',
    }


def _canonical(value):
    # Reuse the strict integer-only Trinite serializer; ETQ omits its final LF.
    return json_bytes(value)[:-1]


def _hash(value, domain=None):
    raw = _canonical(value)
    return identity(raw if domain is None else domain.encode('utf-8') + b'\0' + raw)[7:]


def commitments():
    document, graph = event_document(), lifted_graph()
    return {
        'preservedV2BasisFixtureSha256': _hash(selected_basis()),
        'preservedV2AdjacencyFixtureSha256': _hash(base_adjacency()),
        'preservedV2DegreePotentialNumeratorsFixtureSha256':
            _hash([s['degreePotentialNumerator'] for s in document['siteRegistry']]),
        'siteRegistrySha256': _hash({
            'schema': 'qsol.etq-303.site-registry/v1', 'modelId': 'ETQ-303',
            'modelVersion': '3.0.0', 'baseModel': 'ETQ-101@2.0.0',
            'siteCount': 101, 'sites': document['siteRegistry'],
        }, 'ETQ/V3/SITE-REGISTRY/v1'),
        'eventsSha256': _hash(document['events'], 'ETQ/V3/EVENTS/v1'),
        'liftedEdgesSha256': _hash(graph['edges'], 'ETQ/V3/LIFTED-EDGES/v1'),
        'eventDocumentSha256': _hash(document, 'ETQ/V3/EVENT-DOCUMENT/v1'),
    }


def event_document_bytes():
    """Canonical compatible event payload, UTF-8 with no trailing newline."""
    return _canonical(event_document())
