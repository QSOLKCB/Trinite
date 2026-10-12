from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from trinite.contracts import ContractError
from trinite.e8_reference import triality
from trinite.e8_reference import etq

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'fixtures/e8-etq-reference/donor-commitments.json').read_text())


class ETQReferenceTests(unittest.TestCase):
    def test_all_seven_published_donor_commitments_match(self):
        self.assertEqual(FIXTURE['source']['commit'], etq.SONIFICATION_COMMIT)
        self.assertEqual(etq.commitments(), FIXTURE['commitments'])

    def test_selector_keeps_labels_and_complete_internal_triality_orbits(self):
        basis = etq.selected_basis()
        self.assertEqual(len(basis), 101)
        self.assertEqual(len({tuple(r) for r in basis}), 101)
        for j, root in enumerate(basis):
            target, phase = etq.internal_step(j)
            self.assertEqual(triality(root), tuple(basis[target]))
            current, accumulated = j, 0
            for _ in range(3):
                current, g = etq.internal_step(current)
                accumulated = (accumulated + g) % 4
            self.assertEqual((current, accumulated), (j, 0))
            if j < 2:
                self.assertEqual((target, phase), (j, 0))

    def test_exact_phase_and_scl_covariant_cycle(self):
        self.assertEqual(etq.SCL_STENCIL, (1, -2, 1))
        self.assertEqual(etq.PHASE_EXPONENTS, (3, 2, 3))
        self.assertEqual(tuple(etq.GAUSSIAN_SYMBOLS[g] for g in etq.PHASE_EXPONENTS),
                         ('-i', '-1', '-i'))
        for q in range(3):
            self.assertEqual(sum(etq.SCL_STENCIL[(q - a) % 3] for a in range(3)), 0)
        self.assertNotEqual(etq.SCL_STENCIL[0], etq.SCL_STENCIL[1])

    def test_root_graph_degrees_and_triality_invariance(self):
        adjacency = etq.base_adjacency()
        degrees = list(map(sum, adjacency))
        self.assertEqual(sum(degrees), 3374)
        self.assertEqual((min(degrees), max(degrees)), (22, 55))
        permutation = [etq.internal_step(j)[0] for j in range(101)]
        for i in range(101):
            self.assertEqual(adjacency[i][i], 0)
            for j in range(101):
                self.assertEqual(adjacency[i][j], adjacency[j][i])
                self.assertEqual(adjacency[permutation[i]][permutation[j]], adjacency[i][j])
        registry = etq.site_registry()
        numerators = [s['degreePotentialNumerator'] for s in registry]
        self.assertEqual(sum(numerators), 0)
        self.assertEqual(max(map(abs, numerators)), 2181)

    def test_distinct_internal_and_external_labels_and_symbolic_note_bijection(self):
        sites = etq.site_registry()
        self.assertEqual([s['internalTrialityLabel'] for s in sites[:2]], [None, None])
        self.assertEqual(Counter(s['internalTrialityLabel'] for s in sites[2:]), {0: 33, 1: 33, 2: 33})
        self.assertEqual({s['v2MidiNote'] for s in sites}, set(range(13, 114)))
        for site in sites:
            j = site['siteIndex']
            for a in range(3):
                self.assertEqual(etq.tensor_address(etq.tensor_index(j, a)), (j, a))
        # Internal triality fixes singlet 0; ETQ-303 selector shift changes its site.
        self.assertEqual(etq.internal_step(0), (0, 0))
        self.assertEqual(etq.monomial_power(0, 0, 1), (1, 1, 3))

    def test_tensor_and_crt_bijections_all_states_and_exact_order(self):
        events = etq.canonical_events()
        self.assertEqual({e['tensorIndex'] for e in events}, set(range(303)))
        for n in range(303):
            j, a = etq.event_address(n)
            self.assertEqual(etq.event_index(j, a), n)
            self.assertEqual(etq.monomial_power(j, a, 303), (j, a, 0))
        for power in range(1, 303):
            j, a, _ = etq.monomial_power(0, 0, power)
            self.assertNotEqual((j, a), (0, 0))

    def test_fast_power_matches_independent_repeated_steps_and_composition(self):
        for initial in range(3):
            j, a, phase = 7, initial, 0
            for power in range(610):
                self.assertEqual(etq.monomial_power(7, initial, power), (j, a, phase))
                phase = (phase + (-etq.SCL_STENCIL[a]) % 4) % 4
                j, a = (j + 1) % 101, (a + 1) % 3
        for n, m in ((0, 1), (101, 202), (303, 17), (2**53 - 9, 8)):
            j, a, p = etq.monomial_power(5, 2, n)
            k, b, q = etq.monomial_power(j, a, m)
            self.assertEqual(etq.monomial_power(5, 2, n + m), (k, b, (p + q) % 4))

    def test_lifted_graph_is_cartesian_product_not_assumed_event_walk(self):
        graph, adjacency = etq.lifted_graph(), etq.base_adjacency()
        self.assertEqual((graph['vertexCount'], graph['edgeCount']), (303, 5364))
        self.assertEqual((graph['minimumDegree'], graph['maximumDegree']), (24, 57))
        self.assertTrue(graph['connected'])
        for edge in graph['edges']:
            j, a = etq.tensor_address(edge['source'])
            k, b = etq.tensor_address(edge['target'])
            self.assertLess(edge['source'], edge['target'])
            if edge['kind'] == 'base-e8-edge':
                self.assertEqual(a, b)
                self.assertEqual(adjacency[j][k], 1)
            else:
                self.assertEqual(j, k)
                self.assertNotEqual(a, b)
        self.assertFalse(etq.event_document()['graphContext']['eventTraversalIsGraphWalk'])

    def test_strict_address_power_domains_and_independent_outputs(self):
        for bad in (-1, 101, True, 1., '1', None):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                etq.tensor_index(bad, 0)
        for bad in (-1, 3, True, 1.):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                etq.tensor_index(0, bad)
        for bad in (-1, 2**53, True, 1.):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                etq.monomial_power(0, 0, bad)
        for fn in (etq.event_address, etq.tensor_address):
            with self.assertRaises(ContractError):
                fn(303)
        sites = etq.site_registry()
        sites[0]['rootDoubled'][0] = 99
        self.assertNotEqual(etq.site_registry()[0]['rootDoubled'][0], 99)

    def test_actual_offline_export_matches_canonical_event_commitment(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/run_e8_reference.py'), '--etq-map'],
                                env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.stdout.endswith(b'\n'))
        self.assertEqual(result.stdout, etq.event_document_bytes())
        digest = hashlib.sha256(b'ETQ/V3/EVENT-DOCUMENT/v1\0' + result.stdout).hexdigest()
        self.assertEqual(digest, FIXTURE['commitments']['eventDocumentSha256'])
        document = json.loads(result.stdout)
        self.assertEqual(len(document['siteRegistry']), 101)
        self.assertEqual(len(document['events']), 303)


if __name__ == '__main__':
    unittest.main()
