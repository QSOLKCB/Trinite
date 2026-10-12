"""Independent structural tests of the frozen, training-free lattice scaffold."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.lattice_scaffold import CHILDREN, construct, protocol, verify

ROOT = Path(__file__).resolve().parents[1]


class LatticeScaffoldTests(unittest.TestCase):
    def test_exact_protocol_and_bounded_cardinalities(self):
        self.assertEqual(protocol()['construction'], 'menger-20-face-grid.v1')
        self.assertEqual(len(CHILDREN), 20)
        for depth, expected in enumerate((1, 20, 400, 8000)):
            with self.subTest(depth=depth):
                record = construct(depth)
                self.assertEqual(record['leaf_count'], expected)
                self.assertEqual(record['side'], 3 ** depth)
                self.assertEqual([l['count'] for l in record['levels']],
                                 [20 ** i for i in range(depth + 1)])
                self.assertEqual(record['geometry_loss'], '0')
                self.assertFalse(record['phase5_ready'])
                self.assertFalse(record['neural_projections_fitted'])
                self.assertEqual(verify(json_bytes(record))['outcome'], 'passed')

    def test_parent_child_ownership_and_occupancy_mask(self):
        r = construct(3)
        for d, level in enumerate(r['levels']):
            coords = [tuple(n['coordinate']) for n in level['nodes']]
            self.assertEqual(len(coords), len(set(coords)))
            self.assertEqual(level['occupancy_identity'], identity(json_bytes([list(c) for c in coords])))
            self.assertTrue(all(all(0 <= v < 3 ** d for v in c) for c in coords))
            if d == 0:
                self.assertEqual(level['nodes'], [{'coordinate': [0, 0, 0],
                                                   'parent_index': None, 'child_rank': None}])
                continue
            parents = r['levels'][d - 1]['nodes']
            for index, node in enumerate(level['nodes']):
                self.assertEqual(node['parent_index'], index // 20)
                self.assertEqual(node['child_rank'], index % 20)
                parent = parents[node['parent_index']]['coordinate']
                local = tuple(c - 3 * p for c, p in zip(node['coordinate'], parent))
                self.assertEqual(local, CHILDREN[node['child_rank']])
                self.assertLessEqual(local.count(1), 1)
        self.assertEqual(len(r['levels'][-1]['nodes']), 8000)

    def test_face_graph_independent_bruteforce_and_degrees(self):
        for depth in (0, 1, 2):
            r = construct(depth)
            coords = [tuple(x['coordinate']) for x in r['levels'][-1]['nodes']]
            expected = [[i, j] for i in range(len(coords)) for j in range(i + 1, len(coords))
                        if sum(abs(a - b) for a, b in zip(coords[i], coords[j])) == 1]
            self.assertEqual(r['edges'], expected)
            degree = [0] * len(coords)
            for i, j in expected:
                degree[i] += 1
                degree[j] += 1
            self.assertEqual(r['degree_histogram'],
                             {str(n): degree.count(n) for n in range(7) if n in degree})
        large = construct(3)
        edges = large['edges']
        self.assertEqual(len(set(map(tuple, edges))), len(edges))
        self.assertTrue(all(i < j < 8000 for i, j in edges))
        self.assertEqual(sum(int(n) * k for n, k in large['degree_histogram'].items()), 2 * len(edges))

    def test_bad_depths_tampering_and_source_bind(self):
        for d in (-1, 4, True, None, 1.0, '2'):
            with self.subTest(depth=d), self.assertRaises(ContractError):
                construct(d)
        original = construct(1)
        for mutate in ('edge', 'count', 'source', 'scope', 'protocol'):
            r = copy.deepcopy(original)
            if mutate == 'edge': r['edges'].append([0, 19])
            if mutate == 'count': r['leaf_count'] += 1
            if mutate == 'source': r['source']['src/trinite/contracts.py'] = 'sha256:' + '0' * 64
            if mutate == 'scope': r['phase5_ready'] = True
            if mutate == 'protocol': r['protocol_identity'] = 'sha256:' + '0' * 64
            with self.subTest(mutate=mutate), self.assertRaises(ContractError):
                verify(json_bytes(r))
        self.assertIn('scripts/generate_lattice_scaffold.py', original['source'])
        with self.assertRaises(ContractError): verify(b'x' * (2 * 1024 * 1024 + 1))
        with self.assertRaises(ContractError): verify(json_bytes(original).replace(b'\n', b'  \n'))

    def test_cli_exclusive_creation_readonly_and_symlink_rejection(self):
        script = ROOT / 'scripts/generate_lattice_scaffold.py'
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); target = temp / 'lattice.json'
            cmd = [sys.executable, str(script)]
            created = subprocess.run([*cmd, '--depth', '1', '--output', str(target)], capture_output=True)
            self.assertEqual(created.returncode, 0, created.stderr)
            before = target.read_bytes()
            receipt = subprocess.run([*cmd, '--verify', str(target)], capture_output=True)
            self.assertEqual(receipt.returncode, 0, receipt.stderr)
            self.assertEqual(parse_json(receipt.stdout)['artifact_identity'], identity(before))
            self.assertEqual(before, target.read_bytes())
            self.assertNotEqual(subprocess.run([*cmd, '--depth', '1', '--output', str(target)],
                                               capture_output=True).returncode, 0)
            link = temp / 'link.json'; link.symlink_to(target)
            self.assertNotEqual(subprocess.run([*cmd, '--verify', str(link)],
                                               capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run([*cmd, '--verify', str(target), '--output', str(temp / 'bad')],
                                               capture_output=True).returncode, 0)
            self.assertFalse((temp / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
