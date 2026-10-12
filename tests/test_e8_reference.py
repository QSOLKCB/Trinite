import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from trinite.contracts import ContractError
from trinite.e8_reference import (project_first_three, reference_report,
                                  root_orbits, roots_doubled, rotate_z_quarter,
                                  triality)
from trinite.geometry import metrics

ROOT = Path(__file__).resolve().parents[1]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class E8ReferenceTests(unittest.TestCase):
    def test_root_counts_norms_parity_and_negation(self):
        roots = roots_doubled()
        self.assertEqual(len(roots), 240)
        self.assertEqual(len(set(roots)), 240)
        self.assertEqual(tuple(sorted(roots)), roots)
        self.assertEqual(sum(0 in r for r in roots), 112)
        self.assertEqual(sum(0 not in r for r in roots), 128)
        for root in roots:
            self.assertEqual(dot(root, root), 8)
            self.assertIn(tuple(-x for x in root), roots)
            if 0 not in root:
                self.assertEqual(sum(x < 0 for x in root) % 2, 0)

    def test_known_vector_spinor_cycle(self):
        vector = (2, 0, 0, 0, 2, 0, 0, 0)
        spinor = (1, 1, 1, -1, 1, 1, 1, -1)
        conjugate = (1, 1, 1, 1, 1, 1, 1, 1)
        self.assertEqual(triality(vector), spinor)
        self.assertEqual(triality(spinor), conjugate)
        self.assertEqual(triality(conjugate), vector)

    def test_exact_order_and_orthogonality_on_full_space(self):
        basis = [tuple(int(i == j) for j in range(8)) for i in range(8)]
        transformed = [triality(e) for e in basis]
        for i, e in enumerate(basis):
            self.assertEqual(triality(triality(transformed[i])), e)
            for j, f in enumerate(basis):
                self.assertEqual(dot(transformed[i], transformed[j]), dot(e, f))
        self.assertTrue(any(triality(e) != e for e in basis))

    def test_every_root_and_pairwise_inner_product_preserved(self):
        roots = roots_doubled()
        images = [triality(r) for r in roots]
        self.assertEqual(set(images), set(roots))
        # Root images are integral in doubled coordinates; no rounding is used.
        self.assertTrue(all(x.denominator == 1 for r in images for x in r))
        images = [tuple(int(x) for x in r) for r in images]
        for i, r in enumerate(roots):
            for j, s in enumerate(roots):
                self.assertEqual(dot(images[i], images[j]), dot(r, s))

    def test_all_orbits_are_disjoint_complete_and_action_ordered(self):
        orbits = root_orbits()
        self.assertEqual(sum(len(o) == 1 for o in orbits), 12)
        self.assertEqual(sum(len(o) == 3 for o in orbits), 76)
        members = [r for orbit in orbits for r in orbit]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(set(members), set(roots_doubled()))
        for orbit in orbits:
            self.assertEqual(orbit[0], min(orbit))
            for i, r in enumerate(orbit):
                self.assertEqual(triality(r), orbit[(i + 1) % len(orbit)])

    def test_rotation_example_and_exact_fourfold_closure(self):
        v = (1, -2, 1)
        self.assertEqual(rotate_z_quarter(v), (2, 1, 1))
        self.assertEqual(dot(v, rotate_z_quarter(v)), 1)
        self.assertEqual(dot(v, v), dot(rotate_z_quarter(v), rotate_z_quarter(v)))
        current = v
        for _ in range(4):
            current = rotate_z_quarter(current)
        self.assertEqual(current, v)
        self.assertNotEqual(rotate_z_quarter(rotate_z_quarter(rotate_z_quarter(v))), v)

    def test_lossy_projection_has_no_universal_3d_triality_action(self):
        # e_3 and zero have equal projections, but different projected T images.
        e3, zero = (0, 0, 0, 1, 0, 0, 0, 0), (0,) * 8
        self.assertEqual(project_first_three(e3), project_first_three(zero))
        self.assertNotEqual(project_first_three(triality(e3)),
                            project_first_three(triality(zero)))
        self.assertEqual(project_first_three(triality(e3)), (Fraction(1, 2),
                                                            Fraction(-1, 2),
                                                            Fraction(-1, 2)))

    def test_exact_types_bounds_and_no_input_mutation(self):
        source = [Fraction(1, 3), 0, 0, 0, 0, 0, 0, 0]
        before = copy.deepcopy(source)
        self.assertEqual(triality(triality(triality(source))), tuple(source))
        self.assertEqual(source, before)
        for bad in (None, [0] * 7, [0] * 9, [True] * 8, [1.] * 8,
                    ['1'] * 8, [2**256] * 8, [Fraction(1, 2**256)] * 8):
            with self.subTest(bad=repr(bad)), self.assertRaises(ContractError):
                triality(bad)
        with self.assertRaises(ContractError):
            rotate_z_quarter([0] * 8)

    def test_stencil_reuses_existing_geometry_with_analytic_cases(self):
        for samples in ([[3], [3], [3]], [[0], [1], [2]]):
            self.assertEqual(metrics(samples)['order_2_hex'], [[0.0.hex()]])
        self.assertEqual(metrics([[0], [1], [4]])['order_2_hex'], [[2.0.hex()]])
        self.assertEqual(metrics([[0], [1]])['order_2_hex'], [])

    def test_offline_script_report_and_import_boundary(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/run_e8_reference.py')],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['root_count'], 240)
        self.assertEqual(report['rotation_example']['dot_product'], 1)
        self.assertEqual(report['second_difference_example']['order_2_hex'], [[2.0.hex(), 0.0.hex(), 0.0.hex()]])
        self.assertEqual(report['learning_benefit'], 'not measured')
        result = subprocess.run([sys.executable, '-S', '-c',
                                 'import trinite.e8_reference; import sys; '
                                 'assert not any(x in sys.modules for x in '
                                 '("torch", "numpy", "random", "trinite.model", "trinite.training"))'],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(reference_report()['geometry_loss'], 0)


if __name__ == '__main__':
    unittest.main()
