"""Run the unchanged, hash-checked upstream numerical tests against its kernel."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from trinite.geometry import alignment, metrics, reference, verify_geometry_pin
from trinite.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]


class ReferenceAdapterTests(unittest.TestCase):
    def test_source_acquisition_and_runtime_receipts(self):
        self.assertEqual(len(verify_geometry_pin(ROOT)['checked_files']), 5)
        self.assertEqual(len(verify_geometry_pin()['checked_files']), 2)
        with patch('trinite.geometry.PIN_IDENTITY', 'unsupported'):
            with self.assertRaises(ContractError): reference()

    def test_analytic_line_circle_null_and_unavailable_cases(self):
        self.assertEqual(float.fromhex(metrics([[0.,0.],[3.,4.],[6.,8.]])['path_length_hex']), 10.)
        circle = metrics([[2.,0.],[0.,2.],[-2.,0.]])
        self.assertEqual(float.fromhex(circle['menger_hex'][0]), .5)
        self.assertEqual(metrics([[1.,2.]])['mean_menger_hex'], None)
        self.assertEqual(metrics([[1.,2.]]*3)['menger_hex'], ['0x0.0p+0'])
        with self.assertRaises(ContractError): alignment([[0.]], [[0.]])

    def test_translation_scale_and_cosine_conventions(self):
        a = [[0.,0.],[1.,0.],[1.,1.]]
        b = [[3+x,4+y] for x,y in a]
        c = [[2*x,2*y] for x,y in a]
        self.assertEqual(metrics(a)['path_length_hex'], metrics(b)['path_length_hex'])
        self.assertEqual(metrics(a)['menger_hex'], metrics(b)['menger_hex'])
        self.assertEqual(float.fromhex(metrics(c)['menger_hex'][0]), float.fromhex(metrics(a)['menger_hex'][0])/2)
        self.assertAlmostEqual(alignment(a,b,mode='error'), 1.)
        self.assertEqual(alignment([[0.],[0.]], [[0.],[1.]], mode='error'), 0.)
        self.assertEqual(alignment([[0.],[1e-30]], [[0.],[-1e-30]], mode='error'), -1.)

    def test_bounds_nonfinite_overflow_and_bool_rejections(self):
        for points in ([], [[True]], [[2**10000]], [[float('nan')]], [[0.],[1.,2.]], [[0.]]*257, [[0.]*129]):
            with self.assertRaises(ContractError): metrics(points)
        with self.assertRaises(ContractError): metrics([[-1e308],[1e308]])
        with self.assertRaises(ContractError): alignment([[0.],[1.]], [[0.],[1.]], order=True)

    def test_external_decimal_settings_cannot_change_measurement(self):
        from decimal import localcontext, ROUND_DOWN
        points = [[0.,0.],[1.,0.],[1.,1.]]
        expected = metrics(points)
        with localcontext() as context:
            context.rounding = ROUND_DOWN; context.prec = 3
            self.assertEqual(metrics(points), expected)
            self.assertEqual(context.rounding, ROUND_DOWN)
            self.assertEqual(context.prec, 3)


def load_tests(loader, tests, pattern):
    verify_geometry_pin(ROOT)
    # Only test import names are remapped. The archived test and kernel bytes
    # remain exactly upstream, and the mapping is restored after loading.
    kernel = reference(); package = types.ModuleType('qsol_geo_reason'); package.__path__ = []
    spec = importlib.util.spec_from_file_location('frozen_geo_tests', ROOT/'tests/geo_upstream/test_geometry.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'qsol_geo_reason': package, 'qsol_geo_reason.geometry': kernel}):
        spec.loader.exec_module(module)
    tests.addTests(loader.loadTestsFromModule(module))
    return tests
