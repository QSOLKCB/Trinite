"""Aizawa source-bound synthetic reference tests; no neural learning assertion."""
import copy
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite import geometry
from aizawa_reference import benchmark as a
import run_aizawa_reference as cli


class AizawaReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_protocol_is_frozen_and_source_closure_is_separate(self):
        self.assertEqual(identity(a.PROTOCOL_PATH.read_bytes()), a.PROTOCOL_IDENTITY)
        self.assertEqual(a.protocol()['schema'], a.PROTOCOL_SCHEMA)
        source = a.source_closure()
        self.assertEqual(set(source), set(a.SOURCE_FILES))
        self.assertIn('src/trinite/_geo_reference.py', source)
        self.assertIn('src/trinite/geometry-pin.json', source)
        self.assertFalse((ROOT / 'src/trinite/aizawa_reference.py').exists())
        self.assertFalse((ROOT / 'src/trinite/aizawa_reference').exists())
        self.assertEqual(a.freeze_request()['geometry_pin'], geometry.verify_geometry_pin())

    def test_first_vector_field_matches_independent_derivative(self):
        p = a.protocol()
        self.assertEqual(a.field((0.1, 0.0, 0.0), a._parameters(p)),
                         (-0.06999999999999999, 0.35000000000000003, 0.59))
        state = (0.1, 0.0, 0.0)
        nxt = a.rk4_step(state, 1e-6, a._parameters(p))
        derivative = a.field(state, a._parameters(p))
        for i in range(3):
            self.assertLess(abs((nxt[i] - state[i]) / 1e-6 - derivative[i]), 1e-5)

    def test_integrator_bounds_fail_closed_and_reject_nonfinite(self):
        p = a.protocol()
        for invalid in (0, 1000, 39999, False, 20000.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ContractError):
                    a.integrate(p, invalid)
        with self.assertRaisesRegex(ContractError, 'nonfinite'):
            a.field((float('inf'), 0.0, 0.0), a._parameters(p))
        with self.assertRaisesRegex(ContractError, 'positive binary64'):
            a.rk4_step((0.1, 0.0, 0.0), 0.0, a._parameters(p))

    def test_exact_stream_length_fine_refinement_and_sampling_grid(self):
        p = a.protocol()
        coarse = a.integrate(p, 20000)
        fine = a.integrate(p, 40000)
        self.assertEqual(coarse['stream_states'], 20001)
        self.assertEqual(fine['stream_states'], 40001)
        self.assertEqual(len(coarse['sampled']), 201)
        self.assertEqual(len(fine['sampled']), 201)
        self.assertEqual(coarse['sampled'][0], [0.1, 0.0, 0.0])
        self.assertEqual(fine['sampled'][0], [0.1, 0.0, 0.0])
        self.assertNotEqual(coarse['full_binary64_stream_identity'], fine['full_binary64_stream_identity'])
        early = a._coordinate_errors(coarse['sampled'], fine['sampled'], 21)
        full = a._coordinate_errors(coarse['sampled'], fine['sampled'], 201)
        self.assertLessEqual(float.fromhex(early['max_abs_coordinate_error_hex']),
                             float(p['numerics']['early_max_pointwise_coordinate_error_upper_bound']))
        self.assertGreaterEqual(float.fromhex(full['max_abs_coordinate_error_hex']),
                                float.fromhex(early['max_abs_coordinate_error_hex']))

    def test_controls_are_independent_deterministic_and_read_only(self):
        p = a.protocol()
        coarse = a.integrate(p, 20000)['sampled']
        before = copy.deepcopy(coarse)
        first = a._controls(coarse, p)
        second = a._controls(coarse, p)
        self.assertEqual(first, second)
        self.assertEqual(coarse, before)
        self.assertEqual([v['name'] for v in first], a.freeze_request()['conditions'])
        for c in first:
            self.assertEqual(len(c['sampled']), 201)
            self.assertEqual(len(c['sampled'][0]), 3)
        for c in first[1:3]:
            self.assertEqual(sorted(c['permutation']), list(range(201)))
            self.assertNotEqual(c['permutation'], list(range(201)))
            self.assertEqual(c['sampled'], [coarse[i] for i in c['permutation']])
        self.assertNotEqual(first[1]['permutation'], first[2]['permutation'])
        self.assertEqual(first[-1]['category'], 'orientation-diagnostic-not-null')

    def test_native_metrics_use_pinned_kernel_and_affine_curvature_is_zero(self):
        p = a.protocol()
        rows = a.integrate(p, 20000)['sampled']
        controls = a._controls(rows, p)
        native, affine, reverse = controls[0], controls[-2], controls[-1]
        self.assertEqual(native['sampled'], rows)
        self.assertEqual(reverse['sampled'], list(reversed(rows)))
        native_metric = geometry.metrics(rows)
        reverse_metric = geometry.metrics(reverse['sampled'])
        self.assertEqual(native_metric['path_length_hex'], reverse_metric['path_length_hex'])
        self.assertEqual(native_metric['menger_hex'], list(reversed(reverse_metric['menger_hex'])))
        affine_metric = geometry.metrics(affine['sampled'])
        self.assertTrue(all(x == 0.0.hex() for x in affine_metric['menger_hex']))
        self.assertEqual(len(native_metric['order_2_hex']), 199)

    def test_full_report_is_source_bound_and_never_promotes_learning(self):
        request = a.freeze_request()
        result = a.measure(request)
        self.assertEqual(result['schema'], a.REPORT_SCHEMA)
        self.assertEqual(result['request_identity'], identity(json_bytes(request)))
        self.assertEqual(result['runtime'], request['runtime'])
        self.assertEqual(result['sources'], request['sources'])
        self.assertEqual(result['evidence_class'], 'SIMULATION')
        self.assertEqual(result['geometry_loss'], '0')
        self.assertFalse(result['phase5_ready'])
        self.assertFalse(result['release_ready'])
        self.assertFalse(result['full_optimizer_replay'])
        self.assertEqual(len(result['controls']), 7)
        self.assertEqual(len(result['refined_points_hex']), 201)
        self.assertEqual(result['controls'][0]['points_hex'][0],
                         [0.1.hex(), 0.0.hex(), 0.0.hex()])
        for control in result['controls']:
            self.assertEqual(len(control['points_hex']), 201)
            self.assertEqual(len(control['metrics']['menger_hex']), 199)
            self.assertTrue(math.isfinite(float.fromhex(control['alignment_to_native_order1_hex'])))
        self.assertLess(len(json_bytes(result)), a.protocol()['limits']['max_report_bytes'])

    def test_cross_environment_and_source_drift_fail_before_integration(self):
        request = a.freeze_request()
        changed = copy.deepcopy(request)
        changed['runtime']['python'] = 'foreign'
        with mock.patch.object(a, 'integrate', side_effect=AssertionError('no outcomes')):
            with self.assertRaisesRegex(ContractError, 'drift'):
                a.measure(changed)
        changed = copy.deepcopy(request)
        changed['sources']['src/trinite/geometry.py'] = identity(b'foreign')
        with mock.patch.object(a, 'integrate', side_effect=AssertionError('no outcomes')):
            with self.assertRaisesRegex(ContractError, 'drift'):
                a.measure(changed)
        changed = copy.deepcopy(request)
        del changed['conditions']
        with self.assertRaises(ContractError):
            a.measure(changed)

    def test_frozen_procedure_and_geometry_pin_tampering_rejected(self):
        original = a.PROTOCOL_IDENTITY
        with mock.patch.object(a, 'PROTOCOL_IDENTITY', identity(b'bad-protocol')):
            with self.assertRaisesRegex(ContractError, 'frozen revision'):
                a.protocol()
        authentic = a.freeze_request()
        with mock.patch.object(a.geometry, 'verify_geometry_pin', return_value={'untrusted':'pin'}):
            with self.assertRaisesRegex(ContractError, 'drift'):
                a.measure(authentic)
        self.assertEqual(a.PROTOCOL_IDENTITY, original)

    def test_early_convergence_is_an_actual_gate(self):
        request = a.freeze_request()
        p = a.protocol()
        good = a.integrate(p, 20000)
        altered = copy.deepcopy(a.integrate(p, 40000))
        altered['sampled'][1][0] += 0.1
        with mock.patch.object(a, 'integrate', side_effect=[good, altered]):
            with self.assertRaisesRegex(ContractError, 'convergence failed'):
                a.measure(request)

    def test_full_request_report_verify_only_and_no_evidence_modification(self):
        request_path = self.dir / 'request.json'
        report_path = self.dir / 'report.json'
        self.assertEqual(cli.main(['--freeze-request', '--output', str(request_path)]), 0)
        frozen = request_path.read_bytes()
        req = parse_json(frozen, canonical=True)
        self.assertEqual(req, a.freeze_request())
        self.assertEqual(cli.main(['--request',str(request_path),'--request-identity',identity(frozen),
                                   '--output',str(report_path)]), 0)
        report_bytes = report_path.read_bytes()
        self.assertEqual(cli.main(['--request',str(request_path),'--request-identity',identity(frozen),
                                   '--output',str(report_path),'--verify-only']), 0)
        self.assertEqual(report_path.read_bytes(), report_bytes)
        self.assertEqual(request_path.read_bytes(), frozen)
        self.assertEqual(cli.main(['--request',str(request_path),'--request-identity',identity(frozen),
                                   '--output',str(report_path)]), 2)  # exclusive
        self.assertEqual(report_path.read_bytes(), report_bytes)

    def test_wrong_external_request_identity_and_report_tampering_rejected(self):
        request_path = self.dir/'request.json'
        request_path.write_bytes(json_bytes(a.freeze_request()))
        output = self.dir/'report.json'
        self.assertEqual(cli.main(['--request',str(request_path),'--request-identity',identity(b'wrong'),
                                   '--output',str(output)]),2)
        self.assertFalse(output.exists())
        self.assertEqual(cli.main(['--request',str(request_path), '--request-identity',identity(request_path.read_bytes()),
                                   '--output',str(output),'--verify-only']),2)
        self.assertFalse(output.exists())

    def test_refuse_repository_output_paths_and_symlinks(self):
        self.assertRaises(ContractError, cli.safe_output,
                          ROOT/'scripts/aizawa_reference/unsafe.json')
        link = self.dir/'alias'
        link.symlink_to(self.dir, target_is_directory=True)
        with self.assertRaises(ContractError):
            cli.safe_output(link/'unsafe.json')
        output = self.dir/'receipt.json'
        output.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            cli.write_exclusive(output, b'replacement')
        self.assertEqual(output.read_bytes(), b'existing')

    def test_failed_file_sync_cleans_new_receipt_and_can_retry(self):
        output = self.dir/'report.json'
        with mock.patch.object(cli.os,'fsync',side_effect=OSError('simulated exhausted filesystem')):
            with self.assertRaises(OSError):
                cli.write_exclusive(output, b'incomplete')
        self.assertFalse(output.exists())
        cli.write_exclusive(output, b'complete')
        self.assertEqual(output.read_bytes(), b'complete')

    def test_cosmo_stencil_is_only_an_exact_integer_control(self):
        # COSMO v0.1.0 cosmovirus.lean: diag_const_zero/diag_affine_zero.
        # This checks the integers locally; it is NOT Lean kernel replay.
        for initial, delta in ((0, 0), (9, -4), (-25, 37)):
            self.assertEqual(initial - 2*(initial+delta) + (initial+2*delta), 0)
        self.assertEqual(0 - 2*1 + 4, 2)


if __name__ == '__main__':
    unittest.main()
