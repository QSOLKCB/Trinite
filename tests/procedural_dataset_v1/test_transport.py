"""Transport failures are not semantic negatives; real-oracle checks are explicit."""
import base64
import sys
import unittest
from unittest.mock import patch

from procedural_dataset_v1 import transport as t
import queue_evaluation_v1 as gate


class TransportTests(unittest.TestCase):
    def test_fixed_dispatch(self):
        for module, op in [('../ternary', 'score'), ('context', 'score'),
                           ('tombstone', '/bin/sh'), ('ternary', 'observe')]:
            with self.assertRaises(gate.GateError):
                t.run(module, op, {})

    def test_input_bounds_before_execution(self):
        with self.assertRaises(gate.GateError):
            t.run('tombstone', 'observe', {'x': 'x' * 1024})
        with self.assertRaises(gate.GateError):
            t.run('ternary', 'score', [])

    def test_transport_failure_preserves_evidence(self):
        evidence = {'exit_code': 3, 'failure': None,
                    'stdout_b64': '', 'stderr_b64': base64.b64encode(b'infrastructure failure\n').decode()}
        failure = gate.ProcessError(evidence)
        with patch.object(t, 'identities', return_value={}), \
             patch.object(gate, 'bounded_process', side_effect=failure):
            with self.assertRaises(gate.ProcessError) as caught:
                t.run('tombstone', 'evaluate-privileged', {})
        self.assertEqual(caught.exception.evidence, evidence)

    def test_stderr_not_a_label(self):
        evidence = {'stderr_b64': base64.b64encode(b'warning').decode()}
        with patch.object(t, 'identities', return_value={}), \
             patch.object(gate, 'bounded_process', return_value=(b'{}\n', evidence)):
            with self.assertRaises(gate.ProcessError) as caught:
                t.run('context', 'observe', {})
            self.assertEqual(caught.exception.evidence['failure'], 'invalid_oracle_response')

    def test_duplicate_response_rejected(self):
        with patch.object(t, 'identities', return_value={}), \
             patch.object(gate, 'bounded_process', return_value=(b'{"x":1,"x":2}\n', {'stderr_b64': ''})):
            with self.assertRaises(gate.ProcessError) as caught:
                t.run('context', 'observe', {})
            self.assertEqual(caught.exception.evidence['failure'], 'invalid_oracle_response')

    def test_live_bounded_primitive_timeout_and_output(self):
        # Software doubles establish process cleanup, not native oracle correctness.
        for program, cap, timeout, reason in [
                ('import time;time.sleep(2)', 32, 0.05, 'timeout'),
                ('import sys;sys.stdout.write("x"*1000)', 32, 2, 'output_limit')]:
            with self.assertRaises(gate.ProcessError) as caught:
                gate.bounded_process([sys.executable, '-c', program], b'', timeout=timeout, cap=cap)
            self.assertEqual(caught.exception.evidence['failure'], reason)

    def test_actual_oracles_and_malformed_inputs(self):
        # No skip: this suite requires explicitly built, source-matching binaries.
        self.assertEqual(set(t.identities()), {'ternary', 'context', 'tombstone'})
        examples = [
            ('ternary', 'score', dict(schema='trinite.ternary-evidence-request.v1',
                 input_kind='generic-public-trits', weights=[2, 3],
                 entries=[dict(id='a', active=True, trits=[1, 0], mask=[True, True])])),
            ('context', 'evaluate-privileged', dict(schema='trinite.context-symbol-request.v1',
                 contexts=2, symbols=1, meanings=2, table=[[0], [1]],
                 context_mask=[True, True], symbol_mask=[True], query=0)),
            ('tombstone', 'evaluate-privileged', dict(schema='trinite.tombstone-request.v1',
                 before=[0, 255], removed=[255], after=[0], offset=1)),
        ]
        results = [t.run(*example) for example in examples]
        self.assertEqual(results[0]['entries'][0]['score'], {'numerator': 2, 'denominator': 5})
        self.assertEqual(results[1]['relation'], 0)
        self.assertEqual(results[2]['label'], 'valid-splice')
        for module, operation, request in examples:
            with self.assertRaises(gate.ProcessError) as caught:
                t.run(module, operation, dict(request, unexpected=1))
            self.assertEqual(caught.exception.evidence['exit_code'], 2)
            self.assertEqual(caught.exception.evidence['stdout_b64'], '')


if __name__ == '__main__':
    unittest.main()
