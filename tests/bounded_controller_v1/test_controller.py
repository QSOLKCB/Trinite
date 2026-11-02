import subprocess
import sys
import unittest
from unittest.mock import patch

import bounded_controller_v1 as c
from trinite.tokenizer import EOS


def request(word=123, horizon=3, mode='stepwise'):
    return dict(schema='trinite.bounded-episode.v1', module='omi',
                observation=c.omi.public(word, 'forward', horizon), mode=mode,
                budget=dict(calls=8, tokens=40, prefix_positions=4920))


class Controller(unittest.TestCase):
    def run_episode(self, req, policy):
        return c.run(c.freeze(req), policy)

    def test_native_constant_control_and_accounting(self):
        result = self.run_episode(request(mode='direct'), c.ConstantPolicy('FFFF'))
        self.assertEqual(result['status'], 'complete')
        ledger = c.omi.ledger(request(mode='direct')['observation'])
        self.assertEqual(result['accounting']['processed_prefix_positions'], ledger['full_prefix_forward_positions'])
        self.assertEqual(result['accounting']['emitted_tokens'], 5)
        self.assertIsNone(result['accounting']['actual_neural_forwards'])

    def test_own_output_only_no_oracle(self):
        seen = []
        policy = c.ConstantPolicy('FFFF')
        def capture(prefix):
            self.assertIsInstance(prefix, tuple)
            seen.append(prefix)
            return policy(prefix)
        with patch.object(c.omi, 'expected', side_effect=AssertionError('oracle leak')), patch.object(c.omi, 'orbit', side_effect=AssertionError('orbit leak')):
            result = self.run_episode(request(), capture)
        self.assertEqual(result['status'], 'repeated_state')
        self.assertEqual(len(result['trace']), 2)
        self.assertIn('FFFF', result['trace'][1]['public_prompt'])
        self.assertNotIn('source', result['trace'][1]['public_prompt'])

    def test_caps_before_dispatch(self):
        req = request(); req['budget']['tokens'] = 4
        with patch.object(c.omi, 'native', side_effect=AssertionError('dispatch')):
            result = self.run_episode(req, lambda _: EOS)
        self.assertEqual(result['status'], 'budget')
        self.assertEqual(result['accounting']['calls'], 0)

    def test_cycle_completion_precedence(self):
        for word, period in ((5239, 1), (99, 2)):
            self.assertEqual(len(c.omi.orbit(word)), period)
            for k, status in ((period, 'complete'), (period+1, 'repeated_state')):
                current = word; output = []
                for _ in range(k):
                    current = c.omi.arithmetic_delta(current)
                    output.extend((*f'{current:04X}'.encode(), EOS))
                stream = iter(output)
                result = self.run_episode(request(word=word, horizon=k), lambda _: next(stream))
                self.assertEqual(result['status'], status)
                self.assertEqual(c.evaluate(result)['correct'], status == 'complete')

    def test_call_and_prefix_caps(self):
        req = request(); req['budget']['calls'] = 1
        result = self.run_episode(req, c.ConstantPolicy('FFFF'))
        self.assertEqual(result['status'], 'budget')
        self.assertEqual(result['accounting']['calls'], 1)
        req = request(); req['budget']['prefix_positions'] = 1
        result = self.run_episode(req, lambda _: self.fail('must not invoke'))
        self.assertEqual(result['status'], 'budget')
        self.assertEqual(result['accounting']['calls'], 0)

    def test_eos_and_malformed(self):
        for policy, status in ((lambda _: 65, 'missing_eos'), (lambda _: EOS, 'malformed_answer'),
                               (lambda _: True, 'malformed_token'), (lambda _: 256, 'malformed_token')):
            result = self.run_episode(request(mode='direct'), policy)
            self.assertEqual(result['status'], status)

    def test_period_and_qutrit(self):
        req = request(mode='direct'); req['observation'] = c.omi.public(0, 'period')
        result = self.run_episode(req, c.ConstantPolicy('8'))
        self.assertEqual(result['accounting']['emitted_tokens'], 2)
        req.update(module='qutrit', observation=c.qutrit.public('0120', 'S1', 'check', '0000'))
        result = self.run_episode(req, c.ConstantPolicy('0'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['accounting']['emitted_tokens'], 2)

    def test_partial_call_accounting_and_exception(self):
        req = request(mode='direct')
        for response in (True, 256, 258, -1, 999, [65]*100):
            result = self.run_episode(req, lambda _, response=response: response)
            counts = result['accounting']
            self.assertEqual((counts['calls'], counts['emitted_tokens'], counts['full_prefix_operations']), (1, 0, 1))
            self.assertEqual(counts['processed_prefix_positions'], len(c.omi.public_only(req['observation'])))
        def fail(_): raise RuntimeError('not exposed to model')
        result = self.run_episode(req, fail)
        self.assertEqual(result['status'], 'policy_error')
        self.assertEqual(result['trace'][0]['rejected_type'], 'RuntimeError')

    def test_cli_rejects_raw_oversize(self):
        payload = c.gate.encode(request()) + b' '*16385
        process = subprocess.run([sys.executable, 'scripts/run_bounded_controller_v1.py', '--constant', 'FFFF'],
                                 input=payload, capture_output=True, cwd=c.ROOT)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, b'')
        self.assertIn(b'request too large', process.stderr)

    def test_source_and_request_fail_closed(self):
        frozen = c.freeze(request()); frozen['sources']['src/trinite/__init__.py'] = 'stale'
        with self.assertRaises(c.gate.GateError): c.run(frozen, lambda _: EOS)
        for key, value in (('extra', 1), ('mode', 'repair')):
            req = request(); req[key] = value
            with self.assertRaises(c.gate.GateError): c.freeze(req)
        req = request(); req['budget']['calls'] = True
        with self.assertRaises(c.gate.GateError): c.freeze(req)

    def test_mixed_horizons_same_target_scripted_policy(self):
        # Exposed fixture-script control, deliberately NOT a learned policy.
        for k in range(1, 8):
            answers = [f'{c.omi.arithmetic_delta(123):04X}']
            for _ in range(k-1): answers.append(f'{c.omi.arithmetic_delta(int(answers[-1],16)):04X}')
            stream = iter([token for answer in answers for token in (*answer.encode(), EOS)])
            step = self.run_episode(request(horizon=k), lambda _: next(stream))
            direct = self.run_episode(request(horizon=k, mode='direct'), c.ConstantPolicy(answers[-1]))
            self.assertEqual(step['answer'], direct['answer'])
            self.assertTrue(c.evaluate(step)['correct'])
            self.assertEqual(step['accounting']['calls'], k)


if __name__ == '__main__': unittest.main()
