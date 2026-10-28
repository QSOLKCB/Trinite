"""Controller failure doubles; never evidence of original-checkpoint admission."""
from argparse import Namespace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from frozen_music_math_v1 import artifacts as a
from frozen_music_math_v1 import workflow as w
from frozen_music_math_v1 import protocol as p
from test_receipt import synthetic_original


class ControllerFailureDoubles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original-source'; self.source.mkdir()
        self.original = self.root / 'original-run'; self.original.mkdir()
        original_receipt = synthetic_original(self.original)
        self.request = self.root / 'request.json'
        self.raw = a.encoded({'synthetic': 'controller-double-only'})
        self.request.write_bytes(self.raw)
        self.bound = {'original': original_receipt}
        self.args = Namespace(authorize_inference=True,
            original_source=str(self.source), original_run=str(self.original),
            original_request_identity=original_receipt['request_identity'],
            original_matrix_identity=original_receipt['matrix_identity'],
            request=str(self.request), request_identity=a.digest(self.raw),
            output=str(self.root / 'probe'))

    def test_authorization_checked_before_read_or_worker(self):
        self.args.authorize_inference = False
        with patch.object(w, 'launch', side_effect=AssertionError('worker called')):
            with self.assertRaisesRegex(ValueError, 'authorization'):
                w.run(self.args)
        self.assertFalse(Path(self.args.output).exists())

    def test_failure_and_partial_output_retained_without_retries(self):
        def failed_worker(action, args, log, extra):
            Path(log).write_bytes(b'synthetic failure log\n')
            (Path(args.output) / 'predictions.jsonl').write_bytes(b'{"partial":true}\n')
            raise ValueError('synthetic worker failure')
        with patch.object(w, 'validate_request', return_value=self.bound), \
                patch.object(w, 'launch', side_effect=failed_worker) as launched:
            result = w.run(self.args)
        self.assertEqual(launched.call_count, 1)
        self.assertEqual(result['outcome'], 'failed')
        output = Path(self.args.output)
        report = a.parse(a.read(output / 'report.json'))
        self.assertEqual(report['failure']['message'], 'synthetic worker failure')
        self.assertIsNone(report['summary'])
        self.assertTrue(report['original_inventory_unchanged'])
        self.assertEqual(a.read(output / 'predictions.jsonl'), b'{"partial":true}\n')
        self.assertEqual(a.inventory(self.original), self.bound['original']['inventory'])

    def test_timeout_preserved_as_failure(self):
        with patch.object(w, 'validate_request', return_value=self.bound), \
                patch.object(w, 'launch', side_effect=subprocess.TimeoutExpired('synthetic-worker', 600)):
            result = w.run(self.args)
        report = a.parse(a.read(Path(self.args.output) / 'report.json'))
        self.assertEqual(result['outcome'], 'failed')
        self.assertEqual(report['failure']['type'], 'TimeoutExpired')

    def test_existing_output_never_replaced(self):
        output = Path(self.args.output); output.mkdir()
        (output / 'keep').write_bytes(b'historical')
        with patch.object(w, 'validate_request', return_value=self.bound), \
                patch.object(w, 'launch', side_effect=AssertionError('worker called')):
            with self.assertRaises(FileExistsError):
                w.run(self.args)
        self.assertEqual(a.read(output / 'keep'), b'historical')

    def test_input_mutation_prevents_success_even_after_worker_return(self):
        def mutate(action, args, log, extra):
            (self.original / 'matrix.json').write_bytes(b'changed by synthetic double')
            raise ValueError('synthetic failure')
        with patch.object(w, 'validate_request', return_value=self.bound), \
                patch.object(w, 'launch', side_effect=mutate):
            result = w.run(self.args)
        report = a.parse(a.read(Path(self.args.output) / 'report.json'))
        self.assertEqual(result['outcome'], 'failed')
        self.assertFalse(report['original_inventory_unchanged'])

    def completed_double(self, action, args, log, extra):
        """Constant synthetic outputs; no native load or model execution."""
        Path(log).write_bytes(b'synthetic controller completion only\n')
        output = Path(args.output)
        rows = [{'subset': subset, 'seed': seed, 'step': step,
                 'question_id': q['question_id'], 'generated_tokens': [54, 257]}
                for subset, seed, step in p.CHECKPOINTS for q in p.input_questions()]
        (output / 'predictions.jsonl').write_bytes(b''.join(a.encoded(row) for row in rows))
        (output / 'worker-completion.json').write_bytes(a.encoded({
            'decodes': 256, 'forward_calls': 512, 'training_performed': False,
            'model_identities_unchanged': True, 'original_inventory_unchanged': True}))

    def test_synthetic_complete_audit_and_prediction_tamper(self):
        with patch.object(w, 'validate_request', return_value=self.bound), \
                patch.object(w, 'launch', side_effect=self.completed_double):
            result = w.run(self.args)
            self.assertEqual(result['outcome'], 'completed')
            self.args.expected_identity = result['report_identity']
            observed = w.audit(self.args)
            self.assertTrue(observed['verified'])
            self.assertFalse(observed['target_inference_performed'])
            output = Path(self.args.output)
            journal = output / 'predictions.jsonl'
            journal.write_bytes(journal.read_bytes().replace(b'[54,257]', b'[55,257]', 1))
            with self.assertRaisesRegex(ValueError, 'inventory'):
                w.audit(self.args)

    def test_report_and_external_identity_tamper(self):
        with patch.object(w, 'validate_request', return_value=self.bound), \
                patch.object(w, 'launch', side_effect=self.completed_double):
            result = w.run(self.args)
            self.args.expected_identity = 'sha256:' + 'f'*64
            with self.assertRaisesRegex(ValueError, 'external report identity'):
                w.audit(self.args)
            path = Path(self.args.output) / 'report.json'
            report = a.parse(a.read(path))
            report['causal_transfer_established'] = True
            raw = a.encoded(report); path.write_bytes(raw)
            self.args.expected_identity = a.digest(raw)
            with self.assertRaisesRegex(ValueError, 'unsupported claim'):
                w.audit(self.args)

    def test_request_validation_checks_original_receipt_structure(self):
        value = {'schema': w.REQUEST_SCHEMA, 'probe_source': w.sources(),
                 'protocol': p.PROTOCOL, 'questions': p.input_questions(),
                 'historical_overlap': p.historical_overlap(w.ROOT),
                 'original': self.bound['original']}
        raw = a.encoded(value)
        self.assertEqual(w.validate_request(raw, a.digest(raw)), value)
        value['original'] = {'inventory': self.bound['original']['inventory']}
        raw = a.encoded(value)
        with self.assertRaisesRegex(ValueError, 'original receipt'):
            w.validate_request(raw, a.digest(raw))

    def test_audit_rejects_attempt_authorization_even_if_rehashed(self):
        with patch.object(w, 'validate_request', return_value=self.bound), \
                patch.object(w, 'launch', side_effect=self.completed_double):
            w.run(self.args)
            output = Path(self.args.output)
            attempt = a.parse(a.read(output / 'attempt.json'))
            attempt['authorization'] = 'not-authorized'
            (output / 'attempt.json').write_bytes(a.encoded(attempt))
            report = a.parse(a.read(output / 'report.json'))
            report['files']['attempt.json'] = a.digest(a.read(output / 'attempt.json'))
            raw = a.encoded(report); (output / 'report.json').write_bytes(raw)
            self.args.expected_identity = a.digest(raw)
            with self.assertRaisesRegex(ValueError, 'attempt authorization'):
                w.audit(self.args)

    def test_original_binding_args_are_mandatory(self):
        self.args.original_matrix_identity = None
        with self.assertRaisesRegex(ValueError, 'independent identities'):
            w.original_args(self.args)


if __name__ == '__main__':
    unittest.main()
