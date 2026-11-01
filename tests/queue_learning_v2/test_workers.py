"""Worker lifecycle doubles: never execute an optimizer update."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from queue_learning_v1.artifacts import read_json
from queue_learning_v2 import workers


class WorkerLifecycle(unittest.TestCase):
    def execute(self, *, fit=True, failure=None, retry=False):
        from queue_learning_v2 import admission, numerical, protocol
        import trinite.training
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(frozen=str(root / 'frozen'), freeze_sha='a' * 64,
                anchors=str(root / 'anchors'), anchors_sha='b' * 64,
                approval=str(root / 'approval'), out=str(root / 'out'), seed=17, fit=fit,
                fit_checkpoint='fit', fit_gate='gate', fit_gate_sha='c' * 64)
            state = SimpleNamespace(step=0, target_tokens=0, model=object())
            calls, clock = [], [0.0]
            def step(s, *, stop_after):
                calls.append(stop_after)
                if failure == 'exception':
                    raise ValueError('synthetic worker failure')
                s.step = stop_after
                s.target_tokens = (2816 if fit else 5600) * stop_after // (128 if fit else 256)
                if failure == 'timeout':
                    clock[0] = 120.0
                if failure == 'incomplete':
                    s.step -= 1
            patches = [
                patch.object(admission, 'frozen_context', return_value=(root / 'frozen', {'environment': {}}, {})),
                patch.object(admission, 'authorize'), patch.object(admission, 'training_rows', return_value=[]),
                patch.object(numerical, 'configure', return_value={}),
                patch.object(numerical, 'state_for', return_value=state),
                patch.object(numerical, 'snapshot', return_value=(b'tensors', b'metadata')),
                patch.object(numerical, 'schedule', return_value={}),
                patch.object(protocol, 'plan', return_value=SimpleNamespace(steps=128 if fit else 256)),
                patch.object(protocol, 'fit_report', return_value={'meets_fit_criterion': True}),
                patch.object(workers, 'diagnostics', return_value={}),
                patch.object(workers.time, 'monotonic', side_effect=lambda: clock[0]),
                patch.object(trinite.training, 'run_steps', side_effect=step),
            ]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                if not fit:
                    stack.enter_context(patch('queue_learning_v2.controller.validate_fit_certificate'))
                if failure:
                    with self.assertRaises(ValueError):
                        workers.train_worker(args)
                else:
                    workers.train_worker(args)
                if retry:
                    prior_calls = calls[:]
                    args.out = str(root / 'retry-out')
                    with self.assertRaises(FileExistsError):
                        workers.train_worker(args)
                    self.assertEqual(calls, prior_calls)
                    self.assertTrue((root / 'retry-out/failure-detail.json').exists())
            out = root / 'out'
            if failure:
                self.assertFalse((out / 'training.json').exists())
                self.assertTrue((out / 'failure-detail.json').exists())
                self.assertTrue((out / 'partial.safetensors').exists())
                self.assertEqual(calls, [1])
            else:
                record = read_json(out / 'training.json')
                self.assertEqual(record['step'], 128 if fit else 256)
                self.assertEqual(calls, list(range(1, record['step'] + 1)))
                self.assertEqual(read_json(out / 'progress.json')['phase'], 'completed')
                self.assertTrue((out / 'initial-diagnostics.json').exists())
                self.assertTrue((out / 'final-diagnostics.json').exists())
            self.assertTrue((out / 'resources.json').exists())

    def test_same_freeze_attempt_cannot_retry_after_success(self):
        self.execute(retry=True)

    def test_same_freeze_attempt_cannot_retry_after_failure(self):
        self.execute(failure='exception', retry=True)

    def test_checkpoint_external_anchor_rejected_before_loading(self):
        from queue_learning_v2 import numerical
        with patch.object(numerical, 'state_for') as initialize:
            with self.assertRaises(ValueError):
                numerical.restore(b'tensor', b'{}', [], 17, 'a' * 64, ['b' * 64, 'c' * 64])
            initialize.assert_not_called()

    def test_foreign_checkpoint_schema_rejected(self):
        import queue_evaluation_v1 as gate
        from trinite.contracts import json_bytes
        from queue_learning_v2 import numerical
        old = json_bytes({'schema': 'trinite.queue-learning-checkpoint.v1'})
        fresh = json_bytes({'schema': 'trinite.queue-learning-checkpoint.v2'})
        payload = b'engineering tensor double'
        with patch.object(numerical, 'state_for', return_value=object()), \
             patch.object(numerical, 'snapshot', return_value=(payload, fresh)):
            with self.assertRaises(ValueError):
                numerical.restore(payload, old, [], 17, 'a' * 64,
                                  [gate.digest(payload), gate.digest(old)])

    def test_detached_diagnostic_reconciliation(self):
        from queue_learning_v2 import diagnostics
        base = {'answer': {'targets': 2, 'nll_sum_hex': (2.0).hex()},
                'eos': {'targets': 1, 'nll_sum_hex': (1.0).hex()},
                'native': {'targets': 3, 'mean_nll_hex': (1.0).hex()}}
        with patch.object(diagnostics, 'supervision', return_value=base), \
             patch.object(diagnostics, 'generation', return_value={}):
            self.assertEqual(workers.diagnostics(object(), [])['supervision'], base)
            for bad in (float('nan'), float('inf'), 5.0):
                base['native']['mean_nll_hex'] = bad.hex()
                with self.assertRaises(ValueError):
                    workers.diagnostics(object(), [])
            base['native']['mean_nll_hex'] = (1.0).hex()
            base['native']['targets'] = 2
            with self.assertRaises(ValueError):
                workers.diagnostics(object(), [])

    def test_fit_early_success_still_executes_exact_horizon(self):
        self.execute()

    def test_main_exact_horizon(self):
        self.execute(fit=False)

    def test_failure_retained_without_retry(self):
        self.execute(failure='exception')

    def test_update_deadline_at_limit_rejected(self):
        self.execute(failure='timeout')

    def test_incomplete_update_rejected(self):
        self.execute(failure='incomplete')


if __name__ == '__main__':
    unittest.main()
