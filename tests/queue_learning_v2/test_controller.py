"""Synthetic controller lifecycle; no numerical updates or scientific outcomes."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import read_json, write_json
from queue_learning_v2 import controller, numerical, protocol


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frozen = self.root / 'frozen'
        self.frozen.mkdir()
        self.rows = [{k: row[k] for k in ('id', 'family', 'split')} |
                     {'observation_sha256': gate.digest(b'{}\n')} for row in protocol.episodes()]
        self.heldout = [r for r in self.rows if r['split'] != 'train']
        for row in self.heldout:
            public = self.frozen / row['id'] / 'public'
            public.mkdir(parents=True)
            gate.write(public / 'observation.json', b'{}\n')
        self.args = SimpleNamespace(frozen=self.frozen, out=self.root / 'run',
            run_anchors=self.root / 'run-anchors', freeze_sha='a' * 64,
            anchors=self.root / 'preparation.json', anchors_sha='b' * 64,
            approval=self.root / 'approval.json')
        self.events = []
        self.fail_worker = None
        self.fail_checkpoint = None
        self.partial_failure = False
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(controller.admission, 'frozen_context',
            return_value=(self.frozen, {'episodes': self.rows}, {})))
        self.authorize = self.patches.enter_context(patch.object(controller.admission, 'authorize'))
        self.launch = self.patches.enter_context(patch.object(controller, 'launch', side_effect=self.worker))
        self.checkpoint = self.patches.enter_context(patch.object(controller, 'validate_checkpoint', side_effect=self.validate))
        self.patches.enter_context(patch.object(controller.evaluation, 'constants', side_effect=self.constants))
        self.patches.enter_context(patch.object(controller.evaluation, 'causal', return_value={'status': 'abstain'}))
        self.commit = self.patches.enter_context(patch.object(controller.evaluation, 'commit_all', side_effect=self.commit_all))
        self.score = self.patches.enter_context(patch.object(controller.evaluation, 'score_all', side_effect=self.score_all))

    def worker(self, arguments, seconds, progress=None):
        name = Path(arguments[arguments.index('--out') + 1]).name
        if arguments[0] == '_validate':
            checkpoint = Path(arguments[arguments.index('--checkpoint') + 1])
            try:
                entry = self.validate(checkpoint, 17, '--fit' in arguments, self.args.freeze_sha)
            except ValueError:
                return {'exit_code': 2, 'failure': None}
            folder = Path(arguments[arguments.index('--out') + 1])
            folder.mkdir()
            write_json(folder / 'validated.json', entry)
            return {'exit_code': 0, 'failure': None}
        self.events.append('launch:' + name)
        self.assertGreater(seconds, 0)
        self.assertLessEqual(seconds, 900)
        if name == self.fail_worker:
            if self.partial_failure and arguments[0] == '_infer':
                folder = Path(arguments[arguments.index('--out') + 1])
                folder.mkdir()
                gate.write(folder / 'attempts.jsonl', b''.join(gate.encode({'id': r['id']}) for r in self.heldout[:2]))
                gate.write(folder / 'raw.jsonl', gate.encode({'id': self.heldout[0]['id'], 'raw': {'status': 'abstain'}}))
            return {'exit_code': -9, 'failure': 'timeout'}
        folder = Path(arguments[arguments.index('--out') + 1])
        folder.mkdir()
        if arguments[0] == '_train':
            if name != 'fit':
                self.assertIn('--fit-gate-sha', arguments)
                self.assertTrue((self.args.run_anchors / 'fit-gate.json').exists())
            write_json(folder / 'training.json', {'synthetic': True})
            write_json(folder / 'train-data.json', [{'split': 'train', 'answer': 1}] * 256)
        else:
            self.assertTrue((self.args.run_anchors / 'checkpoints.json').exists())
            gate.write(folder / 'attempts.jsonl', b''.join(gate.encode({'id': r['id']}) for r in self.heldout))
            gate.write(folder / 'raw.jsonl', b''.join(gate.encode({'id': r['id'], 'raw': {'status': 'abstain'}})
                                                     for r in self.heldout))
        return {'exit_code': 0, 'failure': None}

    def validate(self, folder, seed, fit, binding):
        self.events.append('validate:' + folder.name)
        if folder.name == self.fail_checkpoint:
            raise ValueError('seven of eight / incomplete / nonfinite synthetic checkpoint')
        return {'checkpoint_sha256': ['c' * 64, 'd' * 64],
                'train_data_sha256': 'e' * 64, 'training_sha256': gate.digest(gate.read(folder / 'training.json'))}

    def constants(self, rows):
        self.events.append('constants')
        self.assertEqual(len(rows), 256)
        self.assertTrue(all(r['split'] == 'train' for r in rows))
        return {'median': 1, 'mode': 1}

    def commit_all(self, frozen, rows, predictions, out, barrier):
        self.events.append('commit-all')
        self.assertEqual(len(predictions), 640)
        self.assertEqual(set(predictions), {(method, r['id']) for method in controller.evaluation.METHODS for r in self.heldout})
        self.assertTrue(all(v['status'] != 'not_attempted' for v in predictions.values()))
        return '1' * 64

    def score_all(self, *args):
        self.assertEqual(self.events[-1], 'commit-all')
        self.events.append('score-all')
        raw = {'schema': 'trinite.queue-learning-summary.v1', 'synthetic_engineering_only': True,
               'numeric_exact_match': {'numerator': 0, 'denominator': 0}}
        args[-1].mkdir()
        write_json(args[-1] / 'summary.json', raw)
        return raw

    def test_full_exact_sequence_and_all_output_barrier(self):
        result = controller.run(self.args)
        self.assertFalse(result['failed_workers'])
        self.assertEqual(self.events, ['launch:fit', 'validate:fit', 'launch:seed17', 'validate:seed17',
            'launch:seed29', 'validate:seed29', 'constants', 'launch:infer-17', 'launch:infer-29', 'commit-all', 'score-all'])
        self.assertEqual([r['status'] for r in read_json(self.args.out / 'attempts.json')], ['completed'] * 3)
        raw = read_json(self.args.out / 'scores/summary.json')
        normalized = read_json(self.args.out / 'scores/summary-v2.json')
        self.assertEqual(raw['numeric_exact_match'], {'numerator': 0, 'denominator': 0})
        self.assertIsNone(normalized['numeric_exact_match'])
        self.assertEqual(normalized['schema'], 'trinite.queue-learning-summary.v2')
        self.assertEqual(read_json(self.args.out / 'run.json')['summary'], normalized)

    def test_failed_fit_checkpoint_stops_both_main_cells(self):
        self.fail_checkpoint = 'fit'
        result = controller.run(self.args)
        self.assertTrue(result['failed_workers'])
        self.assertEqual(self.events, ['launch:fit', 'validate:fit'])
        self.assertEqual([r['status'] for r in read_json(self.args.out / 'attempts.json')], ['failure', 'not_attempted', 'not_attempted'])
        self.commit.assert_not_called()
        self.score.assert_not_called()
        summary = read_json(self.args.out / 'run.json')['summary']
        for method in controller.evaluation.METHODS:
            for split in ('validation', 'test'):
                self.assertEqual(summary['methods'][method][split]['planned'], 64)
                self.assertEqual(summary['methods'][method][split]['statuses']['not_attempted'], 64)

    def test_worker_failure_is_terminal_without_retry_or_scores(self):
        self.fail_worker = 'seed17'
        result = controller.run(self.args)
        self.assertTrue(result['failed_workers'])
        self.assertEqual(self.events, ['launch:fit', 'validate:fit', 'launch:seed17'])
        self.assertEqual([r['status'] for r in read_json(self.args.out / 'attempts.json')], ['completed', 'timeout', 'not_attempted'])
        self.score.assert_not_called()

    def test_inference_failure_prevents_all_heldout_scoring(self):
        self.fail_worker = 'infer-17'
        result = controller.run(self.args)
        self.assertTrue(result['failed_workers'])
        self.assertNotIn('launch:infer-29', self.events)
        self.commit.assert_not_called()
        self.score.assert_not_called()

    def test_partial_inference_retains_raw_failure_and_unattempted_denominators(self):
        self.fail_worker, self.partial_failure = 'infer-17', True
        result = controller.run(self.args)
        self.assertTrue(result['failed_workers'])
        self.score.assert_not_called()
        summary = read_json(self.args.out / 'run.json')['summary']
        statuses = [summary['methods']['seed17'][split]['statuses'] for split in ('validation', 'test')]
        self.assertEqual(sum(s['abstain'] for s in statuses), 1)
        self.assertEqual(sum(s['timeout'] for s in statuses), 1)
        self.assertEqual(sum(s['not_attempted'] for s in statuses), 126)
        self.assertTrue((self.args.out / 'infer-17/raw.jsonl').exists())

    def test_fit_timeout_prevents_main_attempts(self):
        self.fail_worker = 'fit'
        self.assertTrue(controller.run(self.args)['failed_workers'])
        self.assertEqual(self.events, ['launch:fit'])
        self.score.assert_not_called()

    def test_whole_study_deadline_stops_before_first_attempt(self):
        with patch.object(controller.time, 'monotonic', side_effect=[0., 2700., 2701.]):
            self.assertTrue(controller.run(self.args)['failed_workers'])
        self.launch.assert_not_called()
        self.score.assert_not_called()

    def test_approval_failure_launches_nothing(self):
        self.authorize.side_effect = ValueError('exact approval missing')
        result = controller.run(self.args)
        self.assertTrue(result['failed_workers'])
        self.launch.assert_not_called()
        self.score.assert_not_called()

    def test_consumed_freeze_cannot_retry_in_new_output_directory(self):
        self.fail_worker = 'fit'
        self.assertTrue(controller.run(self.args)['failed_workers'])
        self.launch.reset_mock()
        self.args.out = self.root / 'retry'
        self.args.run_anchors = self.root / 'retry-anchors'
        self.assertTrue(controller.run(self.args)['failed_workers'])
        self.launch.assert_not_called()


class CheckpointGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.rows = [{'identity': f'row-{i}', 'answer': i} for i in range(8)]
        self.report = {'ids': [r['identity'] for r in self.rows], 'generation': {
            'planned': 8, 'exact': 8, 'records': [{'id': r['identity'], 'exact': True,
                'tokens': [48 + r['answer'], 257], 'parsed': {'status': 'prediction'}} for r in self.rows]}}
        diagnostics = {}
        for label, step in [('initial', 0), ('final', 128)]:
            diagnostics[label + '_sha256'] = write_json(self.folder / (label + '-diagnostics.json'), {'step': step})
        self.info = {'outcome': 'completed', 'fit': True, 'seed': 17, 'step': 128,
            'target_tokens': 2816, 'targets': 2816, 'update_seconds_hex': (1.).hex(),
            'train_data_sha256': write_json(self.folder / 'train-data.json', self.rows),
            'checkpoint_sha256': ['a' * 64, 'b' * 64], 'schedule': {'synthetic': True},
            'diagnostics': diagnostics, 'fit_report': self.report}
        gate.write(self.folder / 'tensors.safetensors', b'engineering-placeholder')
        gate.write(self.folder / 'metadata.json', b'{}\n')
        self.state = SimpleNamespace(step=128, target_tokens=2816, model=object())
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(numerical, 'restore', return_value=self.state))
        self.stack.enter_context(patch.object(numerical, 'schedule', return_value={'synthetic': True}))
        self.stack.enter_context(patch.object(protocol, 'fit_report', side_effect=lambda *args: self.report))
        self.stack.enter_context(patch.object(controller, 'fit_rows', return_value=self.rows))

    def validate(self):
        path = self.folder / 'training.json'
        if path.exists():
            path.unlink()  # This is only a synthetic test fixture, never a retained run.
        write_json(path, self.info)
        return controller.validate_checkpoint(self.folder, 17, True, 'f' * 64)

    def test_only_final_eight_canonical_eos_outputs_pass(self):
        self.assertIn('training_sha256', self.validate())

    def test_seven_of_eight_report_cannot_pass(self):
        self.report['generation']['exact'] = 7
        with self.assertRaises(ValueError):
            self.validate()

    def test_early_success_or_nonfinite_timing_rejected(self):
        for field, value in [('step', 1), ('step', 127), ('update_seconds_hex', 'nan'),
                             ('update_seconds_hex', (120.).hex()), ('target_tokens', 2815)]:
            original = self.info[field]
            self.info[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.validate()
            self.info[field] = original

    def test_claimed_exact_without_eos_is_rejected(self):
        self.report['generation']['records'][0]['tokens'] = [48]
        with self.assertRaises(ValueError):
            self.validate()

    def test_safe_restored_step_must_match_report(self):
        self.state.step = 127
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
