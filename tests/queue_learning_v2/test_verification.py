"""Read-only verifier ordering with synthetic receipts; no numerical workers."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from queue_learning_v1.artifacts import inventory, write_json, read_json
from trinite.contracts import json_bytes
from queue_learning_v2 import verification as v
from queue_learning_v2.controller import ATTEMPTS, empty_predictions, stopped_summary
from queue_learning_v2.protocol import episodes


class VerificationTests(unittest.TestCase):
    @staticmethod
    def receipt():
        return {'exit_code': 0, 'failure': None,
                'limits': {'rss_bytes_per_process': 1 << 30, 'aggregate_rss_bytes': 2 << 30,
                           'address_space_bytes_per_process': 4 << 30},
                'wall_seconds_hex': (1.).hex(), 'deadline_seconds_hex': (2700.).hex(),
                'peak_process_rss_bytes': 1024, 'peak_aggregate_rss_bytes': 2048, 'reaped': []}

    def fixture(self, root, *, incomplete=False, success=False):
        destination, anchors = root / 'run', root / 'anchors'
        destination.mkdir(); anchors.mkdir()
        manifest = {'episodes': episodes()}
        if not incomplete:
            write_json(destination / 'supervisor.json', self.receipt())
        if incomplete:
            write_json(destination / 'supervisor.json', {'exit_code': -9, 'failure': 'timeout'})
        else:
            planned = [{'name': n, 'seed': s, 'fit': f, 'planned_updates': u, 'planned_targets': t,
                        'status': 'not_attempted'} for n, s, f, u, t in ATTEMPTS]
            write_json(destination / 'planned-attempts.json', planned)
            actual = [dict(x) for x in planned]
            if success:
                for row in actual:
                    row['status'] = 'completed'
            else:
                actual[0]['status'] = 'failure'
            write_json(destination / 'attempts.json', actual)
            write_json(destination / 'predictions.json', [
                {'method': m, 'id': identity, 'raw': raw}
                for (m, identity), raw in sorted(empty_predictions(manifest).items())])
            write_json(destination / 'run.json', {'schema': 'trinite.queue-learning-run.v2',
                'freeze_sha256': 'a' * 64, 'optimizer_replay': False, 'learning_claim': False,
                'stopped': None if success else 'fit failed', 'workers': {},
                'barrier_sha256': 'b' * 64,
                'scoring_started': success, 'scoring_completed': success,
                'summary': stopped_summary(empty_predictions(manifest)) |
                           {'heldout_scored': success, 'scoring_completed': success}})
        sha = write_json(anchors / 'final-inventory.json', inventory(destination))
        args = SimpleNamespace(run=str(destination), run_anchors=str(anchors), run_sha=sha,
            frozen=str(root / 'frozen'), freeze_sha='a' * 64,
            anchors=str(root / 'prepares'), anchors_sha='c' * 64)
        return args, manifest

    def reseal(self, args):
        path = Path(args.run_anchors) / 'final-inventory.json'
        path.unlink()
        args.run_sha = write_json(path, inventory(Path(args.run)))

    def test_clean_stopped_exit_two_still_checks_stopped_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifest = self.fixture(Path(folder))
            receipt = self.receipt(); receipt['exit_code'] = 2
            (Path(args.run) / 'supervisor.json').write_bytes(json_bytes(receipt))
            self.reseal(args)
            with patch.object(v, '_public_context', return_value=(Path(args.frozen), manifest)), \
                 patch.object(v.admission, 'frozen_context', side_effect=AssertionError('no private verification')):
                result = v.verify(args)
            self.assertIn('closed stopped attempt', result['scope'])
            self.assertFalse(result['complete_study'])
            path = Path(args.run) / 'run.json'
            record = read_json(path); record['summary']['heldout_scored'] = True
            path.write_bytes(json_bytes(record)); self.reseal(args)
            with patch.object(v, '_public_context', return_value=(Path(args.frozen), manifest)):
                with self.assertRaisesRegex(ValueError, 'stopped v2 summary'):
                    v.verify(args)

    def test_outer_failure_overrides_completed_inner_run(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifest = self.fixture(Path(folder), success=True)
            receipt = self.receipt(); receipt['failure'] = 'rss_limit'
            (Path(args.run) / 'supervisor.json').write_bytes(json_bytes(receipt))
            self.reseal(args)
            with patch.object(v, '_public_context', return_value=(Path(args.frozen), manifest)), \
                 patch.object(v.evaluation, 'public_barrier', return_value=[]), \
                 patch.object(v.admission, 'frozen_context', side_effect=AssertionError('no private verification')):
                result = v.verify(args)
            self.assertFalse(result['complete_study'])
            self.assertFalse(result['heldout_verified'])
            self.assertIn('outer resource', result['scope'])

    def test_partial_scoring_retained_without_truth_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifest = self.fixture(Path(folder))
            path = Path(args.run) / 'run.json'
            record = read_json(path)
            record['scoring_started'] = True
            record['summary']['heldout_scored'] = True
            path.write_bytes(json_bytes(record))
            scores = Path(args.run) / 'scores'; scores.mkdir()
            write_json(scores / 'partial.json', {'retained': 'partial scoring failure'})
            self.reseal(args)
            with patch.object(v, '_public_context', return_value=(Path(args.frozen), manifest)), \
                 patch.object(v.evaluation, 'public_barrier', return_value=[]) as barrier, \
                 patch.object(v.admission, 'frozen_context', side_effect=AssertionError('no private verification')), \
                 patch.object(v.gate, 'verify', side_effect=AssertionError('no score rerun')):
                result = v.verify(args)
            barrier.assert_called_once()
            self.assertIn('partial scores retained unverified', result['scope'])
            self.assertFalse(result['heldout_verified'])

    def test_success_receipt_requires_exact_resource_limits_and_wall(self):
        receipt = self.receipt()
        self.assertTrue(v._healthy(receipt, 2700))
        for field, value in [('peak_process_rss_bytes', 1 << 30),
                             ('peak_aggregate_rss_bytes', 2 << 30),
                             ('wall_seconds_hex', (2700.).hex())]:
            altered = dict(receipt); altered[field] = value
            self.assertFalse(v._healthy(altered, 2700))
        altered = dict(receipt); altered['limits'] = {}
        self.assertFalse(v._healthy(altered, 2700))
        altered = dict(receipt); altered['wall_seconds_hex'] = float('nan').hex()
        with self.assertRaises(ValueError):
            v._healthy(altered, 2700)

    def test_stopped_verification_reads_no_truth_and_changes_no_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifest = self.fixture(Path(folder))
            before = inventory(Path(args.run))
            with patch.object(v, '_public_context', return_value=(Path(args.frozen), manifest)), \
                 patch.object(v.admission, 'frozen_context', side_effect=AssertionError('private preparations forbidden')), \
                 patch.object(v.gate, 'run', side_effect=AssertionError('truth read forbidden')), \
                 patch.object(v.gate, 'verify', side_effect=AssertionError('truth verification forbidden')):
                result = v.verify(args)
            self.assertFalse(result['complete_study'])
            self.assertFalse(result['heldout_verified'])
            self.assertEqual(before, inventory(Path(args.run)))

    def test_killed_study_is_only_incomplete_inventory_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifest = self.fixture(Path(folder), incomplete=True)
            with patch.object(v.admission, 'frozen_context', side_effect=AssertionError('inventory only')):
                result = v.verify(args)
            self.assertFalse(result['complete_study'])
            self.assertIn('incomplete', result['scope'])

    def test_changed_artifacts_rejected_before_frozen_context(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = self.fixture(Path(folder))
            (Path(args.run) / 'extra').write_text('changed')
            with patch.object(v.admission, 'frozen_context') as context:
                with self.assertRaises(ValueError):
                    v.verify(args)
                context.assert_not_called()

    def test_private_preparations_only_after_successful_public_barrier(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifest = self.fixture(Path(folder), success=True)
            order = []
            def barrier(*args):
                order.append('public barrier')
                return []
            def context(*args):
                order.append('private preparations')
                raise ValueError('stop after ordering assertion')
            with patch.object(v, '_public_context', return_value=(Path(args.frozen), manifest)), \
                 patch.object(v.evaluation, 'public_barrier', side_effect=barrier), \
                 patch.object(v.admission, 'frozen_context', side_effect=context), \
                 patch.object(v, '_successful_attempt_receipts'):
                with self.assertRaisesRegex(ValueError, 'stop after ordering'):
                    v.verify(args)
            self.assertEqual(order, ['public barrier', 'private preparations'])

    def test_bad_public_barrier_precedes_checkpoint_and_truth_access(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifest = self.fixture(Path(folder), success=True)
            with patch.object(v, '_public_context', return_value=(Path(args.frozen), manifest)), \
                 patch.object(v.admission, 'frozen_context', side_effect=AssertionError('private preparations forbidden')), \
                 patch.object(v.evaluation, 'public_barrier', side_effect=ValueError('bad public barrier')), \
                 patch.object(v, 'validate_checkpoint') as checkpoint, \
                 patch.object(v.gate, 'verify') as truth:
                with self.assertRaisesRegex(ValueError, 'bad public barrier'):
                    v.verify(args)
                checkpoint.assert_not_called(); truth.assert_not_called()


if __name__ == '__main__':
    unittest.main()
