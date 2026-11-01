"""Engineering-only admission tests: synthetic receipts, no optimizer/target run."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import write_json
from queue_learning_v2 import admission
from queue_learning_v2.spec import PROTOCOL


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.git = {'clean': True, 'head': 'a' * 40, 'main': 'a' * 40}
        self.source = {'fake.py': 'b' * 64}
        self.environment = {'synthetic': True}
        self.proof = {'schema': 'trinite.queue-learning-feasibility.v2',
            'sources': self.source, 'git': self.git, 'binary_sha256': 'c' * 64,
            'protocol': PROTOCOL, 'peak_rss_kib': 4096,
            'supervisor': {'exit_code': 0, 'failure': None,
                'limits': {'rss_bytes_per_process': 1073741824, 'aggregate_rss_bytes': 2147483648,
                           'address_space_bytes_per_process': 4294967296},
                'peak_process_rss_bytes': 1024, 'peak_aggregate_rss_bytes': 2048,
                'wall_seconds_hex': (20.).hex(), 'deadline_seconds_hex': (900.).hex()},
            'admission_import_setup_seconds_hex': (2.).hex(),
            'conservative_whole_attempt_seconds_hex': (28.).hex(),
            'kernel': {'environment': self.environment, 'parameters': 24768,
                'optimizer_updates': 0, 'gradient_calls': 3,
                'kernel_seconds_hex': [(0.01).hex()] * 3,
                'conservative_256_update_estimate_seconds_hex': (0.01 * 256 * 4).hex(),
                'diagnostic_io_seconds_hex': (1.).hex(),
                'conservative_whole_attempt_seconds_hex': (20.).hex()}}

    def validate(self, proof=None):
        with patch.object(gate, 'binary_identity', return_value='c' * 64):
            admission.validate_feasibility(proof or self.proof, self.source, self.git, self.environment)

    def test_fresh_feasibility(self):
        self.validate()

    def test_feasibility_rejects_old_schema_sources_runtime_and_git(self):
        for field, value in [('schema', 'trinite.queue-learning-feasibility.v1'),
                             ('sources', {}), ('git', dict(self.git, head='d' * 40)),
                             ('binary_sha256', 'e' * 64), ('protocol', {})]:
            with self.subTest(field=field):
                proof = deepcopy(self.proof)
                proof[field] = value
                with self.assertRaises(ValueError):
                    self.validate(proof)
        proof = deepcopy(self.proof)
        proof['kernel']['environment'] = {}
        with self.assertRaises(ValueError):
            self.validate(proof)

    def test_feasibility_rejects_steps_timing_nonfinite_and_memory(self):
        for key, value in [('optimizer_updates', 1), ('optimizer_updates', False),
                           ('gradient_calls', 2), ('parameters', 24769),
                           ('kernel_seconds_hex', ['nan'] * 3),
                           ('conservative_256_update_estimate_seconds_hex', (10.).hex()),
                           ('diagnostic_io_seconds_hex', (-1.).hex()),
                           ('conservative_whole_attempt_seconds_hex', (900.).hex()),
                           ('conservative_whole_attempt_seconds_hex', (11.).hex())]:
            with self.subTest(key=key, value=value):
                proof = deepcopy(self.proof)
                proof['kernel'][key] = value
                with self.assertRaises(ValueError):
                    self.validate(proof)
        self.proof['peak_rss_kib'] = 1048576
        with self.assertRaises(ValueError):
            self.validate()

    def test_unsealed_failed_or_overlimit_supervision_cannot_freeze(self):
        proof = deepcopy(self.proof)
        del proof['supervisor']
        with self.assertRaises(ValueError):
            self.validate(proof)
        for key, value in [('exit_code', 1), ('failure', 'process_rss'),
                           ('limits', {}), ('peak_process_rss_bytes', 1073741824),
                           ('peak_aggregate_rss_bytes', 2147483648),
                           ('wall_seconds_hex', (900.).hex()), ('deadline_seconds_hex', (901.).hex())]:
            proof = deepcopy(self.proof)
            proof['supervisor'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(proof)

    def test_exact_source_specific_inventory(self):
        rows = admission.episodes()
        record = admission.record(rows, self.source)
        self.assertEqual(record['counts'], {'train': 256, 'validation': 64, 'test': 64, 'families': 48})
        admission.validate_admission(record, rows, self.source)
        for altered in [rows[:-1], list(reversed(rows)), admission.historical()]:
            with self.assertRaises(ValueError):
                admission.record(altered, self.source)
        with self.assertRaises(ValueError):
            admission.validate_admission(record, rows, {})

    def test_v1_or_wrong_freeze_approval_cannot_authorize(self):
        freeze_sha = 'f' * 64
        approval = {'schema': 'trinite.queue-learning-owner-approval.v2', 'freeze_sha256': freeze_sha,
            'implementation_commit': self.git['head'], 'approved_by': 'Trent Slade',
            'reviewed_merged_green': True,
            'authorization': 'execute queue-learning-v2 fit seed 17 once; only after final 8/8 at step 128, main seeds 17 and 29 once at step 256'}
        with tempfile.TemporaryDirectory() as temporary, patch.object(admission, 'git_identity', return_value=self.git):
            path = Path(temporary) / 'approval.json'
            write_json(path, approval)
            admission.authorize({'git': self.git}, freeze_sha, path)
            for key, value in [('schema', 'trinite.queue-learning-owner-approval.v1'),
                               ('freeze_sha256', '0' * 64), ('reviewed_merged_green', False),
                               ('authorization', 'execute queue-learning-v1 seeds 17 and 29 once')]:
                bad = dict(approval, **{key: value})
                with patch.object(admission, 'read_json', return_value=bad), self.assertRaises(ValueError):
                    admission.authorize({'git': self.git}, freeze_sha, path)
            for git in [dict(self.git, clean=False), dict(self.git, main=None), dict(self.git, head='0' * 40)]:
                with patch.object(admission, 'git_identity', return_value=git), self.assertRaises(ValueError):
                    admission.authorize({'git': self.git}, freeze_sha, path)

    def test_training_relabel_rejected_before_target_operation(self):
        rows = admission.episodes()
        manifest = {'episodes': [{k: r[k] for k in ('id', 'family', 'split')} for r in rows]}
        manifest['episodes'][-1]['split'] = 'train'
        with patch.object(admission, 'sources', return_value=self.source), \
             patch.object(admission, 'read_json', return_value=admission.record(rows, self.source)), \
             patch.object(admission.previous, 'training_rows') as target:
            with self.assertRaises(ValueError):
                admission.training_rows(Path('/unused'), manifest, {})
            target.assert_not_called()


if __name__ == '__main__':
    unittest.main()
