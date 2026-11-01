"""Real Rust observation/admission boundary; synthetic feasibility, NOT a study.

The only execution doubles are Git cleanliness and numerical environment. The
synthetic timing/supervision receipt is explicitly test input, not measured
feasibility, approval, optimizer replay or scientific evidence. Every request,
observation, prepare anchor, source hash and binary identity uses the real gate.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import read_json, write_json
from queue_learning_v2 import admission, numerical


class FreezeBoundaryTests(unittest.TestCase):
    def test_real_observation_only_freeze_replay_and_tamper_rejection(self):
        actual_run = gate.run
        operations = []

        def observe_only(operation, *args, **kwargs):
            self.assertEqual(operation, 'observe', 'test must never request any training or held-out target')
            operations.append(operation)
            return actual_run(operation, *args, **kwargs)

        head = admission.git_identity()['head']
        synthetic_git = {'head': head, 'main': head, 'clean': True}
        synthetic_environment = {'engineering_test_only': 'no numerical backend configured'}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(gate, 'run', side_effect=observe_only), \
             patch.object(admission, 'git_identity', return_value=synthetic_git), \
             patch.object(numerical, 'configure', return_value=synthetic_environment):
            root = Path(temporary)
            rows = admission.episodes()
            observed = admission.observed_inventory(rows)
            structure = admission.structural(rows, observed)
            self.assertEqual(structure['fresh_overlap_with_exposed_v1'], 0)
            self.assertEqual(structure['cross_split_related_families'], 0)
            self.assertEqual(structure['episodes'], 384)
            self.assertEqual(structure['requests_sha256'],
                '33d54a234d3320acdee9fccbe461af696003bf3d3b2871d43f9233d25d28c076')
            proof = {'schema': 'trinite.queue-learning-feasibility.v2',
                'sources': admission.sources(), 'git': synthetic_git,
                'binary_sha256': gate.binary_identity(), 'protocol': admission.PROTOCOL,
                'structural': structure, 'peak_rss_kib': 4096,
                'admission_import_setup_seconds_hex': (2.).hex(),
                'conservative_whole_attempt_seconds_hex': (28.).hex(),
                'kernel': {'environment': synthetic_environment, 'parameters': 24768,
                    'optimizer_updates': 0, 'gradient_calls': 3,
                    'kernel_seconds_hex': [(0.01).hex()] * 3,
                    'conservative_256_update_estimate_seconds_hex': (0.01 * 256 * 4).hex(),
                    'diagnostic_io_seconds_hex': (1.).hex(),
                    'conservative_whole_attempt_seconds_hex': (20.).hex()},
                'supervisor': {'exit_code': 0, 'failure': None,
                    'limits': {'rss_bytes_per_process': 1073741824, 'aggregate_rss_bytes': 2147483648,
                               'address_space_bytes_per_process': 4294967296},
                    'peak_process_rss_bytes': 1024, 'peak_aggregate_rss_bytes': 2048,
                    'wall_seconds_hex': (20.).hex(), 'deadline_seconds_hex': (900.).hex()}}
            proof_path = root / 'synthetic-engineering-feasibility.json'
            proof_sha = write_json(proof_path, proof)
            frozen, anchors = root / 'frozen', root / 'anchors.json'
            result = admission.freeze(proof_path, proof_sha, frozen, anchors)
            freeze_sha, anchors_sha = result['freeze_sha256'], result['preparation_anchors_sha256']

            def verify(fsha=freeze_sha, asha=anchors_sha):
                return admission.frozen_context(frozen, fsha, anchors, asha)

            _, manifest, prepares = verify()
            self.assertEqual(len(manifest['episodes']), 384)
            self.assertEqual(len(prepares), 384)
            self.assertEqual({s: sum(r['split'] == s for r in manifest['episodes'])
                              for s in ('train', 'validation', 'test')},
                             {'train': 256, 'validation': 64, 'test': 64})
            self.assertEqual(manifest['sources'], admission.sources())
            self.assertEqual(manifest['binary_sha256'], gate.binary_identity())
            self.assertEqual(manifest['structural'], structure)
            first = rows[0]['id']
            # These are temporary engineering fixtures, never retained run bytes.
            for name in (f'{first}/request.json', f'{first}/public/observation.json',
                         f'{first}/private/manifest.json', 'splits.json'):
                with self.subTest(tamper=name):
                    path = frozen / name
                    original = path.read_bytes()
                    try:
                        path.write_bytes(b'{}\n')
                        with self.assertRaises(ValueError):
                            verify()
                    finally:
                        path.write_bytes(original)
            with self.assertRaises(ValueError):
                verify(asha='0' * 64)
            journal = anchors.with_suffix('.jsonl')
            original = journal.read_bytes()
            try:
                journal.write_bytes(original[:-1])
                with self.assertRaises(ValueError):
                    verify()
            finally:
                journal.write_bytes(original)
            # Even recomputing all local file/manifest/anchor hashes cannot make
            # a foreign split recipe pass exact regeneration.
            paths = [frozen / 'splits.json', frozen / 'manifest.json', anchors]
            saved = {path: path.read_bytes() for path in paths}
            try:
                splits = read_json(paths[0])
                splits[next(iter(splits))] = 'test' if splits[next(iter(splits))] != 'test' else 'train'
                paths[0].write_bytes(gate.encode(splits))
                changed = read_json(paths[1])
                changed['files']['splits.json'] = gate.digest(paths[0].read_bytes())
                paths[1].write_bytes(gate.encode(changed))
                new_freeze_sha = gate.digest(paths[1].read_bytes())
                changed_anchors = read_json(anchors)
                changed_anchors['freeze_sha256'] = new_freeze_sha
                anchors.write_bytes(gate.encode(changed_anchors))
                with self.assertRaisesRegex(ValueError, 'split/protocol mismatch'):
                    verify(new_freeze_sha, gate.digest(anchors.read_bytes()))
            finally:
                for path, raw in saved.items():
                    path.write_bytes(raw)
            self.assertGreaterEqual(len(operations), 2304)
            self.assertEqual(set(operations), {'observe'})


if __name__ == '__main__':
    unittest.main()
