"""Bounded boundary tests with no target model construction, training or inference."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_learning_v1 import request, controller, workflow


class WorkflowTests(unittest.TestCase):
    def test_read_bounds_links_and_observed_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'value.json'
            path.write_bytes(b'{}\n')
            self.assertEqual(request.read(path, 3), b'{}\n')
            with self.assertRaises(ContractError):
                request.read(path, 2)
            link = root / 'link.json'
            link.symlink_to(path)
            with self.assertRaises(ContractError):
                request.read(link)
            parent = root / 'linked'
            parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ContractError):
                request.read(parent / path.name)
            original = os.open
            def replace_before_open(candidate, *args, **kwargs):
                replacement = root / 'replacement'
                replacement.write_bytes(b'[]\n')
                replacement.replace(path)
                return original(candidate, *args, **kwargs)
            with patch('music_learning_v1.request.os.open', side_effect=replace_before_open):
                with self.assertRaisesRegex(ContractError, 'changed at open'):
                    request.read(path)

    def test_request_identity_and_changed_expected_source(self):
        value = {'source': {'codec': 'sha256:' + '0' * 64}, 'protocol': 'fixed'}
        raw = json_bytes(value)
        with patch('music_learning_v1.request.core', return_value=value):
            self.assertEqual(request.validate(raw, identity(raw)), value)
            with self.assertRaises(ContractError):
                request.validate(raw, 'sha256:' + 'f' * 64)
            with self.assertRaises(ContractError):
                request.validate(raw + b' ', identity(raw + b' '))
        changed = {'source': {'codec': 'sha256:' + '1' * 64}, 'protocol': 'fixed'}
        with patch('music_learning_v1.request.core', return_value=changed):
            with self.assertRaisesRegex(ContractError, 'source/environment/admission/protocol'):
                request.validate(raw, identity(raw))

    def test_loaded_foreign_module_rejected_by_source_binding(self):
        from music_learning_v1 import codec
        with patch.object(codec, '__file__', '/tmp/foreign/music_learning_v1/codec.py'):
            with self.assertRaisesRegex(ContractError, 'outside bound checkout'):
                request.sources()

    def test_preflight_is_real_admission_without_model(self):
        code = ('import sys; from music_learning_v1.controller import main; '
                'assert main(["preflight"]) == 0; '
                'assert not any(n in sys.modules for n in ["torch", "numpy", '
                '"trinite.model", "trinite.training", "symbolic_music_v1.render"])')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"model_executed":false', result.stdout)

    def test_missing_authorization_fails_before_runtime_configuration(self):
        for action in ('run', '_worker'):
            with patch('music_learning_v1.controller.configure', side_effect=AssertionError('must not configure')):
                with redirect_stderr(StringIO()) as errors:
                    self.assertEqual(controller.main([action]), 1)
                self.assertIn('separate explicit owner authorization', errors.getvalue())

    def test_freeze_rejects_symlinked_parent_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / 'real'
            real.mkdir()
            link = root / 'linked'
            link.symlink_to(real, target_is_directory=True)
            with patch('music_learning_v1.controller.configure'), \
                 patch('music_learning_v1.controller.core', return_value={'test': 'boundary-only'}), \
                 patch('music_learning_v1.controller.publish', side_effect=AssertionError('must not publish')):
                with redirect_stderr(StringIO()):
                    self.assertEqual(controller.main(['freeze', '--request', str(link / 'request.json')]), 1)
            self.assertEqual(list(real.iterdir()), [])

    def test_output_protects_source_request_and_symlink_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requested = root / 'request.json'
            requested.write_bytes(b'{}\n')
            for output in (request.ROOT, request.ROOT / 'child', root):
                with self.subTest(output=output), self.assertRaises(ContractError):
                    controller.protect(output, requested)
            linked = root / 'linked'
            linked.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ContractError):
                controller.protect(linked / 'output', requested)
            self.assertEqual(controller.protect(root / 'safe-output', requested),
                             (root / 'safe-output', requested.resolve()))

    def test_real_admitted_custody_with_explicit_non_numerical_double(self):
        from music_learning_v1.admission import dataset
        from trinite.observation import verify_observation
        raw, admitted = dataset()
        request_value = {'admission_identity': identity(admitted),
                         'test_scope': 'custody conformance only; synthetic checkpoint bytes'}
        request_raw = json_bytes(request_value)
        request_identity = identity(request_raw)
        def fake_trial(seed, admission_identity, request_identity, write, journal):
            # Deliberately NOT checkpoints, model predictions or learning evidence.
            for name in sorted(workflow.artifact_names() - {'report.json', 'predictions.jsonl'}):
                write(name, json_bytes({'test_artifact': name, 'not_numerical_evidence': True}))
            journal('untrained', 'train', {'test_double': True, 'retained_failure': True})
            return {'test_double': True, 'not_numerical_evidence': True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'attempt'
            root.mkdir()
            info = root.stat()
            with patch('music_learning_v1.workflow.validate', return_value=request_value), \
                 patch('music_learning_v1.numerical.run_trial', side_effect=fake_trial):
                commitment = workflow.run(root, request_raw, request_identity, 17,
                                          (info.st_dev, info.st_ino))
            completion_raw = (root / 'completion.json').read_bytes()
            self.assertEqual(identity(completion_raw), commitment)
            completion = parse_json(completion_raw, canonical=True)
            self.assertEqual(set(completion['artifacts']), workflow.artifact_names())
            verification = verify_observation(root / 'provenance')
            self.assertTrue(verification['integrity_verified'])
            self.assertEqual(verification['errors'], [])
            self.assertEqual(verification['known_missing_artifacts'], 0)
            manifest = parse_json((root / 'provenance/manifest.json').read_bytes(), canonical=True)
            indexes = []
            for item in manifest['core']['artifacts']:
                payload = (root / 'provenance/artifacts/sha256' / item['content_identity'][7:]).read_bytes()
                value = parse_json(payload, canonical=True)
                if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1':
                    indexes.append(value['artifacts'])
            self.assertEqual(indexes, [{**completion['artifacts'], 'request.json': request_identity,
                                       'dataset.json': identity(raw), 'admission.json': identity(admitted)}])
            # Tampering must fail before any model/checkpoint restore is reached.
            path = root / 'untrained.safetensors'
            path.write_bytes(b'tampered synthetic conformance bytes')
            with patch('music_learning_v1.workflow.validate', return_value=request_value), \
                 patch('music_learning_v1.numerical.restore', side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(ContractError, 'retained artifact identity differs'):
                    workflow.audit(root, request_raw, request_identity, 17, commitment)
            # Original custody bytes are still intact; the altered outer file is not resealed.
            self.assertTrue(verify_observation(root / 'provenance')['integrity_verified'])

    def test_direct_worker_cannot_write_inside_source(self):
        # Inode handoff is not authorization to treat source as an output directory.
        info = request.ROOT.stat()
        with patch('music_learning_v1.workflow.validate', return_value={'admission_identity': 'x'}), \
             patch('music_learning_v1.workflow.require_native_admission', return_value=(b'{}', b'{}')), \
             patch('music_learning_v1.workflow.Store', side_effect=AssertionError('must not open source for writing')):
            with self.assertRaises(ContractError):
                workflow.run(request.ROOT, b'{}', 'x', 17, (info.st_dev, info.st_ino))


if __name__ == '__main__':
    unittest.main()
