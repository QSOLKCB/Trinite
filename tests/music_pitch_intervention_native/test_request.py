"""Real environment/source admission, without constructing any model."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from music_pitch_intervention_v1 import request
from music_pitch_intervention_v1.controller import configure
from trinite.contracts import identity, json_bytes


class RequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_actual_freeze_bound_source_and_no_model(self):
        with tempfile.TemporaryDirectory() as d, mock.patch('trinite.model.Decoder', side_effect=AssertionError('no model')), mock.patch('trinite.training.create_optimizer', side_effect=AssertionError('no optimizer')):
            destination = Path(d) / 'request.json'
            result = request.freeze(destination)
            raw = destination.read_bytes()
            self.assertFalse(result['training_performed'])
            self.assertFalse(result['target_inference_performed'])
            self.assertEqual(result['request_identity'], identity(raw))
            value = request.validate(raw, result['request_identity'])
            self.assertEqual(len(value['cells']), 4)
            self.assertIn('scripts/music_pitch_intervention_v1/numerical.py', value['source']['files'])
            with self.assertRaises(FileExistsError): request.freeze(destination)
            self.assertEqual(destination.read_bytes(), raw)

    def test_changed_request_rejected_even_with_new_identity(self):
        value = request.core()
        value['protocol'] = {**value['protocol'], 'steps': 1}
        raw = json_bytes(value)
        with self.assertRaises(ValueError): request.validate(raw, identity(raw))
        raw = json_bytes(request.core())
        with self.assertRaises(ValueError): request.validate(raw, 'sha256:' + '0'*64)

    def test_source_and_environment_changes_reject(self):
        raw = json_bytes(request.core())
        with mock.patch.object(request, 'sources', return_value={'changed': True}):
            with self.assertRaises(ValueError): request.validate(raw, identity(raw))
        with mock.patch('trinite.training.require_environment', return_value={'changed': True}):
            with self.assertRaises(ValueError): request.validate(raw, identity(raw))

if __name__ == '__main__': unittest.main()
