"""Request identity and CPU/GPU protocol freeze, without importing numerical code."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from trinite.contracts import identity, json_bytes, parse_json
from music_device_trial_v1 import request, runtime, spec

class RequestTests(unittest.TestCase):
    def test_freeze_and_validate_pins_source_environment_and_fixed_limits(self):
        with tempfile.TemporaryDirectory() as d,patch.object(request,'sources',return_value={'commit':'a','tree':'b','files':{}}),patch.object(runtime,'require_environment',return_value={'scope':'synthetic metadata only'}):
            path=Path(d)/'request.json';r=request.freeze(path);raw=path.read_bytes()
            self.assertEqual(r['request_identity'],identity(raw));self.assertFalse(r['model_executed'])
            value=request.validate(raw,r['request_identity'])
            self.assertEqual(set(value['cells']),{'cpu','cuda'})
            self.assertEqual(value['protocol']['parity_tolerances'],spec.PARITY_TOLERANCES)
            changed=parse_json(raw,canonical=True);changed['protocol']['steps']=641
            changed_raw=json_bytes(changed)
            with self.assertRaises(ValueError):request.validate(changed_raw,identity(changed_raw))
            with self.assertRaises(ValueError):request.validate(raw,'sha256:'+'0'*64)
            with self.assertRaises((ValueError,OSError)):request.freeze(path)
    def test_actual_clean_source_freeze_without_backend_imports(self):
        import builtins
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'numpy', 'safetensors') or name in (
                    'trinite.model', 'trinite.training', 'music_device_trial_v1.numerical'):
                raise AssertionError('freeze must not construct or execute numerical work')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as d, patch.object(runtime, 'require_environment',
                return_value={'scope': 'synthetic metadata; no hardware tested'}), \
                patch.object(builtins, '__import__', side_effect=guarded):
            path = Path(d) / 'request.json'
            receipt = request.freeze(path)
            value = request.validate(path.read_bytes(), receipt['request_identity'])
            self.assertEqual(len(value['source']['commit']), 40)
            self.assertEqual(len(value['source']['tree']), 40)
            for name in ('requirements.music-device.lock',
                         'scripts/music_device_trial_v1/runtime-lock.json',
                         'scripts/run_music_device_trial.py'):
                self.assertIn(name, value['source']['files'])
            self.assertFalse(receipt['model_executed'])

    def test_environment_change_rejects_request(self):
        with patch.object(request,'sources',return_value={}),patch.object(runtime,'require_environment',return_value={'id':'before'}):raw=json_bytes(request.core())
        with patch.object(request,'sources',return_value={}),patch.object(runtime,'require_environment',return_value={'id':'after'}):
            with self.assertRaisesRegex(ValueError,'environment'):request.validate(raw,identity(raw))
    def test_unsupported_freeze_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as d,patch.object(request,'sources',return_value={}),patch.object(runtime,'require_environment',side_effect=runtime.UnsupportedRuntime('synthetic no CUDA')):
            path=Path(d)/'request.json'
            with self.assertRaises(runtime.UnsupportedRuntime):request.freeze(path)
            self.assertFalse(path.exists())
    def test_linked_destination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);real=root/'real';real.mkdir();link=root/'linked';link.symlink_to(real,target_is_directory=True)
            with self.assertRaises(ValueError):request.freeze(link/'request.json')

if __name__=='__main__':unittest.main()
