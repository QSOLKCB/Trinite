"""V2 request identity and unchanged numerical authority; no target model work."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from trinite.contracts import identity, json_bytes, parse_json
from music_placement_v2 import request, spec, reporting
from music_placement_v1 import spec as original_spec, reporting as original_reporting


class RequestTests(unittest.TestCase):
    def test_same_workload_scorer_and_admission_with_distinct_orchestration(self):
        for name in ('dataset', 'require_admission', 'plan_for', 'train_rows',
                     'diagnostic_rows', 'development_rows', 'schedule_receipt'):
            self.assertIs(getattr(spec, name), getattr(original_spec, name))
        self.assertIs(reporting.score_cell, original_reporting.score_cell)
        self.assertIs(reporting.matrix_summary, original_reporting.matrix_summary)
        protocol = dict(spec.PROTOCOL)
        self.assertEqual(protocol.pop('schema'), 'trinite.music-placement-protocol.v2')
        protocol.pop('numerical_semantics'); protocol.pop('partial_custody')
        self.assertEqual(protocol, {k:v for k,v in original_spec.PROTOCOL.items() if k!='schema'})

    def test_new_request_rejects_old_schema_identity_and_protocol_mutation(self):
        with tempfile.TemporaryDirectory() as d, patch.object(request, 'sources', return_value={}), \
                patch('trinite.training.require_environment', return_value={'synthetic': True}):
            destination=Path(d)/'request.json'; receipt=request.freeze(destination)
            raw=destination.read_bytes(); value=request.validate(raw,receipt['request_identity'])
            self.assertEqual(value['schema'],'trinite.music-placement-request.v2')
            self.assertFalse(receipt['training_performed'])
            self.assertFalse(receipt['target_inference_performed'])
            for mutation in ('schema','protocol'):
                changed=parse_json(raw,canonical=True)
                if mutation=='schema': changed['schema']='trinite.music-placement-request.v1'
                else: changed['protocol']['steps']=641
                payload=json_bytes(changed)
                with self.assertRaises(ValueError): request.validate(payload,identity(payload))
            with self.assertRaises(ValueError): request.validate(raw,'sha256:'+'0'*64)
            with self.assertRaises(OSError): request.freeze(destination)

    def test_foreign_v2_module_is_not_ignored_by_source_guard(self):
        import sys
        from types import ModuleType
        modules = {name:module for name,module in sys.modules.items()
                   if (name=='trinite' or name.startswith('trinite.')
                       or name.split('.')[0].endswith(('_v1','_v2'))
                       or name.startswith('provenance_'))}
        allowed = {str(Path(module.__file__).absolute()): 'synthetic'
                   for module in modules.values() if getattr(module,'__file__',None)}
        foreign = ModuleType('synthetic_foreign_v2')
        foreign.__file__ = '/outside/foreign.py'
        with patch.object(request, 'check', return_value={'files':allowed}), \
             patch.dict(sys.modules, {'synthetic_foreign_v2':foreign}):
            with self.assertRaisesRegex(ValueError, 'synthetic_foreign_v2'):
                request.sources()

    def test_actual_clean_source_freeze_no_model_or_optimizer(self):
        # Run from final committed source, as enforced by the actual admission.
        from music_placement_v2.controller import configure
        configure()
        with tempfile.TemporaryDirectory() as d, \
                patch('trinite.model.Decoder', side_effect=AssertionError('no model')), \
                patch('trinite.training.create_optimizer', side_effect=AssertionError('no optimizer')):
            destination=Path(d)/'request.json'
            receipt=request.freeze(destination)
            value=request.validate(destination.read_bytes(),receipt['request_identity'])
            for name in ('scripts/music_placement_v2/controller.py',
                         'scripts/music_placement_v2/request.py',
                         'scripts/music_placement_v1/numerical.py',
                         'scripts/music_placement_v1/spec.py',
                         'scripts/run_music_placement_v2.py',
                         'docs/MUSIC-ROBUSTNESS-V2.md'):
                self.assertIn(name,value['source']['files'])
            self.assertEqual(value['schema'],'trinite.music-placement-request.v2')

if __name__=='__main__': unittest.main()
