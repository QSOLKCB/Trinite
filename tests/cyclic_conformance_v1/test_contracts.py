import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from cyclic_conformance_v1 import spec, runner, checks
from cyclic_conformance_v1.reference import CyclicElement, from_sexagesimal, to_sexagesimal, CyclicGroupConformanceError


class TestContracts(unittest.TestCase):
    def test_fixed_suite_bound_and_provenance(self):
        req = spec.request()
        self.assertLess(len(req['cases']), req['limits']['max_cases'])
        self.assertEqual(max(c['n'] for c in req['cases']), 360)
        rows = list(checks.rows(req))
        self.assertEqual(len(rows), len(req['cases']))
        for row in rows:
            self.assertEqual(json.loads(row['rendered_text']), row['formal_specification'])
            self.assertEqual(row['generator_revision'], req['generator_revision'])
            self.assertEqual(row['request_identity'], spec.identity(spec.canonical(req)))
            self.assertIn('conformance-only', row['split_procedure'])
            self.assertEqual(row['source_id'], spec.SOURCE_ID)
            self.assertIsNone(row['seed'])

    def test_counter_oracle_does_not_call_class_add_or_gcd(self):
        with patch.object(CyclicElement, 'add', side_effect=RuntimeError('not oracle')), patch('math.gcd', side_effect=RuntimeError('not oracle')):
            self.assertEqual(checks.counter_order(12, 60), 5)
            self.assertEqual(checks.counter_order(12, 61), 61)
            self.assertEqual(checks.counter_order(0, 1), 1)
        for k,n in ((0,361),(0,0),(-1,60),(True,60)):
            with self.assertRaises(ValueError): checks.counter_order(k,n)

    def test_permissive_codec_normalizes_spelling(self):
        self.assertEqual(from_sexagesimal((0,0,1)), 1)
        self.assertEqual(to_sexagesimal(from_sexagesimal((0,0,1))), (1,))
        for invalid in ('12', iter([1]), 1, None):
            with self.assertRaises(CyclicGroupConformanceError): from_sexagesimal(invalid)

    def test_explicit_displacement_both_endpoints(self):
        zero, half = CyclicElement(0, 60), CyclicElement(30, 60)
        self.assertEqual(zero.signed_displacement_to(half,'negative'), -30)
        self.assertEqual(zero.signed_displacement_to(half,'positive'), 30)
        self.assertEqual(CyclicElement(0,1).signed_displacement_to(CyclicElement(0,1)),0)

    def test_failed_attempt_retains_partial_rows(self):
        req=spec.request()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'attempt'
            original=checks.rows
            def broken(value):
                yield next(original(value))
                raise ValueError('intentional finite check failure')
            with patch.object(checks,'rows',broken): result=runner.execute(req,folder)
            self.assertEqual(result['status'],'failed')
            self.assertIn('intentional',result['error'])
            self.assertEqual(len((folder/'rows.jsonl').read_bytes().splitlines()),1)
            self.assertEqual(json.loads((folder/'result.json').read_bytes())['status'],'failed')
            before={p.name:p.read_bytes() for p in folder.iterdir()}
            with self.assertRaises(FileExistsError): runner.execute(req,folder)
            self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})

    def test_result_write_failure_never_creates_success_receipt(self):
        req=spec.request()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'attempt'; publish=runner.publish
            def fail(path,raw):
                if path.name=='result.json': raise OSError('intentional result write failure')
                return publish(path,raw)
            with patch.object(runner,'publish',fail), self.assertRaises(OSError): runner.execute(req,folder)
            self.assertTrue((folder/'rows.jsonl').exists())
            self.assertFalse((folder/'result.json').exists())

    def test_isolated_launcher_requires_independent_request(self):
        launcher=spec.ROOT/'scripts/check_cyclic_conformance.py'
        with tempfile.TemporaryDirectory() as tmp:
            request=Path(tmp)/'request.json'
            frozen=subprocess.run([sys.executable,'-I','-S',str(launcher),'freeze','--request',str(request)],capture_output=True,check=False)
            self.assertEqual(frozen.returncode,0,frozen.stderr)
            rejected=subprocess.run([sys.executable,'-I','-S',str(launcher),'check','--request',str(request),'--request-identity','0'*64,'--output-dir',str(Path(tmp)/'bad')],capture_output=True,check=False)
            self.assertEqual(rejected.returncode,1)
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_changes_to_request_are_rejected(self):
        req=spec.request(); req['cases'][0]['n']=999
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            runner.execute(req,Path(tmp)/'attempt')
