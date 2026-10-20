import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nexus_conformance_v1 import reference as ref, spec, runner


class Contracts(unittest.TestCase):
    def test_donor_bytes_and_separate_revisions(self):
        lock=runner.check_sources()
        self.assertEqual({x['revision'] for x in lock.values()},
                         {'24c7b5bb8b158d84e08b10ccc8fdd16b6889b7e4','a9486c34eebd395c8060f8b2b03885eff10115c6'})
        self.assertIn('Mozilla Public License', (runner.UPSTREAM/'LICENSE').read_text())

    def test_graph_exact_structure(self):
        self.assertEqual((len(ref.VERTICES),len(ref.EDGES),len(ref.FACES)),(12,24,8))
        for colour in range(4):
            self.assertEqual(sorted(v for e,c in zip(ref.EDGES,ref.COLOURS) if c==colour for v in e),list(range(12)))
        self.assertEqual(ref.COLOURS,(0,1,2,3,1,3,2,0,2,3,3,2,0,1,1,0,0,1,1,0,2,3,3,2))

    def test_invalid_domains_fail_closed(self):
        for seed,rounds,state in [(True,1,[0]*24),(-1,1,[0]*24),(2**32,1,[0]*24),
                                  (0,129,[0]*24),(0,-1,[0]*24),(0,1,[0]*36),
                                  (0,1,[4]*24),(0,1,[False]*24)]:
            with self.assertRaises(ValueError): ref.evolve(seed,rounds,state)

        with self.assertRaises(ValueError): ref.evolve(0,0,[0]*24,'inverse')

    def test_both_inverse_directions_on_all_bank_states(self):
        for c in spec.cases():
            with self.subTest(seed=c['seed'],rounds=c['rounds'],cells=c['cells']):
                states=spec.expected_states(c)
                self.assertEqual(states[1],c['state']);self.assertEqual(states[3],c['state'])
        # Periodic rings and u32 round wrapping are deliberately distinct boundaries.
        for cells in (2,6):
            state=[i%4 for i in range(cells*12)]
            for r in (0,1,2**32-1):
                self.assertEqual(ref.step(2**32-1,r,ref.step(2**32-1,r,state),True),state)

    def test_complete_plan_coverage_and_no_heldout_partition(self):
        req=spec.request()
        self.assertEqual(len(req['plan_coverage']),64)
        self.assertEqual(req['split'],'conformance-only')
        self.assertEqual(len(req['cases']),1732)
        self.assertEqual(len({spec.identity(spec.canonical(c)) for c in req['cases']}),1732)

    def test_every_item_records_and_binds_lineage(self):
        req=spec.request(); seen=set()
        for row in spec.items(req):
            data=dict(row); item_id=data.pop('item_id')
            self.assertEqual(item_id,spec.identity(spec.canonical(data)))
            self.assertNotIn(item_id,seen);seen.add(item_id)
            for field in ('source_id','generator_revision','seed','seed_procedure','split_procedure',
                          'family_id','formal_specification','rendered_text','expected_answer','verifier_outcome','transformation_lineage'):
                self.assertIn(field,row)
            formal=json.loads(row['rendered_text'])
            self.assertEqual(formal,row['formal_specification'])
            actual=ref.evolve(formal['seed'],formal['rounds'],formal['input'],formal['direction']=='inverse')
            expected=actual if formal['target']=='state' else actual[formal['index']]
            self.assertEqual(row['expected_answer'],expected)
            self.assertEqual(row['split'],'conformance-only')
        self.assertEqual(len(seen),6928)

    def test_exclusive_atomic_result_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'result.json'
            runner.publish(path,b'original')
            with self.assertRaises(FileExistsError):runner.publish(path,b'changed')
            self.assertEqual(path.read_bytes(),b'original')
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()),['result.json'])

    def test_failed_link_does_not_publish_partial_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'result.json'
            with patch.object(runner.os,'link',side_effect=OSError('synthetic I/O failure')):
                with self.assertRaises(OSError):runner.publish(path,b'{"status":"passed"}')
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()),[])

    def test_toolchain_failure_retains_failed_result(self):
        req=spec.request()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'attempt'
            result=runner.execute(req,folder,str(Path(temp)/'missing-rustc'),'lean')
            self.assertEqual(result['status'],'failed')
            self.assertEqual(json.loads((folder/'result.json').read_bytes()),result)
            self.assertTrue((folder/'request.json').is_file())
            self.assertTrue((folder/'input.txt').is_file())
            self.assertIn('child_peak_rss_kib',result['resources'])

    def test_source_request_change_fails_before_donor_execution(self):
        req=spec.request();req['bounds']['max_rounds']=127
        with tempfile.TemporaryDirectory() as temp:
            result=runner.execute(req,Path(temp)/'attempt','not-executed','not-executed')
            self.assertEqual(result['commands'],[])
            self.assertEqual(result['status'],'failed')

    def test_isolation_flags_and_frozen_identity(self):
        script=spec.ROOT/'scripts/check_nexus_conformance.py'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'request.json'
            rejected=subprocess.run([sys.executable,str(script),'freeze','--request',str(path)],capture_output=True)
            self.assertNotEqual(rejected.returncode,0);self.assertFalse(path.exists())
            good=subprocess.run([sys.executable,'-I','-S',str(script),'freeze','--request',str(path)],capture_output=True)
            self.assertEqual(good.returncode,0,good.stderr)
            self.assertEqual(good.stdout.strip().decode(),hashlib.sha256(path.read_bytes()).hexdigest())
            output=Path(temp)/'attempt'
            bad=subprocess.run([sys.executable,'-I','-S',str(script),'check','--request',str(path),
                                '--request-identity','0'*64,'--output-dir',str(output)],capture_output=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
