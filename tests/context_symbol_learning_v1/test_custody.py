"""Archive, hash, source, resources and read-only replay boundary checks."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import inventory,read_json,write_json
from context_symbol_learning_v1 import custody,verification,spec


class CustodyTests(unittest.TestCase):
    def test_source_archive_closes_current_runtime(self):
        source=spec.sources()
        for name in ('src/trinite/training.py','src/trinite/comparison_checkpoint.py',
            'scripts/queue_learning_v2/resources.py','scripts/procedural_dataset_v1/admission.py',
            'scripts/run_context_symbol_learning_v1.py','external/context-symbol-relation-v1/src/lib.rs'):
            self.assertIn(name,source)
        raw=custody.archive(source)
        custody.verify_archive(raw,source)
        altered=BytesIO()
        with zipfile.ZipFile(altered,'w') as z:
            z.writestr('../escape','bad')
        with self.assertRaises(ValueError):
            custody.verify_archive(altered.getvalue(),source)

    def test_stopped_outer_custody_is_read_only_and_anchor_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            write_json(run/'supervisor.json',dict(exit_code=-9,failure='timeout'))
            write_json(run/'partial-log.json',dict(reason='engineering interruption'))
            anchor=root/'anchor.json'
            sha=write_json(anchor,dict(schema='trinite.context-symbol-evidence-anchor.v1',freeze_sha256='a'*64,files=inventory(run)))
            args=SimpleNamespace(run=run,run_anchor=anchor,run_sha=sha,frozen=root/'frozen',
                freeze_sha='a'*64,anchors=root/'prep.json',anchors_sha='b'*64)
            before=inventory(run)
            with patch.object(custody,'context',return_value=({},{})):
                result=verification.verify(args)
                self.assertTrue(result['verified']);self.assertFalse(result['complete'])
                self.assertEqual(inventory(run),before)
                (run/'partial-log.json').write_bytes(b'{}\n')
                with self.assertRaises(ValueError):
                    verification.verify(args)

    def test_actual_process_cleanup_and_budget_defaults(self):
        from queue_learning_v2.resources import supervise
        result=supervise([sys.executable,'-c','import os,time;os.close(1);os.close(2);time.sleep(10)'],.1)
        self.assertEqual(result['failure'],'timeout')
        self.assertTrue(result['reaped'])
        self.assertEqual(result['limits']['address_space_bytes_per_process'],4<<30)

    def test_internal_worker_cannot_bypass_supervisor(self):
        result=subprocess.run([sys.executable,'-I','-B',str(spec.ROOT/'scripts/run_context_symbol_learning_v1.py'),
            '_attempt'],capture_output=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'internal worker requires',result.stderr)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        from context_symbol_learning_v1 import data,numerical
        from procedural_dataset_v1.cases import expected_target
        rows=data.inventory()
        train=data.training_rows(rows,lambda r:expected_target('context',r))
        self.content=dict(rows=rows,train=train,public=[dict(id=r['id'],observation=data.observation(r)) for r in rows],
            audit=data.audit(rows),schedule={n:numerical.schedule(train,s,f) for n,s,f,_,_ in spec.ATTEMPTS})
        self.source=spec.sources()
        self.git=dict(head='c'*40,main='c'*40,clean=True)
        self.rt=dict(environment=dict(synthetic_engineering_runtime=True))
        self.proof=dict(schema='trinite.context-symbol-feasibility.v1',sources=self.source,git=self.git,runtime=self.rt,
            protocol=spec.PROTOCOL,audit=self.content['audit'],schedule=self.content['schedule'],optimizer_updates=0,
            kernel=dict(parameters=24768,optimizer_updates=0,gradient_calls=3,environment=self.rt['environment'],
                kernel_seconds_hex=[(.001).hex()]*3,update_projection_hex=(.001*256*4).hex(),
                diagnostic_io_seconds_hex=(.25).hex(),whole_projection_hex=(.001*256*4+1).hex()),
            setup_seconds_hex=(.5).hex(),whole_projection_hex=(.001*256*4+3).hex(),
            supervisor=dict(exit_code=0,failure=None,wall_seconds_hex=(2.).hex(),
                peak_process_rss_bytes=1<<20,peak_aggregate_rss_bytes=2<<20,
                limits=dict(rss_bytes_per_process=1<<30,aggregate_rss_bytes=2<<30,address_space_bytes_per_process=4<<30)))

    def test_feasibility_budget_runtime_source_and_supervisor_rejections(self):
        custody.validate_feasibility(self.proof,self.source,self.git,self.rt)
        for field,value in [('optimizer_updates',1),('sources',{}),('runtime',{}),('git',{}),
                            ('whole_projection_hex',(900.).hex())]:
            altered=deepcopy(self.proof);altered[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                custody.validate_feasibility(altered,self.source,self.git,self.rt)
        for field,value in [('failure','timeout'),('peak_process_rss_bytes',1<<30),('peak_aggregate_rss_bytes',2<<30)]:
            altered=deepcopy(self.proof);altered['supervisor'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                custody.validate_feasibility(altered,self.source,self.git,self.rt)

    def test_source_bound_test_double_freeze_roundtrip_and_resealed_label_tamper(self):
        from trinite.contracts import json_bytes
        proof=self.root/'proof.json';sha=write_json(proof,self.proof)
        frozen=self.root/'frozen';anchor=self.root/'prep.json'
        with (patch.object(custody,'runtime',return_value=self.rt),
              patch.object(custody,'clean_main',return_value=self.git),
              patch.object(custody,'prepare_content',return_value=self.content)):
            result=custody.freeze(proof,sha,frozen,anchor)
            before=inventory(frozen)
            custody.context(frozen,result['freeze_sha256'],anchor,result['anchors_sha256'])
            self.assertEqual(inventory(frozen),before)
            with patch.object(custody,'runtime',side_effect=AssertionError('portable verification must not import native runtime')):
                custody.context(frozen,result['freeze_sha256'],anchor,result['anchors_sha256'],live=False,scores_only=True)
            self.assertEqual(inventory(frozen),before)
            value=read_json(frozen/'data.json')
            value['train'][0]['answer']='wrong'
            (frozen/'data.json').write_bytes(json_bytes(value))
            manifest=read_json(frozen/'manifest.json')
            manifest['files']['data.json']=gate.digest(gate.read(frozen/'data.json',4<<20))
            (frozen/'manifest.json').write_bytes(json_bytes(manifest))
            new_sha=gate.digest(gate.read(frozen/'manifest.json',4<<20))
            anchors=read_json(anchor);anchors['freeze_sha256']=new_sha
            anchor.write_bytes(json_bytes(anchors))
            new_anchor=gate.digest(gate.read(anchor,4<<20))
            with self.assertRaisesRegex(ValueError,'training rows differ'):
                custody.context(frozen,new_sha,anchor,new_anchor)


if __name__=='__main__':
    unittest.main()
