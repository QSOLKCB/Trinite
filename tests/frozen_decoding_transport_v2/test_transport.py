import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from trinite.contracts import json_bytes, identity
from frozen_decoding_transport_v2 import artifacts, runner


class ArtifactTests(unittest.TestCase):
    def test_json_bytes_is_already_jsonl(self):
        records=[json_bytes({'cell':i}) for i in range(4)]
        self.assertTrue(all(r.endswith(b'\n') and not r.endswith(b'\n\n') for r in records))
        self.assertEqual([json.loads(line) for line in b''.join(records).splitlines()],[{'cell':i} for i in range(4)])

    def test_complete_atomic_publish(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json';link=artifacts.os.link
            def inspect(temp,dest):
                self.assertFalse(p.exists());self.assertEqual(Path(temp).read_bytes(),b'complete')
                link(temp,dest)
            with patch.object(artifacts.os,'link',side_effect=inspect):artifacts.publish(p,b'complete')
            self.assertEqual(p.read_bytes(),b'complete');self.assertEqual(list(Path(d).iterdir()),[p])

    def test_short_writes_and_zero_progress(self):
        class Short(io.BytesIO):
            def write(self,raw):return super().write(raw[:2])
        out=Short();artifacts.write_all(out,b'abcdef');self.assertEqual(out.getvalue(),b'abcdef')
        with patch.object(out,'write',return_value=0):
            with self.assertRaises(OSError):artifacts.write_all(out,b'x')

    def test_existing_file_and_symlinks_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'output';target=root/'target';target.write_bytes(b'old')
            for kind in ('file','link','dangling'):
                if kind=='file':p.write_bytes(b'old')
                else:p.symlink_to(target if kind=='link' else root/'absent')
                with self.assertRaises(FileExistsError):artifacts.publish(p,b'new')
                if kind=='file':self.assertEqual(p.read_bytes(),b'old')
                else:self.assertTrue(p.is_symlink())
                self.assertEqual(target.read_bytes(),b'old');p.unlink()
                self.assertEqual(list(root.glob('.*.tmp')),[])

    def test_concurrent_destination_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'output';link=artifacts.os.link
            def race(temp,dest):
                p.write_bytes(b'competitor');link(temp,dest)
            with patch.object(artifacts.os,'link',side_effect=race):
                with self.assertRaises(FileExistsError):artifacts.publish(p,b'ours')
            self.assertEqual(p.read_bytes(),b'competitor');self.assertEqual(list(Path(d).iterdir()),[p])

    def test_failure_and_interrupt_clean_unpublished_temp(self):
        for stage in ('write','fsync','link'):
            for error in (OSError('fault'),KeyboardInterrupt()):
                with self.subTest(stage=stage,error=type(error).__name__), tempfile.TemporaryDirectory() as d:
                    p=Path(d)/'output'
                    def partial_write(stream,raw):stream.write(raw[:2]);raise error
                    target=patch.object(artifacts,'write_all',side_effect=partial_write) if stage=='write' else patch.object(artifacts.os,stage,side_effect=error)
                    with target:
                        with self.assertRaises(type(error)):artifacts.publish(p,b'complete')
                    self.assertFalse(p.exists());self.assertEqual(list(Path(d).iterdir()),[])


class RunnerTests(unittest.TestCase):
    def call(self,args):
        with contextlib.redirect_stdout(io.StringIO()):return runner.main(args)

    def setup_request(self,root):
        p=root/'request.json';p.write_bytes(json_bytes({'bound':True}))
        return p,identity(p.read_bytes())

    def run_args(self,p,key,out):return ['run','--request',str(p),'--request-identity',key,'--output',str(out)]

    def test_same_stem_outputs_have_distinct_sidecars_and_valid_jsonl(self):
        def produce(on_cell=None,timings=None):
            for i in range(4):on_cell({'cell':i})
            return {'inference':'unchanged'}
        with tempfile.TemporaryDirectory() as d, patch.object(runner,'request',return_value={'bound':True}),patch.object(runner,'report',side_effect=produce):
            root=Path(d);p,key=self.setup_request(root)
            for name in ('report.json','report.txt'):
                out=root/name;self.assertEqual(self.call(self.run_args(p,key,out)),0)
                cells=next(root.glob(name+'.attempt-*.cells.jsonl'))
                self.assertEqual([json.loads(x) for x in cells.read_bytes().splitlines()],[{'cell':i} for i in range(4)])
                self.assertEqual(len(list(root.glob(name+'.attempt-*.timings.json'))),1)
            self.assertEqual(len(list(root.glob('report*'))),6)

    def test_producer_failure_keeps_completed_cells_without_report(self):
        def fail(on_cell=None,timings=None):on_cell({'done':1});raise OSError('producer failed')
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'request',return_value={'bound':True}),patch.object(runner,'report',side_effect=fail):
            root=Path(d);p,key=self.setup_request(root);out=root/'report.json'
            self.assertEqual(self.call(self.run_args(p,key,out)),1);self.assertFalse(out.exists())
            self.assertEqual(next(root.glob('report.json.attempt-*.cells.jsonl')).read_bytes(),json_bytes({'done':1}))
            self.assertEqual(list(root.glob('report.json.attempt-*.timings.json')),[])

    def test_existing_report_rejects_before_generation(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'request',return_value={'bound':True}),patch.object(runner,'report') as report:
            root=Path(d);p,key=self.setup_request(root);out=root/'report.json';out.write_bytes(b'old')
            self.assertEqual(self.call(self.run_args(p,key,out)),1);report.assert_not_called();self.assertEqual(out.read_bytes(),b'old')

    def test_retry_same_output_after_producer_failure(self):
        def fail(on_cell=None,timings=None):on_cell({'done':1});raise OSError('failed')
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'request',return_value={'bound':True}):
            root=Path(d);p,key=self.setup_request(root);out=root/'report.json'
            with patch.object(runner,'report',side_effect=fail):self.assertEqual(self.call(self.run_args(p,key,out)),1)
            old=next(root.glob('report.json.attempt-*.cells.jsonl'));old_bytes=old.read_bytes()
            with patch.object(runner,'report',return_value={'complete':True}):self.assertEqual(self.call(self.run_args(p,key,out)),0)
            self.assertEqual(old.read_bytes(),old_bytes);self.assertEqual(len(list(root.glob('report.json.attempt-*.cells.jsonl'))),2)
            self.assertTrue(out.is_file())

    def test_freeze_write_failure_and_interrupt_never_publish(self):
        for error in (OSError('disk full'),KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as d,patch.object(runner,'request',return_value={'bound':True}),patch.object(artifacts,'write_all',side_effect=error):
                p=Path(d)/'request.json'
                if isinstance(error,OSError):self.assertEqual(self.call(['freeze','--request',str(p)]),1)
                else:
                    with self.assertRaises(KeyboardInterrupt):self.call(['freeze','--request',str(p)])
                self.assertEqual(list(Path(d).iterdir()),[])

    def test_final_report_failure_and_race_preserve_destinations(self):
        real_publish=artifacts.publish
        for action in ('write_failure','race'):
            with tempfile.TemporaryDirectory() as d,patch.object(runner,'request',return_value={'bound':True}),patch.object(runner,'report',return_value={'inference':'unchanged'}):
                root=Path(d);p,key=self.setup_request(root);out=root/'report.json'
                def fail_final(path,raw):
                    if path!=out:return real_publish(path,raw)
                    if action=='race':out.write_bytes(b'competitor');return real_publish(path,raw)
                    with patch.object(artifacts,'write_all',side_effect=OSError('disk full')):return real_publish(path,raw)
                with patch.object(runner,'publish',side_effect=fail_final):self.assertEqual(self.call(self.run_args(p,key,out)),1)
                if action=='race':self.assertEqual(out.read_bytes(),b'competitor')
                else:
                    self.assertFalse(out.exists())
                    self.assertEqual(self.call(self.run_args(p,key,out)),0)
                    self.assertTrue(out.is_file())
                    self.assertEqual(len(list(root.glob('report.json.attempt-*.cells.jsonl'))),2)
                self.assertEqual(list(root.glob('.*.tmp')),[])

if __name__=='__main__':unittest.main()
