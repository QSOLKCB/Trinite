"""Synthetic only: never invoke actual NVIDIA utilities or Torch CUDA calls."""
import ast
import hashlib
import contextlib
import io
import stat
from types import SimpleNamespace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from cuda_availability_v1 import check as c


def process(raw=b'', status='completed', code=0, stderr=False):
    return {'status':status,'returncode':code,'stdout':raw,'stderr_seen':stderr}


class DiagnosticTests(unittest.TestCase):
    def test_visibility_only_categories_no_values(self):
        for value,expected in [(None,'unset'),('','empty'),('-1','minus-one'),
                               ('0,2','index-list'),('GPU-secret-identity','other-redacted'),
                               ('/home/private','other-redacted'),('0'*300,'other-redacted')]:
            with patch.dict(os.environ,{},clear=True):
                if value is not None:os.environ['CUDA_VISIBLE_DEVICES']=value
                self.assertEqual(c.visibility_class(),expected)

    def test_metadata_is_fixed_distribution_subset_and_absence_is_not_import_failure(self):
        data={'torch':{'status':'absent'},'numpy':{'status':'present','version':'2.3.5'},
              'safetensors':{'status':'metadata_error'}}
        with patch.object(c,'bounded_process',return_value=process(json.dumps(data).encode())) as run:
            result=c.metadata_query()
        self.assertEqual(result['distributions'],data)
        argv=run.call_args.args[0]
        self.assertEqual(argv[:4],[sys.executable,'-I','-B','-c'])
        self.assertNotIn('import torch',argv[4])
        for bad in ('/home/user/private','1.0\nsecret','1.0/'+'a'*100):
            data['numpy']['version']=bad
            with patch.object(c,'bounded_process',return_value=process(json.dumps(data).encode())):
                self.assertEqual(c.metadata_query()['status'],'invalid_response')

    def test_venv_interpreter_symlink_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'python';p.symlink_to(sys.executable)
            with patch.object(c.sys,'executable',str(p)),patch.object(c,'bounded_process',return_value=process(status='spawn_failed')) as run:
                c.cuda_query()
                self.assertEqual(run.call_args.args[0][0],str(p))

    def test_smi_allowlisted_query_and_separate_outcomes(self):
        for status in ('tool_unavailable','unsafe_executable'):
            with patch.object(c,'_trusted_smi',return_value=(status,None)),patch.object(c,'bounded_process') as run:
                self.assertEqual(c.smi_query(),{'status':status});run.assert_not_called()
        with patch.object(c,'_trusted_smi',return_value=('available','/trusted/nvidia-smi')), \
             patch.object(c,'bounded_process',return_value=process(b'580.95.05, NVIDIA GeForce RTX 5060 Ti, 16303\n')) as run:
            result=c.smi_query()
            self.assertEqual(run.call_args.args[0],['/trusted/nvidia-smi',*c.SMI_ARGUMENTS])
            self.assertEqual(result['devices'][0]['total_memory_mib'],16303)
        with patch.object(c,'sys') as fake:
            fake.platform='win32';self.assertEqual(c.smi_query()['status'],'unsupported_platform')

    def test_smi_noise_private_fields_and_extra_columns_are_not_exported(self):
        for data in (b'bad driver /home/secret',b'580.1, NVIDIA GPU-secret-uuid, 16000\n',
                     b'580.1, NVIDIA RTX, 16000, GPU-secret\n',b'580.1, NVIDIA RTX, NaN\n',
                     b'580.1, /home/private, 16000\n'):
            with patch.object(c,'_trusted_smi',return_value=('available','/trusted/nvidia-smi')), \
                 patch.object(c,'bounded_process',return_value=process(data,stderr=True)):
                result=c.smi_query()
                self.assertEqual(result['status'],'invalid_response')
                self.assertNotIn('private',json.dumps(result));self.assertNotIn('secret',json.dumps(result))

    def test_torch_query_absent_visible_unavailable_and_driver_error_are_distinct(self):
        base={'torch_version':'2.8.0+cu128','compiled_cuda':'12.8'}
        cases=[({'status':'torch_import_failed'},'torch_unavailable'),
               ({**base,'status':'query_failed'},'query_failed'),
               ({**base,'status':'no_visible_cuda','available':False,'device_count':0},'no_visible_cuda'),
               ({**base,'status':'visible_cuda','available':True,'device_count':1,'devices':[
                   {'index':0,'name':'NVIDIA RTX 5060 Ti','total_memory_bytes':16*1024**3,'capability':[12,0]}]},'visible_cuda')]
        for data,expected in cases:
            with patch.object(c,'bounded_process',return_value=process(json.dumps(data).encode())):
                self.assertEqual(c.cuda_query()['status'],expected)
        for bad in ({**base,'status':'visible_cuda','available':True,'device_count':17},
                    {**base,'status':'no_visible_cuda','available':True,'device_count':1},
                    {**base,'status':'query_failed','exception':'/home/private'},
                    {**base,'status':'query_failed','torch_version':'/home/private'}):
            with patch.object(c,'bounded_process',return_value=process(json.dumps(bad).encode())):
                self.assertEqual(c.cuda_query()['status'],'invalid_response')

    def test_default_never_requests_torch_and_report_bound_to_sources(self):
        with patch.object(c,'metadata_query',return_value={'status':'completed'}), \
             patch.object(c,'smi_query',return_value={'status':'tool_unavailable'}), \
             patch.object(c,'cuda_query') as cuda:
            receipt=c.diagnose()
            cuda.assert_not_called()
        report=receipt['report']
        self.assertEqual(receipt['report_identity'],c._identity(c._json(report)))
        self.assertEqual(set(report['source_files']),set(c.SOURCE_FILES))
        for field in ('hardware_attested','gpu_health_established','trial_admission_established','model_executed',
                      'tensor_allocation_requested','training_performed','repair_attempted'):
            self.assertIs(report[field],False)
        self.assertEqual(report['root_cause'],'unknown')
        self.assertNotIn(sys.executable,json.dumps(report))

    def test_source_mutation_rejected(self):
        with patch.object(c,'source_snapshot',side_effect=[{'a':'b'},{'a':'c'}]), \
             patch.object(c,'metadata_query'),patch.object(c,'smi_query'):
            with self.assertRaises(ValueError):c.diagnose()

    def test_noisy_child_and_timeout_are_bounded_and_stderr_discarded(self):
        start=time.monotonic()
        result=c.bounded_process([sys.executable,'-I','-B','-c',
            "import os;os.write(2,b'/home/private secret'*10000)"],timeout=2,limit=1024)
        self.assertEqual(result['status'],'output_limit');self.assertEqual(result['stdout'],b'')
        result=c.bounded_process([sys.executable,'-I','-B','-c','import time;time.sleep(30)'],timeout=.15)
        self.assertEqual(result['status'],'timed_out');self.assertLess(time.monotonic()-start,4)

    def test_both_pipes_drain_and_failure_does_not_publish_partial_stdout(self):
        result=c.bounded_process([sys.executable,'-I','-B','-c',
            "import os;os.write(1,b'ok');os.write(2,b'/home/private');raise SystemExit(7)"])
        self.assertEqual(result['status'],'command_failed');self.assertEqual(result['returncode'],7)
        self.assertEqual(result['stdout'],b'');self.assertTrue(result['stderr_seen'])
        result=c.bounded_process([sys.executable,'-I','-B','-c',"print('ok')"])
        self.assertEqual(result['stdout'],b'ok\n')

    def test_child_grandchild_pipe_cannot_defeat_deadline(self):
        code="import subprocess,sys;subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)'])"
        start=time.monotonic();result=c.bounded_process([sys.executable,'-I','-B','-c',code],timeout=.2)
        self.assertEqual(result['status'],'timed_out');self.assertLess(time.monotonic()-start,3)

    def test_interrupt_cleanup_kills_waits_and_closes_both_pipes(self):
        created=[];real_popen=subprocess.Popen
        def spawning(*args,**kwargs):
            p=real_popen(*args,**kwargs);created.append(p);return p
        selector=c.selectors.DefaultSelector()
        with patch.object(c.selectors,'DefaultSelector',return_value=selector), \
             patch.object(selector,'select',side_effect=KeyboardInterrupt), \
             patch.object(c.subprocess,'Popen',side_effect=spawning):
            with self.assertRaises(KeyboardInterrupt):
                c.bounded_process([sys.executable,'-I','-B','-c','import time;time.sleep(30)'])
        self.assertIsNotNone(created[0].poll())
        self.assertTrue(created[0].stdout.closed);self.assertTrue(created[0].stderr.closed)

    def test_combined_limit_and_closed_pipes_still_bound_wait(self):
        result=c.bounded_process([sys.executable,'-I','-B','-c',
            "import os;os.write(1,b'x'*600);os.write(2,b'y'*600)"],limit=1000)
        self.assertEqual(result['status'],'output_limit')
        result=c.bounded_process([sys.executable,'-I','-B','-c',
            'import os,time;os.close(1);os.close(2);time.sleep(30)'],timeout=.2)
        self.assertEqual(result['status'],'timed_out')

    def test_trusted_executable_rejects_foreign_owner_writable_and_missing(self):
        target=Path('/usr/lib/nvidia/nvidia-smi')
        def safe_lstat(path):
            return SimpleNamespace(st_uid=0,st_mode=(stat.S_IFREG|0o755) if path.name=='nvidia-smi' else (stat.S_IFDIR|0o755))
        with patch.object(Path,'lstat',safe_lstat),patch.object(Path,'resolve',return_value=target), \
             patch.object(Path,'stat',return_value=SimpleNamespace(st_mode=stat.S_IFREG|0o755)):
            self.assertEqual(c._trusted_smi(),('available',str(target)))
        for uid,mode in ((1000,stat.S_IFREG|0o755),(0,stat.S_IFREG|0o777)):
            with patch.object(Path,'lstat',return_value=SimpleNamespace(st_uid=uid,st_mode=mode)):
                self.assertEqual(c._trusted_smi()[0],'unsafe_executable')
        with patch.object(Path,'lstat',side_effect=FileNotFoundError):
            self.assertEqual(c._trusted_smi()[0],'tool_unavailable')

    def test_optional_torch_child_uses_fake_metadata_only(self):
        calls=[]
        fake=SimpleNamespace(__version__='2.8.0+cu128',version=SimpleNamespace(cuda='12.8'),
             cuda=SimpleNamespace(is_available=lambda: calls.append('available') or False,
                                  device_count=lambda: calls.append('count') or 0))
        output=io.StringIO()
        with patch.dict(sys.modules,{'torch':fake}),contextlib.redirect_stdout(output):
            exec(compile(c.TORCH_CODE,'<fixed synthetic query>','exec'),{})
        self.assertEqual(calls,['available','count'])
        self.assertEqual(json.loads(output.getvalue())['status'],'no_visible_cuda')

    def test_source_reader_refuses_link_oversize_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'x';p.write_bytes(b'x')
            link=root/'link';link.symlink_to(p)
            for target in (root,link):
                with self.assertRaises(ValueError):c._read_source(target)
            p.write_bytes(b'x'*(c.SOURCE_LIMIT+1))
            with self.assertRaises(ValueError):c._read_source(p)

    def test_child_programs_contain_no_tensor_model_update_or_install_calls(self):
        for code in (c.METADATA_CODE,c.TORCH_CODE):
            tree=ast.parse(code)
            attributes={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
            self.assertFalse(attributes & {'tensor','empty','zeros','ones','load','save','backward','step','to','compile'})
            self.assertNotIn('subprocess',code)
        self.assertNotIn('environ',c.TORCH_CODE)

    def test_launcher_guard_before_path_import_and_unknown_arguments_sanitized(self):
        script=c.ROOT/'scripts/check_cuda_availability.py'
        for flags in ([],['-I'],['-B']):
            result=subprocess.run([sys.executable,*flags,str(script),'--help'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Use python -I -B',result.stderr)
        result=subprocess.run([sys.executable,'-I','-B',str(script),'--private=/home/secret'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertNotIn('/home/secret',result.stderr)
        result=subprocess.run([sys.executable,'-I','-B',str(script),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0);self.assertIn('--query-cuda',result.stdout)

    def test_launcher_cache_and_stdlib_shadow_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in c.SOURCE_FILES:
                destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(c.ROOT/name,destination)
            sentinel=root/'executed'
            (root/'scripts/pathlib.py').write_text('open('+repr(str(sentinel))+',"w").write("bad")')
            script=root/'scripts/check_cuda_availability.py'
            result=subprocess.run([sys.executable,'-I','-B',str(script),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0);self.assertFalse(sentinel.exists())
            (root/'scripts/cuda_availability_v1/__pycache__').mkdir()
            result=subprocess.run([sys.executable,'-I','-B',str(script),'--help'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('caches',result.stderr)
            self.assertFalse(sentinel.exists())

    def test_launcher_has_no_project_path_insertion_or_native_import(self):
        text=(c.ROOT/'scripts/check_cuda_availability.py').read_text()
        self.assertNotIn('sys.path',text);self.assertNotIn('import torch',text)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
