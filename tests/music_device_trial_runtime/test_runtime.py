"""Runtime policy tests use explicit software doubles, never CUDA hardware."""
from contextlib import ExitStack
import os
import builtins
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from music_device_trial_v1 import runtime as r


def fake_torch():
    props = SimpleNamespace(name='synthetic SM120 device', total_memory=16*1024**3,
                            multi_processor_count=36, uuid='synthetic-only')
    memory = SimpleNamespace(get_per_process_memory_fraction=Mock(return_value=.5),
                             set_per_process_memory_fraction=Mock())
    cuda = SimpleNamespace(is_available=Mock(return_value=True), device_count=Mock(return_value=1),
        get_device_properties=Mock(return_value=props), get_device_capability=Mock(return_value=(12, 0)),
        get_arch_list=Mock(return_value=['sm_90','sm_120']), is_initialized=Mock(return_value=False),
        current_device=Mock(return_value=0), current_stream=Mock(return_value='default'),
        default_stream=Mock(return_value='default'), set_device=Mock(), synchronize=Mock(), memory=memory)
    return SimpleNamespace(__version__='2.8.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=cuda,
        float32='torch.float32', get_default_dtype=Mock(return_value='torch.float32'),
        get_default_device=Mock(return_value='cpu'), get_num_threads=Mock(return_value=1),
        get_num_interop_threads=Mock(return_value=1), set_num_threads=Mock(), set_default_dtype=Mock(),
        use_deterministic_algorithms=Mock(), are_deterministic_algorithms_enabled=Mock(return_value=True),
        is_deterministic_algorithms_warn_only_enabled=Mock(return_value=False),
        is_autocast_enabled=Mock(return_value=False), set_float32_matmul_precision=Mock(),
        get_float32_matmul_precision=Mock(return_value='highest'),
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False, deterministic=True,
            allow_tf32=False, version=Mock(return_value=91002)),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cpu=SimpleNamespace(get_cpu_capability=Mock(return_value='synthetic'))))


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.real_host=r._host; self.real_driver=r._driver
        self.t=fake_torch()
        self.stack.enter_context(patch.object(r,'_torch',return_value=self.t))
        self.stack.enter_context(patch.object(r,'_host',return_value={'python':'synthetic3.12'}))
        self.stack.enter_context(patch.object(r,'_packages',return_value=dict(r.PACKAGES)))
        self.stack.enter_context(patch.object(r,'_driver',return_value='synthetic driver'))
        self.stack.enter_context(patch.object(r,'_cpu',return_value={'model name':'synthetic CPU'}))
        self.stack.enter_context(patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':r.WORKSPACE},clear=True))

    def test_same_cuda_environment_required_for_both_cells(self):
        cpu=r.require_environment('cpu'); gpu=r.require_environment('cuda')
        self.assertEqual({k:v for k,v in cpu.items() if k!='backend'},
                         {k:v for k,v in gpu.items() if k!='backend'})
        self.assertEqual(cpu['allocator_limit_bytes'],8*1024**3)
        self.assertEqual(cpu['hardware']['capability'],[12,0])
        self.t.cuda.is_available.return_value=False
        for backend in ('cpu','cuda'):
            with self.assertRaises(r.UnsupportedRuntime):r.require_environment(backend)

    def test_unpinned_package_cpu_wheel_and_wrong_platform_reject(self):
        with patch.object(r,'_packages',return_value={'torch':'2.8.0+cpu'}):
            with self.assertRaises(r.UnsupportedRuntime):r.require_environment('cpu')
        self.t.__version__='2.8.0+cpu'
        with self.assertRaises(r.UnsupportedRuntime):r.require_environment('cpu')
        self.t.cuda.is_available.assert_not_called()

    def test_device_architecture_memory_and_multi_gpu_reject(self):
        mutations=[(self.t.cuda.device_count,2),(self.t.cuda.get_device_capability,(8,9)),
                   (self.t.cuda.get_arch_list,['compute_120','sm_90'])]
        for method,value in mutations:
            before=method.return_value;method.return_value=value
            with self.assertRaises(r.UnsupportedRuntime):r.require_environment('cuda')
            method.return_value=before
        p=self.t.cuda.get_device_properties.return_value
        for total in (8*1024**3,24*1024**3,True):
            p.total_memory=total
            with self.assertRaises(r.UnsupportedRuntime):r.require_environment('cuda')

    def test_drift_flags_and_allocator_are_rejected_not_repaired(self):
        methods=[(self.t.get_num_threads,2),(self.t.get_default_dtype,'torch.float64'),
            (self.t.get_default_device,'cuda:0'),(self.t.are_deterministic_algorithms_enabled,False),
            (self.t.is_deterministic_algorithms_warn_only_enabled,True),
            (self.t.get_float32_matmul_precision,'high'),(self.t.is_autocast_enabled,True),
            (self.t.cuda.current_device,1),(self.t.cuda.current_stream,'other'),
            (self.t.cuda.memory.get_per_process_memory_fraction,1.0)]
        for method,value in methods:
            before=method.return_value;method.return_value=value
            with self.assertRaises(r.UnsupportedRuntime):r.require_environment('cuda')
            method.return_value=before
        for owner,name,value in [(self.t.backends.cudnn,'benchmark',True),
                                  (self.t.backends.cudnn,'deterministic',False),
                                  (self.t.backends.cudnn,'allow_tf32',True),
                                  (self.t.backends.cuda.matmul,'allow_tf32',True)]:
            before=getattr(owner,name);setattr(owner,name,value)
            with self.assertRaises(r.UnsupportedRuntime):r.require_environment('cpu')
            setattr(owner,name,before)
        self.t.set_num_threads.assert_not_called()

    def test_configure_before_work_and_no_repair_reentrant(self):
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG')
        r.configure()
        self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],r.WORKSPACE)
        self.t.use_deterministic_algorithms.assert_called_once_with(True,warn_only=False)
        self.t.cuda.memory.set_per_process_memory_fraction.assert_called_once_with(.5,0)
        self.t.cuda.is_initialized.return_value=True
        self.t.backends.cuda.matmul.allow_tf32=True
        with self.assertRaises(r.UnsupportedRuntime):r.configure()
        self.assertTrue(self.t.backends.cuda.matmul.allow_tf32)

    def test_conflicting_workspace_rejected_before_torch(self):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
        with self.assertRaises(r.UnsupportedRuntime):r.configure()
        r._torch.assert_not_called()

    def test_hidden_precision_allocator_and_late_workspace_overrides_reject(self):
        for name in ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE',
                     'CUDA_LAUNCH_BLOCKING', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF'):
            with patch.dict(os.environ, {name: '1'}):
                with self.assertRaises(r.UnsupportedRuntime):r.require_environment('cuda')
                with self.assertRaises(r.UnsupportedRuntime):r.configure()
        self.t.cuda.is_initialized.return_value=True
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG')
        with self.assertRaises(r.UnsupportedRuntime):r.configure()

    def test_synchronize_explicit_backend_only(self):
        r.synchronize('cpu');self.t.cuda.synchronize.assert_not_called()
        r.synchronize('cuda');self.t.cuda.synchronize.assert_called_once_with(0)
        for value in ('auto','cuda:0','gpu',None,True):
            with self.assertRaises(r.UnsupportedRuntime):r.synchronize(value)

    def test_driver_and_visible_device_mapping_are_recorded_for_request_binding(self):
        before=r.require_environment('cuda')
        with patch.object(r,'_driver',return_value='another driver'):
            self.assertNotEqual(before,r.require_environment('cuda'))
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-other'}):
            self.assertNotEqual(before,r.require_environment('cuda'))
        with patch.object(Path,'open',side_effect=OSError('missing')):
            with self.assertRaises(r.UnsupportedRuntime):self.real_driver()
        with patch.object(Path,'open',return_value=io.BytesIO(b'x'*4097)):
            with self.assertRaises(r.UnsupportedRuntime):self.real_driver()

    def test_supported_host_is_explicit_not_guessed(self):
        with patch.object(r.platform,'system',return_value='Windows'):
            with self.assertRaises(r.UnsupportedRuntime):self.real_host()
        with patch.object(r.sys,'version_info',(3,13,0)):
            with self.assertRaises(r.UnsupportedRuntime):self.real_host()

    def test_stable_identity_no_volatile_memory_queries(self):
        first=r.require_environment('cpu');second=r.require_environment('cpu')
        self.assertEqual(first,second)
        self.assertNotIn('free_memory',first['hardware'])


class ImportBoundaryTests(unittest.TestCase):
    def test_unavailable_torch_and_shared_libraries_are_unsupported(self):
        original=builtins.__import__
        for error in (ImportError('missing torch'), OSError('missing CUDA library'),
                      RuntimeError('incompatible runtime')):
            def importing(name,*args,**kwargs):
                if name=='torch':raise error
                return original(name,*args,**kwargs)
            with patch.object(builtins,'__import__',side_effect=importing):
                with self.assertRaises(r.UnsupportedRuntime) as caught:r._torch()
                self.assertIs(caught.exception.__cause__,error)
        with patch.object(builtins,'__import__',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):r._torch()


class DependencyLockTests(unittest.TestCase):
    def test_exact_complete_separate_lock_and_no_hardware_claim(self):
        root=Path(__file__).resolve().parents[2]
        data=json.loads((root/'scripts/music_device_trial_v1/runtime-lock.json').read_bytes())
        self.assertFalse(data['hardware_tested'])
        self.assertFalse(data['wheel_payloads_downloaded'])
        packages=data['packages']
        self.assertEqual({n:v['version'] for n,v in packages.items()},r.PACKAGES)
        text=(root/'requirements.music-device.lock').read_text()
        self.assertIn('--require-hashes',text)
        for name,p in packages.items():
            self.assertRegex(p['sha256'],r'^[a-f0-9]{64}$')
            self.assertIn(name+' @ '+p['url'],text)
            self.assertIn('--hash=sha256:'+p['sha256'],text)
            self.assertTrue(p['url'].startswith(('https://download.pytorch.org/',
                'https://download-r2.pytorch.org/', 'https://pypi.nvidia.com/',
                'https://files.pythonhosted.org/')))
            for requirement in p['required_dependencies']:
                dependency=re.match(r'[A-Za-z0-9_-]+',requirement).group().lower().replace('_','-')
                self.assertIn(dependency,packages)
        self.assertEqual(packages['torch']['sha256'],
            '4354fc05bb79b208d6995a04ca1ceef6a9547b1c4334435574353d381c55087c')


if __name__=='__main__':unittest.main()
