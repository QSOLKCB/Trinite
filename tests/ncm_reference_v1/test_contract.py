"""Software-only tests. Fakes never produce retained CUDA evidence."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ncm_reference_v1 import core, evidence, runner, runtime


class Contract(unittest.TestCase):
    def test_request_regenerates_and_binds_transitive_closure(self):
        req = core.request()
        self.assertEqual(req, core.request())
        self.assertEqual(req['inputs'], core.inputs())
        self.assertEqual(req['protocol']['admission']['outcome'], 'admitted only for exposed external engineering smoke')
        self.assertIn('scripts/ncm_reference_v1/vendor/models/quant.py', req['source'])
        self.assertIn('scripts/run_ncm_reference.py', req['source'])
        self.assertEqual(len(req['inputs']), 4)

    def test_safe_loader_has_no_unsafe_fallback(self):
        source = (core.ROOT/'runtime.py').read_text()
        tree = ast.parse(source)
        calls = [x for x in ast.walk(tree) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                 and isinstance(x.func.value, ast.Name) and x.func.value.id=='torch' and x.func.attr=='load']
        self.assertEqual(len(calls), 1)
        self.assertTrue(next(k.value.value for k in calls[0].keywords if k.arg=='weights_only'))
        self.assertNotIn('weights_only=False', source)
        self.assertNotIn('from_pretrained(', source)
        self.assertNotIn('pickle.load', source)

    def test_archive_external_hash_and_inventory(self):
        packet = evidence.pack({'abc': b'123'})
        self.assertEqual(evidence.unpack(packet, core.digest(packet)), {'abc': b'123'})
        with self.assertRaises(ValueError):
            evidence.unpack(packet, '0'*64)
        bad = evidence.pack({'../abc': b'123'})
        with self.assertRaises(ValueError):
            evidence.unpack(bad, core.digest(bad))

    def test_failed_run_is_retained_not_success(self):
        req=core.request()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); frozen=root/'request.json'; frozen.write_bytes(core.json_bytes(req))
            # Actual fresh child fails on missing assets before importing torch.
            packet,state=runner.run(root/'failed.zip', root/'missing', frozen, core.digest(frozen.read_bytes()))
            self.assertEqual(state,'failed')
            result=evidence.verify(packet,core.digest(packet))
            self.assertFalse(result['inference_verified'])
            self.assertEqual(result['state'],'failed')
            self.assertIn(b'unreadable regular input',evidence.unpack(packet,core.digest(packet))['worker.log'])
            with self.assertRaises(FileExistsError):
                runner.run(root/'failed.zip',root/'missing',frozen,core.digest(frozen.read_bytes()))

    def test_wrong_request_prevents_worker(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'req';p.write_bytes(b'{}')
            with patch.object(runner.subprocess,'Popen') as launch:
                with self.assertRaises(ValueError):
                    runner.run(Path(d)/'out',d,p,core.digest(b'{}'))
                launch.assert_not_called()

    def test_cpu_and_wrong_dependencies_fail_closed(self):
        class Cuda:
            def is_available(self):return False
        class Torch:cuda=Cuda()
        with patch.object(runtime.importlib.metadata,'version',side_effect=lambda n:runtime.VERSIONS[n]):
            with patch.object(runtime.sys,'version_info',(3,11)):
                with self.assertRaisesRegex(ValueError,'no CPU fallback'):
                    runtime.preflight(Torch(),core.request()['protocol'])
        with patch.object(runtime.importlib.metadata,'version',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'version mismatch'):
                runtime.preflight(Torch(),core.request()['protocol'])



class AdapterDoubles(unittest.TestCase):
    def test_actual_api_call_shapes_and_clue_only_prefix(self):
        class Truth:
            def any(self): return False
            def all(self): return True
            def __or__(self, other): return self
        class Tensor:
            def __init__(self,data): self.data=data; self.shape=(len(data),len(data[0]))
            def __lt__(self,other):return Truth()
            def __ge__(self,other):return Truth()
            def cpu(self):return self
            def tolist(self):return self.data
        class Logits:
            shape=(1,89,12)
            def argmax(self,dim):return Tensor([core.encode_board(core.inputs()[0]['target'])])
        class Cuda:
            def synchronize(self):pass
        class Torch:
            cuda=Cuda();long='long';seen=None
            def tensor(self,data,**kwargs):self.seen=data;return Tensor(data)
            def isfinite(self,value):return Truth()
        class Net:
            def generate(self,idx,prefix,**kwargs):
                self.args=(idx,prefix,kwargs)
                return Tensor([[0]*322])
        class VQ:
            def idx_to_zhats(self,ids,return_zlevels):
                self.args=(ids,return_zlevels);return None,['final-level']
        class Codec:
            vq=VQ()
            def decoder(self,level):assert level=='final-level';return Logits()
        torch=Torch();net=Net();codec=Codec();item=core.inputs()[0]
        result,timing=runtime.infer(torch,net,codec,item['clues'])
        self.assertEqual(torch.seen,[[11]+core.encode_board(item['clues'])+[11]])
        self.assertIsNone(net.args[0]);self.assertEqual(net.args[2],dict(temperature=0.0,top_k=0,top_p=1.0,cfg_scale=0.0,cfg_null_eos=None,use_kv_cache=False))
        self.assertEqual(len(result['latent_ids']),322)
        self.assertEqual(result['raw_ids'],core.encode_board(item['target']))
        self.assertGreaterEqual(timing['total_seconds'],0)

    def test_restricted_load_exception_never_retries(self):
        from unittest.mock import Mock
        payload=b'fake checkpoint fixture, not real weights'
        torch=Mock();torch.load.side_effect=RuntimeError('unsupported global')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weight';p.write_bytes(payload)
            with patch.object(runtime, 'validate_loader_metadata'), self.assertRaisesRegex(RuntimeError,'unsupported global'):
                runtime.safe_state(torch,p,dict(size=len(payload),sha256=core.digest(payload),path='sudoku/easy/tokenizer.pt'),
                                   role='tokenizer', expected_state={'weight': object()}, fixed_buffers={})
        torch.load.assert_called_once()
        self.assertEqual(torch.load.call_args.kwargs,dict(weights_only=True,map_location='cpu'))

if __name__=='__main__':unittest.main()
