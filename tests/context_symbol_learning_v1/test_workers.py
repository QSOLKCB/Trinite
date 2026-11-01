"""Exercise the actual worker lifecycle using a doubled updater/state."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from queue_learning_v1.artifacts import write_json,read_json
from procedural_dataset_v1.cases import expected_target
from context_symbol_learning_v1 import data,numerical,workers


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        rows=data.training_rows(data.inventory(),lambda r:expected_target('context',r))
        self.packet=self.root/'packet.json'
        sha=write_json(self.packet,dict(freeze_sha256='a'*64,train=rows,
            public=[dict(id='engineering-public',observation=data.observation(data.inventory()[0]))]))
        self.args=SimpleNamespace(packet=self.packet,packet_sha=sha,freeze_sha='a'*64,fit=True,seed=17,out=self.root/'attempt')
        self.state=SimpleNamespace(step=0,target_tokens=0,config=SimpleNamespace(steps=128),model=object(),initial_train_loss=(1.).hex())
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(numerical,'configure'))
        self.stack.enter_context(patch.object(numerical,'state_for',return_value=self.state))
        self.stack.enter_context(patch.object(numerical,'snapshot',return_value=(b'synthetic-checkpoint',b'{}\n')))
        self.stack.enter_context(patch.object(numerical,'fit_report',return_value=[]))
        self.require_fit=self.stack.enter_context(patch.object(numerical,'require_fit'))
        self.stack.enter_context(patch.object(workers,'load_checkpoint',return_value=self.state))
        self.stack.enter_context(patch('trinite.training.evaluate',return_value=(1.).hex()))
        self.stack.enter_context(patch('trinite.training.model_identity',return_value='engineering-unchanged'))
        self.updater=self.stack.enter_context(patch('trinite.training.run_steps',side_effect=self.step))
        self.stack.enter_context(patch.object(numerical,'infer',return_value=dict(status='malformed',tokens=[])))
        self.step_calls=[]

    def step(self,state,stop_after):
        self.step_calls.append(stop_after)
        state.step=stop_after
        state.target_tokens=stop_after*19 if self.args.fit else (4900 if stop_after==256 else stop_after*18)

    def test_fit_always_completes_128_updates_and_reloads_final(self):
        workers.attempt(self.args)
        self.assertEqual(self.step_calls,list(range(1,129)))
        self.require_fit.assert_called_once()
        self.assertEqual(read_json(self.args.out/'training.json')['target_tokens'],2432)
        self.assertTrue((self.args.out/'tensors.safetensors').exists())

    def test_main_completes_256_then_infers_after_validated_fit(self):
        self.args.fit=False
        self.state.config.steps=256
        self.args.fit_folder=self.root/'fit';self.args.fit_training_sha='b'*64
        self.args.fit_certificate=self.root/'certificate.json'
        self.args.fit_certificate_sha=write_json(self.args.fit_certificate,dict(synthetic_gate=True))
        with patch.object(workers,'validate_fit',return_value=dict(synthetic_gate=True)) as validate:
            workers.attempt(self.args)
        validate.assert_called_once()
        self.assertEqual(self.step_calls,list(range(1,257)))
        self.assertEqual(read_json(self.args.out/'training.json')['target_tokens'],4900)
        self.assertTrue((self.args.out/'predictions.jsonl').read_bytes())

    def test_partial_updater_failure_preserves_state_without_retry(self):
        self.updater.side_effect=RuntimeError('synthetic optimizer infrastructure failure')
        with self.assertRaises(RuntimeError):
            workers.attempt(self.args)
        self.assertTrue((self.args.out/'partial.safetensors').exists())
        self.assertTrue((self.args.out/'failure.json').exists())
        self.assertTrue((self.args.out/'resources.json').exists())
        self.assertEqual(self.updater.call_count,1)
        self.require_fit.assert_not_called()

    def test_incomplete_update_cannot_pass_or_infer(self):
        self.updater.side_effect=lambda *a,**kw:None
        with self.assertRaises(ValueError):
            workers.attempt(self.args)
        self.assertTrue((self.args.out/'failure.json').exists())
        self.require_fit.assert_not_called()


if __name__=='__main__':
    unittest.main()
