import unittest
from unittest.mock import patch
import torch
from trinite.contracts import ContractError, ModelConfig
from trinite.model import Decoder, Linear
from memory_traffic_v1.native import observe_forward
from memory_traffic_v1.accounting import parameter_accounting


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self,lane='ternary'):
        return Decoder(ModelConfig(context_length=8,blocks=2,width=6,heads=2,
                                   feed_forward_width=9),lane=lane,seed=17).eval()

    def inputs(self):
        return torch.tensor([[256,1,2],[256,3,0]],dtype=torch.int64), torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)

    def test_all_lanes_exact_parity_rng_state_and_shapes(self):
        for lane in ('dense','ternary','four-state'):
            with self.subTest(lane=lane), torch.no_grad():
                model=self.model(lane); ids,mask=self.inputs()
                state={n:p.clone() for n,p in model.named_parameters()}
                rng=torch.get_rng_state().clone()
                before=model(ids,mask).logits
                output,report=observe_forward(model,ids,mask)
                after=model(ids,mask).logits
                self.assertTrue(torch.equal(before,output.logits))
                self.assertTrue(torch.equal(before,after))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                for n,p in model.named_parameters():
                    self.assertTrue(torch.equal(p,state[n])); self.assertIsNone(p.grad)
                self.assertEqual(report['unique_parameter_storage_bytes'],sum(p.numel()*4 for p in model.parameters()))
                self.assertEqual(report['unique_parameter_storage_bytes'],parameter_accounting(model.config,lane)['native_float32_parameter_bytes'])
                self.assertEqual(len(report['linear_outputs']),12)
                for suffix, category in {'q':'q_outputs_all_blocks', 'k':'k_outputs_all_blocks',
                                         'v':'v_outputs_all_blocks', 'o':'o_outputs_all_blocks',
                                         'up':'ffn_up_outputs_all_blocks', 'down':'ffn_down_outputs_all_blocks'}.items():
                    observed=sum(r['logical_output_bytes'] for r in report['linear_outputs'] if r['name'].endswith('.'+suffix))
                    self.assertEqual(observed,report['shape_accounting']['selected_activation_payload_bytes'][category])
                self.assertEqual(report['logits_logical_bytes'],output.logits.numel()*4)
                self.assertIsNone(report['native_cache_payload_bytes'])
                self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_padding_does_not_shrink_allocated_payload(self):
        with torch.inference_mode():
            model=self.model(); ids,mask=self.inputs()
            _,a=observe_forward(model,ids,mask)
            _,b=observe_forward(model,ids,torch.ones_like(mask))
        self.assertEqual(a,b)

    def test_hooks_removed_on_forward_failure(self):
        model=self.model(); ids,mask=self.inputs()
        with torch.no_grad(), patch.object(model.blocks[0].o,'forward',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'): observe_forward(model,ids,mask)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_existing_hook_preserved(self):
        model=self.model(); ids,mask=self.inputs(); calls=[]
        handle=model.blocks[0].q.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad(): observe_forward(model,ids,mask)
        self.assertEqual(calls,[1]); self.assertEqual(len(model.blocks[0].q._forward_hooks),1)
        handle.remove()

    def test_grad_and_training_modes_rejected_without_switching(self):
        model=self.model(); ids,mask=self.inputs()
        with self.assertRaises(ContractError): observe_forward(model,ids,mask)
        self.assertTrue(torch.is_grad_enabled())
        model.train()
        with torch.no_grad(), self.assertRaises(ContractError): observe_forward(model,ids,mask)
        self.assertTrue(model.training)

    def test_invalid_input_leaves_no_hooks(self):
        model=self.model(); ids,mask=self.inputs()
        with torch.no_grad(), self.assertRaises(ContractError): observe_forward(model,ids.float(),mask)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_boundaries_one_and_full_context(self):
        model=self.model()
        for length in (1,8):
            with torch.no_grad():
                ids=torch.full((1,length),256,dtype=torch.int64)
                out,r=observe_forward(model,ids,torch.ones_like(ids,dtype=torch.bool))
                self.assertEqual(r['logits_logical_bytes'],length*259*4)
                self.assertEqual(tuple(out.logits.shape),(1,length,259))


if __name__ == '__main__': unittest.main()
