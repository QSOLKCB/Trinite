"""CPU-only adapter parity; these tests do not certify CUDA hardware."""
import unittest
import torch
from trinite.contracts import ContractError
from trinite.model import Decoder
from trinite.training import target_loss as native_loss
from music_placement_v1.spec import plan_for
from music_device_trial_v1.adapter import DeviceDecoder, model_identity, target_loss, batch
from music_device_trial_v1.numerical import _probe_examples


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_cpu_native_initial_forward_loss_and_gradient_exact(self):
        cfg=plan_for(17).model;native=Decoder(cfg,lane='dense',seed=17);trial=DeviceDecoder(cfg,17,'cpu')
        from trinite.training import model_identity as native_identity
        self.assertEqual(model_identity(trial),native_identity(native))
        self.assertGreaterEqual(float.fromhex(trial.cpu_initialization_seconds_hex),0.)
        self.assertGreaterEqual(float.fromhex(trial.model_transfer_seconds_hex),0.)
        ids,mask,_=batch(_probe_examples(),'cpu')
        self.assertTrue(torch.equal(native(ids[:,:-1],mask[:,:-1]).logits,trial(ids[:,:-1],mask[:,:-1]).logits))
        left,lc=native_loss(native,_probe_examples());right,rc=target_loss(trial,_probe_examples())
        self.assertEqual(lc,rc);self.assertTrue(torch.equal(left,right));left.backward();right.backward()
        for (_,a),(_,b) in zip(native.named_parameters(),trial.named_parameters()):self.assertTrue(torch.equal(a.grad,b.grad))

    def test_parameter_tying_and_invalid_state_rejected(self):
        model=DeviceDecoder(plan_for(17).model,17,'cpu');self.assertIs(model.output_weight,model.token_embedding)
        model.output_weight=torch.nn.Parameter(model.output_weight.clone())
        with self.assertRaises(ContractError):model.validate_state()
        model=DeviceDecoder(plan_for(17).model,17,'cpu');model.blocks[0].q.lane='ternary'
        with self.assertRaises(ContractError):model.validate_state()
        model=DeviceDecoder(plan_for(17).model,17,'cpu');model.register_buffer('extra',torch.zeros(1))
        with self.assertRaises(ContractError):model.validate_state()

    def test_input_shape_dtype_mask_and_nonfinite_fail_closed(self):
        model=DeviceDecoder(plan_for(17).model,17,'cpu');ids,mask,_=batch(_probe_examples(),'cpu')
        for badids,badmask in [(ids.float(),mask),(ids,mask.long()),(ids[:,:0],mask[:,:0])]:
            with self.assertRaises(ContractError):model(badids,badmask)
        mask[:,1]=False
        with self.assertRaises(ContractError):model(ids,mask)
        with torch.no_grad():model.token_embedding[0,0]=float('nan')
        with self.assertRaises(ContractError):model.validate_state()
        with self.assertRaises(ContractError):DeviceDecoder(plan_for(17).model,17,'automatic')

if __name__=='__main__':unittest.main()
