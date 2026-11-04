# SPDX-License-Identifier: MPL-2.0
"""Bounded synthetic native conformance, never the 72-row target experiment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import identity, json_bytes
from teacher_fit_v1 import custody as io, evidence, metrics, native, spec
from teacher_training_v1.numerical import configure


def synthetic_request(seed):
    model=spec.settings()[1]
    rows=[]
    for i in range(72):
        rows.append({'id':identity(str(i).encode()),'prompt':f'X{i}=',
            'answer':str(i%2),'observation':{'task':'check'},'family':'synthetic',
            'supervised_tokens':2})
    return {'seed':seed,'model':model.to_dict(),'native_config':spec.native_config(model,seed),
        'panel':rows,'dataset_train_identity':identity(b'synthetic-only'),
        'dataset_manifest_identity':identity(b'synthetic-manifest'),
        'dataset_rights_identity':identity(b'synthetic-rights'), 'model_parameters':90624,
        'selection':{'visit_ids':[rows[i%72]['id'] for i in range(1024)]}}


class NativeCanary(unittest.TestCase):
    def test_one_synthetic_step_diagnostics_restore_and_rng(self):
        import torch
        from trinite.training import run_steps, model_identity, require_optimizer
        from trinite.comparison_checkpoint import tensor_payload
        from teacher_logit_v1.native import forward_vector
        configure()
        request=synthetic_request(17);anchor=identity(json_bytes(request))
        states=[native.fresh_state(synthetic_request(s),anchor) for s in (17,29,43)]
        self.assertEqual(len({s.initialization_identity for s in states}),3)
        state=states[0];initial=state.initialization_identity
        self.assertFalse(state.optimizer.state);require_optimizer(state)
        before=native.guard(state)
        panel=request['panel'][:2]
        observation=metrics.observe(panel,lambda p:forward_vector(state.model,p)[0])
        native.unchanged(before,native.guard(state))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'checkpoint'
            cp_anchor=native.diagnostic(state,request,anchor,folder,0.)
            checked=evidence.checkpoint(request,anchor,folder,cp_anchor,0)
            self.assertEqual(checked['metadata']['step'],0)
            cold=native.restore(request,anchor,folder)
            self.assertEqual(native.identities(cold),native.identities(state))
            # A forged aggregate remains invalid even with a newly supplied test anchor.
            summary=io.load(folder/'summary.json');summary['groups']['all']['exact']+=1
            (folder/'summary.json').write_bytes(json_bytes(summary));(folder/'manifest.json').unlink()
            forged=io.seal(folder,{'schema':spec.SCHEMA+'.checkpoint-manifest','request_identity':anchor,'step':0})
            with self.assertRaises(Exception):evidence.checkpoint(request,anchor,folder,forged,0)
        run_steps(state,stop_after=1)  # sole optimizer update; synthetic prompts only
        self.assertEqual(state.step,1);self.assertEqual(state.target_tokens,8)
        self.assertNotEqual(model_identity(state.model),initial)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            io.write(folder/'tensors.safetensors',tensor_payload(state))
            io.record(folder/'metadata.json',native.metadata(state,request,anchor,1.))
            restored=native.restore(request,anchor,folder)
            self.assertEqual(native.identities(restored),native.identities(state))
            before=native.guard(restored)
            a=metrics.observe(panel,lambda p:forward_vector(state.model,p)[0])
            b=metrics.observe(panel,lambda p:forward_vector(restored.model,p)[0])
            self.assertEqual(a,b)
            native.unchanged(before,native.guard(restored))
            with torch.no_grad():next(restored.model.parameters()).add_(1)
            with self.assertRaises(Exception):native.unchanged(before,native.guard(restored))

    def test_backend_substitution_rejected_before_allocation(self):
        from trinite import training
        from trinite.model import Decoder
        configure();real=training.environment()
        for key,bad in (('torch','2.8.0+cu128'),('default_device','cuda'),('default_dtype','torch.float64'),('machine','aarch64')):
            changed=dict(real);changed[key]=bad
            with patch.object(training,'environment',return_value=changed),patch('trinite.model.Decoder') as model:
                with self.assertRaises(Exception):native.fresh_state(synthetic_request(17),identity(b'test'))
                model.assert_not_called()


if __name__=='__main__':unittest.main()
