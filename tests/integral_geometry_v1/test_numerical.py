"""Conformance only: one update on training rows, no held-out quality run."""
import unittest
from unittest.mock import patch
import torch
from trinite.training import run_steps, model_identity, require_environment
from trinite.tokenizer import EOS
from integral_geometry_v1.data import examples
from integral_geometry_v1.experiment import create, training_data, decoded, score
from integral_geometry_v1.spec import PROTOCOL

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class MatchedCPU(unittest.TestCase):
    def test_paired_initialization_and_update(self):
        require_environment()
        a,b=create(0,'representation_control'),create(0,'geometry')
        self.assertEqual(a.initialization_identity,b.initialization_identity)
        self.assertEqual(a.config,b.config)
        self.assertEqual(a.model.config.context_length,128)
        for x,y in zip(a.data['train'],b.data['train']):
            self.assertEqual(x['identity'],y['identity'])
            self.assertEqual([t for t,m in zip(x['ids'],x['loss']) if m],
                             [t for t,m in zip(y['ids'],y['loss']) if m])
        initial=model_identity(a.model)
        run_steps(a,stop_after=1)
        run_steps(b,stop_after=1)
        self.assertEqual(a.target_tokens,b.target_tokens)
        self.assertNotEqual(model_identity(a.model),initial)
        self.assertEqual(a.step,1)
        self.assertEqual(a.validation,[]) # no held-out diagnostic or selection

    def test_generation_requires_eos_and_no_gold_fallback(self):
        self.assertIsNone(decoded([49]))
        self.assertIsNone(decoded([258,EOS]))
        self.assertEqual(decoded([49,EOS]),'1')
        calls=[]
        def fake(model,prompt,cap):
            calls.append(prompt)
            return [ord('?'),EOS]
        with patch('integral_geometry_v1.experiment.greedy',fake):
            result=score(None)
        self.assertEqual(len(calls),2*len(examples()))
        self.assertTrue(all(not r['end_to_end_tokens'] and not r['end_to_end'] for r in result['records']))
        self.assertTrue(all(not r['setup_semantic'] for r in result['records']))

    def test_valid_wrong_setup_is_forwarded_without_repair(self):
        calls=[]
        wrong='I(0;1;99)'
        def fake(model,prompt,cap):
            calls.append(prompt)
            return [*wrong.encode(),EOS] if not prompt.startswith('eval[') else [49,EOS]
        with patch('integral_geometry_v1.experiment.greedy',fake):
            result=score(None)
        self.assertEqual(len(calls),3*len(examples()))
        self.assertTrue(all(calls[i]=='eval['+wrong+']=' for i in range(2,len(calls),3)))
        self.assertTrue(all(not r['setup_semantic'] for r in result['records']))

if __name__=='__main__':unittest.main()
