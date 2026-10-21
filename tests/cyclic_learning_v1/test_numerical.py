import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from cyclic_learning_v1.data import examples
from cyclic_learning_v1.experiment import create,grade,training_data,report
from cyclic_learning_v1.spec import request
from trinite.contracts import ContractError
from trinite.tokenizer import EOS
from trinite.training import run_steps,model_identity,require_environment

class Numerical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        require_environment()

    def test_paired_one_real_update_not_target_experiment(self):
        a=create(0,'decimal');b=create(0,'sexagesimal')
        self.assertEqual(a.initialization_identity,b.initialization_identity)
        self.assertEqual([r['identity'] for r in a.data['train']],[r['identity'] for r in b.data['train']])
        for x,y in zip(a.data['train'],b.data['train']):
            self.assertEqual(x['loss'],y['loss']);self.assertEqual(x['mask'],y['mask'])
        for state in (a,b):
            run_steps(state,stop_after=1)
            self.assertEqual(state.step,1);self.assertEqual(state.target_tokens,16)
            self.assertNotEqual(model_identity(state.model),state.initialization_identity)

    def test_no_heldout_labels_in_training_or_diagnostics(self):
        allowed={r['identity'] for r in examples() if r['split']=='train'}
        for arm in ('decimal','sexagesimal'):
            data=training_data(1,arm)
            self.assertEqual({r['identity'] for r in data['train']},allowed)
            self.assertEqual(data['validation'],data['train'])

    def test_raw_scoring_no_repair(self):
        self.assertEqual(grade([48,48,55,EOS],'007'),{'canonical':True,'exact':True})
        for tokens in ([55,EOS],[48,48,55],[48,48,55,EOS,EOS],[48,48,55,256]):
            self.assertFalse(grade(tokens,'007')['exact'])
        self.assertFalse(grade([48,48,56,EOS],'007')['exact'])

    def test_request_mismatch_before_any_training(self):
        with self.assertRaises(ContractError):report(expected_request={'wrong':'request'})

    def test_completed_training_retained_before_later_failure(self):
        # Controller sequencing test only; these synthetic values are not evidence.
        state=SimpleNamespace(initialization_identity='paired',model=None,step=32,
                              target_tokens=512,history=[{'synthetic':'test only'}])
        frozen=request(require_environment())
        records=[]
        with patch('cyclic_learning_v1.experiment.create',return_value=state), \
             patch('cyclic_learning_v1.experiment.model_identity',return_value='test only'), \
             patch('cyclic_learning_v1.experiment.run_steps',side_effect=[None,RuntimeError('later failure')]), \
             patch('cyclic_learning_v1.experiment.score') as scoring:
            with self.assertRaisesRegex(RuntimeError,'later failure'):
                report(expected_request=frozen,on_cell=records.append)
            scoring.assert_not_called()
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['phase'],'training_completed')
        self.assertEqual(records[0]['steps'],32)
