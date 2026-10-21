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

    def test_untrained_baseline_preserves_trained_state_and_rng(self):
        from cyclic_learning_v1.experiment import untrained_score
        from trinite.inspection import tensor_identity
        state=create(0,'decimal')
        run_steps(state,stop_after=1)
        before=model_identity(state.model)
        history=list(state.history)
        optimizer={name:{k:tensor_identity(v) for k,v in state.optimizer.state[p].items()}
                   for name,p in state.model.named_parameters()}
        rng=torch.get_rng_state().clone()
        # Two training-pool rows only: bounded scorer conformance, not the target run.
        rows=[r for r in examples() if r['split']=='train'][:2]
        with patch('cyclic_learning_v1.experiment.examples',return_value=rows):
            for arm in ('decimal','sexagesimal'):
                baseline=untrained_score(state,arm)
                self.assertEqual(baseline['model_identity'],state.initialization_identity)
                self.assertEqual([r['identity'] for r in baseline['scores']['records']],
                                 [r['identity'] for r in rows])
        self.assertEqual(model_identity(state.model),before)
        self.assertEqual(state.history,history)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(optimizer,{name:{k:tensor_identity(v) for k,v in state.optimizer.state[p].items()}
                                   for name,p in state.model.named_parameters()})
        state.initialization_identity='invalid'
        with patch('cyclic_learning_v1.experiment.score') as scoring:
            with self.assertRaisesRegex(ContractError,'recreated initial'):
                untrained_score(state,'decimal')
            scoring.assert_not_called()

    def test_baselines_scored_per_arm_only_after_all_training(self):
        completed=[];calls=[]
        def synthetic_update(state):
            # Sequencing fixture, not executed training/evidence.
            completed.append(state)
            state.step=32;state.target_tokens=512;state.cursor=128
        def synthetic_score(model,arm):
            self.assertEqual(len(completed),4)
            calls.append(arm)
            return {'records':[{'identity':'synthetic-row'}],
                    'groups':{s:{'rows':1,'exact':0,'canonical':0}
                              for s in ('train','validation','test')}}
        with patch('cyclic_learning_v1.experiment.run_steps',side_effect=synthetic_update), \
             patch('cyclic_learning_v1.experiment.score',side_effect=synthetic_score):
            result=report(expected_request=request(require_environment()))
        self.assertEqual(result['schema'],'trinite.cyclic-learning-report.v2')
        self.assertEqual(calls,['decimal','decimal','sexagesimal','sexagesimal']*2)
        for cell in result['cells']:
            self.assertEqual(cell['untrained']['model_identity'],cell['initial_model_identity'])
            self.assertEqual(cell['final_minus_untrained_counts']['test'],{'exact':0,'canonical':0})
