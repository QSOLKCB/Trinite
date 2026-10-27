"""Software-double wiring controls with real regenerated admission.

These do not execute a Decoder, optimizer, tensor loader, or model inference and
must not be reported as numerical/checkpoint compatibility or learning evidence.
"""
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity
from music_learning_v1 import numerical as n
from music_learning_v1.admission import dataset, audit_bytes


def module(name, **items):
    result = ModuleType(name)
    result.__dict__.update(items)
    return result


class RuntimeDoubleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.receipt = dataset()
        cls.admission_id = identity(cls.receipt)

    def gate(self, supplied):
        audit_bytes(self.raw, self.receipt)
        if supplied != self.admission_id:
            raise ContractError('test gate rejects foreign admission')

    @contextmanager
    def doubles(self):
        events = []
        def model(config, *, lane, seed):
            events.append(('model', lane, seed))
            return SimpleNamespace(config=config, lane=lane)
        def state(config, lane, model, optimizer, data, did, aid, iid, loss):
            return SimpleNamespace(config=config, lane=lane, model=model, optimizer=optimizer,
                data=data, dataset_identity=did, manifest_identity=aid,
                initialization_identity=iid, initial_train_loss=loss, step=0,
                cursor=0, target_tokens=0, history=[], validation=[])
        def optimizer(model, config):
            events.append(('optimizer', model, config))
            return 'optimizer-double'
        def evaluate(model, rows, batch):
            events.append(('evaluate', tuple(r['identity'] for r in rows), batch))
            return '0x0.0p+0'
        def update(state):
            events.append(('update', state))
            state.step = state.config.steps
            state.target_tokens = 123
        modules = {
            'music_learning_v1.workflow': module('music_learning_v1.workflow', require_native_admission=self.gate),
            'trinite.model': module('trinite.model', Decoder=model),
            'trinite.training': module('trinite.training', TrainingState=state, create_optimizer=optimizer,
                evaluate=evaluate, model_identity=lambda model: 'model-double',
                require_environment=lambda: events.append(('environment',)), run_steps=update),
            'trinite.comparison_checkpoint': module('trinite.comparison_checkpoint',
                tensor_payload=lambda state: b'not-real-tensors',
                progress=lambda state: events.append(('progress', state.step)),
                restore_tensors=lambda state, raw, model_id: (events.append(('restore', raw, model_id)) or state)),
        }
        with patch.dict(sys.modules, modules):
            yield events

    def test_state_and_restore_delegation_with_real_admission(self):
        with self.doubles() as events:
            state = n.state_for(17, self.admission_id)
            self.assertEqual(state.dataset_identity, identity(self.raw))
            self.assertEqual(state.manifest_identity, self.admission_id)
            self.assertEqual(state.lane, 'dense')
            self.assertEqual({r['identity'] for r in state.data['train']},
                             {r['example_identity'] for r in n.train_rows()})
            self.assertEqual(state.data['train'], state.data['validation'])
            self.assertEqual(len(state.data['train']), 480)
            self.assertEqual(events[1], ('model', 'dense', 17))
            payload, metadata = n.snapshot(state, 'request-double')
            restored = n.restore(payload, metadata, 17, 'request-double', self.admission_id,
                                 (identity(payload), identity(metadata)))
            self.assertEqual(restored.initialization_identity, 'model-double')
            self.assertIn(('restore', payload, 'model-double'), events)
            with self.assertRaises(ContractError):
                n.restore(payload, metadata, 17, 'foreign-request', self.admission_id,
                          (identity(payload), identity(metadata)))
            with self.assertRaises(ContractError):
                n.restore(payload, metadata, 17, 'request-double', self.admission_id,
                          ('wrong', identity(metadata)))
            before = len(events)
            with self.assertRaises(ContractError):
                n.state_for(17, 'foreign-admission')
            self.assertEqual(len(events), before)

    def test_score_journals_before_next_row_and_preserves_partial_failure(self):
        rows, native_split = n.view('test')
        train = [r for r in rows if r['split'] == 'train']
        selected = [r for r in rows if r['split'] == 'test'][:2]
        calls, retained = [], []
        def scorer(model, inputs, split, cap):
            self.assertEqual(inputs[:-1], train)
            self.assertEqual((split, cap), ('test', 64))
            calls.append(inputs[-1]['example_identity'])
            if len(calls) == 2:
                self.assertEqual(len(retained), 1)
                raise RuntimeError('synthetic scoring interruption')
            item = inputs[-1]
            return {}, [{'example_identity': item['example_identity'], 'split': 'test'}]
        with self.doubles(), patch.dict(sys.modules, {'trinite.foundations': module(
                'trinite.foundations', score_examples=scorer,
                check_prediction_examples=lambda *args: {})}), patch.object(n, 'view', return_value=(train + selected, native_split)):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                n.score(SimpleNamespace(model='model-double'), 'test', retained.append)
        self.assertEqual(retained, [{'example_identity': selected[0]['example_identity'], 'split': 'test'}])

    def test_run_retains_untrained_and_failure_before_later_stages(self):
        written, journal = {}, []
        def write(name, raw):
            self.assertNotIn(name, written)
            written[name] = raw
        def score(state, selector, emit):
            emit({'synthetic': True, 'selector': selector})
            if state.step and selector == 'validation':
                raise RuntimeError('synthetic final scoring failure')
            return {'model_identity': 'model-double'}
        state = SimpleNamespace(config=n.plan_for(17), step=0, target_tokens=0)
        with self.doubles(), patch.object(n, 'state_for', return_value=state), \
             patch.object(n, 'score', side_effect=score), \
             patch.object(n, 'snapshot', return_value=(b'tensor-double', b'{}')), \
             patch('music_learning_v1.reporting.describe', return_value={'synthetic': True}):
            with self.assertRaisesRegex(RuntimeError, 'final scoring failure'):
                n.run_trial(17, self.admission_id, 'request-double', write,
                            lambda stage, split, row: journal.append((stage, split, row)))
        self.assertIn('untrained.safetensors', written)
        self.assertIn('trained-train.json', written)
        self.assertNotIn('trained-test.json', written)
        self.assertIn('failure.json', written)
        self.assertEqual(len(journal), 5)
        self.assertEqual(journal[-1][0:2], ('trained', 'validation'))

    def test_elapsed_update_budget_retains_checkpoint_and_failure(self):
        written = {}
        state = SimpleNamespace(config=n.plan_for(17), step=0, target_tokens=0)
        with self.doubles(), patch.object(n, 'state_for', return_value=state), \
             patch.object(n, 'score', return_value={'model_identity': 'model-double'}), \
             patch.object(n, 'snapshot', return_value=(b'tensor-double', b'{}')), \
             patch('music_learning_v1.reporting.describe', return_value={'synthetic': True}), \
             patch('time.monotonic', side_effect=[0.0, 121.0]):
            with self.assertRaisesRegex(ContractError, 'elapsed budget'):
                n.run_trial(17, self.admission_id, 'request-double',
                    lambda name, raw: written.__setitem__(name, raw), lambda *args: None)
        self.assertIn('update-failure.safetensors', written)
        self.assertIn('update-failure-checkpoint.json', written)
        self.assertIn('failure.json', written)
        self.assertNotIn('trained-train.json', written)


if __name__ == '__main__':
    unittest.main()
