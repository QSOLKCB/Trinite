import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from trinite.contracts import ContractError, identity, json_bytes
from batch_composition_v1 import experiment as e, spec
from batch_composition_v1.comparison import comparisons


def rows():
    return [{'item_id': 'a', 'split': 'train', 'prompt': 'ab', 'prompt_identity': identity(b'ab')},
            {'item_id': 'b', 'split': 'validation', 'prompt': 'longer', 'prompt_identity': identity(b'longer')}]


class FakeModel(nn.Module):
    def __init__(self, contaminated=False):
        super().__init__()
        self.token_embedding = nn.Parameter(torch.zeros(1))
        self.output_weight = self.token_embedding
        self.contaminated = contaminated
        self.eval()

    def forward(self, ids, mask):
        value = torch.zeros((*ids.shape, 259), dtype=torch.float32)
        value[:, :, 257] = 1
        if self.contaminated and len(ids) > 1:
            value[:, :, 3] = 0.1
        value *= mask[:, :, None]
        return type('Output', (), {'logits': value})()


def fake_historical(data):
    result = []
    for seed, arm in spec.CELLS:
        modes = {mode: {'predictions': [{'example_identity': r['item_id'], 'generated_tokens': tokens,
                                        'stop_reason': 'eos'} for r in data]}
                 for mode, tokens in (('greedy', [257]), ('syntax', [48, 257]))}
        result.append({'seed': seed, 'arm': arm, 'modes': modes})
    return {'cells': result}


class ConformanceTests(unittest.TestCase):
    def run_fake_report(self, contaminated=False, on_cell=None, on_layout=None):
        data = rows()
        source = {'cells': [{'seed': seed, 'arm': arm, 'final_checkpoint': {}} for seed, arm in spec.CELLS]}
        with patch.object(e, 'request', return_value={'fixed': True}), patch.object(e, 'workload', return_value=data), \
             patch.object(e, 'references', return_value={cell: [[48,257]]*len(data) for cell in spec.CELLS}), \
             patch.object(e, 'reference_report', return_value=fake_historical(data)), patch.object(e, 'original', return_value=source), \
             patch.object(e, 'load_model', side_effect=lambda _: FakeModel(contaminated)):
            return e.report(on_cell=on_cell, on_layout=on_layout)

    def test_full_controller_uses_all_endpoints_layouts_and_separate_lanes(self):
        partial, layouts = [], []
        report = self.run_fake_report(on_cell=partial.append, on_layout=layouts.append)
        self.assertTrue(report['conforms'])
        self.assertEqual(len(partial), 4)
        self.assertEqual(len(layouts), 32)
        self.assertEqual(report['training_updates'], 0)
        self.assertFalse(report['test_evaluated'])
        for cell in report['cells']:
            self.assertEqual(list(cell['layouts']), list(spec.LAYOUTS))
            self.assertEqual(len(cell['comparisons']), 13)
            self.assertTrue(all(cell['historical_singleton_parity'].values()))
            self.assertLessEqual(cell['budget']['active_decisions'], cell['budget']['row_evaluations'])

    def test_numeric_failure_retains_all_cells_instead_of_aborting_or_selecting(self):
        partial = []
        report = self.run_fake_report(contaminated=True, on_cell=partial.append)
        self.assertFalse(report['conforms'])
        self.assertEqual(len(partial), 4)
        for cell in report['cells']:
            self.assertFalse(cell['numeric_pass'])
            self.assertTrue(cell['decision_pass'])
            self.assertEqual(len(cell['layouts']), 8)

    def test_comparisons_reject_missing_extra_modes_layouts_and_row_corruption(self):
        report = self.run_fake_report()
        original = report['cells'][0]['layouts']
        ids = [r['item_id'] for r in rows()]
        mutations = [lambda x: x.pop('singleton_min'), lambda x: x.update(extra=x['singleton_min']),
                     lambda x: x['singleton_min']['generated'].pop('greedy'),
                     lambda x: x['batch8_32']['forced'].reverse(),
                     lambda x: x['batch8_32']['forced'].append(x['batch8_32']['forced'][0])]
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            with self.assertRaises(ContractError):
                comparisons(changed, ids)

    def test_observer_callback_cannot_change_rng(self):
        state = torch.get_rng_state().clone()
        try:
            with self.assertRaisesRegex(ContractError, 'RNG'):
                self.run_fake_report(on_layout=lambda _: torch.rand(1))
        finally:
            torch.set_rng_state(state)

    def test_protocol_constants_and_source_closure_are_bound_without_target_run(self):
        protocol = spec.protocol()
        self.assertEqual(protocol['baseline_commit'], spec.BASE)
        self.assertEqual([x['id'] for x in protocol['layouts']], list(spec.LAYOUTS))
        self.assertEqual(protocol['budgets']['max_worker_seconds'], spec.MAX_SECONDS)
        self.assertEqual(protocol['budgets']['max_report_bytes'], spec.MAX_REPORT)
        self.assertEqual(protocol['numeric_gate']['atol'], '0.000002')
        source = spec.sources()
        for path in ('scripts/batch_composition_v1/experiment.py', 'scripts/batch_composition_v1/comparison.py',
                     'scripts/batch_composition_v1/runner.py', 'scripts/run_batch_composition.py',
                     'scripts/frozen_decoding_transport_v2/artifacts.py', 'src/trinite/model.py',
                     'tests/batch_composition_review/test_numeric_review.py', '.github/workflows/batch-composition.yml'):
            self.assertEqual(source[path], identity((spec.ROOT/path).read_bytes()))

if __name__ == '__main__':
    unittest.main()
