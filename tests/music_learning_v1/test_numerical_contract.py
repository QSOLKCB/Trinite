"""Frozen plan/adapter conformance only: no native model construction or updates."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

from trinite.contracts import ContractError
from music_learning_v1.numerical import (SEEDS, examples, plan_for, train_rows,
                                        schedule_receipt, encoded, view)


class NumericalContractTests(unittest.TestCase):
    def test_fixed_plan_and_full_train_cycle(self):
        receipts = []
        for seed in SEEDS:
            plan = plan_for(seed)
            self.assertEqual((plan.steps, plan.batch_size, plan.model.context_length), (64, 8, 256))
            self.assertEqual(plan.model.parameter_count, 24768)
            self.assertEqual(plan.model.geometry_loss, '0')
            receipt = schedule_receipt(seed)
            self.assertEqual((receipt['unique_train_examples'], receipt['scheduled_visits'],
                              receipt['unique_scheduled_examples']), (480, 512, 480))
            self.assertLessEqual(receipt['target_tokens'], plan.max_target_tokens)
            receipts.append(receipt)
        self.assertEqual(receipts[0], receipts[1])
        for seed in (True, 0, 18, '17'):
            with self.assertRaises(ContractError):
                plan_for(seed)

    def test_only_train_updates_and_masks(self):
        rows = train_rows()
        self.assertEqual({r['split'] for r in rows}, {'train'})
        for row, value in zip(rows, encoded(rows)):
            self.assertEqual(sum(value['loss']), len(row['answer']) + 1)
            self.assertFalse(any(value['loss'][:len(row['prompt']) + 1]))
        train_families = {r['family_id'] for r in rows}
        for selector in ('validation', 'test'):
            selected, split = view(selector)
            self.assertEqual(split, 'test')
            self.assertFalse(train_families & {r['family_id'] for r in selected if r['split'] == 'test'})
            self.assertEqual(sum(r['split'] == 'test' for r in selected), 240)
        with self.assertRaises(ContractError):
            view('combined')

    def test_stdlib_import_never_loads_models(self):
        code = ('import sys; from music_learning_v1.numerical import examples; examples(); '
                'assert not any(x in sys.modules for x in '
                '["torch", "trinite.model", "trinite.training", "trinite.geometry"])')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_authoritative_primitives_not_reimplemented(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / 'scripts/music_learning_v1/numerical.py').read_text())
        imports = {(node.module, alias.name) for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) for alias in node.names}
        for item in [('trinite.training', 'TrainingState'), ('trinite.training', 'create_optimizer'),
                     ('trinite.training', 'run_steps'), ('trinite.foundations', 'score_examples'),
                     ('trinite.foundations', 'check_prediction_examples'),
                     ('trinite.comparison_checkpoint', 'restore_tensors')]:
            self.assertIn(item, imports)


if __name__ == '__main__':
    unittest.main()
