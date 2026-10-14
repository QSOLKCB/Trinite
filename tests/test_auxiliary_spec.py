"""Candidate gate arithmetic and fail-closed profile, without model calls."""
import copy
import unittest

from trinite.contracts import ContractError
from trinite.procedural_qec_auxiliary.spec import (
    ARMS, assess_training_counts, cell_name, cells, settings, specification)


class AuxiliarySpecTests(unittest.TestCase):
    def counts(self):
        return {cell_name(*cell): {task: {'correct': threshold, 'examples': total}
                for task, (threshold, total) in specification()['gates'].items()}
                for cell in cells()}

    def test_absolute_gate_and_strict_paired_fold_contrast_are_separate(self):
        counts = self.counts()
        result = assess_training_counts(counts)
        self.assertTrue(result['treatment_training_gate_pass'])
        self.assertFalse(result['positive_fold_contrast_both_seeds'])
        counts[cell_name(0, ARMS[1])]['fold-sum']['correct'] += 1
        self.assertFalse(assess_training_counts(counts)['positive_fold_contrast_both_seeds'])
        counts[cell_name(1, ARMS[1])]['fold-sum']['correct'] += 1
        self.assertTrue(assess_training_counts(counts)['positive_fold_contrast_both_seeds'])
        counts[cell_name(0, ARMS[1])]['divide']['correct'] -= 1
        result = assess_training_counts(counts)
        self.assertFalse(result['treatment_training_gate_pass'])
        self.assertTrue(result['positive_fold_contrast_both_seeds'])
        self.assertFalse(result['evidence_verified'])
        self.assertFalse(result['learning_adequacy'])

    def test_missing_cells_extra_tasks_repeated_denominators_and_invalid_counts_reject(self):
        for mutate in (
            lambda c: c.pop(cell_name(0, ARMS[0])),
            lambda c: c[cell_name(0, ARMS[0])].pop('gf3'),
            lambda c: c[cell_name(0, ARMS[0])].update(extra={'correct': 0, 'examples': 1}),
            lambda c: c[cell_name(0, ARMS[0])]['add'].update(examples=132),
            lambda c: c[cell_name(0, ARMS[0])]['add'].update(correct=True),
            lambda c: c[cell_name(0, ARMS[0])]['add'].update(correct=67),
        ):
            counts = self.counts()
            mutate(counts)
            with self.assertRaises(ContractError):
                assess_training_counts(counts)

    def test_research_is_unfrozen_and_disabled(self):
        self.assertFalse(specification()['research_execution_enabled'])
        self.assertEqual(settings('engineering')['steps'], 16)
        for value in ('research', 'smoke', {}, None, True):
            with self.assertRaises(ContractError):
                settings(value)


if __name__ == '__main__':
    unittest.main()
