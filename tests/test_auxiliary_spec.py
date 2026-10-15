"""Candidate gate arithmetic and fail-closed profile, without model calls."""
import copy
import unittest

from trinite.contracts import ContractError
from trinite.procedural_qec_auxiliary.spec import (
    ARMS, assess_training_counts, cell_name, cells, diagnostic_rows,
    metadata_bytes, metadata_limit, profile_settings, report_assessment, schema, settings, specification)


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

    def scores(self):
        result = {}
        for mode in ('final', 'prefix', 'trace'):
            tasks = {task: {'examples': total, 'correct': threshold}
                     for task, (threshold, total) in specification()['gates'].items()
                     if mode == 'final' or task in ('fold-sum', 'gf3')}
            result[mode] = {'tasks': tasks, 'examples': sum(r['examples'] for r in tasks.values()),
                            'correct': sum(r['correct'] for r in tasks.values())}
        return result

    def test_dormant_report_all_modes_exact_denominators_and_no_milestone_claim(self):
        scores = self.scores()
        result = report_assessment('research', scores)
        self.assertTrue(result['training_gate_evaluated'])
        self.assertTrue(result['training_gate_pass'])
        for key in ('held_out_scored', 'learning_adequacy', 'phase5_ready',
                    'format_selection_eligible', 'release_ready'):
            self.assertIs(result[key], False)
        scores['final']['tasks']['fold-sum']['correct'] -= 1
        scores['final']['correct'] -= 1
        self.assertEqual(report_assessment('research', scores)['training_task_failures'], ['fold-sum'])
        self.assertFalse(report_assessment('engineering', {})['training_gate_evaluated'])
        for mutate in (
            lambda x: x.pop('trace'),
            lambda x: x['prefix']['tasks']['gf3'].update(examples=108),
            lambda x: x['trace'].update(examples=2408),
            lambda x: x['final'].update(correct=0),
            lambda x: x['final']['tasks']['add'].update(correct=True),
        ):
            changed = self.scores()
            mutate(changed)
            with self.assertRaises(ContractError):
                report_assessment('research', changed)

    def test_candidate_plan_data_and_distinct_schemas_do_not_enable_execution(self):
        self.assertEqual(profile_settings('research')['steps'], 8192)
        self.assertEqual(profile_settings('research')['validation_every'], 2048)
        for kind in ('request', 'checkpoint', 'report', 'verification-request', 'summary'):
            self.assertIn('-research-', schema(kind, 'research'))
            self.assertNotIn('-research-', schema(kind, 'engineering'))
        with self.assertRaises(ContractError):
            settings('research')

    def test_full_candidate_history_fits_explicit_bound_without_relaxing_engineering(self):
        # Synthetic serialization only: every native update retains eight row IDs.
        history = [{'step': i + 1, 'target_tokens': 300, 'examples': ['sha256:' + 'a' * 64] * 8,
                    'loss_hex': '0x1.0000000000000p+0',
                    'gradient_norm_hex': '0x1.0000000000000p+0'} for i in range(8192)]
        record = {'history': history, 'context_padding': 'x' * (128 * 1024)}
        raw = metadata_bytes(record, 'research')
        self.assertGreater(len(raw), metadata_limit('engineering'))
        self.assertLess(len(raw), metadata_limit('research'))
        with self.assertRaises(ContractError):
            metadata_bytes(record, 'engineering')
        with self.assertRaises(ContractError):
            metadata_bytes({'extra': 'x' * metadata_limit('research')}, 'research')

    def test_unique_diagnostic_selection_excludes_held_out_and_rejects_repeated_rows(self):
        rows = []
        for mode in ('final', 'prefix', 'trace'):
            for task, (_, total) in specification()['gates'].items():
                if mode != 'final' and task not in ('fold-sum', 'gf3'):
                    continue
                for number in range(total):
                    rows.append({'task': task, 'mode': mode, 'split': 'train',
                        'example_identity': mode + task + str(number),
                        'semantic_identity': str(number).zfill(4)})
        rows += [{**rows[0], 'split': split, 'example_identity': split} for split in ('validation', 'test')]
        for mode, expected in (('final', 1468), ('prefix', 1204), ('trace', 1204)):
            result = diagnostic_rows(rows, 'research', mode)
            self.assertEqual(len(result), expected)
            self.assertEqual(len({r['example_identity'] for r in result}), expected)
            self.assertTrue(all(r['split'] == 'train' for r in result))
        self.assertEqual(len(diagnostic_rows(rows, 'engineering', 'final')), 6)
        with self.assertRaises(ContractError):
            diagnostic_rows(rows + [rows[0]], 'research', 'final')
        with self.assertRaises(ContractError):
            diagnostic_rows(rows[1:], 'research', 'final')


if __name__ == '__main__':
    unittest.main()
