"""Frozen questions and synthetic scoring conformance, never target performance."""
import copy
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch

from frozen_music_math_v1 import protocol as p


def synthetic_records(answer='6'):
    """Fabricated outputs for scorer tests; these do not come from a model."""
    gold = p.gold_answers()
    return [{'subset': subset, 'seed': seed, 'step': step,
             'question_id': q['question_id'],
             'generated_tokens': [*str(gold[q['question_id']] if answer is None else answer).encode(), p.EOS]}
            for subset, seed, step in p.CHECKPOINTS for q in p.input_questions()]


class ProtocolContracts(unittest.TestCase):
    def test_input_only_inventory_and_budget(self):
        with patch.object(p, 'gold_answers', side_effect=AssertionError('gold input leak')):
            questions = p.input_questions()
        self.assertEqual(len(questions), 32)
        self.assertEqual(len({q['question_id'] for q in questions}), 32)
        self.assertEqual(len({q['prompt'] for q in questions}), 32)
        self.assertEqual(sorted(Counter(q['stratum'] for q in questions).values()), [8]*4)
        for q in questions:
            self.assertEqual(set(q), {'question_id', 'stratum', 'prompt'})
            self.assertRegex(q['prompt'], r'^\d+\+\d+=$')
            self.assertLessEqual(1 + len(q['prompt'].encode()) + p.CAP, p.CONTEXT)
        self.assertEqual(len(p.CHECKPOINTS) * len(questions) * p.CAP, 4096)
        self.assertEqual(p.validate_questions(questions), questions)
        questions[0]['answer'] = '3'
        with self.assertRaises(ValueError):
            p.validate_questions(questions)

    def test_gold_is_independent_integer_addition(self):
        gold = p.gold_answers()
        for q in p.input_questions():
            a, b = map(int, q['prompt'][:-1].split('+'))
            self.assertEqual(gold[q['question_id']], str(a+b))
        self.assertTrue(any(len(answer) == 3 for answer in gold.values()))
        self.assertEqual(Counter(gold.values())['6'], 2)
        self.assertEqual(Counter(gold.values())['14'], 1)

    def test_synthetic_perfect_scores_label_harness_only(self):
        summary = p.score_records(synthetic_records(None))
        self.assertEqual(summary['decodes'], 256)
        self.assertLessEqual(summary['forward_calls'], 4096)
        self.assertTrue(all(c['correct'] == 32 for c in summary['cells'].values()))
        self.assertTrue(all(c['malformed'] == 0 for c in summary['cells'].values()))
        self.assertTrue(all(c['correct_difference'] == 0 for c in summary['paired'].values()))
        for name in ('fresh_confirmation', 'learning_adequate', 'phase5_ready', 'release_ready'):
            self.assertIs(summary[name], False)

    def test_constants_do_not_look_like_generalization(self):
        for answer, correct in [('6', 2), ('14', 1), ('5/4', 0), ('7/4', 0)]:
            with self.subTest(answer=answer):
                summary = p.score_records(synthetic_records(answer))
                for cell in summary['cells'].values():
                    self.assertEqual(cell['correct'], correct)
                    self.assertEqual(cell['unique_raw_outputs'], 1)
                    self.assertEqual(cell['answers_other_than_6_or_14']['correct'], 0)
                self.assertEqual(summary['fixed_answer_baselines'][answer]['correct'], correct)
                self.assertEqual(summary['max_constant_baseline']['correct'], 2)

    def test_exact_eos_no_repair(self):
        rows = synthetic_records(None)
        rows[0]['generated_tokens'] = [ord('/')] * 16
        rows[1]['generated_tokens'] = [ord('0'), ord('4'), p.EOS]
        rows[2]['generated_tokens'] = [256, p.EOS]
        rows[3]['generated_tokens'] = [p.EOS]
        cell = p.score_records(rows)['cells']['one-17-000']
        self.assertEqual(cell['correct'], 28)
        self.assertEqual(cell['cap_terminated'], 1)
        self.assertEqual(cell['malformed'], 4)
        for bad in ([ord('3')], [], [p.EOS, ord('3')], [True, p.EOS], [259, p.EOS], [ord('1')]*17):
            rows = synthetic_records(None)
            rows[0]['generated_tokens'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                p.score_records(rows)

    def test_complete_unique_ordered_inventory_required(self):
        rows = synthetic_records()
        bads = [rows[:-1], rows + [rows[0]], list(reversed(rows))]
        duplicate = copy.deepcopy(rows); duplicate[1] = duplicate[0]; bads.append(duplicate)
        for field, value in [('seed', True), ('step', False), ('subset', 1), ('question_id', 0)]:
            changed = copy.deepcopy(rows); changed[0][field] = value; bads.append(changed)
        extra = copy.deepcopy(rows); extra[0]['answer'] = '6'; bads.append(extra)
        for bad in bads:
            with self.assertRaises(ValueError):
                p.score_records(bad)

    def test_paired_comparison(self):
        rows = synthetic_records('6')
        gold = p.gold_answers()
        for row in rows:
            if row['subset'] == 'one' and row['seed'] == 17 and row['step'] == 64:
                row['generated_tokens'] = [*gold[row['question_id']].encode(), p.EOS]
        pair = p.score_records(rows)['paired']['one-17']
        self.assertEqual(pair, {'initial_correct': 2, 'final_correct': 32,
            'correct_difference': 30, 'newly_correct': 30, 'lost_correct': 0,
            'changed_raw_outputs': 30})

    def test_overlap_discloses_known_history_not_freshness(self):
        root = Path(__file__).resolve().parents[2]
        result = p.historical_overlap(root)
        self.assertFalse(result['global_freshness_established'])
        self.assertEqual(len(result['questions']), 32)
        self.assertEqual(sum(q['known_balanced_addition_family'] for q in result['questions']), 17)
        self.assertEqual(result['excluded_questions'], [])
        self.assertIn('excludes archives', result['scope'])
        self.assertTrue(result['fixture_sources'])


if __name__ == '__main__':
    unittest.main()
