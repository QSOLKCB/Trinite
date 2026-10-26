"""Synthetic retained outcomes, not predictions from target checkpoints."""
import copy
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, ModelConfig
from trinite.tokenizer import EOS
from trinite.foundations import metrics
from position_study_v1 import numerical, reporting, data
from position_study_v1.adapter import StudyDecoder, addressing


def observations(selector, answer):
    from trinite.foundations import majority_answers
    predictions=[]
    for width,rows,split in numerical.scoring_groups('control',selector):
        baseline=majority_answers(rows)['add']
        for r in rows:
            if r['split']!=split:continue
            expected=[*r['answer'].encode(),EOS];emitted=answer(r)
            base=[*baseline.encode(),EOS]
            predictions.append({'example_identity':r['example_identity'],'family_id':r['family_id'],
                'task':'add','split':split,'expected_tokens':expected,'generated_tokens':emitted,
                'correct':emitted==expected,'baseline_tokens':base,'baseline_correct':base==expected})
    return {'schema':numerical.PROTOCOL['scoring']['schema'],
            'model_identity':'synthetic','addressing':addressing('sequence'),
            'summary':metrics(predictions),'predictions':predictions}


class ReportingTests(unittest.TestCase):
    def test_constant_wrong_answers_never_pass_even_with_perfect_consistency(self):
        for s in data.STRATA:
            record=observations(s,lambda r:[57]*(r['width']+1)+[EOS])
            numerical.check_score(record,'control',s,'synthetic')
            group=reporting.group(reporting.annotate(data.examples(s),record))
            self.assertEqual(group['total']['correct'],0)
            for kind in ('operand_reversal','equal_sum'):
                p=group['perturbations'][kind]
                self.assertGreater(p['pair_count'],0)
                self.assertEqual(p['consistent'],p['pair_count'])
                self.assertEqual(p['consistent_and_correct'],0)
            inc=group['perturbations']['increment']
            self.assertGreater(inc['pair_count'],0);self.assertEqual(inc['consistent'],0)
            self.assertEqual(inc['numeric_relation'],0)

    def test_oracle_is_only_a_non_neural_scoring_control(self):
        for s in ('train',*data.STRATA):
            record=observations(s,lambda r:[*r['answer'].encode(),EOS])
            numerical.check_score(record,'control',s,'synthetic')
            group=reporting.group(reporting.annotate(data.examples(s),record))
            self.assertEqual(group['total']['correct'],len(data.examples(s)))
            self.assertEqual(group['total']['all_renderings_correct_families'],len(data.FAMILIES[s]))
            for p in group['perturbations'].values():
                self.assertEqual(p['both_correct'],p['pair_count'])
                self.assertEqual(p['numeric_relation'],p['pair_count'])

    def test_explicit_new_reversed_padded_eos_classification(self):
        expected=[48,48,49,EOS]  # reversed100
        self.assertEqual(reporting.classify(expected,expected),'exact_correct')
        self.assertEqual(reporting.classify([49,48,49,EOS],expected),'wrong_value')
        self.assertEqual(reporting.classify([EOS],expected),'immediate_eos')
        self.assertEqual(reporting.classify([57,EOS],expected),'premature_eos')
        self.assertEqual(reporting.classify([255,EOS],expected),'malformed_output')
        self.assertEqual(reporting.classify([255,48,49,48],expected),'missing_eos')
        self.assertTrue(reporting.output_flags([255,48,49,48],expected)['malformed'])

    def test_numeric_diagnostic_does_not_rescue_noncanonical_early_eos(self):
        record=observations('unseen',lambda r:[*str(r['formal']['a']+r['formal']['b'])[::-1].encode(),EOS])
        numerical.check_score(record,'control','unseen','synthetic')
        rows=reporting.annotate(data.examples('unseen'),record)
        self.assertTrue(all(r['numeric_correct'] for r in rows))
        self.assertGreater(sum(r['output_flags']['early_eos'] for r in rows),0)
        self.assertLess(sum(r['prediction']['correct'] for r in rows),len(rows))
        self.assertTrue(any(r['fixed_width_decoded'] is None and r['numeric_correct'] for r in rows))

    def test_impossible_termination_unknown_duplicate_and_optimistic_scores_reject(self):
        good=observations('length4',lambda r:[57]*(r['width']+1)+[EOS])
        for out in ([48],[EOS,48],[EOS,EOS],[48]*7):
            bad=copy.deepcopy(good);bad['predictions'][0]['generated_tokens']=out
            with self.assertRaises(ContractError):numerical.check_score(bad,'control','length4','synthetic')
        for bad in (copy.deepcopy(good),copy.deepcopy(good),copy.deepcopy(good)):
            bad['predictions'].append(copy.deepcopy(bad['predictions'][0]))
            with self.assertRaises(ContractError):numerical.check_score(bad,'control','length4','synthetic')
        bad=copy.deepcopy(good);bad['summary']['correct']=23
        with self.assertRaises(ContractError):numerical.check_score(bad,'control','length4','synthetic')
        bad=copy.deepcopy(good);bad['addressing']=addressing('coupled')
        with self.assertRaises(ContractError):numerical.check_score(bad,'control','length4','synthetic')

    def test_real_scorer_uses_only_prompt_and_prompt_width_cap_on_malformed_output(self):
        # Intercept unchanged greedy's call boundary, not a target experiment.
        model=StudyDecoder(ModelConfig(context_length=64,blocks=1,width=8,heads=2,feed_forward_width=16),mode='sequence')
        class State:pass
        state=State();state.model=model
        def generate(m,prompt,cap):
            self.assertIs(m,model);self.assertEqual(cap,data.parse_prompt(prompt)[2]+2)
            return [255]*cap
        with patch('trinite.foundations.greedy',side_effect=generate) as call:
            for s in ('train',*data.STRATA):
                r=numerical.score(state,'control',s)
                numerical.check_score(r,'control',s,numerical.model_identity(model))
                self.assertEqual(r['summary']['correct'],0)
                self.assertTrue(all(len(p['generated_tokens'])>=3 for p in r['predictions']))
            self.assertEqual(call.call_count,256)

    def test_descriptive_criteria_retain_per_seed_negative_and_incomplete(self):
        train=observations('train',lambda r:[*r['answer'].encode(),EOS])
        dev={s:observations(s,lambda r:[*r['answer'].encode(),EOS]) for s in data.STRATA}
        positive=reporting.describe('control',train,dev)
        negative=copy.deepcopy(positive)
        for s in data.STRATA:negative['development'][s]['total']['all_renderings_correct_families']=0
        pairs={f'{a}-{s}':copy.deepcopy(positive if a=='candidate' else negative)
               for a in ('control','candidate') for s in range(3)}
        self.assertTrue(reporting.criteria(pairs)['all_descriptive_criteria_met'])
        pairs['candidate-2']=negative
        result=reporting.criteria(pairs)
        self.assertFalse(result['all_descriptive_criteria_met']);self.assertFalse(result['per_seed']['2']['unseen_gain'])
        del pairs['candidate-2']
        self.assertFalse(reporting.criteria(pairs)['all_descriptive_criteria_met'])


if __name__=='__main__':unittest.main()
