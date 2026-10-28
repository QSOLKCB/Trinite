"""Synthetic conformance, never target model observations."""
import copy
from collections import Counter
import unittest
from trinite.contracts import ContractError, identity, json_bytes
from music_learning_v1.codec import encode_example, parse_prompt
from music_contrast_v1 import spec, reporting


def fake_predictions(selector, answer=None):
    result=[]
    for row in spec.rows(selector,selector=='heldout'):
        gold=[*row['answer'].encode(),257]
        tokens=gold if answer is None else [*answer.encode(),257]
        result.append({'example_identity':row['example_identity'],'family_id':row['family_id'],
            'task':'attacks','split':row['split'],'expected_tokens':gold,
            'generated_tokens':tokens,'correct':tokens==gold,
            'baseline_tokens':[55,257],'baseline_correct':gold==[55,257]})
    return {'predictions':result}


class CorpusContract(unittest.TestCase):
    def test_deterministic_admission_and_exact_membership(self):
        a,b=spec.dataset();self.assertEqual((a,b),spec.dataset())
        self.assertEqual((a,b),spec.require_admission(identity(b)))
        with self.assertRaises(ContractError):spec.require_admission(identity(a))
        self.assertEqual(len(spec.rows()),40);self.assertEqual(len(spec.rows('heldout')),40)
        self.assertEqual(len(spec.rows('heldout',True)),120)

    def test_no_family_or_prompt_overlap(self):
        train=spec.rows();test=spec.rows('heldout',True)
        self.assertFalse({r['family_id'] for r in train}&{r['family_id'] for r in test})
        self.assertFalse({r['prompt'] for r in train}&{r['prompt'] for r in test})
        # Removing ALL event pitch and metadata still leaves disjoint rhythm structure.
        skeleton=lambda r:tuple(tuple(tuple(e['duration']) for e in bar) for bar in r['formal']['bars'])
        self.assertFalse({skeleton(r) for r in train}&{skeleton(r) for r in test})
        train_partitions={tuple(sorted(p[2])) for p in spec.PARENTS if p[1]=='train'}
        heldout_partitions={tuple(sorted(p[2])) for p in spec.PARENTS if p[1]=='heldout'}
        self.assertFalse(train_partitions & heldout_partitions)

    def test_meters_and_quartet_parity(self):
        for selector in ('train','heldout'):
            values=spec.rows(selector)
            self.assertEqual({r['meter'] for r in values},{'3/4','4/4','5/8','7/8','8/16'})
            for group in {r['group_id'] for r in values}:
                q=[r for r in values if r['group_id']==group]
                self.assertEqual(Counter(r['answer'] for r in q),Counter({'6':1,'7':2,'8':1}))
                self.assertEqual(len({r['answer_start'] for r in q}),1)
                self.assertEqual(len({r['prompt'].split(';')[0:4].__repr__() for r in q}),1)
                self.assertEqual(len({len(r['prompt']) for r in q}),1)

    def test_masks_and_invariance(self):
        for row in spec.rows('heldout',True)+spec.rows():
            item=encode_example({k:row[k] for k in ('task','prompt','answer','text','answer_start')})
            self.assertEqual(sum(item.loss_mask),2)
            self.assertEqual(item.loss_mask[-2:],(True,True))
            self.assertEqual(parse_prompt(row['prompt'])['score'],row['formal'])
            self.assertLessEqual(len(row['prompt'])+1+64,256)

    def test_schedule_fixed_exposure(self):
        receipt=spec.schedule_receipt()
        self.assertEqual(receipt['scheduled_visits'],512)
        self.assertEqual(receipt['target_tokens'],1024)
        self.assertEqual(receipt['live_input_tokens'],34812)
        self.assertEqual(Counter(receipt['visits_per_example'].values()),Counter({13:32,12:8}))
        self.assertEqual(receipt['answer_histogram'],{'8':128,'7':256,'6':128})

    def test_changed_rows_rejected(self):
        changed=copy.deepcopy(spec.rows());changed[0]['answer']='9'
        with self.assertRaises(ContractError):spec.validate_rows(changed)
        with self.assertRaises(ContractError):spec.validate_rows(spec.rows('heldout'))
        with self.assertRaises(ContractError):spec.rows('train',True)


class ReportingContract(unittest.TestCase):
    def test_synthetic_oracle_positive(self):
        report=reporting.summarize(json_bytes(fake_predictions('heldout')),'heldout')
        self.assertEqual(report['all']['exact_plus_eos'],120)
        self.assertEqual(report['all']['complete_groups'],30)
        self.assertEqual(report['all']['both_correct_edges'],120)
        self.assertEqual(report['invariance']['all_three_correct'],40)
        self.assertEqual(report['invariance']['identical_raw_outputs'],40)

    def test_constant_is_not_contrast_success(self):
        report=reporting.summarize(json_bytes(fake_predictions('heldout','7')),'heldout')
        self.assertEqual(report['all']['exact_plus_eos'],60)
        self.assertEqual(report['all']['complete_groups'],0)
        self.assertEqual(report['all']['both_correct_edges'],0)
        self.assertEqual(report['all']['changed_output_edges'],0)
        self.assertEqual(report['invariance']['identical_raw_outputs'],40)
        self.assertEqual(report['invariance']['all_three_correct'],20)
        for key in ('global','task_meter','header','length','header_length'):
            self.assertEqual(report['baselines'][key]['complete_groups'],0)
            self.assertEqual(report['baselines'][key]['exact_plus_eos'],60)

    def test_forged_or_partial_predictions_rejected(self):
        cases=[]
        v=fake_predictions('train');v['predictions'].pop();cases.append(v)
        v=fake_predictions('train');v['predictions'][0]['generated_tokens']=[257,257];cases.append(v)
        v=fake_predictions('train');v['predictions'][0]['generated_tokens']=[56];cases.append(v)
        v=fake_predictions('train');v['predictions'][0]['correct']=False;cases.append(v)
        for value in cases:
            with self.assertRaises(ContractError):reporting.summarize(json_bytes(value),'train')

    def test_wrong_changed_answers_not_rewarded(self):
        value=fake_predictions('train')
        for i,p in enumerate(value['predictions']):
            p['generated_tokens']=[48+i%4,257];p['correct']=False
        report=reporting.summarize(json_bytes(value),'train')
        self.assertEqual(report['all']['both_correct_edges'],0)
        self.assertEqual(report['all']['complete_groups'],0)
        self.assertEqual(report['all']['changed_output_edges'],40)

if __name__=='__main__':unittest.main()
