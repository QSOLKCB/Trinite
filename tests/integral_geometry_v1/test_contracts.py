import copy
from fractions import Fraction as Q
import unittest
from trinite.contracts import ContractError, identity, json_bytes
from integral_geometry_v1.data import examples, validate, admission, independent_value, polynomial
from integral_geometry_v1.math import (integral, parse_setup, setup_correct, answer_correct,
                                      evaluate_setup, render)
from integral_geometry_v1.spec import PROTOCOL, targets, schedule

class ExactGeometry(unittest.TestCase):
    def test_complete_admission_and_split(self):
        rows=examples()
        self.assertTrue(validate(rows))
        self.assertEqual(len(rows),90)
        self.assertEqual({s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
                         {'train':54,'validation':12,'test':24})
        self.assertEqual(admission()['dataset_identity'],admission()['dataset_identity'])
        families={}
        for r in rows:
            families.setdefault(r['family'],set()).add(r['split'])
            self.assertFalse(r['split']=='train' and r['novel_transform'])
        self.assertTrue(all(len(v)==1 for v in families.values()))
        for field in ('answer','setup','split','family','control_prompt','identity',
                      'generator_revision','generation_seed','split_procedure','verifier_outcome'):
            changed=copy.deepcopy(rows)
            changed[0][field]='forged'
            with self.assertRaises(ContractError):validate(changed)

    def test_each_item_binds_provenance_and_preserves_semantic_order(self):
        from pathlib import Path
        from integral_geometry_v1 import data
        added={'ordering_identity','generator_revision','generation_seed','generation_procedure',
               'split_seed','split_procedure','verifier_outcome','identity'}
        for row in examples():
            self.assertEqual(row['ordering_identity'],identity(json_bytes({k:v for k,v in row.items() if k not in added})))
            self.assertEqual(row['identity'],identity(json_bytes({k:v for k,v in row.items() if k!='identity'})))
            self.assertIsNone(row['generation_seed'])
            self.assertIsNone(row['split_seed'])
            self.assertIn('no RNG',row['generation_procedure'])
            self.assertEqual(set(row['verifier_outcome'].values()),{'pass'})
            for name,digest in row['generator_revision'].items():
                self.assertEqual(digest,identity(Path(data.__file__).with_name(name).read_bytes()))
        self.assertEqual([targets(seed) for seed in PROTOCOL['seeds']],[1320,1333])

    def test_independent_reference_cases(self):
        # Uniform density on [-1/2,1] has mass3/4 and centroid1/4.
        self.assertEqual(independent_value('area',0,0,Q(-1,2),Q(3,2),Q(1,2)),Q(3,4))
        self.assertEqual(independent_value('centroid_x',0,0,Q(-1,2),Q(3,2),Q(1,2)),Q(1,4))
        # f=1+x+x² on [0,1]: mass11/6, moment13/12, centroid13/22.
        self.assertEqual(evaluate_setup('I(0;1;1,1,1)'),Q(11,6))
        self.assertEqual(evaluate_setup('I(0;1;0,1,1,1)/I(0;1;1,1,1)'),Q(13,22))
        for b in range(3):
            for c in range(3):
                base=independent_value('centroid_x',b,c,Q(0),Q(1),Q(1))
                transformed=independent_value('centroid_x',b,c,Q(-2),Q(3,2),Q(7))
                self.assertEqual(transformed,-2+Q(3,2)*base)
                p=polynomial(b,c,Q(-2),Q(3,2),Q(7))
                self.assertEqual(integral((Q(0),*p),Q(-2),Q(-1,2))/integral(p,Q(-2),Q(-1,2)),transformed)

    def test_semantics_and_rejections(self):
        self.assertTrue(setup_correct('I(0/2;2/2;2/2,0)','I(0;1;1)'))
        self.assertFalse(setup_correct('I(0;1;1)','I(0;2;1/2)'))
        self.assertTrue(answer_correct('2/4','1/2'))
        for s in ('',None,'nan','I(1;0;1)','I(0;1;1,2,3,4,5)',
                  'I(0;1;1)/I(0;2;1)','I(0;1;1);import os','I(0;1;1/0)'):
            self.assertFalse(setup_correct(s,'I(0;1;1)'))
        with self.assertRaises(ContractError):evaluate_setup('I(0;1;1)/I(0;1;0)')
        with self.assertRaises(ContractError):parse_setup('I(0;1;'+'1'*100+')')

    def test_schedule_is_train_only_bounded(self):
        for seed in PROTOCOL['seeds']:
            self.assertTrue(all(r['split']=='train' for r,k in schedule(seed)))
            self.assertLessEqual(targets(seed),PROTOCOL['max_target_tokens'])
            self.assertEqual(len(schedule(seed)),108)

if __name__=='__main__':unittest.main()
