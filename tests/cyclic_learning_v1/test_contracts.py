import copy
import unittest
from cyclic_learning_v1.data import examples,validate,orbit,counter,parse_prompt,admission,validate_admission,ARMS
from cyclic_learning_v1.spec import schedule,targets,request
from trinite.contracts import ContractError

class Contracts(unittest.TestCase):
    def test_exact_admitted_items_and_counts(self):
        rows=examples();validate(rows)
        self.assertEqual(len(rows),288)
        self.assertEqual([sum(r['split']==s for r in rows) for s in ('train','validation','test')],[192,48,48])
        self.assertEqual(admission()['accepted_families'],24)
        validate_admission(admission())
        changed=admission();changed['content_identity']='wrong'
        with self.assertRaises(ContractError):validate_admission(changed)
        for row in rows:
            self.assertEqual(int(row['answer']),counter(row['a'],row['b']))
            for field in ('generator_revision','seed','split_procedure','formal_spec','verifier','lineage','rights_basis'):
                self.assertIn(field,row)
            for arm in ARMS:self.assertEqual(parse_prompt(row['prompts'][arm]),(row['a'],row['b']))
        bad=copy.deepcopy(rows);bad[0]['verifier']['outcome']='unchecked'
        with self.assertRaises(ContractError):validate(bad)

    def test_orbit_closure_and_split_isolation(self):
        families={}
        for row in examples():
            key=tuple(orbit(row['a'],row['b'])[0])
            families.setdefault(key,set()).add(row['split'])
        self.assertEqual(len(families),24)
        self.assertTrue(all(len(s)==1 for s in families.values()))
        for a,b in ((7,11),(0,0),(180,180)):
            original=orbit(a,b)
            for x,y,z in original:self.assertEqual(orbit(x,y),original)

    def test_schedule_and_frozen_request(self):
        for seed in (0,1):
            rows=schedule(seed)
            self.assertEqual(len(rows),192)
            self.assertEqual(len({r['identity'] for r in rows}),192)
            self.assertTrue(all(r['split']=='train' for r in rows))
            self.assertEqual(targets(seed),512)
        frozen=request({'test':'not numerical evidence'})
        self.assertEqual(frozen['items'],examples())
        self.assertEqual(len(frozen['schedules']['0']['scheduled_visits']),128)
        self.assertIn('scripts/cyclic_conformance_v1/reference.py',frozen['source'])

    def test_parser_rejects_noncanonical_and_out_of_range(self):
        for text in ('r60:rot360[00:60,00:01]=','r10:rot360[00360,00001]=',
                     'r10:rot360[00001,00002]=junk','r60:rot360[00001,00002]='):
            with self.assertRaises(ContractError):parse_prompt(text)
