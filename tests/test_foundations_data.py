"""Finite independent labels, complete admission, parser and family isolation."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import unittest

from trinite.contracts import ContractError, json_bytes, parse_json
from trinite.foundations_data import dataset, audit_bytes, items, validate_admission
from trinite.foundations_oracle import solve, parse_prompt, address
from trinite.foundations_plan import FoundationsPlan

ROOT=Path(__file__).resolve().parents[1]


class FoundationsDataTests(unittest.TestCase):
    def test_retained_fixtures_full_oracle_replay(self):
        for kind,count in (('arithmetic',392),('fold',1550),('lattice',270)):
            raw,manifest=dataset(kind);report=audit_bytes(raw,manifest)
            root=ROOT/'fixtures/foundations-data-v1'/kind
            self.assertEqual((root/'dataset.json').read_bytes(),raw)
            self.assertEqual((root/'manifest.json').read_bytes(),manifest)
            self.assertEqual(sum(report['counts'].values()),count)
            for e in parse_json(raw)['examples']:
                self.assertEqual(parse_prompt(kind,e['prompt']),e['formal'])
                self.assertEqual(solve(kind,e['formal']),e['answer'])

    def test_families_cover_sign_permutation_carrier_and_corruption_variants(self):
        for kind in ('arithmetic','fold','lattice'):
            data=parse_json(dataset(kind)[0]);families=defaultdict(set);items_by_id=defaultdict(list)
            for e in data['examples']:
                families[e['family_id']].add(e['split']);items_by_id[e['semantic_identity']].append(e)
            self.assertTrue(all(len(v)==1 for v in families.values()))
            self.assertEqual({e['split'] for e in data['examples']},{'train','validation','test'})
            self.assertTrue(all(len(v)==2 and {e['carrier'] for e in v}=={'infix','fields'} for v in items_by_id.values()))
            def family(spec):
                return next(e['family_id'] for e in data['examples'] if e['formal']==spec)
            if kind=='arithmetic':
                self.assertEqual(family({'task':'add','a':-3,'b':1}),family({'task':'multiply','a':1,'b':3}))
            if kind=='fold':
                self.assertEqual(family({'task':'fold','values':[-2,1]}),family({'task':'fold','values':[0,-1,2,0]}))

    def test_independent_oracle_boundaries_and_known_answers(self):
        for a,b,answer in ((-2,4,'outside'),(0,0,'ERR'),(-3,2,'-3/2'),(2,-2,'-1')):
            spec={'task':'divide','a':a,'b':b}
            if answer=='outside':
                with self.assertRaises(ContractError):solve('arithmetic',spec)
            else:self.assertEqual(solve('arithmetic',spec),answer)
        self.assertEqual(solve('fold',{'task':'fold','values':[-2,0,1]}),'3 -1 5')
        self.assertEqual(solve('lattice',{'task':'traversal','index':1}),'L[1,2,2]')
        for text in ('L[3,0,0]','L[０,0,0]','L[0,0,0]\n','../L[0,0,0]'):
            with self.assertRaises(ContractError):address(text)
        for kind,prompt in (('fold','F:1,2,3,4,5='),('arithmetic','A:True,1='),
                            ('lattice','T:27='),('lattice','R:XXX='),('arithmetic','A:1,2=\n')):
            with self.assertRaises(ContractError):parse_prompt(kind,prompt)

    def test_admission_standalone_rejects_contradictory_fields_and_evidence(self):
        raw,manifest=dataset('arithmetic');record=parse_json(manifest)['admission']
        validate_admission(record,'arithmetic')
        for field,value in (('outcome','rejected'),('author',''),('reviewed_on','not-a-date'),
                            ('rights_basis','copied prose'),('scope','all internet text'),('supporting_reference',{})):
            changed={**record,field:value}
            with self.assertRaises(ContractError):validate_admission(changed,'arithmetic')
            mutated=parse_json(manifest);mutated['admission']=changed
            with self.assertRaises(ContractError):audit_bytes(raw,json_bytes(mutated))
        for field in ('answer','prompt','family_id','split','formal'):
            changed=parse_json(raw);changed['examples'][0][field]='altered'
            with self.assertRaises(ContractError):audit_bytes(json_bytes(changed),manifest)

    def test_separate_plan_limits_do_not_relax_native_limits(self):
        plan=FoundationsPlan()
        for fields in ({'steps':1025},{'batch_size':9},{'max_target_tokens':True},
                       {'learning_rate':'nan'},{'schema':'trinite.training-plan.v1'}):
            with self.assertRaises(ContractError):replace(plan,**fields)
        self.assertEqual(FoundationsPlan.from_dict(plan.to_dict()),plan)


if __name__=='__main__':unittest.main()
