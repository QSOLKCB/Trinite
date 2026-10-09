import itertools
from pathlib import Path
import unittest

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.game_data import dataset, audit_bytes, validate_admission, orbit, specs, label, render
from trinite.game_oracle import solve, parse_prompt

ROOT=Path(__file__).resolve().parents[1]


class GameDataTests(unittest.TestCase):
    def test_retained_corpus_and_admission_regenerate_exactly(self):
        raw,manifest=dataset();root=ROOT/'fixtures/game-data-v1'
        self.assertEqual((root/'dataset.json').read_bytes(),raw)
        self.assertEqual((root/'manifest.json').read_bytes(),manifest)
        checked=audit_bytes(raw,manifest);validate_admission(checked['admission'])
        self.assertFalse(checked['training_integrated']);self.assertFalse(checked['fresh_evaluation'])
        self.assertEqual(sum(checked['counts'].values()),256*11)
        for name,value in (('outcome','rejected'),('rights_basis','copied prose'),('reviewed_on','not-a-date')):
            record={**checked['admission'],name:value}
            with self.assertRaises(ContractError):validate_admission(record)

    def test_exhaustive_labels_prompts_and_family_isolation(self):
        raw,manifest=dataset();rows=parse_json(raw,canonical=True)['examples'];seen={}
        for row in rows:
            self.assertEqual(solve(row['formal']),row['answer'])
            self.assertEqual(parse_prompt(row['prompt']),row['formal'])
            family=orbit(row['formal']['payoffs'])
            if family in seen:self.assertEqual(seen[family],row['split'])
            seen[family]=row['split']
        self.assertEqual(set(seen.values()),{'train','validation','test'})

    def test_every_example_has_bound_generator_seed_and_transformation_lineage(self):
        raw,manifest=dataset();rows=parse_json(raw,canonical=True)['examples']
        checked=parse_json(manifest,canonical=True)
        revision=identity(json_bytes(checked['admission']['supporting_reference']['implementation']))
        families={row['family_id'] for row in rows}
        ranked=sorted(families,key=lambda f:identity(json_bytes({'policy':'trinite.game-data.v1','seed':41,'family':f})))
        train_count=len(ranked)*8//10;validation_count=max(1,len(ranked)//10)
        splits={f:'train' if i<train_count else 'validation' if i<train_count+validation_count else 'test'
                for i,f in enumerate(ranked)}
        for row in rows:
            self.assertEqual(row['generator_revision'],revision)
            self.assertIs(type(row['seed']),int);self.assertEqual(row['seed'],41)
            self.assertEqual(row['split'],splits[row['family_id']])
            self.assertEqual(row['transformation_lineage'],{
                'formal_identity':identity(json_bytes(row['formal'])),
                'carrier':'trinite.game-symbols.v1',
                'family_rule':'action-relabel-player-exchange-v1',
                'split_rule':'sha256-family-rank-80-10-remainder-v1'})
            self.assertEqual(row['example_identity'],identity(json_bytes({
                k:v for k,v in row.items() if k!='example_identity'})))

    def test_missing_or_forged_lineage_rejects_with_refreshed_receipts(self):
        raw,manifest=dataset()
        forged={'generator_revision':'sha256:'+'0'*64,'seed':42,
                'transformation_lineage':None}
        for name,value in forged.items():
            for remove in (False,True):
                with self.subTest(field=name,removed=remove):
                    data=parse_json(raw,canonical=True);row=data['examples'][0]
                    if remove:del row[name]
                    elif name=='transformation_lineage':
                        row[name]={**row[name],'formal_identity':'sha256:'+'0'*64}
                    else:row[name]=value
                    row['example_identity']=identity(json_bytes({k:v for k,v in row.items() if k!='example_identity'}))
                    data['examples'].sort(key=lambda r:r['example_identity'])
                    changed=json_bytes(data);record=parse_json(manifest,canonical=True)
                    record['dataset_identity']=identity(changed)
                    record['admission']['evidence']['dataset_identity']=identity(changed)
                    record['admission']['supporting_reference']['evidence_identity']=identity(json_bytes(record['admission']['evidence']))
                    with self.assertRaises(ContractError):audit_bytes(changed,json_bytes(record))

    def test_ties_strict_dominance_and_no_pure_equilibrium(self):
        # Matching pennies has no pure Nash equilibrium; equal payoffs have four.
        zero={'payoffs':[0]*8,'task':'pure-nash'}
        self.assertEqual(solve(zero),'00,01,10,11')
        pennies={'payoffs':[1,0,0,1,0,1,1,0],'task':'pure-nash'}
        self.assertEqual(solve(pennies),'-')
        self.assertEqual(solve({**zero,'task':'row-best','column':0}),'0,1')
        self.assertEqual(solve({**zero,'task':'strict-dominant','player':0}),'-')
        dominant={'payoffs':[1,0,1,1,0,0,0,1]}
        self.assertEqual(solve({**dominant,'task':'strict-dominant','player':0}),'0')
        self.assertEqual(solve({**dominant,'task':'strict-dominant','player':1}),'1')
        self.assertEqual(solve({**dominant,'task':'pure-nash'}),'01')

    def test_orbit_is_closed_under_action_relabelling_and_player_exchange(self):
        for values in itertools.product((0,1),repeat=8):
            row_swap=values[4:]+values[:4]
            column_swap=values[2:4]+values[:2]+values[6:]+values[4:6]
            player_swap=tuple(values[2*(2*c+r)+p] for r,c in itertools.product((0,1),repeat=2) for p in (1,0))
            self.assertEqual(orbit(values),orbit(row_swap))
            self.assertEqual(orbit(values),orbit(column_swap))
            self.assertEqual(orbit(values),orbit(player_swap))

    def test_invalid_specs_and_prompt_ambiguity_reject(self):
        base={'task':'payoff','payoffs':[0]*8,'row':0,'column':0}
        for record in ({**base,'row':True},{**base,'payoffs':[False]*8},{**base,'payoffs':[2]*8},
                       {**base,'extra':0},{**base,'payoffs':[0]*7}):
            with self.assertRaises(ContractError):solve(record)
        for prompt in ('N:00000000:0=','P:00000000=','P:00000000:0,2=',' N:00000000=', 'R:00000000:0,1='):
            with self.assertRaises(ContractError):parse_prompt(prompt)
        raw,manifest=dataset();data=parse_json(raw,canonical=True);data['examples'][0]['answer']='forged'
        with self.assertRaises(ContractError):audit_bytes(json_bytes(data),manifest)


if __name__=='__main__':unittest.main()
