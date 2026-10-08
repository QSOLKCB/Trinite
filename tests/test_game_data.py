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
