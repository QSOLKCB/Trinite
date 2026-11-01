"""Formal data and custody boundaries; no scientific optimizer execution."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from context_symbol_learning_v1 import data,numerical,scoring,spec
from procedural_dataset_v1 import cases
from trinite.tokenizer import ByteTokenizer


def train():
    return data.training_rows(data.inventory(),lambda r:cases.expected_target('context',r))


class DataTests(unittest.TestCase):
    def test_exact_inventory_and_orbit_grouping(self):
        rows=data.inventory()
        audit=data.audit(rows)
        self.assertEqual(audit['requests_sha256'],'21b0311fc629bd95f52a01c0e57c756383599ea4b286a4d01b8c58d799c130d5')
        self.assertEqual(audit['counts'],dict(train=360,validation=144,test=144))
        self.assertEqual(audit['semantic_orbits'],18)
        self.assertEqual(audit['max_complete_tokens'],98)
        for f in {r['family'] for r in rows}:
            self.assertEqual(len({r['split'] for r in rows if r['family']==f}),1)
        self.assertEqual(data.orbit([[0,1,2],[2,1,0]]),data.orbit([[2,0,1],[1,0,2]]))
        changed=deepcopy(rows)
        next(r for r in changed if r['split']!='train')['split']='train'
        with self.assertRaises(ValueError):
            data.audit(changed)

    def test_native_compatible_world_semantics_and_masks(self):
        r=dict(schema=cases.REQUEST_SCHEMAS['context'],contexts=2,symbols=2,meanings=2,
            table=[[0,1],[1,1]],context_mask=[True,True],symbol_mask=[True,False],query=0)
        self.assertEqual(cases.expected_target('context',r),'0')
        r['symbol_mask']=[False,True]
        self.assertEqual(cases.expected_target('context',r),'-1')
        r['query']=1
        self.assertEqual(cases.expected_target('context',r),'1')
        r['context_mask']=[False,False]
        with self.assertRaises(ValueError):
            data.prompt(cases.public_observation('context',r))

    def test_public_only_lossless_and_exact_eos_mask(self):
        obs=data.observation(data.inventory()[0])
        prefix=data.prompt(obs)
        module,decoded=cases.decode_public(prefix[:-2])
        self.assertEqual((module,decoded),('context',obs))
        for answer in ('-1','0','1'):
            row=data.encode(obs,answer)
            self.assertEqual([t for t,m in zip(row['ids'],row['loss']) if m],[*answer.encode('ascii'),257])
            self.assertEqual(row['loss'][:1+len(prefix)],[0]*(1+len(prefix)))
            self.assertEqual(row['loss'][-1],1)
        for key in ('label','family','seed','filename','provenance'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                data.prompt(dict(obs,**{key:'forbidden'}))
        self.assertNotIn('c0000',prefix)

    def test_256_boundary_never_truncates(self):
        self.assertEqual(len(ByteTokenizer(256).encode('x'*254).input_ids),256)
        with self.assertRaises(ValueError):
            ByteTokenizer(256).encode('x'*255)
        obs=data.observation(data.inventory()[0])
        obs.update(contexts=8,symbols=8,meanings=8,table=[list(range(8))]*8,
                   context_mask=[True]+[False]*7,symbol_mask=[True]+[False]*7)
        with self.assertRaises(ValueError):
            data.encode(obs,'-1')

    def test_train_filter_precedes_actual_privileged_function(self):
        seen=[]
        wanted={cases.canonical(r['request']) for r in data.inventory() if r['split']=='train'}
        def oracle(request):
            self.assertIn(cases.canonical(request),wanted)
            seen.append(request)
            return cases.expected_target('context',request)
        rows=data.training_rows(data.inventory(),oracle)
        self.assertEqual(len(seen),360)
        self.assertEqual(len(rows),360)
        self.assertEqual({r['answer'] for r in data.fit_rows(rows)},{'-1','0','1'})
        with self.assertRaises(ValueError):
            data.training_rows(data.inventory(),lambda r:'wrong')

    def test_exact_schedule_and_global_token_ceiling(self):
        rows=train()
        self.assertEqual(numerical.schedule(rows,17,True)['targets'],2432)
        self.assertEqual(numerical.schedule(rows,17)['targets'],4900)
        self.assertEqual(sum(t for _,_,_,_,t in spec.ATTEMPTS),12232)
        self.assertEqual(sum(u for _,_,_,u,_ in spec.ATTEMPTS),640)
        with patch.object(numerical,'plan',return_value=type('P',(),dict(steps=256,batch_size=8,max_target_tokens=4096))()):
            with self.assertRaises(ValueError):
                numerical.schedule(rows,17)

    def test_empty_malformed_missing_eos_are_not_uncertainty(self):
        for tokens in ([],[257],[48],[49,48,257],[45,257],[256,48,257],[48,257,257],[False,257]):
            with self.subTest(tokens=tokens):
                self.assertEqual(numerical.parse_output(tokens)['status'],'malformed')
        self.assertEqual(numerical.parse_output([48,257])['value'],'0')

    def test_full_denominator_and_pair_failures(self):
        predictions=scoring.empty()
        # Arbitrary synthetic targets test aggregation, not scientific outcomes.
        targets={r['id']:'0' for r in data.inventory() if r['split']!='train'}
        for row in predictions:
            row['raw']=dict(status='malformed',tokens=[])
        result=scoring.summary(predictions,targets)
        value=result['methods']['main-17']['test']
        self.assertEqual(value['exact_accuracy'],dict(numerator=0,denominator=144))
        self.assertEqual(value['statuses']['malformed'],144)
        self.assertEqual(value['macro_f1'],dict(numerator=0,denominator=1))
        self.assertEqual(value['context_switch_consistency']['all_pairs']['numerator'],0)
        self.assertEqual(result['methods']['oracle']['test']['exact_accuracy'],dict(numerator=144,denominator=144))
        self.assertFalse(result['descriptive_gain_over_both_baselines']['test'])

    def test_majority_is_train_only_tie_smallest_and_context_shortcut(self):
        self.assertEqual(scoring.majority([dict(answer='1'),dict(answer='-1')]),'-1')
        obs=data.observation(data.inventory()[0])
        other=dict(obs,context_mask=[not x for x in obs['context_mask']])
        self.assertEqual(scoring.shortcut(obs),scoring.shortcut(other))

    def test_final_fit_gate_uses_only_exact_step_and_recomputed_answers(self):
        rows=train()
        state=type('S',(),dict(step=128,target_tokens=2432,model=object()))()
        with patch.object(numerical,'fit_report',return_value=[dict(exact=True)]*8) as report:
            numerical.require_fit(state,rows)
            report.assert_called_once_with(state.model,rows)
            for step in (0,1,127,129):
                state.step=step
                with self.assertRaises(ValueError):
                    numerical.require_fit(state,rows)
            state.step=128
            report.return_value=[dict(exact=True)]*7+[dict(exact=False)]
            with self.assertRaises(ValueError):
                numerical.require_fit(state,rows)


if __name__=='__main__':
    unittest.main()
