"""Independent finite accounting checks; synthetic records, no model inference.

A constant-zero counterexample proves that agreement must not imply correctness.
These checks exercise evidence accounting, not any actual checkpoint capability.
"""
import copy
import unittest
from test_measurements import report, FAKE_MODEL
from failure_map_v1 import measurements as m
from trinite.contracts import ContractError
from trinite.tokenizer import EOS


class IndependentScientificAccountingTests(unittest.TestCase):
    def test_full_corpus_pairs_constant_output_and_context_mutations(self):
        for arm,seen,prior in [('control',20,318),('candidate',40,338)]:
            r,old,train=report(arm)
            rows=r['rows']; s=r['summary']
            assert len(rows)==338 and s['families']==91 and s['pass']==338
            assert s['seen_anchors']['rows']==seen
            assert sum(x['reproducibility']=='match' for x in rows)==prior
            assert sum(x['reproducibility']=='not_previously_scored' for x in rows)==338-prior
            assert len(s['coverage_cells'])==48 and sum(c['rows'] for c in s['coverage_cells'])==338
            assert s['metamorphic']['carrier']['pair_count']==169
            assert s['metamorphic']['operand_order']['pair_count']==156
            for x in rows:
                assert x['answer']==str(x['a']+x['b'])
                assert x['prediction']['correct']
                assert x['zero_answer']==(x['a']==x['b']==0)
            by={x['example_identity']:x for x in rows}
            for name,entry in s['metamorphic'].items():
                ids=set()
                for p in entry['pairs']:
                    l,h=by[p['left']],by[p['right']]
                    assert l['answer']==h['answer'] and p['both_correct'] and p['consistent']
                    assert (p['left'],p['right']) not in ids
                    ids.add((p['left'],p['right']))
                    if name=='carrier': assert (l['a'],l['b'])==(h['a'],h['b']) and l['carrier']!=h['carrier']
                    else: assert (l['a'],l['b'])==(h['b'],h['a']) and l['a']!=l['b'] and l['carrier']==h['carrier']
            for path in ['status','scientific_exposure','answer','seen_in_this_arm','zero_answer']:
                bad=copy.deepcopy(r); bad['rows'][0][path]='tampered'
                try:m.validate_report(bad,arm,0,FAKE_MODEL,old,train)
                except ContractError:pass
                else:raise AssertionError(path)
            constant,_,_=report(arm,emit=lambda x:[ord('0'),EOS])
            cs=constant['summary']; assert cs['raw_correct']==2 and cs['pass']==2 and cs['fail']==336
            assert cs['metamorphic']['carrier']['consistent']==169
            assert cs['metamorphic']['carrier']['both_correct']==1
            assert cs['metamorphic']['operand_order']['consistent']==156
            assert cs['metamorphic']['operand_order']['both_correct']==0
            all_wrong,old_wrong,train_wrong=report(arm,emit=lambda x:[EOS])
            assert all_wrong['summary']['pass']==0 and all_wrong['summary']['fail']==338
            assert all_wrong['summary']['untested']==0
            assert all_wrong['summary']['reproducibility'].get('mismatch',0)==0
            assert m.validate_report(all_wrong,arm,0,FAKE_MODEL,old_wrong,train_wrong)==all_wrong['summary']
            mismatch,_,_=report(arm,emit=lambda x:[ord('0'),EOS],prior_emit=lambda x:[ord('1'),EOS])
            assert mismatch['summary']['untested']==prior
            assert mismatch['summary']['reproducibility']['mismatch']==prior


if __name__ == "__main__":
    unittest.main()
