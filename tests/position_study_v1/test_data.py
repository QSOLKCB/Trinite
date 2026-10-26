"""Exact frozen counts and labels, without training or importing a backend."""
from collections import Counter
import unittest

from trinite.contracts import ContractError, json_bytes, parse_json
from trinite.tokenizer import ByteTokenizer
from position_study_v1 import data
from position_study_v1.protocol import PROTOCOL, STRATA


class DataTests(unittest.TestCase):
    def test_exact_inventory_and_independent_labels(self):
        raw,manifest = data.dataset()
        self.assertEqual(data.audit_bytes(raw,manifest)['counts'],PROTOCOL['inventory'])
        for r in data.all_rows():
            a,b,w = data.parse_prompt(r['prompt'])
            self.assertEqual(r['answer'],str(a+b).zfill(w+1)[::-1])
            self.assertEqual(data.reference(a,b,w),r['answer'])
            self.assertEqual(len(r['answer']),w+1)
            self.assertEqual(r['family_id'],data.family(b,a))
            self.assertLessEqual(len(ByteTokenizer(64).encode(r['text']).input_ids),17)

    def test_semantic_families_and_all_admitted_renderings_stay_together(self):
        seen = {}
        for r in data.all_rows():
            a,b = r['formal']['a'],r['formal']['b']
            key = tuple(sorted((a,b)))
            if key in seen: self.assertEqual(seen[key],r['stratum'])
            seen[key] = r['stratum']
            variants = [x for x in data.all_rows() if x['family_id']==r['family_id']]
            self.assertEqual({(x['formal']['a'],x['formal']['b'],x['width']) for x in variants},
                {(a,b,w) for a,b in {(a,b),(b,a)} for w in data.widths(a,b)})
        self.assertEqual(len(seen),112)

    def test_exact_schedule_and_token_budget(self):
        left,right = data.schedule('control'),data.schedule('candidate')
        self.assertEqual(left,right)
        self.assertEqual(len(left),160)
        self.assertEqual(Counter(r['width'] for r in left),{1:32,2:128})
        self.assertEqual(Counter(Counter(r['family_id'] for r in left).values()),{4:16,2:48})
        enc = [ByteTokenizer(64).encode(r['text'],answer_start=len(r['prompt'])) for r in left]
        self.assertEqual(sum(len(e.input_ids)-1 for e in enc)*200,300800)
        actual = sum(sum(min(len(e.input_ids),max(len(v.input_ids) for v in enc[i:i+8])-1)
            for e in enc[i:i+8]) for i in range(0,160,8))*200
        self.assertEqual(actual,307200)
        self.assertEqual(sum(sum(e.loss_mask) for e in enc)*200,121600)
        data.validate_matching()

    def test_no_probe_in_schedule_or_training_diagnostics(self):
        probe = {r['example_identity'] for s in STRATA for r in data.examples(s)}
        for arm in ('control','candidate'):
            self.assertFalse(probe & {r['example_identity'] for r in data.schedule(arm)})
            self.assertFalse(probe & {r['example_identity'] for r in data.examples(arm)})

    def test_row_exposure_and_fixed_offsets_are_not_significance_claims(self):
        left,right = data.exposure('control'),data.exposure('candidate')
        self.assertEqual(set(left['training_position_row_lookups']),set(map(str,range(10))))
        self.assertEqual(right['training_position_row_lookups'],{'0':102400,'1':96000,'2':83200,'3':25600})
        self.assertEqual(sum(right['training_position_row_lookups'].values()),307200)
        self.assertEqual(right['live_unscored_eos_input_tokens'],6400)
        self.assertEqual(left['cycle_identity'],right['cycle_identity'])
        self.assertEqual(right['training_position_row_roles']['3'],{'answer-slot-2':25600})

    def test_static_probe_pair_inventory_and_context_bounds_are_frozen(self):
        from position_study_v1.reporting import design_summary
        summary=design_summary()
        for s,record in summary.items():
            self.assertEqual(record['pair_counts'],PROTOCOL['perturbation_pair_counts'][s])
            self.assertLessEqual(record['full_prefix_max_length'],17)
            for rows in record['position_rows_at_full_cap'].values():
                self.assertGreaterEqual(min(rows),0);self.assertLess(max(rows),64)

    def test_admission_rejects_labels_membership_rights_and_noncanonical_bytes(self):
        raw,manifest = data.dataset()
        for field,value in [('answer','999'),('stratum','train'),('family_id','foreign')]:
            record = parse_json(raw,canonical=True); record['examples'][-1][field] = value
            with self.assertRaises(ContractError): data.audit_bytes(json_bytes(record),manifest)
        record = parse_json(manifest,canonical=True);record['admission']['outcome']='rejected'
        with self.assertRaises(ContractError): data.audit_bytes(raw,json_bytes(record))
        with self.assertRaises(ContractError): data.audit_bytes(raw+b' ',manifest)

    def test_only_supported_prompt_grammar(self):
        for p in ('A:1,2=','1+02=','1 +2=','-1+2=','1+2=3','1+2','00000+00000='):
            with self.assertRaises(ContractError): data.parse_prompt(p)


if __name__=='__main__': unittest.main()
