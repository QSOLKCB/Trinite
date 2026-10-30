"""Finite paired admission and exposure contracts; no model execution."""
from collections import Counter
from copy import deepcopy
import unittest
from unittest import mock

from trinite.contracts import ContractError, identity, json_bytes
from music_contrast_v1.spec import rows as original_rows
from music_exposure_v1.spec import plan_for as original_plan, schedule_receipt as old_schedule
from music_placement_v1 import spec


class SpecTests(unittest.TestCase):
    def test_exact_frozen_matrix_and_budgets(self):
        self.assertEqual(spec.CELLS, (('control',17),('swap',17),('swap',29),('control',29)))
        self.assertEqual(spec.PROTOCOL['aggregate_updates'],2560)
        self.assertEqual(spec.PROTOCOL['aggregate_target_tokens'],40960)
        self.assertEqual(spec.PROTOCOL['max_decodes'],800)
        self.assertEqual(spec.PROTOCOL['max_decode_forwards'],51200)
        self.assertIsNone(spec.PROTOCOL['accuracy_pass_threshold'])
        for seed in spec.SEEDS:
            self.assertEqual(spec.plan_for(seed),original_plan(seed))
        for args in [('best',17),('control',True),('swap',18)]:
            with self.assertRaises(ContractError): spec.cell_name(*args)
        for seed in (True,18,'17'):
            with self.assertRaises(ContractError): spec.plan_for(seed)

    def test_train_control_duplicate_is_exact_original_order(self):
        old=original_rows(); control=spec.train_rows('control')
        self.assertEqual(control,old+old)
        self.assertEqual(spec.diagnostic_rows()[:40],old)
        self.assertEqual(spec.development_rows(),original_rows('heldout',True))
        self.assertEqual(spec.schedule_receipt('control')['schedule_identity'],old_schedule(640)['schedule_identity'])
        self.assertEqual(spec.schedule_receipt('control')['live_input_tokens'],old_schedule(640)['live_input_tokens'])

    def test_only_valid_joint_placement_templates_change(self):
        a,b=spec.train_rows('control'),spec.train_rows('swap')
        self.assertEqual(a[:40],b[:40])
        for left,right in zip(a,b):
            self.assertEqual((left['answer'],len(left['prompt']),left['parent_id'],left['variant_id']),
                             (right['answer'],len(right['prompt']),right['parent_id'],right['variant_id']))
            lf,rf=deepcopy(left['formal']),deepcopy(right['formal'])
            for score in (lf,rf):
                for bar in score['bars']:
                    for event in bar:
                        self.assertIn(event['pitch'],(None,60))
                        event['pitch']=60;event['tie_in']=False;event['tie_out']=False
            self.assertEqual(lf,rf)
        self.assertEqual(len({r['prompt'] for r in a}),40)
        self.assertEqual(len({r['prompt'] for r in b}),70)
        self.assertEqual(sum(x['prompt']==y['prompt'] for x,y in zip(b[:40],b[40:])),10)
        for row in b[40:]:
            events=[e for bar in row['formal']['bars'] for e in bar]
            tie,rest={'a':(4,2),'b':(0,6)}[row['parent_id']]
            self.assertEqual([i for i,e in enumerate(events) if e['pitch'] is None], [rest] if row['variant_id'][0]=='1' else [])
            self.assertEqual([i for i,e in enumerate(events) if e['tie_out']], [tie] if row['variant_id'][1]=='1' else [])

    def test_exposure_matches_structure_not_canonical_forms(self):
        for arm,unique,count in [('control',40,128),('swap',70,64)]:
            receipt=spec.schedule_receipt(arm)
            self.assertEqual(receipt['scheduled_visits'],5120)
            self.assertEqual(receipt['target_tokens'],10240)
            self.assertEqual(receipt['unique_surface_rows'],unique)
            self.assertEqual(set(receipt['visits_per_surface'].values()),{128} if arm=='control' else {64,128})
            if arm=='swap':self.assertEqual(Counter(receipt['visits_per_surface'].values()),{64:60,128:10})
            self.assertEqual(set(receipt['visits_per_structure'].values()),{128})
            self.assertEqual(receipt['answer_histogram'],{'6':1280,'7':2560,'8':1280})
        c,p=spec.schedule_receipt('control'),spec.schedule_receipt('swap')
        self.assertEqual(c['live_input_tokens'],p['live_input_tokens'])
        self.assertEqual(c['visits_per_structure'],p['visits_per_structure'])

    def test_admission_is_recomputed_disjoint_and_precise(self):
        raw,receipt=spec.dataset()
        self.assertEqual(spec.require_admission(identity(receipt)),(raw,receipt))
        with self.assertRaises(ContractError):spec.require_admission('sha256:'+'0'*64)
        diag,dev=spec.diagnostic_rows(),spec.development_rows()
        self.assertEqual({r['parent_id'] for r in diag},{'a','b'})
        self.assertEqual({r['parent_id'] for r in dev},{'c','d'})
        self.assertFalse({r['prompt'] for r in diag}&{r['prompt'] for r in dev})
        self.assertEqual(Counter(r['sibling'] for r in diag),{'canonical':40,'swap':40})
        self.assertEqual(Counter(r['sibling'] for r in dev),{'canonical':40,'pitch':40,'tempo':40})
        bad=spec.train_rows('swap');bad[-1]['split']='test'
        old=spec.train_rows
        with mock.patch.object(spec,'train_rows',side_effect=lambda arm:bad if arm=='swap' else old(arm)):
            with self.assertRaisesRegex(ContractError,'entered training'):spec.dataset()

    def test_second_factor_or_structural_order_drift_rejected(self):
        old=spec.train_rows
        for change in ('tempo','order'):
            bad=old('swap')
            if change=='tempo':bad[-1]['formal']['tempo_qpm']=180
            else:bad[-1],bad[-2]=bad[-2],bad[-1]
            with mock.patch.object(spec,'train_rows',side_effect=lambda arm:bad if arm=='swap' else old(arm)):
                with self.assertRaises(ContractError):spec.dataset()
        for arm in ('extra',1,True):
            with self.assertRaises(ContractError):spec.train_rows(arm)

if __name__=='__main__':unittest.main()
