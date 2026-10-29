"""Reports-only preservation integrity and scorer rejection tests; no models."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from music_exposure_result_v1 import check

PRED = 'train-17--step-064-predictions.json'
JOURNAL = 'train-17--step-064-journal.jsonl'

class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Path(self.tmp.name)/'fixture'
        shutil.copytree(check.FIXTURE,self.fixture)

    def test_all_observations_gate_and_readonly(self):
        before=check.snapshot(self.fixture)
        r=check.verify(self.fixture)
        self.assertEqual(r['observations'],240)
        self.assertEqual(r['duplicate_journal_rows'],240)
        self.assertEqual(r['final_exact_per_seed'],{'17':40,'29':40})
        self.assertIs(r['both_seeds_fit'],True)
        self.assertEqual(r['historical_step0_64_files_equal'],8)
        self.assertEqual(r['heldout_model_evaluation'],'unrun')
        for key in ('original_raw_custody_verified','inference_replayed',
                    'optimizer_replayed','fresh_confirmation','learning_adequate','phase5_ready','release_ready'):
            self.assertIs(r[key],False)
        self.assertEqual(check.snapshot(self.fixture),before)
        self.assertNotIn('torch',sys.modules)

    def test_all_member_bytes_bound(self):
        for name in check.FIXTURE_HASHES:
            with self.subTest(name=name):
                path=self.fixture/name;raw=path.read_bytes();path.write_bytes(raw+b' ')
                with self.assertRaisesRegex(ValueError,'bytes changed'):check.verify(self.fixture)
                path.write_bytes(raw)

    def test_missing_and_extra(self):
        (self.fixture/'README.md').rename(self.fixture/'extra')
        with self.assertRaisesRegex(ValueError,'membership'):check.verify(self.fixture)

    def test_extra_directory(self):
        (self.fixture/'unexpected').mkdir()
        with self.assertRaisesRegex(ValueError,'membership'):check.verify(self.fixture)

    def test_linked_member(self):
        p=self.fixture/PRED;p.unlink();p.symlink_to(check.FIXTURE/PRED)
        with self.assertRaisesRegex(ValueError,'regular'):check.verify(self.fixture)

    def test_linked_root(self):
        p=Path(self.tmp.name)/'link';p.symlink_to(self.fixture,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'non-link'):check.verify(p)

    def test_linked_ancestor(self):
        p=Path(self.tmp.name)/'link';p.symlink_to(self.fixture.parent,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'ancestor'):check.verify(p/'fixture')

    def test_nonregular(self):
        p=self.fixture/PRED;p.unlink();p.mkdir()
        with self.assertRaisesRegex(ValueError,'regular'):check.verify(self.fixture)

    def test_oversized(self):
        p=self.fixture/PRED;p.write_bytes(b'x'*(check.LIMIT+1))
        with self.assertRaisesRegex(ValueError,'bounded'):check.verify(self.fixture)

    def test_source_tamper(self):
        read=check.read
        def tamper(p):
            raw=read(p)
            return raw+b' ' if str(p).endswith('/reporting.py') else raw
        with mock.patch.object(check,'read',side_effect=tamper):
            with self.assertRaisesRegex(ValueError,'source changed'):check.verify(self.fixture)

    def test_foreign_loaded_module(self):
        with mock.patch.dict(sys.modules,{'music_contrast_v1.reporting':types.SimpleNamespace(__file__='/foreign/reporting.py')}):
            with self.assertRaisesRegex(ValueError,'foreign'):check.sources()

    def source_copy(self):
        root=Path(self.tmp.name)/'checkout'
        for relative in check.SOURCES:
            target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(check.ROOT/relative,target)
        return root

    def test_package_shadow_before_execution(self):
        root=self.source_copy();p=root/'scripts/music_contrast_v1/reporting';p.mkdir()
        marker=Path(self.tmp.name)/'executed'
        (p/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'shadow'):check.sources()
        self.assertFalse(marker.exists())

    def test_cache_rejected(self):
        root=self.source_copy();(root/'src/trinite/__pycache__').mkdir()
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'cache'):check.sources()

    def test_foreign_resolution_before_execution(self):
        p=Path(self.tmp.name)/'trinite';p.mkdir();marker=p/'executed'
        (p/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name=='trinite' or name.startswith('trinite.'):del sys.modules[name]
            with mock.patch.object(sys,'path',[self.tmp.name,*sys.path]):
                with self.assertRaisesRegex(ValueError,'resolution'):check.sources()
        self.assertFalse(marker.exists())

    def test_journal_mismatch(self):
        before=check.snapshot(self.fixture);before[JOURNAL]=b'{}\n'
        with self.assertRaisesRegex(ValueError,'journal'):check.score(before)

    def test_context_and_label_rejected(self):
        check.sources()
        from trinite.contracts import json_bytes
        for key,value in [('correct',True),('split','test'),('expected_tokens',[54,257]),('generated_tokens',[True,257])]:
            before=check.snapshot(self.fixture);records=json.loads(before[PRED])['predictions'];records[0][key]=value
            before[PRED]=json_bytes({'predictions':records});before[JOURNAL]=b''.join(json_bytes(r) for r in records)
            with self.subTest(key=key):
                with self.assertRaises(ValueError):check.score(before)

    def test_missing_duplicate_and_extra_rows(self):
        check.sources()
        from trinite.contracts import json_bytes
        for mode in ('missing','duplicate','extra'):
            before=check.snapshot(self.fixture);records=json.loads(before[PRED])['predictions']
            if mode=='missing':records.pop()
            elif mode=='extra':records.append(records[0])
            else:records[-1]=records[0]
            before[PRED]=json_bytes({'predictions':records});before[JOURNAL]=b''.join(json_bytes(r) for r in records)
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):check.score(before)

    def test_negative_replacement_is_not_preserved_positive(self):
        check.sources()
        from trinite.contracts import json_bytes
        before=check.snapshot(self.fixture)
        for seed in (17,29):
            prefix=f'train-{seed}--step-640'
            records=json.loads(before[prefix+'-predictions.json'])['predictions']
            for r in records:r['generated_tokens']=[55,257];r['correct']=r['expected_tokens']==[55,257]
            before[prefix+'-predictions.json']=json_bytes({'predictions':records})
            before[prefix+'-journal.jsonl']=b''.join(json_bytes(r) for r in records)
        with self.assertRaisesRegex(ValueError,'trajectory differs'):check.score(before)

    def test_exact_and_majority_not_just_eos(self):
        summary=check.score(check.snapshot(self.fixture))
        final=summary['observations'][PRED]
        self.assertEqual(final['all']['eos_terminated'],40)
        self.assertEqual(final['all']['exact_plus_eos'],20)
        self.assertEqual(final['all']['complete_groups'],0)
        self.assertEqual(final['all']['both_correct_edges'],0)
        self.assertEqual(final['all']['changed_output_edges'],0)
        self.assertEqual(final['baselines']['global']['exact_plus_eos'],20)
        self.assertEqual(summary['decode_forwards'],5440)
        final=summary['observations']['train-17--step-640-predictions.json']['all']
        self.assertEqual((final['exact_plus_eos'],final['complete_groups'],final['both_correct_edges']),(40,10,40))

    def test_mutation_during_scoring(self):
        score=check.score
        def changed(before):
            result=score(before);(self.fixture/'README.md').write_text('changed');return result
        with mock.patch.object(check,'score',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'during verification'):check.verify(self.fixture)


    def test_historical_first_eight_files_have_fixed_hash_and_equal_bytes(self):
        before=check.snapshot(self.fixture)
        self.assertEqual(len(check.HISTORICAL_HASHES),8)
        self.assertEqual(check.historical_parity(before),check.HISTORICAL_HASHES)
        for name,expected in check.HISTORICAL_HASHES.items():
            self.assertEqual(check.digest(before[name]),expected)
            self.assertEqual(before[name],check.read(check.HISTORICAL/name))

    def test_historical_mutation_rejected_even_when_both_sets_match(self):
        before=check.snapshot(self.fixture)
        name=next(iter(check.HISTORICAL_HASHES))
        before[name]+=b' '
        original=check.read
        def changed(path):
            return before[name] if Path(path)==check.HISTORICAL/name else original(path)
        with mock.patch.object(check,'read',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'historical'):
                check.historical_parity(before)

    def test_historical_changes_during_verification_rejected(self):
        original=check.score
        calls=[]
        historical=check.historical_parity
        def changed(before):
            result=original(before);calls.append(True);return result
        def check_again(before):
            if calls:raise ValueError('historical changed during verification')
            return historical(before)
        with mock.patch.object(check,'score',side_effect=changed),mock.patch.object(check,'historical_parity',side_effect=check_again):
            with self.assertRaisesRegex(ValueError,'historical changed'):
                check.verify(self.fixture)

    def test_source_closure_includes_exposure_and_transitive_dependencies(self):
        for name in ('music_exposure_v1.reporting','music_exposure_v1.spec',
                     'music_learning_v1.numerical','trinite.foundations_plan',
                     'music_contrast_v1.reporting','music_contrast_v1.spec'):
            self.assertIn(name,check.MODULES)
            relative=check.MODULES[name]
            self.assertEqual(check.digest(check.read(check.ROOT/relative)),check.SOURCES[relative])
        check.sources()
        self.assertNotIn('torch',sys.modules)

    def test_boolean_in_place_of_integer_and_integer_in_place_of_boolean_rejected(self):
        check.sources()
        from trinite.contracts import json_bytes
        for key,value in [('correct',0),('baseline_correct',False),('expected_tokens',[True,257]),('generated_tokens',[True,257])]:
            before=check.snapshot(self.fixture);records=json.loads(before[PRED])['predictions']
            # Select a gold7 row for baseline_correct=True, then replace it withFalse.
            index=1 if key=='baseline_correct' else 0
            records[index][key]=value
            before[PRED]=json_bytes({'predictions':records});before[JOURNAL]=b''.join(json_bytes(r) for r in records)
            with self.subTest(key=key):
                with self.assertRaises(ValueError):check.score(before)

    def test_noncanonical_and_unknown_prediction_schema_rejected(self):
        check.sources()
        from trinite.contracts import json_bytes
        before=check.snapshot(self.fixture)
        before[PRED]=json.dumps(json.loads(before[PRED]),indent=2).encode()
        with self.assertRaises(ValueError):check.score(before)
        before=check.snapshot(self.fixture);data=json.loads(before[PRED]);data['extra']=True
        before[PRED]=json_bytes(data)
        with self.assertRaises(ValueError):check.score(before)

    def test_summary_integer_boolean_substitution_rejected_even_if_repinned(self):
        summary=json.loads((self.fixture/'summary.json').read_bytes())
        summary['both_seeds_fit']=1
        from trinite.contracts import json_bytes
        raw=json_bytes(summary);(self.fixture/'summary.json').write_bytes(raw)
        hashes={**check.FIXTURE_HASHES,'summary.json':check.digest(raw)}
        with mock.patch.object(check,'FIXTURE_HASHES',hashes):
            with self.assertRaisesRegex(ValueError,'derived summary'):
                check.verify(self.fixture)

    def test_no_private_handoff_or_run_commitments_published(self):
        self.assertNotIn('HANDOFF.md',check.FIXTURE_HASHES)
        self.assertNotIn('REPORT.md',check.FIXTURE_HASHES)
        self.assertNotIn('SHA256SUMS',check.FIXTURE_HASHES)
        provenance=json.loads((self.fixture/'provenance.json').read_bytes())
        self.assertIn('owner_reported_execution',provenance)
        self.assertIn('missing_original_artifacts',provenance)
        for key in ('environment','request_identity','matrix_identity','archive_sha256','audit_receipt_sha256'):
            self.assertNotIn(key,provenance)


    def test_regenerated_schedule_is_static_not_execution_evidence(self):
        summary=check.score(check.snapshot(self.fixture))
        schedules=summary['declared_schedules']
        self.assertEqual(set(schedules),{'0','64','640'})
        self.assertEqual(schedules['0']['scheduled_visits'],0)
        self.assertEqual(schedules['64']['scheduled_visits'],512)
        self.assertEqual(schedules['64']['target_tokens'],1024)
        self.assertEqual(sorted(schedules['64']['visits_per_example'].values()),[12]*8+[13]*32)
        self.assertEqual(schedules['640']['scheduled_visits'],5120)
        self.assertEqual(schedules['640']['target_tokens'],10240)
        self.assertEqual(schedules['640']['live_input_tokens'],348160)
        self.assertEqual(list(schedules['640']['visits_per_example'].values()),[128]*40)
        self.assertIn('not observed optimizer execution',summary['schedule_scope'])


    def test_both_final_seed_histograms_and_all_strata_are_exact(self):
        summary=check.score(check.snapshot(self.fixture))
        for seed in (17,29):
            final=summary['observations'][f'train-{seed}--step-640-predictions.json']
            self.assertEqual(final['unique_raw_outputs'],3)
            self.assertEqual(final['raw_output_histogram'],[
                {'tokens':[54,257],'rows':10},{'tokens':[55,257],'rows':20},{'tokens':[56,257],'rows':10}])
            for label,count in (('6',10),('7',20),('8',10)):
                self.assertEqual(final['by_answer'][label],{'rows':count,'exact_plus_eos':count})
            for parent in ('a','b'):
                self.assertEqual(final['by_parent'][parent],{'rows':20,'exact_plus_eos':20})
            self.assertEqual(len(final['by_meter']),5)
            for meter in final['by_meter'].values():
                self.assertEqual((meter['rows'],meter['exact_plus_eos'],meter['complete_groups'],meter['both_correct_edges']),(8,8,2,8))

if __name__=='__main__':unittest.main()
