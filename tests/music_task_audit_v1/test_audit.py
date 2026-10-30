"""Read-only support/error audit conformance; no training or backend calls."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from music_task_audit_v1 import check


class AuditBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'checkout'

    def closed_copy(self):
        for relative in {*check.SOURCES, *check.INPUT_HASHES}:
            target = self.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(check.ROOT/relative, target)
        shutil.copytree(check.ROOT/'scripts/music_task_audit_v1',
                        self.root/'scripts/music_task_audit_v1', dirs_exist_ok=True)
        shutil.copyfile(check.ROOT/'scripts/check_music_task_audit.py',
                        self.root/'scripts/check_music_task_audit.py')
        return self.root

    def launch(self, root=None, args=(), flags=('-I','-B')):
        return subprocess.run([sys.executable, *flags,
            str((root or check.ROOT)/'scripts/check_music_task_audit.py'), *args],
            capture_output=True, text=True)

    def test_all_evidence_pins_reject_before_analysis(self):
        original = check.read
        for relative in check.INPUT_HASHES:
            with self.subTest(relative=relative):
                def read(path):
                    raw = original(path)
                    return raw+b' ' if Path(path) == check.ROOT/relative else raw
                with mock.patch.object(check,'read',side_effect=read):
                    with self.assertRaisesRegex(ValueError, 'evidence bytes'): check.snapshot()

    def test_all_source_pins_reject_before_project_import(self):
        original = check.read
        for relative in check.SOURCES:
            with self.subTest(relative=relative):
                def read(path):
                    raw = original(path)
                    return raw+b' ' if Path(path) == check.ROOT/relative else raw
                with mock.patch.object(check,'read',side_effect=read), mock.patch.object(
                        check.importlib, 'import_module', side_effect=AssertionError('must reject first')):
                    with self.assertRaisesRegex(ValueError,'source changed'): check.verify()

    def test_analysis_source_pin_rejects(self):
        original = check.read
        for relative in check.ANALYSIS_HASHES:
            def read(path):
                raw = original(path)
                return raw+b' ' if Path(path) == check.ROOT/relative else raw
            with mock.patch.object(check,'read',side_effect=read):
                with self.assertRaisesRegex(ValueError,'source (changed|differs)'): check.verify()

    def test_read_rejects_symlink_ancestor_leaf_and_nonregular(self):
        target=Path(self.tmp.name)/'data';target.mkdir();file=target/'value';file.write_bytes(b'{}\n')
        alias=Path(self.tmp.name)/'alias';alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):check.read(alias/'value')
        link=target/'link';link.symlink_to(file)
        with self.assertRaises(ValueError):check.read(link)
        with self.assertRaises(ValueError):check.read(target)
        file.write_bytes(b'x'*(check.LIMIT+1))
        with self.assertRaisesRegex(ValueError,'bounded'):check.read(file)

    def test_missing_evidence_and_resealed_matrix_manifest_reject(self):
        root=self.closed_copy()
        path=root/'fixtures/music-placement-result-v1/cpu/matrix.json'
        raw=path.read_bytes();value=json.loads(raw);value['outcome']='incomplete'
        changed=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();path.write_bytes(changed)
        manifest=root/'fixtures/music-placement-result-v1/source-SHA256SUMS'
        manifest.write_text(manifest.read_text().replace(check.digest(raw)[7:],check.digest(changed)[7:]))
        result=self.launch(root);self.assertNotEqual(result.returncode,0)
        self.assertIn('evidence bytes',result.stderr)
        path.unlink();result=self.launch(root);self.assertNotEqual(result.returncode,0)

    def test_extra_historical_fixture_member_rejects(self):
        root=self.closed_copy()
        (root/'fixtures/music-placement-result-v1/unexpected.json').write_text('{}')
        result=self.launch(root);self.assertNotEqual(result.returncode,0)
        self.assertIn('membership',result.stderr)

    def test_source_package_shadow_and_cache_reject_before_execution(self):
        root=self.closed_copy();shadow=root/'scripts/music_placement_v1/spec';shadow.mkdir()
        marker=root/'executed';(shadow/'__init__.py').write_text('open('+repr(str(marker))+',"w").write("x")\n')
        result=self.launch(root);self.assertNotEqual(result.returncode,0);self.assertFalse(marker.exists())
        shutil.rmtree(shadow)
        shadow=root/'scripts/music_task_audit_v1/analysis';shadow.mkdir()
        (shadow/'__init__.py').write_text('open('+repr(str(marker))+',"w").write("x")\n')
        result=self.launch(root);self.assertNotEqual(result.returncode,0);self.assertFalse(marker.exists())
        shutil.rmtree(shadow);(root/'src/trinite/__pycache__').mkdir()
        result=self.launch(root);self.assertNotEqual(result.returncode,0);self.assertIn('cache',result.stderr)

    def test_foreign_loaded_module_rejected(self):
        with mock.patch.dict(sys.modules,{'music_placement_v1.spec':types.SimpleNamespace(__file__='/foreign.py')}):
            with self.assertRaisesRegex(ValueError,'foreign'):check.sources()

    def test_foreign_import_origin_rejected_before_initializer(self):
        package=Path(self.tmp.name)/'trinite';package.mkdir();marker=package/'executed'
        (package/'__init__.py').write_text('open('+repr(str(marker))+',"w").write("x")\n')
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name=='trinite' or name.startswith('trinite.'):del sys.modules[name]
            with mock.patch.object(sys,'path',[self.tmp.name,*sys.path]):
                with self.assertRaisesRegex(ValueError,'resolution'):check.sources()
        self.assertFalse(marker.exists())

    def test_launcher_requires_flags_before_pathlib(self):
        root=Path(self.tmp.name);script=root/'check.py'
        shutil.copyfile(check.ROOT/'scripts/check_music_task_audit.py',script)
        marker=root/'executed';(root/'pathlib.py').write_text('open('+repr(str(marker))+',"w").write("x")\n')
        for flags in ([],['-B'],['-I']):
            result=subprocess.run([sys.executable,*flags,str(script)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('Use python -I -B',result.stderr)
            self.assertFalse(marker.exists());self.assertFalse((root/'__pycache__').exists())

    def test_own_package_origin_and_symlink_rejected(self):
        root=self.closed_copy();marker=root/'executed';shadow=root/'src/music_task_audit_v1';shadow.mkdir()
        (shadow/'__init__.py').write_text('open('+repr(str(marker))+',"w").write("x")\n')
        result=self.launch(root);self.assertNotEqual(result.returncode,0);self.assertFalse(marker.exists())
        shutil.rmtree(shadow);package=root/'scripts/music_task_audit_v1';shutil.rmtree(package)
        package.symlink_to(check.ROOT/'scripts/music_task_audit_v1',target_is_directory=True)
        result=self.launch(root);self.assertNotEqual(result.returncode,0);self.assertIn('linked verifier',result.stderr)

    def test_stdlib_shadows_do_not_execute(self):
        root=self.closed_copy();marker=root/'executed'
        for name in ('hashlib','json','subprocess'):
            (root/'src'/f'{name}.py').write_text('open('+repr(str(marker))+',"w").write("x")\nraise RuntimeError("shadow")\n')
        result=self.launch(root);self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(marker.exists());self.assertFalse(any(root.rglob('__pycache__')))

    def test_evidence_changed_after_analysis_is_rejected(self):
        before=check.snapshot();after=dict(before)
        name=next(iter(after));after[name]=after[name]+b' '
        with mock.patch.object(check,'snapshot',side_effect=[before,after]):
            with self.assertRaisesRegex(ValueError,'changed during audit'):check.verify()

    def test_no_training_or_alternate_input_cli(self):
        for args in (['run'],['--fixture',self.tmp.name],['--authorize-run'],['--checkpoint','x']):
            result=self.launch(args=args);self.assertNotEqual(result.returncode,0)
            self.assertIn('takes no arguments',result.stderr)

    def test_readonly_receipt_does_not_claim_execution_or_causation(self):
        before=check.snapshot();result=check.verify()
        self.assertEqual(result['schema'],'trinite.music-task-audit.v1')
        self.assertEqual(result['evidence_class'],'DERIVED')
        for key in ('model_imported','inference_replayed','training_performed','causal_mechanism_established',
                    'fresh_confirmation','full_original_custody_verified','hardware_attested',
                    'learning_adequate','release_ready'):
            self.assertIs(result[key],False)
        self.assertEqual(set(result['preservation_receipts']),set(check.VERIFIERS))
        self.assertEqual(check.snapshot(),before)
        for name in ('torch','numpy','trinite.model','trinite.training','music_device_trial_v2.runtime'):
            self.assertNotIn(name,sys.modules)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        check.sources()
        from music_task_audit_v1 import analysis
        self.a=analysis
        before=check.snapshot()
        self.inputs={name:before[name] for name in analysis.INPUT_PATHS}

    def test_role_specific_support_not_union_of_positions(self):
        report=self.a.build_report(self.inputs)
        for pool in ('original','pitch_control','pitch_treatment','placement_control','placement_treatment'):
            support=report['support'][pool]['roles']
            self.assertEqual(support['tie_start']['absolute_index'],[0,4])
            self.assertEqual(support['rest']['absolute_index'],[2,6])
            mismatch=report['support_mismatch'][pool]
            self.assertEqual(mismatch['tie_start']['development_absolute_indices'],[1,5])
            self.assertEqual(mismatch['rest']['development_absolute_indices'],[0,7])
            self.assertEqual(mismatch['rest']['unseen_development_absolute_indices'],[0,7])
        # Index zero is supported as a training tie start, not as a training rest.
        self.assertIn(0,report['support']['original']['roles']['tie_start']['absolute_index'])
        self.assertNotIn(0,report['support']['original']['roles']['rest']['absolute_index'])

    def test_surface_duplicates_and_no_development_leakage(self):
        report=self.a.build_report(self.inputs)
        self.assertEqual({k:report['placement'][k] for k in ('diagnostic_slots','unique_prompts','changed_input_pairs','identical_input_pairs')},
                         {'diagnostic_slots':80,'unique_prompts':70,'changed_input_pairs':30,'identical_input_pairs':10})
        self.assertEqual(report['support']['placement_control']['rows'],80)
        self.assertEqual(report['support']['placement_control']['unique_prompts'],40)
        self.assertEqual(report['support']['placement_treatment']['unique_prompts'],70)
        self.assertEqual(report['corpus']['exact_train_development_prompt_overlap'],0)
        self.assertEqual(report['corpus']['training_structural_parents'],['a','b'])
        self.assertEqual(report['corpus']['exposed_development_structural_parents'],['c','d'])
        self.assertTrue(report['corpus']['oracle_and_actual_prompt_target_mask_alignment_verified'])
        self.assertTrue(all(value is False for value in report['claims'].values()))

    def test_finite_valid_support_is_inventory_not_new_curriculum(self):
        result=self.a.finite_placement_universe()
        self.assertEqual(result['valid_adjacent_tie_plus_disjoint_rest_placements'],42)
        self.assertEqual(result['without_cross_bar_ties'],36)
        self.assertEqual(sum(row['cross_bar_tie'] for row in result['placements']),6)
        for row in result['placements']:
            self.assertNotIn(row['rest_index'],(row['tie_start_index'],row['tie_end_index']))
            self.assertEqual(row['tie_end_index'],row['tie_start_index']+1)
        self.assertIn('not generated curriculum',result['scope'])

    def test_oracle_prompt_and_mask_disagreement_rejected(self):
        from copy import deepcopy
        row=self.a.contrast.rows()[0]
        for key,value in (('answer','0'),('answer_start',0),('text',row['text']+'x'),('prompt',row['prompt']+'x')):
            changed=deepcopy(row);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.a.row_features(changed)
        encoded=self.a.encode_example({k:row[k] for k in ('task','prompt','answer','text','answer_start')})
        wrong=types.SimpleNamespace(input_ids=encoded.input_ids,loss_mask=[1]*len(encoded.loss_mask),attention_mask=encoded.attention_mask)
        with mock.patch.object(self.a,'encode_example',return_value=wrong):
            with self.assertRaisesRegex(ValueError,'mask alignment'):self.a.row_features(row)

    def test_retained_errors_remain_descriptive_correlated_evidence(self):
        report=self.a.build_report(self.inputs)
        retained=report['retained'];self.assertGreater(retained['total_retained_records'],0)
        self.assertIn('not independent subjects',retained['scope'])
        relevant=[s for s in retained['sections'] if s['stage']=='development' and s['view']=='canonical']
        self.assertTrue(relevant)
        for section in relevant:
            self.assertEqual(section['by_variant']['10']['correct'],0)
            self.assertEqual(len(section['rest_only_cases']),10)
            self.assertTrue(all(not case['correct'] and case['gold']==[55,257] for case in section['rest_only_cases']))
            self.assertEqual(len(section['same_gold_tie_only_vs_rest_only']),10)
        self.assertFalse(report['claims']['causal_mechanism_identified'])
        self.assertFalse(report['claims']['independent_replication'])

    def test_analysis_rejects_missing_extra_duplicate_bad_raw_records(self):
        with self.assertRaises(ValueError):self.a.build_report({})
        with self.assertRaises(ValueError):self.a.build_report({**self.inputs,'unexpected':b''})
        name=self.a.INPUT_PATHS[0];rows=[json.loads(line) for line in self.inputs[name].splitlines()]
        for changed in (rows[:-1],rows+rows[-1:],[{**rows[0],'generated_tokens':[True,257]},*rows[1:]],
                        [{**rows[0],'example_identity':'sha256:'+'0'*64},*rows[1:]]):
            data={**self.inputs,name:b''.join((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode() for r in changed)}
            with self.assertRaises(ValueError):self.a.build_report(data)


if __name__=='__main__':unittest.main()
