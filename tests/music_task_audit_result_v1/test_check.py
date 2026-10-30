"""Read-only preservation and adversarial scoring checks; no target model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from music_task_audit_result_v1 import check

PRED = 'music-task-audit.json'
REPORT = 'cuda-availability.json'


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                       allow_nan=False) + '\n').encode()


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Path(self.tmp.name) / 'fixture'
        shutil.copytree(check.FIXTURE, self.fixture)


    def test_every_fixture_member_pinned_before_scoring(self):
        for name in check.FIXTURE_HASHES:
            with self.subTest(name=name):
                path = self.fixture/name; raw = path.read_bytes(); path.write_bytes(raw+b' ')
                with mock.patch.object(check, 'score', side_effect=AssertionError('must not score')):
                    with self.assertRaisesRegex(ValueError, 'bytes changed'): check.verify(self.fixture)
                path.write_bytes(raw)


    def test_missing_extra_nested_members(self):
        for mode in ('missing', 'extra', 'deep'):
            with self.subTest(mode=mode):
                if mode == 'missing': (self.fixture/'README.md').rename(self.fixture/'renamed')
                elif mode == 'extra': (self.fixture/'extra').write_text('x')
                else: (self.fixture/'deep').mkdir()
                with self.assertRaisesRegex(ValueError, 'membership'): check.verify(self.fixture)
                if mode == 'missing': (self.fixture/'renamed').rename(self.fixture/'README.md')
                elif mode == 'extra': (self.fixture/'extra').unlink()
                else: (self.fixture/'deep').rmdir()


    def test_oversized_and_nonregular(self):
        path = self.fixture/PRED; path.write_bytes(b'x'*(check.LIMIT+1))
        with self.assertRaisesRegex(ValueError, 'bounded'): check.verify(self.fixture)
        path.unlink(); path.mkdir()
        with self.assertRaises(ValueError): check.verify(self.fixture)




    def test_source_tamper_before_import(self):
        original=check.read
        def read(path):
            raw=original(path)
            return raw+b' ' if str(path).endswith('/analysis.py') else raw
        with mock.patch.object(check,'read',side_effect=read):
            with self.assertRaisesRegex(ValueError,'source changed'):check.verify(self.fixture)

    def source_copy(self):
        root=Path(self.tmp.name)/'checkout'
        check.sources()
        from music_task_audit_v1 import check as original_audit
        for relative in {*check.SOURCES, *check.DIAGNOSTIC_SOURCES, *original_audit.SOURCES, *original_audit.INPUT_HASHES}:
            target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(check.ROOT/relative,target)
        return root

    def test_package_shadow_cache_and_foreign_module(self):
        root=self.source_copy(); shadow=root/'scripts/music_task_audit_v1/analysis';shadow.mkdir()
        marker=Path(self.tmp.name)/'executed'
        (shadow/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.dict(sys.modules), mock.patch.object(check,'ROOT',root):
            for name in list(sys.modules):
                if name=='music_task_audit_v1' or name.startswith('music_task_audit_v1.'):
                    del sys.modules[name]
            with self.assertRaisesRegex(ValueError,'shadow'):check.sources()
        self.assertFalse(marker.exists());shutil.rmtree(shadow)
        (root/'scripts/music_task_audit_v1/__pycache__').mkdir()
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'cache'):check.sources()
        with mock.patch.dict(sys.modules,{'music_task_audit_v1.analysis':types.SimpleNamespace(__file__='/foreign.py')}):
            with self.assertRaisesRegex(ValueError,'foreign'):check.sources()

    def test_foreign_resolution_rejected_without_execution(self):
        p=Path(self.tmp.name)/'music_task_audit_v1';p.mkdir();marker=p/'executed'
        (p/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name=='music_task_audit_v1' or name.startswith('music_task_audit_v1.'):del sys.modules[name]
            with mock.patch.object(sys,'path',[self.tmp.name,*sys.path]):
                with self.assertRaisesRegex(ValueError,'resolution'):check.sources()
        self.assertFalse(marker.exists())

    def test_fixture_changes_during_verification(self):
        original=check.score
        def score(before):
            result=original(before);(self.fixture/'README.md').write_text('changed');return result
        with mock.patch.object(check,'score',side_effect=score):
            with self.assertRaisesRegex(ValueError,'during verification'):check.verify(self.fixture)



    def test_launcher_requires_flags_before_pathlib(self):
        root=Path(self.tmp.name)/'launch';root.mkdir()
        script=root/'check.py';shutil.copyfile(check.ROOT/'scripts/check_music_task_audit_result.py',script)
        marker=root/'executed'
        (root/'pathlib.py').write_text('open('+repr(str(marker))+',"w").write("x")\n')
        for flags in ([],['-B'],['-I']):
            result=subprocess.run([sys.executable,*flags,str(script)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Use python -I -B',result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((root/'__pycache__').exists())

    def test_launcher_rejects_own_package_and_check_shadows_before_execution(self):
        for mode in ('src-package', 'check-directory'):
            with self.subTest(mode=mode):
                root=Path(self.tmp.name)/mode
                (root/'src').mkdir(parents=True)
                package=root/'scripts/music_task_audit_result_v1'
                package.mkdir(parents=True)
                shutil.copyfile(check.ROOT/'scripts/check_music_task_audit_result.py', root/'scripts/check.py')
                shutil.copyfile(check.ROOT/'scripts/music_task_audit_result_v1/__init__.py', package/'__init__.py')
                (package/'check.py').write_text('raise RuntimeError("legitimate verifier must not execute")\n')
                marker=root/'executed'
                shadow=(root/'src/music_task_audit_result_v1') if mode=='src-package' else package/'check'
                shadow.mkdir()
                (shadow/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
                result=subprocess.run([sys.executable,'-I','-B',str(root/'scripts/check.py')],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('foreign verifier',result.stderr)
                self.assertFalse(marker.exists())
                self.assertFalse(any(root.rglob('__pycache__')))

    def test_launcher_rejects_linked_own_package_root(self):
        root=Path(self.tmp.name)/'linked-launch'
        (root/'scripts').mkdir(parents=True)
        shutil.copyfile(check.ROOT/'scripts/check_music_task_audit_result.py',root/'scripts/check.py')
        (root/'scripts/music_task_audit_result_v1').symlink_to(check.ROOT/'scripts/music_task_audit_result_v1',target_is_directory=True)
        result=subprocess.run([sys.executable,'-I','-B',str(root/'scripts/check.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('linked verifier package',result.stderr)





    def test_launcher_preloads_stdlib_before_project_path_shadows(self):
        root = self.source_copy()
        package = root/'scripts/music_task_audit_result_v1'
        shutil.copytree(check.ROOT/'scripts/music_task_audit_result_v1', package)
        script = root/'scripts/check_music_task_audit_result.py'
        shutil.copyfile(check.ROOT/'scripts/check_music_task_audit_result.py', script)
        shutil.copytree(check.FIXTURE, root/'fixtures/music-task-audit-result-v1')
        marker = root/'executed'
        for name in ('hashlib', 'json', 'subprocess'):
            (root/'src'/f'{name}.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise RuntimeError("shadow executed")\n')
        result = subprocess.run([sys.executable, '-I', '-B', str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['verified'])
        self.assertFalse(marker.exists())
        self.assertFalse(any(root.rglob('__pycache__')))


    def test_launcher_rejects_run_and_alternate_fixture_selectors(self):
        script = check.ROOT/'scripts/check_music_task_audit_result.py'
        for arguments in (['run'], ['--fixture', str(self.fixture)], ['--authorize-run']):
            result = subprocess.run([sys.executable, '-I', '-B', str(script), *arguments],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('takes no arguments', result.stderr)














    def test_symlink_root_ancestor_leaf_and_fifo_rejected(self):
        alias=Path(self.tmp.name)/'alias';alias.symlink_to(self.fixture,target_is_directory=True)
        with self.assertRaises(ValueError):check.verify(alias)
        with self.assertRaises(ValueError):check.read(alias/PRED)
        path=self.fixture/PRED;path.unlink();path.symlink_to(check.FIXTURE/PRED)
        with self.assertRaises(ValueError):check.verify(self.fixture)
        path.unlink();os.mkfifo(path)
        with self.assertRaises(ValueError):check.read(path)

    def test_resealed_original_and_manifest_cannot_replace_receipt(self):
        before=check.snapshot(self.fixture)
        wrapper=json.loads(before[REPORT]);wrapper['report']['hardware_attested']=True
        wrapper['report_identity']=check.digest(encoded(wrapper['report']))
        raw=encoded(wrapper);(self.fixture/REPORT).write_bytes(raw)
        manifest=self.fixture/'source-SHA256SUMS'
        manifest.write_text(manifest.read_text().replace(check.ORIGINAL_HASHES[REPORT][7:],check.digest(raw)[7:]))
        with mock.patch.object(check,'score',side_effect=AssertionError('must reject before score')):
            with self.assertRaises(ValueError):check.verify(self.fixture)

    def test_full_audit_semantic_change_rejected_even_with_resealed_hash(self):
        before=check.snapshot(self.fixture);old=before[PRED];obj=json.loads(old)
        obj['analysis']['placement']['unique_prompts']=80;before[PRED]=encoded(obj)
        before['source-SHA256SUMS']=before['source-SHA256SUMS'].replace(check.digest(old)[7:].encode(),check.digest(before[PRED])[7:].encode())
        with self.assertRaisesRegex(ValueError,'complete offline audit report differs'):check.score(before)


    def test_cuda_report_identity_schema_and_false_claims(self):
        original=check.snapshot(self.fixture)
        mutations=(('schema','unknown'),('hardware_attested',True),('gpu_health_established',True),
                   ('trial_admission_established',True),('training_performed',True),
                   ('model_executed',True),('repair_attempted',True),('root_cause','malware'),
                   ('tensor_allocation_requested',True),('unexpected',True))
        for key,value in mutations:
            with self.subTest(key=key):
                before=dict(original);wrapper=json.loads(before[REPORT]);wrapper['report'][key]=value
                wrapper['report_identity']=check.digest(encoded(wrapper['report']));before[REPORT]=encoded(wrapper)
                with self.assertRaisesRegex(ValueError,'CUDA'):check.validate_cuda(wrapper)
        before=dict(original);wrapper=json.loads(before[REPORT]);wrapper['report_identity']='sha256:'+'0'*64
        before[REPORT]=encoded(wrapper)
        with self.assertRaises(ValueError):check.score(before)
        before=dict(original);before[REPORT]=before[REPORT].replace(b'"platform":"linux"',b'"platform":"linux","platform":"linux"')
        with self.assertRaises(ValueError):check.score(before)

    def test_cuda_source_status_and_skipped_utility_not_malware(self):
        original=check.snapshot(self.fixture)
        for section,field,value in (('nvidia_smi','returncode',0),('nvidia_smi','status','completed'),
                                    ('nvidia_smi','malware_detected',True),('cuda_query','available',True),
                                    ('cuda_query','device_count',1),('cuda_query','status','visible_cuda')):
            with self.subTest(section=section,field=field):
                before=dict(original);wrapper=json.loads(before[REPORT]);wrapper['report'][section][field]=value
                wrapper['report_identity']=check.digest(encoded(wrapper['report']));before[REPORT]=encoded(wrapper)
                with self.assertRaisesRegex(ValueError,'CUDA'):check.validate_cuda(wrapper)
        before=dict(original);wrapper=json.loads(before[REPORT]);name=next(iter(wrapper['report']['source_files']))
        wrapper['report']['source_files'][name]='sha256:'+'0'*64
        wrapper['report_identity']=check.digest(encoded(wrapper['report']));before[REPORT]=encoded(wrapper)
        with self.assertRaises(ValueError):check.score(before)

    def test_exit_receipts_are_exact_zero_bytes_not_execution_attestation(self):
        for name in ('music-task-audit.exit','cuda-availability.exit'):
            for value in (b'1\n',b'00\n',b'0',b'0\nextra\n'):
                before=check.snapshot(self.fixture);before[name]=value
                with self.assertRaises(ValueError):check.score(before)

    def test_diagnostic_source_authenticated_but_never_executed(self):
        original=check.read
        for relative in check.DIAGNOSTIC_SOURCES:
            def read(path):
                raw=original(path)
                return raw+b' ' if Path(path)==check.ROOT/relative else raw
            with mock.patch.object(check,'read',side_effect=read):
                with self.assertRaises(ValueError):check.verify(self.fixture)

    def test_verified_readonly_report_reproduction_and_limits(self):
        before=check.snapshot(self.fixture);result=check.verify(self.fixture)
        self.assertEqual(result['schema'],'trinite.music-task-audit-retained-check.v1')
        self.assertIs(result['complete_offline_report_reproduced'],True)
        self.assertEqual(result['original_audit_exit_receipt'],0)
        self.assertEqual(result['cuda_diagnostic_exit_receipt'],0)
        self.assertEqual(result['retained_record_occurrences'],3520)
        self.assertEqual(result['retained_sections'],88)
        self.assertEqual(result['structural_development_parents'],2)
        self.assertEqual(result['nvidia_smi_status'],'unsafe_executable')
        self.assertIs(result['nvidia_smi_executed_in_supplied_receipt'],False)
        self.assertEqual(result['cuda_query_status'],'no_visible_cuda')
        self.assertIs(result['cuda_available'],False)
        self.assertEqual(result['cuda_device_count'],0)
        self.assertIs(result['cuda_query_stderr_observed'],True)
        for key in ('actual_execution_attested','hardware_attested','runtime_attested','inference_replayed',
                    'training_performed','causal_mechanism_established','absence_of_other_attempts_attested',
                    'full_original_custody_verified'):
            self.assertIs(result[key],False)
        self.assertEqual(check.snapshot(self.fixture),before)
        for module in ('torch','numpy','trinite.model','trinite.training','cuda_availability_v1',
                       'cuda_availability_v1.check','music_device_trial_v2.runtime'):
            self.assertNotIn(module,sys.modules)


    def test_provenance_cannot_promote_observations_even_if_repinned(self):
        path=self.fixture/'provenance-DERIVED.json';original=path.read_bytes()
        for claim in ('actual_invocation_attested','gpu_health_established','gpu_root_cause_established',
                      'hardware_attested','runtime_replayed','historical_namespace_equivalence_attested'):
            with self.subTest(claim=claim):
                value=json.loads(original);value['claims'][claim]=True;path.write_bytes(encoded(value))
                pins={**check.FIXTURE_HASHES,'provenance-DERIVED.json':check.digest(path.read_bytes())}
                with mock.patch.object(check,'FIXTURE_HASHES',pins):
                    with self.assertRaisesRegex(ValueError,'provenance'):check.verify(self.fixture)
                path.write_bytes(original)


    def test_manifest_members_and_omitted_hashes_even_if_repinned(self):
        original=check.snapshot(self.fixture)
        lines=original['source-SHA256SUMS'].decode('ascii').splitlines()
        omitted=next(name for name in check.OMITTED_MANIFEST_HASHES)
        omitted_index=next(i for i,line in enumerate(lines) if line.endswith('  '+omitted))
        bad_omitted=list(lines);bad_omitted[omitted_index]='0'*64+'  '+omitted
        upper=list(lines);upper[0]=upper[0][:64].upper()+upper[0][64:]
        malformed=list(lines);malformed[0]='z'*64+malformed[0][64:]
        for changed in (lines[:-1],lines+['0'*64+'  extra.txt'],upper,malformed,bad_omitted):
            with self.subTest(first=changed[0]):
                before=dict(original);before['source-SHA256SUMS']=('\n'.join(changed)+'\n').encode('ascii')
                pins={**check.ORIGINAL_HASHES,'source-SHA256SUMS':check.digest(before['source-SHA256SUMS'])}
                with mock.patch.object(check,'ORIGINAL_HASHES',pins):
                    with self.assertRaisesRegex(ValueError,'manifest|commitment'):check.score(before)


if __name__=='__main__': unittest.main()
