"""Read-only preservation and adversarial scoring checks; no target model."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from music_robustness_result_v1 import check

PRED = 'cpu/development/predictions.jsonl'
REPORT = 'cpu/matrix.json'


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

    def test_resealed_report_manifest_rejected_before_scoring(self):
        report = json.loads((self.fixture/REPORT).read_bytes()); report['outcome'] = 'incomplete'
        raw = encoded(report); (self.fixture/REPORT).write_bytes(raw)
        manifest = self.fixture/'source-SHA256SUMS'
        manifest.write_text(manifest.read_text().replace(check.ORIGINAL_HASHES[REPORT][7:], check.digest(raw)[7:]))
        with mock.patch.object(check, 'score', side_effect=AssertionError('must not score')):
            with self.assertRaisesRegex(ValueError, 'bytes changed'): check.verify(self.fixture)

    def test_missing_extra_nested_members(self):
        for mode in ('missing', 'extra', 'deep'):
            with self.subTest(mode=mode):
                if mode == 'missing': (self.fixture/'README.md').rename(self.fixture/'renamed')
                elif mode == 'extra': (self.fixture/'extra').write_text('x')
                else: (self.fixture/'cpu/development/deep').mkdir()
                with self.assertRaisesRegex(ValueError, 'membership'): check.verify(self.fixture)
                if mode == 'missing': (self.fixture/'renamed').rename(self.fixture/'README.md')
                elif mode == 'extra': (self.fixture/'extra').unlink()
                else: (self.fixture/'cpu/development/deep').rmdir()

    def test_symlink_root_ancestor_leaf_and_directory(self):
        alias = Path(self.tmp.name)/'alias'; alias.symlink_to(self.fixture, target_is_directory=True)
        with self.assertRaises(ValueError): check.verify(alias)
        with self.assertRaises(ValueError): check.read(alias/PRED)
        path = self.fixture/PRED; path.unlink(); path.symlink_to(check.FIXTURE/PRED)
        with self.assertRaises(ValueError): check.verify(self.fixture)
        path.unlink(); shutil.copyfile(check.FIXTURE/PRED, path)
        shutil.rmtree(self.fixture/'cpu')
        (self.fixture/'cpu').symlink_to(check.FIXTURE/'cpu', target_is_directory=True)
        with self.assertRaises(ValueError): check.verify(self.fixture)

    def test_oversized_and_nonregular(self):
        path = self.fixture/PRED; path.write_bytes(b'x'*(check.LIMIT+1))
        with self.assertRaisesRegex(ValueError, 'bounded'): check.verify(self.fixture)
        path.unlink(); path.mkdir()
        with self.assertRaises(ValueError): check.verify(self.fixture)

    def test_invalid_journal_order_coverage_tokens_and_fields(self):
        original = check.snapshot(self.fixture)
        rows = [json.loads(line) for line in original[PRED].splitlines()]
        variants = [rows[:-1], rows+[rows[-1]], [rows[1], rows[0], *rows[2:]],
                    [{**rows[0], 'seed': 29}, *rows[1:]],
                    [{**rows[0], 'generated_tokens': [True,257]}, *rows[1:]],
                    [{**rows[0], 'extra': 1}, *rows[1:]],
                    [{**rows[0], 'generated_tokens': [55]}, *rows[1:]]]
        for changed in variants:
            with self.subTest(first=changed[0]):
                before = dict(original, **{PRED: b''.join(encoded(r) for r in changed)})
                with self.assertRaises(ValueError): check.score(before)

    def test_noncanonical_duplicate_json_rejected(self):
        before = check.snapshot(self.fixture)
        before[PRED] = before[PRED].replace(b'"seed":17', b'"seed":17,"seed":17', 1)
        with self.assertRaises(ValueError): check.score(before)


    def test_original_manifest_link_rejected(self):
        before=check.snapshot(self.fixture)
        before['source-SHA256SUMS']=b'0'*64+b'  '+PRED.encode()+b'\n'
        with self.assertRaisesRegex(ValueError,'manifest'):check.score(before)


    def test_source_tamper_before_import(self):
        original=check.read
        def read(path):
            raw=original(path)
            return raw+b' ' if str(path).endswith('/spec.py') else raw
        with mock.patch.object(check,'read',side_effect=read):
            with self.assertRaisesRegex(ValueError,'source changed'):check.verify(self.fixture)

    def source_copy(self):
        root=Path(self.tmp.name)/'checkout'
        for relative in {*check.SOURCES, *check.REFERENCE_SOURCES}:
            target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(check.ROOT/relative,target)
        return root

    def test_package_shadow_cache_and_foreign_module(self):
        root=self.source_copy(); shadow=root/'scripts/music_placement_v1/spec';shadow.mkdir()
        marker=Path(self.tmp.name)/'executed'
        (shadow/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'shadow'):check.sources()
        self.assertFalse(marker.exists());shutil.rmtree(shadow)
        (root/'src/trinite/__pycache__').mkdir()
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'cache'):check.sources()
        with mock.patch.dict(sys.modules,{'music_placement_v1.spec':types.SimpleNamespace(__file__='/foreign.py')}):
            with self.assertRaisesRegex(ValueError,'foreign'):check.sources()

    def test_foreign_resolution_rejected_without_execution(self):
        p=Path(self.tmp.name)/'trinite';p.mkdir();marker=p/'executed'
        (p/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name=='trinite' or name.startswith('trinite.'):del sys.modules[name]
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
        script=root/'check.py';shutil.copyfile(check.ROOT/'scripts/check_music_robustness_result.py',script)
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
                package=root/'scripts/music_robustness_result_v1'
                package.mkdir(parents=True)
                shutil.copyfile(check.ROOT/'scripts/check_music_robustness_result.py', root/'scripts/check.py')
                shutil.copyfile(check.ROOT/'scripts/music_robustness_result_v1/__init__.py', package/'__init__.py')
                (package/'check.py').write_text('raise RuntimeError("legitimate verifier must not execute")\n')
                marker=root/'executed'
                shadow=(root/'src/music_robustness_result_v1') if mode=='src-package' else package/'check'
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
        shutil.copyfile(check.ROOT/'scripts/check_music_robustness_result.py',root/'scripts/check.py')
        (root/'scripts/music_robustness_result_v1').symlink_to(check.ROOT/'scripts/music_robustness_result_v1',target_is_directory=True)
        result=subprocess.run([sys.executable,'-I','-B',str(root/'scripts/check.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('linked verifier package',result.stderr)



    def test_resealed_semantic_gate_attempt_claim_and_score_rejected(self):
        cases = [('cpu/control-17/report.json', 'step', 64),
                 ('cpu/control-17/report.json', 'target_tokens', 10241),
                 ('cpu/control-17/report.json', 'initialization_identity', 'sha256:'+'0'*64),
                 ('cpu/control-17/attempt.json', 'outcome', 'failed'),
                 ('cpu/control-17/completion.json', 'cell', 'swap-17'),
                 ('cpu/development-gate.json', 'selection', 'best seed'),
                 (REPORT, 'fresh_confirmation', True),
                 (REPORT, 'summary', {}),
                 ('cpu/development/report.json', 'input_files_unchanged', False),
                 ('cpu/audit.stdout', 'verified', 1),
                 ('cpu/audit.stdout', 'optimizer_replayed', True)]
        for name, key, value in cases:
            with self.subTest(name=name, key=key):
                before = check.snapshot(self.fixture)
                obj = json.loads(before[name]); obj[key] = value
                before[name] = encoded(obj)
                with self.assertRaises(ValueError): check.score(before)

    def test_every_diagnostic_cell_and_all_four_development_cells_required(self):
        for cell in ('control-17', 'swap-17', 'swap-29', 'control-29'):
            key = f'cpu/{cell}/diagnostic-predictions.jsonl'
            before = check.snapshot(self.fixture)
            rows = [json.loads(line) for line in before[key].splitlines()]
            for changed in (rows[:-1], rows[40:]+rows[:40],
                            [{**rows[0], 'arm': 'other'}, *rows[1:]]):
                with self.subTest(cell=cell):
                    modified = dict(before, **{key: b''.join(encoded(r) for r in changed)})
                    with self.assertRaises(ValueError): check.score(modified)
        before = check.snapshot(self.fixture)
        rows = [json.loads(line) for line in before[PRED].splitlines()]
        for changed in (rows[:360], rows[120:]+rows[:120]):
            with self.assertRaises(ValueError):
                check.score(dict(before, **{PRED: b''.join(encoded(r) for r in changed)}))



    def test_launcher_preloads_stdlib_before_project_path_shadows(self):
        root = self.source_copy()
        package = root/'scripts/music_robustness_result_v1'
        shutil.copytree(check.ROOT/'scripts/music_robustness_result_v1', package)
        script = root/'scripts/check_music_robustness_result.py'
        shutil.copyfile(check.ROOT/'scripts/check_music_robustness_result.py', script)
        shutil.copytree(check.FIXTURE, root/'fixtures/music-robustness-result-v1')
        for relative in check.HISTORICAL_HASHES:
            target = root/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(check.ROOT/relative, target)
        marker = root/'executed'
        for name in ('hashlib', 'json', 'subprocess'):
            (root/'src'/f'{name}.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise RuntimeError("shadow executed")\n')
        result = subprocess.run([sys.executable, '-I', '-B', str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['verified'])
        self.assertFalse(marker.exists())
        self.assertFalse(any(root.rglob('__pycache__')))


    def test_launcher_rejects_run_and_alternate_fixture_selectors(self):
        script = check.ROOT/'scripts/check_music_robustness_result.py'
        for arguments in (['run'], ['--fixture', str(self.fixture)], ['--authorize-run']):
            result = subprocess.run([sys.executable, '-I', '-B', str(script), *arguments],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('takes no arguments', result.stderr)








    def test_verified_cpu_repeat_is_not_completed_gpu_study(self):
        before = check.snapshot(self.fixture)
        result = check.verify(self.fixture)
        self.assertTrue(result['verified'])
        self.assertEqual(result['overall_outcome'], 'incomplete')
        self.assertEqual(result['cpu_outcome_in_supplied_report'], 'completed')
        self.assertEqual(result['device_error_in_supplied_stderr'],
                         {'outcome': 'unsupported', 'type': 'UnsupportedRuntime'})
        self.assertEqual(result['decodes_in_journal'], 800)
        self.assertEqual(result['generated_tokens'], 1600)
        self.assertEqual(result['available_matrix_bound_files'], 21)
        self.assertEqual(result['omitted_matrix_bound_files'], 22)
        self.assertEqual(result['historical_raw_records_equal'], 800)
        self.assertIs(result['historical_summary_equal'], True)
        self.assertEqual(result['canonical_exact_by_cell'],
                         {'control-17': 16, 'swap-17': 16, 'swap-29': 16, 'control-29': 15})
        for key in ('actual_execution_attested', 'hardware_attested', 'timing_attested',
                    'device_invocation_attested', 'device_error_source_bound', 'allocator_fix_validated',
                    'partial_custody_failure_path_validated',
                    'other_attempts_absence_attested', 'inference_replayed', 'optimizer_replayed',
                    'fresh_confirmation', 'original_raw_custody_verified'):
            self.assertIs(result[key], False)
        for key in ('timing_comparison', 'gpu_score', 'device_canonical_exact_by_backend',
                    'allocator_limit_bytes', 'allocator_effective_bytes', 'recorded_probe_checks'):
            self.assertNotIn(key, result)
        self.assertEqual(check.snapshot(self.fixture), before)
        for module in ('torch', 'numpy', 'trinite.training', 'trinite.model',
                       'music_placement_v1.numerical', 'music_placement_v2.numerical',
                       'music_device_trial_v2.runtime', 'music_device_trial_v2.numerical'):
            self.assertNotIn(module, sys.modules)

    def test_standalone_device_error_does_not_admit_new_claims(self):
        original = check.snapshot(self.fixture)
        name = 'device/freeze.stderr'
        for key, value in (('outcome', 'completed'), ('type', 'ValueError'),
                           ('detail', 'allocator validated'), ('request_identity', 'sha256:'+'0'*64),
                           ('allocator_effective_bytes', 8589934592), ('gpu_score', 40)):
            with self.subTest(key=key):
                before = dict(original); obj = json.loads(before[name]); obj[key] = value
                before[name] = encoded(obj)
                with self.assertRaises(ValueError): check.score(before)
        for name in ('device/request.json', 'device/matrix.json', 'device/probe.json', 'device/audit.stdout'):
            path = self.fixture/name; path.write_bytes(b'{}\n')
            with self.assertRaises(ValueError): check.verify(self.fixture)
            path.unlink()

    def test_historical_journals_and_matrix_are_independently_pinned(self):
        original = check.read
        for relative in check.HISTORICAL_HASHES:
            with self.subTest(relative=relative):
                def read(path):
                    raw = original(path)
                    return raw+b' ' if Path(path) == check.ROOT/relative else raw
                with mock.patch.object(check, 'read', side_effect=read):
                    with self.assertRaises(ValueError): check.verify(self.fixture)

    def test_scientific_denominators_and_repeat_summary(self):
        summary = check.score(check.snapshot(self.fixture))
        self.assertEqual(summary['decodes'], 800)
        self.assertEqual(summary['decode_forwards'], 1600)
        self.assertEqual(summary['development']['structural_parent_count'], 2)
        for cell, value in summary['diagnostic']['by_cell'].items():
            self.assertEqual(value['rendering_rows'], 80)
            self.assertEqual(value['unique_prompt_surfaces'], 70)
            self.assertEqual(value['raw']['exact_plus_eos'], 80)
            strata = value['transformation_pairs']['swap']['by_input_change']
            self.assertEqual(strata['changed_input'], {'pairs': 30, 'identical_raw_outputs': 30, 'both_correct': 30})
            self.assertEqual(strata['identical_input'], {'pairs': 10, 'identical_raw_outputs': 10, 'both_correct': 10})
        for value in summary['development']['by_cell'].values():
            self.assertEqual(value['canonical_controls']['global']['exact_plus_eos'], 20)
            self.assertEqual(value['canonical_strata']['variant_id']['10']['exact_plus_eos'], 0)
            for view in ('canonical', 'pitch', 'tempo'):
                self.assertEqual(value['by_sibling'][view]['groups'], 10)
                self.assertEqual(value['by_sibling'][view]['complete_groups'], 0)

    def test_all_available_completion_links_required(self):
        original = check.snapshot(self.fixture)
        for name in original:
            if not name.endswith('completion.json'):
                continue
            with self.subTest(name=name):
                before = dict(original); obj = json.loads(before[name])
                field = next(iter(obj['artifacts']))
                obj['artifacts'][field] = 'sha256:' + '0' * 64
                before[name] = encoded(obj)
                with self.assertRaises(ValueError): check.score(before)


    def test_provenance_cannot_upgrade_unsupported_even_if_repinned(self):
        path = self.fixture/'provenance-DERIVED.json'; original = path.read_bytes()
        for key, value in (('actual_execution_attested', True), ('full_run_custody', 1),
                           ('hardware_attested', True), ('timing_attested', True),
                           ('allocator_policy_validated', True),
                           ('device_performance_comparison_available', True),
                           ('overall_outcome_in_owner_handoff', 'completed'),
                           ('device_outcome_in_supplied_stderr', 'completed'),
                           ('device_receipt_scope', 'source-bound request'),
                           ('producer_commit', '0'*40), ('unexpected_claim', True)):
            with self.subTest(key=key):
                obj = json.loads(original); obj[key] = value; path.write_bytes(encoded(obj))
                pins = {**check.FIXTURE_HASHES, 'provenance-DERIVED.json': check.digest(path.read_bytes())}
                with mock.patch.object(check, 'FIXTURE_HASHES', pins):
                    with self.assertRaises(ValueError): check.verify(self.fixture)
                path.write_bytes(original)
        obj = json.loads(original); obj['cpu']['matrix_bound_omitted'] = []
        path.write_bytes(encoded(obj))
        pins = {**check.FIXTURE_HASHES, 'provenance-DERIVED.json': check.digest(path.read_bytes())}
        with mock.patch.object(check, 'FIXTURE_HASHES', pins):
            with self.assertRaises(ValueError): check.verify(self.fixture)


if __name__=='__main__': unittest.main()
