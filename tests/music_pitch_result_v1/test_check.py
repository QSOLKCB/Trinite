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

from music_pitch_result_v1 import check

PRED = 'original-run/development/predictions.jsonl'
REPORT = 'original-run/matrix.json'


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
                else: (self.fixture/'original-run/development/deep').mkdir()
                with self.assertRaisesRegex(ValueError, 'membership'): check.verify(self.fixture)
                if mode == 'missing': (self.fixture/'renamed').rename(self.fixture/'README.md')
                elif mode == 'extra': (self.fixture/'extra').unlink()
                else: (self.fixture/'original-run/development/deep').rmdir()

    def test_symlink_root_ancestor_leaf_and_directory(self):
        alias = Path(self.tmp.name)/'alias'; alias.symlink_to(self.fixture, target_is_directory=True)
        with self.assertRaises(ValueError): check.verify(alias)
        with self.assertRaises(ValueError): check.read(alias/PRED)
        path = self.fixture/PRED; path.unlink(); path.symlink_to(check.FIXTURE/PRED)
        with self.assertRaises(ValueError): check.verify(self.fixture)
        path.unlink(); shutil.copyfile(check.FIXTURE/PRED, path)
        shutil.rmtree(self.fixture/'original-run')
        (self.fixture/'original-run').symlink_to(check.FIXTURE/'original-run', target_is_directory=True)
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
        for relative in check.SOURCES:
            target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(check.ROOT/relative,target)
        return root

    def test_package_shadow_cache_and_foreign_module(self):
        root=self.source_copy(); shadow=root/'scripts/music_pitch_intervention_v1/spec';shadow.mkdir()
        marker=Path(self.tmp.name)/'executed'
        (shadow/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'shadow'):check.sources()
        self.assertFalse(marker.exists());shutil.rmtree(shadow)
        (root/'src/trinite/__pycache__').mkdir()
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'cache'):check.sources()
        with mock.patch.dict(sys.modules,{'music_pitch_intervention_v1.spec':types.SimpleNamespace(__file__='/foreign.py')}):
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
        script=root/'check.py';shutil.copyfile(check.ROOT/'scripts/check_music_pitch_result.py',script)
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
                package=root/'scripts/music_pitch_result_v1'
                package.mkdir(parents=True)
                shutil.copyfile(check.ROOT/'scripts/check_music_pitch_result.py', root/'scripts/check.py')
                shutil.copyfile(check.ROOT/'scripts/music_pitch_result_v1/__init__.py', package/'__init__.py')
                (package/'check.py').write_text('raise RuntimeError("legitimate verifier must not execute")\n')
                marker=root/'executed'
                shadow=(root/'src/music_pitch_result_v1') if mode=='src-package' else package/'check'
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
        shutil.copyfile(check.ROOT/'scripts/check_music_pitch_result.py',root/'scripts/check.py')
        (root/'scripts/music_pitch_result_v1').symlink_to(check.ROOT/'scripts/music_pitch_result_v1',target_is_directory=True)
        result=subprocess.run([sys.executable,'-I','-B',str(root/'scripts/check.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('linked verifier package',result.stderr)


    def test_verified_read_only_no_native_imports(self):
        before = check.snapshot(self.fixture)
        result = check.verify(self.fixture)
        self.assertTrue(result['verified'])
        self.assertEqual(result['canonical_exact_by_cell'],
                         {'control-17': 16, 'pitch-17': 14, 'pitch-29': 15, 'control-29': 15})
        self.assertEqual(result['decodes_in_journal'], 800)
        self.assertEqual(result['generated_tokens'], 1600)
        self.assertEqual(result['available_matrix_bound_files'], 21)
        self.assertEqual(result['omitted_matrix_bound_files'], 22)
        for key in ('actual_execution_attested', 'original_raw_custody_verified',
                    'inference_replayed', 'optimizer_replayed', 'fresh_confirmation',
                    'learning_adequate', 'release_ready'):
            self.assertIs(result[key], False)
        self.assertEqual(check.snapshot(self.fixture), before)
        for module in ('torch', 'trinite.training', 'trinite.model',
                       'music_pitch_intervention_v1.numerical',
                       'music_pitch_intervention_v1.controller',
                       'music_pitch_intervention_v1.workflow'):
            self.assertNotIn(module, sys.modules)

    def test_resealed_semantic_gate_attempt_claim_and_score_rejected(self):
        cases = [('original-run/control-17/report.json', 'step', 64),
                 ('original-run/control-17/report.json', 'target_tokens', 10241),
                 ('original-run/control-17/report.json', 'initialization_identity', 'sha256:'+'0'*64),
                 ('original-run/control-17/attempt.json', 'outcome', 'failed'),
                 ('original-run/control-17/completion.json', 'cell', 'pitch-17'),
                 ('original-run/development-gate.json', 'selection', 'best seed'),
                 (REPORT, 'fresh_confirmation', True),
                 (REPORT, 'summary', {}),
                 ('original-run/development/report.json', 'input_files_unchanged', False),
                 ('original-run/audit.stdout', 'verified', 1),
                 ('original-run/audit.stdout', 'optimizer_replayed', True)]
        for name, key, value in cases:
            with self.subTest(name=name, key=key):
                before = check.snapshot(self.fixture)
                obj = json.loads(before[name]); obj[key] = value
                before[name] = encoded(obj)
                with self.assertRaises(ValueError): check.score(before)

    def test_every_diagnostic_cell_and_all_four_development_cells_required(self):
        for cell in ('control-17', 'pitch-17', 'pitch-29', 'control-29'):
            key = f'original-run/{cell}/diagnostic-predictions.jsonl'
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

    def test_declared_mixed_patterns_and_controls(self):
        summary = check.score(check.snapshot(self.fixture))
        self.assertEqual(summary['decodes'], 800)
        self.assertEqual(summary['decode_forwards'], 1600)
        for seed, canonical, pitch in [('17', 14, 14), ('29', 15, 15)]:
            treatment = summary['development']['by_cell']['pitch-'+seed]
            control = summary['development']['by_cell']['control-'+seed]
            self.assertEqual(treatment['canonical_primary']['exact_plus_eos'], canonical)
            self.assertEqual(treatment['canonical_controls']['global']['exact_plus_eos'], 20)
            self.assertEqual(treatment['by_sibling']['pitch']['exact_plus_eos'], pitch)
            self.assertEqual(treatment['transformation_pairs']['pitch']['identical_raw_outputs'], 40)
            self.assertEqual(treatment['transformation_pairs']['pitch']['both_correct'], canonical)
            self.assertEqual(control['transformation_pairs']['pitch']['identical_raw_outputs'], 4)
            for cell in (treatment, control):
                for view in ('canonical', 'pitch', 'tempo'):
                    self.assertEqual(cell['by_sibling'][view]['complete_groups'], 0)
                    self.assertEqual(cell['by_sibling'][view]['groups'], 10)
                self.assertEqual(cell['canonical_strata']['variant_id']['10']['exact_plus_eos'], 0)
            self.assertEqual(summary['historical_control_reproducibility']['by_seed'][seed]['all'],
                             {'rows': 120, 'identical_raw_outputs': 120})
        self.assertTrue(summary['paired_by_seed']['29']['lost_control_perfect_training_fit'])
        self.assertFalse(summary['paired_by_seed']['17']['lost_control_perfect_training_fit'])
        self.assertEqual(summary['diagnostic']['by_cell']['pitch-29']['canonical_primary']['exact_plus_eos'], 38)
        for cell in ('pitch-17', 'pitch-29'):
            self.assertEqual(summary['diagnostic']['by_cell'][cell]['by_sibling']['pitch']['exact_plus_eos'], 40)

    def test_historical_output_tamper_rejected(self):
        original = check.read
        def read(path):
            raw = original(path)
            return raw+b' ' if 'frozen-music-eval-result-v1' in str(path) else raw
        with mock.patch.object(check, 'read', side_effect=read):
            with self.assertRaises(ValueError): check.verify(self.fixture)

    def test_launcher_preloads_stdlib_before_project_path_shadows(self):
        root = self.source_copy()
        package = root/'scripts/music_pitch_result_v1'
        shutil.copytree(check.ROOT/'scripts/music_pitch_result_v1', package)
        script = root/'scripts/check_music_pitch_result.py'
        shutil.copyfile(check.ROOT/'scripts/check_music_pitch_result.py', script)
        shutil.copytree(check.FIXTURE, root/'fixtures/music-pitch-result-v1')
        historical = 'fixtures/frozen-music-eval-result-v1/original-attempt/predictions.jsonl'
        target = root/historical; target.parent.mkdir(parents=True)
        shutil.copyfile(check.ROOT/historical, target)
        marker = root/'executed'
        for name in ('hashlib', 'json', 'subprocess'):
            (root/'src'/f'{name}.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise RuntimeError("shadow executed")\n')
        result = subprocess.run([sys.executable, '-I', '-B', str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['verified'])
        self.assertFalse(marker.exists())
        self.assertFalse(any(root.rglob('__pycache__')))

    def test_provenance_scope_and_missing_inventory_even_if_repinned(self):
        path = self.fixture/'provenance-DERIVED.json'
        original = path.read_bytes()
        for key, value in (('actual_execution_attested', True), ('full_run_custody', 1),
                           ('matrix_bound_omitted', []), ('inference_replayed', True),
                           ('producer_commit', '0'*40)):
            with self.subTest(key=key):
                obj = json.loads(original); obj[key] = value
                path.write_bytes(encoded(obj))
                pins = {**check.FIXTURE_HASHES, 'provenance-DERIVED.json': check.digest(path.read_bytes())}
                with mock.patch.object(check, 'FIXTURE_HASHES', pins):
                    with self.assertRaises(ValueError): check.verify(self.fixture)
                path.write_bytes(original)

    def test_launcher_rejects_run_and_alternate_fixture_selectors(self):
        script = check.ROOT/'scripts/check_music_pitch_result.py'
        for arguments in (['run'], ['--fixture', str(self.fixture)], ['--authorize-run']):
            result = subprocess.run([sys.executable, '-I', '-B', str(script), *arguments],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('takes no arguments', result.stderr)

if __name__=='__main__': unittest.main()
