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

from frozen_music_eval_result_v1 import check

PRED = 'original-attempt/predictions.jsonl'
REPORT = 'original-attempt/report.json'


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                       allow_nan=False) + '\n').encode()


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Path(self.tmp.name) / 'fixture'
        shutil.copytree(check.FIXTURE, self.fixture)

    def test_verified_read_only_no_native_imports(self):
        before = check.snapshot(self.fixture)
        result = check.verify(self.fixture)
        self.assertEqual(result['canonical_exact_per_seed'], {'17': 16, '29': 15})
        self.assertEqual(result['decodes_in_journal'], 320)
        self.assertEqual(result['generated_tokens'], 640)
        self.assertEqual(result['historical_training_predictions_equal'], 80)
        self.assertEqual(result['heldout_structural_parents'], 2)
        for key in ('actual_execution_attested', 'original_raw_custody_verified',
                    'inference_replayed', 'optimizer_replayed', 'fresh_confirmation',
                    'learning_adequate', 'release_ready'):
            self.assertIs(result[key], False)
        self.assertEqual(check.snapshot(self.fixture), before)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('trinite.training', sys.modules)
        self.assertNotIn('frozen_music_eval_v1.worker', sys.modules)

    def test_every_fixture_member_pinned_before_scoring(self):
        for name in check.FIXTURE_HASHES:
            with self.subTest(name=name):
                path = self.fixture/name; raw = path.read_bytes(); path.write_bytes(raw+b' ')
                with mock.patch.object(check, 'score', side_effect=AssertionError('must not score')):
                    with self.assertRaisesRegex(ValueError, 'bytes changed'): check.verify(self.fixture)
                path.write_bytes(raw)

    def test_resealed_report_manifest_rejected_before_scoring(self):
        report = json.loads((self.fixture/REPORT).read_bytes()); report['fresh_confirmation'] = True
        raw = encoded(report); (self.fixture/REPORT).write_bytes(raw)
        manifest = self.fixture/'original-SHA256SUMS.txt'
        manifest.write_text(manifest.read_text().replace(check.ORIGINAL_HASHES[REPORT][7:], check.digest(raw)[7:]))
        with mock.patch.object(check, 'score', side_effect=AssertionError('must not score')):
            with self.assertRaisesRegex(ValueError, 'bytes changed'): check.verify(self.fixture)

    def test_missing_extra_nested_members(self):
        for mode in ('missing', 'extra', 'deep'):
            with self.subTest(mode=mode):
                if mode == 'missing': (self.fixture/'README.md').rename(self.fixture/'renamed')
                elif mode == 'extra': (self.fixture/'extra').write_text('x')
                else: (self.fixture/'original-attempt/deep').mkdir()
                with self.assertRaisesRegex(ValueError, 'membership'): check.verify(self.fixture)
                if mode == 'missing': (self.fixture/'renamed').rename(self.fixture/'README.md')
                elif mode == 'extra': (self.fixture/'extra').unlink()
                else: (self.fixture/'original-attempt/deep').rmdir()

    def test_symlink_root_ancestor_leaf_and_directory(self):
        alias = Path(self.tmp.name)/'alias'; alias.symlink_to(self.fixture, target_is_directory=True)
        with self.assertRaises(ValueError): check.verify(alias)
        with self.assertRaises(ValueError): check.read(alias/PRED)
        path = self.fixture/PRED; path.unlink(); path.symlink_to(check.FIXTURE/PRED)
        with self.assertRaises(ValueError): check.verify(self.fixture)
        path.unlink(); shutil.copyfile(check.FIXTURE/PRED, path)
        shutil.rmtree(self.fixture/'original-attempt')
        (self.fixture/'original-attempt').symlink_to(check.FIXTURE/'original-attempt', target_is_directory=True)
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

    def test_resealed_semantic_completion_gate_claim_and_score_rejected(self):
        cases = [('original-attempt/worker-completion.json','decodes',True),
                 ('original-attempt/worker-completion.json','training_performed',True),
                 ('original-attempt/replay-gate.json','both_training_fit',1),
                 (REPORT,'fresh_confirmation',True), (REPORT,'original_inventory_unchanged',False),
                 (REPORT,'summary',{}), ('audit-receipt.json','verified',1),
                 ('original-attempt/attempt.json','selection','best seed')]
        for name,key,value in cases:
            with self.subTest(name=name,key=key):
                before = check.snapshot(self.fixture); obj=json.loads(before[name]); obj[key]=value
                before[name]=encoded(obj)
                with self.assertRaises(ValueError): check.score(before)

    def test_original_manifest_link_rejected(self):
        before=check.snapshot(self.fixture)
        before['original-SHA256SUMS.txt']=b'0'*64+b'  '+PRED.encode()+b'\n'
        with self.assertRaisesRegex(ValueError,'manifest'):check.score(before)

    def test_historical_prediction_tamper_rejected(self):
        original=check.read
        def read(path):
            raw=original(path)
            return raw+b' ' if Path(path).parent==check.HISTORICAL else raw
        with mock.patch.object(check,'read',side_effect=read):
            with self.assertRaisesRegex(ValueError,'historical'):check.verify(self.fixture)

    def test_source_tamper_before_import(self):
        original=check.read
        def read(path):
            raw=original(path)
            return raw+b' ' if str(path).endswith('/protocol.py') else raw
        with mock.patch.object(check,'read',side_effect=read):
            with self.assertRaisesRegex(ValueError,'source changed'):check.verify(self.fixture)

    def source_copy(self):
        root=Path(self.tmp.name)/'checkout'
        for relative in check.SOURCES:
            target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(check.ROOT/relative,target)
        return root

    def test_package_shadow_cache_and_foreign_module(self):
        root=self.source_copy(); shadow=root/'scripts/frozen_music_eval_v1/protocol';shadow.mkdir()
        marker=Path(self.tmp.name)/'executed'
        (shadow/'__init__.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'shadow'):check.sources()
        self.assertFalse(marker.exists());shutil.rmtree(shadow)
        (root/'src/trinite/__pycache__').mkdir()
        with mock.patch.object(check,'ROOT',root):
            with self.assertRaisesRegex(ValueError,'cache'):check.sources()
        with mock.patch.dict(sys.modules,{'frozen_music_eval_v1.protocol':types.SimpleNamespace(__file__='/foreign.py')}):
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

    def test_provenance_claim_even_if_repinned(self):
        p=self.fixture/'provenance-DERIVED.json';obj=json.loads(p.read_bytes());obj['inference_replayed']=1
        p.write_bytes(encoded(obj))
        with mock.patch.object(check,'FIXTURE_HASHES',{**check.FIXTURE_HASHES,'provenance-DERIVED.json':check.digest(p.read_bytes())}):
            with self.assertRaisesRegex(ValueError,'provenance'):check.verify(self.fixture)

    def test_declared_negative_patterns(self):
        summary=check.score(check.snapshot(self.fixture))
        for seed,canonical,pitch in [('17',16,10),('29',15,14)]:
            held=summary['by_seed'][seed]['heldout']
            self.assertEqual(held['canonical_primary']['exact_plus_eos'],canonical)
            self.assertEqual(held['canonical_controls']['global']['exact_plus_eos'],20)
            self.assertEqual(held['by_sibling']['pitch']['exact_plus_eos'],pitch)
            self.assertEqual(held['transformation_pairs']['pitch']['identical_raw_outputs'],4)
            self.assertEqual(held['transformation_pairs']['tempo']['identical_raw_outputs'],40)
            for sibling in ('canonical','pitch','tempo'):
                self.assertEqual(held['by_sibling'][sibling]['complete_groups'],0)
                self.assertEqual(held['by_sibling'][sibling]['groups'],10)
            self.assertEqual(held['canonical_strata']['variant_id']['10']['exact_plus_eos'],0)

    def test_launcher_requires_flags_before_pathlib(self):
        root=Path(self.tmp.name)/'launch';root.mkdir()
        script=root/'check.py';shutil.copyfile(check.ROOT/'scripts/check_frozen_music_eval_result.py',script)
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
                package=root/'scripts/frozen_music_eval_result_v1'
                package.mkdir(parents=True)
                shutil.copyfile(check.ROOT/'scripts/check_frozen_music_eval_result.py', root/'scripts/check.py')
                shutil.copyfile(check.ROOT/'scripts/frozen_music_eval_result_v1/__init__.py', package/'__init__.py')
                (package/'check.py').write_text('raise RuntimeError("legitimate verifier must not execute")\n')
                marker=root/'executed'
                shadow=(root/'src/frozen_music_eval_result_v1') if mode=='src-package' else package/'check'
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
        shutil.copyfile(check.ROOT/'scripts/check_frozen_music_eval_result.py',root/'scripts/check.py')
        (root/'scripts/frozen_music_eval_result_v1').symlink_to(check.ROOT/'scripts/frozen_music_eval_result_v1',target_is_directory=True)
        result=subprocess.run([sys.executable,'-I','-B',str(root/'scripts/check.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('linked verifier package',result.stderr)

    def test_launcher_preloads_stdlib_before_project_path_shadows(self):
        root=self.source_copy()
        package=root/'scripts/frozen_music_eval_result_v1'
        shutil.copytree(check.ROOT/'scripts/frozen_music_eval_result_v1',package)
        script=root/'scripts/check_frozen_music_eval_result.py'
        shutil.copyfile(check.ROOT/'scripts/check_frozen_music_eval_result.py',script)
        shutil.copytree(check.FIXTURE,root/'fixtures/frozen-music-eval-result-v1')
        historical=root/'fixtures/music-exposure-result-20261029-v1';historical.mkdir()
        for name in check.HISTORICAL_HASHES:shutil.copyfile(check.HISTORICAL/name,historical/name)
        marker=root/'executed'
        for name in ('hashlib','subprocess'):
            (root/'src'/f'{name}.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise RuntimeError("shadow executed")\n')
        result=subprocess.run([sys.executable,'-I','-B',str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['verified'])
        self.assertFalse(marker.exists())
        self.assertFalse(any(root.rglob('__pycache__')))

if __name__=='__main__': unittest.main()
