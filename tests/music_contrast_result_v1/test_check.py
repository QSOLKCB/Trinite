"""Reports-only preservation integrity and scorer rejection tests; no models."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from music_contrast_result_v1 import check

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
        self.assertEqual(r['observations'],160)
        self.assertEqual(r['duplicate_journal_rows'],160)
        self.assertEqual(r['final_exact_per_seed'],{'17':20,'29':20})
        self.assertEqual(r['heldout_model_evaluation'],'unrun')
        for key in ('training_gate_passed','original_raw_custody_verified','inference_replayed',
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

    def test_positive_replacement_is_not_preserved_negative(self):
        check.sources()
        from trinite.contracts import json_bytes
        before=check.snapshot(self.fixture)
        for seed in (17,29):
            prefix=f'train-{seed}--step-064'
            records=json.loads(before[prefix+'-predictions.json'])['predictions']
            for r in records:r['generated_tokens']=r['expected_tokens'];r['correct']=True
            before[prefix+'-predictions.json']=json_bytes({'predictions':records})
            before[prefix+'-journal.jsonl']=b''.join(json_bytes(r) for r in records)
        with self.assertRaisesRegex(ValueError,'unexpectedly passes'):check.score(before)

    def test_exact_and_majority_not_just_eos(self):
        summary=check.score(check.snapshot(self.fixture))
        final=summary['observations'][PRED]
        self.assertEqual(final['all']['eos_terminated'],40)
        self.assertEqual(final['all']['exact_plus_eos'],20)
        self.assertEqual(final['all']['complete_groups'],0)
        self.assertEqual(final['all']['both_correct_edges'],0)
        self.assertEqual(final['all']['changed_output_edges'],0)
        self.assertEqual(final['baselines']['global']['exact_plus_eos'],20)
        self.assertEqual(summary['decode_forwards'],5280)

    def test_mutation_during_scoring(self):
        score=check.score
        def changed(before):
            result=score(before);(self.fixture/'README.md').write_text('changed');return result
        with mock.patch.object(check,'score',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'during verification'):check.verify(self.fixture)

if __name__=='__main__':unittest.main()
