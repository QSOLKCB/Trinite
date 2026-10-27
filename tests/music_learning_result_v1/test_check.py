"""Original-byte and adversarial partial-evidence checks; no numerical execution."""
from pathlib import Path
import copy
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from trinite.contracts import ContractError, identity
from music_learning_result_v1 import check
from unseen_evaluation_v1.packet import inventory


class PublicEvidence(unittest.TestCase):
    def copied(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = Path(directory.name) / 'fixture'
        shutil.copytree(check.FIXTURE, fixture)
        return fixture

    def test_complete_originals_are_readonly(self):
        before = inventory(check.FIXTURE)
        with ExitStack() as stack:
            for target in ('trinite.model.Decoder', 'trinite.training.run_steps',
                           'music_learning_v1.numerical.restore',
                           'music_learning_v1.numerical.state_for'):
                stack.enter_context(patch(target, side_effect=AssertionError('numerical execution forbidden')))
            result = check.verify()
        self.assertEqual(result['prediction_records'], 3840)
        self.assertEqual(result['original_files'], 25)
        for name in (*check.GATES, 'closed_custody', 'checkpoint_tensors_verified',
                     'inference_replayed', 'optimizer_replayed'):
            self.assertIs(result[name], False)
        for cell in result['cells'].values():
            for stage in cell.values():
                for split in stage.values():
                    self.assertEqual(split['correct'], 0)
                    self.assertEqual(split['terminal_eos'], 0)
        self.assertEqual(inventory(check.FIXTURE), before)

    def test_extra_and_missing_rejected(self):
        for kind in ('extra', 'missing'):
            with self.subTest(kind=kind):
                fixture = self.copied()
                if kind == 'extra':
                    (fixture / 'private-environment.json').write_text('{}')
                else:
                    (fixture / 'dense-17/report.json').unlink()
                with self.assertRaisesRegex(ContractError, 'membership'):
                    check.verify(fixture)

    def test_changed_original_and_receipt_rejected(self):
        for name in ('matrix.json', 'source-bindings.json', 'dense-17/trained-train.json',
                     'dense-29/predictions.jsonl', 'dense-17/completion.json'):
            with self.subTest(name=name):
                fixture = self.copied()
                path = fixture / name
                path.write_bytes(path.read_bytes() + b' ')
                with self.assertRaises(ContractError):
                    check.verify(fixture)

    def test_symlink_file_directory_and_ancestor_rejected(self):
        fixture = self.copied()
        original = fixture / 'matrix.json'
        original.unlink()
        original.symlink_to(check.FIXTURE / 'matrix.json')
        with self.assertRaises(ContractError):
            check.verify(fixture)
        fixture = self.copied()
        shutil.rmtree(fixture / 'dense-17')
        (fixture / 'dense-17').symlink_to(check.FIXTURE / 'dense-17', target_is_directory=True)
        with self.assertRaises(ContractError):
            check.verify(fixture)
        alias = fixture.parent / 'alias'
        alias.symlink_to(fixture.parent, target_is_directory=True)
        with self.assertRaises(ContractError):
            check.verify(alias / 'fixture')

    def test_source_hash_and_foreign_import_rejected(self):
        name = 'scripts/music_learning_v1/codec.py'
        with self.assertRaisesRegex(ContractError, 'producer source differs'):
            check.source_check({name: 'sha256:' + '0' * 64})
        from music_learning_v1 import codec
        with patch.object(codec, '__file__', '/tmp/foreign/codec.py'):
            with self.assertRaisesRegex(ContractError, 'foreign root'):
                check.source_check({name: identity(check.read(check.ROOT / name))})
        with self.assertRaisesRegex(ContractError, 'invalid source path'):
            check.source_check({'../escape': 'unused'})

    def test_read_rejects_special_or_oversized_members(self):
        fixture = self.copied()
        with self.assertRaises(ContractError):
            check.read(fixture)
        path = fixture / 'large'
        with path.open('wb') as stream:
            stream.truncate(4 * 1024 * 1024 + 1)
        with self.assertRaises(ContractError):
            check.read(path)

    def test_observed_mutation_rejected(self):
        fixture = self.copied()
        real = check.source_check
        def mutate_after_inventory(sources):
            real(sources)
            path = fixture / 'README.md'
            path.write_bytes(path.read_bytes() + b'\nobserved mutation\n')
        with patch.object(check, 'source_check', side_effect=mutate_after_inventory):
            with self.assertRaisesRegex(ContractError, 'packet changed'):
                check.verify(fixture)

    def test_authoritative_scorer_rejects_corrupted_predictions(self):
        from trinite.contracts import parse_json
        original = parse_json(check.read(check.FIXTURE / 'dense-17/trained-train.json'), canonical=True)
        for kind in ('correct', 'duplicate', 'expected'):
            record = copy.deepcopy(original)
            if kind == 'correct':
                record['predictions'][0]['correct'] = True
            elif kind == 'duplicate':
                record['predictions'][1] = copy.deepcopy(record['predictions'][0])
            else:
                record['predictions'][0]['expected_tokens'] = [0]
            with self.subTest(kind=kind), self.assertRaises(ContractError):
                check.check_score(record, 'train', record['model_identity'])

    def test_checker_does_not_call_numerical_execution(self):
        source = check.read(check.ROOT / 'scripts/music_learning_result_v1/check.py').decode()
        for forbidden in ('state_for(', 'restore(', 'run_trial(', 'run_steps(', 'Decoder(', 'torch.load('):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
