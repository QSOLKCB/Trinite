"""Historical evidence checks and synthetic analysis controls; no model execution."""
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from coverage_diagnostic_v1 import analysis as a
from run_coverage_diagnostic import main


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = a.analyze()
        cls.groups = a.corpus()

    def fixture_copy(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        target = Path(temp.name)/'subset'
        shutil.copytree(a.FIXTURE, target)
        return target

    def test_original_commitment_and_complete_membership(self):
        self.assertEqual(identity((a.FIXTURE/'completion.json').read_bytes()), a.COMPLETION)
        self.assertEqual(len(a.retained()), 128)
        self.assertEqual(len(list(a.FIXTURE.iterdir())), 131)  # 130 exact bytes + README

    def test_split_outcomes_and_eos_separation(self):
        for split, correct, non_eos, outputs in [('validation', 96, 40, 6), ('test', 101, 45, 8)]:
            s = self.report['by_split'][split]['outcome']
            self.assertEqual((s['rows'], s['families'], s['semantic_items']), (64, 1, 32))
            self.assertEqual(s['raw_answer_and_eos_correct'], 0)
            self.assertEqual((s['teacher_forced_correct'], s['teacher_forced_targets']), (correct, 176))
            self.assertEqual((s['teacher_forced_answer_correct'], s['teacher_forced_answer_targets']), (non_eos, 112))
            self.assertEqual((s['teacher_forced_eos_correct'], s['teacher_forced_eos_targets']), (56, 64))
            self.assertEqual(s['distinct_generated_outputs'], outputs)

    def test_every_operation_fails(self):
        for split in self.report['by_split'].values():
            for result in split['by_task'].values():
                self.assertEqual(result['outcome']['rows'], 16)
                self.assertEqual(result['outcome']['raw_answer_and_eos_correct'], 0)
                self.assertEqual(result['generated_answer_seen_in_training'], 16)

    def test_same_task_answer_support_not_global_vocabulary(self):
        for split in self.report['by_split'].values():
            for task, result in split['by_task'].items():
                support = result['support_counts']
                self.assertEqual(support['gold_answer_seen_in_same_task_training'],
                                 16 if task in ('add', 'subtract') else 0)
                self.assertEqual(support['operand_values_seen_in_same_task_training'], 16)
                self.assertEqual(support['family_seen_in_training'], 0)
                self.assertEqual(support['operand_range_extrapolation'], 0)
                self.assertEqual(support['operand_digit_length_extrapolation'], 0)

    def test_frozen_families(self):
        self.assertEqual(self.report['by_split']['validation']['operand_magnitude_families'], [[1, 2]])
        self.assertEqual(self.report['by_split']['test']['operand_magnitude_families'], [[1, 3]])
        for field in ('example_identity', 'semantic_identity', 'family_id', 'prompt'):
            sets = [{e[field] for e in self.groups[s]} for s in ('train', 'validation', 'test')]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_observation_rows_have_no_missing_or_duplicate_ids(self):
        ids = [r['example_identity'] for r in self.report['row_support']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [e['example_identity'] for e in self.groups['validation']+self.groups['test']])

    def test_synthetic_support_distinguishes_output_length_from_novelty(self):
        train = [{'task':'add', 'answer':'2', 'formal':{'a':1,'b':1}, 'family_id':'known'}]
        item = {'task':'add', 'answer':'3', 'formal':{'a':1,'b':2}, 'family_id':'new'}
        flags = a.support(item, train)
        self.assertFalse(flags['gold_answer_seen_in_same_task_training'])
        self.assertTrue(flags['answer_byte_length_seen_in_same_task_training'])
        self.assertTrue(flags['operand_range_extrapolation'])
        self.assertFalse(flags['operand_digit_length_extrapolation'])
        item['formal']['b'] = 10
        item['answer'] = '11'
        flags = a.support(item, train)
        self.assertTrue(flags['operand_digit_length_extrapolation'])
        self.assertFalse(flags['answer_byte_length_seen_in_same_task_training'])

    def test_synthetic_missing_task_support_rejected(self):
        with self.assertRaises(ContractError):
            a.support({'task':'add'}, [])

    def test_completion_tampering_rejected(self):
        fixture = self.fixture_copy()
        with (fixture/'completion.json').open('ab') as stream: stream.write(b' ')
        with self.assertRaisesRegex(ContractError, 'completion identity'):
            a.retained(fixture)

    def test_index_tampering_rejected(self):
        fixture = self.fixture_copy()
        (fixture/'row-index.json').write_bytes(b'{}\n')
        with self.assertRaisesRegex(ContractError, 'index identity'):
            a.retained(fixture)

    def test_row_tampering_rejected(self):
        fixture = self.fixture_copy()
        value = parse_json((fixture/'row-0000.json').read_bytes(), canonical=True)
        value['row']['generated_tokens'] = value['row']['expected_tokens']
        (fixture/'row-0000.json').write_bytes(json_bytes(value))
        with self.assertRaisesRegex(ContractError, 'row identity'):
            a.retained(fixture)

    def test_missing_failed_row_rejected(self):
        fixture = self.fixture_copy()
        (fixture/'row-0000.json').unlink()
        with self.assertRaisesRegex(ContractError, 'membership'):
            a.retained(fixture)

    def test_extra_member_rejected(self):
        fixture = self.fixture_copy()
        (fixture/'private-request.json').write_text('{}')
        with self.assertRaisesRegex(ContractError, 'membership'):
            a.retained(fixture)

    def test_symlink_rejected(self):
        if not hasattr(os, 'symlink'): self.skipTest('OS has no symlink API')
        fixture = self.fixture_copy()
        row = fixture/'row-0000.json'
        raw = row.read_bytes(); row.unlink()
        other = fixture.parent/'other.json'; other.write_bytes(raw)
        try: row.symlink_to(other)
        except OSError: self.skipTest('OS does not permit symlink creation')
        with self.assertRaises(ContractError): a.retained(fixture)

    def test_bounded_reader_rejects_oversize(self):
        with self.assertRaises(ContractError): a.read_bytes(a.FIXTURE/'completion.json', limit=1)

    def test_fixture_unchanged_after_analysis(self):
        before = a.inventory(a.FIXTURE)
        a.analyze()
        self.assertEqual(a.inventory(a.FIXTURE), before)

    def test_changed_corpus_pin_rejected(self):
        with patch.object(a, 'CORPUS', 'sha256:'+'0'*64):
            with self.assertRaisesRegex(ContractError, 'corpus identity'): a.corpus()

    def test_label_reconciliation_even_after_transport_stage(self):
        records = copy.deepcopy(a.retained())
        records[0]['row']['expected_tokens'][0] = 0
        with patch.object(a, 'retained', return_value=records):
            with self.assertRaisesRegex(ContractError, 'labels'): a.analyze()

    def test_membership_reconciliation_even_after_transport_stage(self):
        records = copy.deepcopy(a.retained())
        records[0], records[1] = records[1], records[0]
        with patch.object(a, 'retained', return_value=records):
            with self.assertRaisesRegex(ContractError, 'admitted corpus'): a.analyze()

    def test_wrong_import_root_rejected(self):
        with patch.object(sys.modules['trinite.contracts'], '__file__', '/different/contracts.py'):
            with self.assertRaisesRegex(ContractError, 'different source root'): a.analyze()

    def test_source_change_rejected(self):
        real = a.read_bytes
        calls = 0
        def changed(path, limit=1024*1024):
            nonlocal calls
            if Path(path) == a.ROOT/a.SOURCE_NAMES[0]:
                calls += 1
                if calls > 1: return b'changed source'
            return real(path, limit)
        with patch.object(a, 'read_bytes', side_effect=changed):
            with self.assertRaisesRegex(ContractError, 'source changed'): a.analyze()

    def test_report_source_closure_and_conservative_flags(self):
        self.assertEqual(set(self.report['sources']), set(a.SOURCE_NAMES))
        for flag in ('predictions_replayed', 'training_executed', 'fresh_confirmation',
                     'learning_adequate', 'phase5_ready', 'release_ready'):
            self.assertIs(self.report[flag], False)
        proposal = self.report['proposal']
        self.assertEqual(proposal['operation'], 'add')
        self.assertIs(proposal['training_enabled'], False)
        self.assertEqual(len(proposal['required_separate_strata']), 3)

    def test_public_original_schema_privacy_allowlist(self):
        row_keys = {'answer_and_eos_correct','answer_correct','authoritative_loss_hex','carrier',
            'context_length','example_identity','expected_tokens','family_id','first_answer',
            'generated_tokens','malformed_reason','non_neural_controls','output_tokens',
            'premature_eos','scored_targets','semantic_identity','split','stop_reason','task'}
        target_keys = {'answer_offset','argmax_token','correct','eos_probability_hex',
            'example_identity','gold_probability_hex','gold_rank','gold_token','is_eos',
            'loss_hex','target_position'}
        for value in a.retained():
            self.assertEqual(set(value['row']), row_keys)
            for target in value['targets']+[value['row']['first_answer']]:
                self.assertEqual(set(target), target_keys)
            self.assertEqual(set(value['row']['non_neural_controls']), {
                'constant_eos_tokens','exact_oracle_tokens','training_lookup_tokens','training_majority_tokens'})
        for path in a.FIXTURE.glob('*.json'):
            raw = path.read_bytes()
            for private in (b'/home/', b'/tmp/', b'username', b'hostname', b'environment', b'inode', b'device'):
                self.assertNotIn(private, raw)

    def test_no_numerical_backend_imported(self):
        script = "from coverage_diagnostic_v1.analysis import analyze; analyze(); import sys; assert not any(n in sys.modules for n in ('torch','numpy','trinite.model','trinite.training'))"
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=a.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_exclusive_create_and_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'report.json'
            stdout = io.StringIO()
            with patch('sys.stdout', stdout): self.assertEqual(main(['report','--output',str(out)]), 0)
            expected = stdout.getvalue().strip()
            self.assertEqual(expected, identity(out.read_bytes()))
            with patch('sys.stdout', io.StringIO()):
                self.assertEqual(main(['verify','--report',str(out),'--expected-identity',expected]), 0)
            old = out.read_bytes()
            with patch('sys.stderr', io.StringIO()): self.assertEqual(main(['report','--output',str(out)]), 1)
            self.assertEqual(out.read_bytes(), old)

    def test_cli_rejects_wrong_commitment_or_resealed_modified_report(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'report.json'
            out.write_bytes(json_bytes(self.report))
            with patch('sys.stderr', io.StringIO()):
                self.assertEqual(main(['verify','--report',str(out),'--expected-identity','sha256:'+'0'*64]), 1)
                changed = copy.deepcopy(self.report); changed['sources'] = {}
                raw = json_bytes(changed); out.write_bytes(raw)
                self.assertEqual(main(['verify','--report',str(out),'--expected-identity',identity(raw)]), 1)

    def test_cli_refuses_fixture_output(self):
        target = a.FIXTURE/'must-not-create.json'
        with patch('sys.stderr', io.StringIO()): self.assertEqual(main(['report','--output',str(target)]), 1)
        self.assertFalse(target.exists())

    def test_deterministic_report(self):
        self.assertEqual(json_bytes(a.analyze()), json_bytes(a.analyze()))


if __name__ == '__main__': unittest.main()
