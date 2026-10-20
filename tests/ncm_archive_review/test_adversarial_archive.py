"""Independent data-only adversarial review of the retained public NCM packet."""
import io
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_ncm_rtx_archive as archive


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def reseal_inventory(members):
    """Test-local mutant only; never modifies or reseals a historical fixture."""
    members['inventory.json'] = encoded({
        name: archive.digest(data) for name, data in members.items()
        if name != 'inventory.json'
    })
    return members


def zip_bytes(entries):
    result = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(result, 'w', compression=zipfile.ZIP_DEFLATED) as stream:
            for name, value in entries:
                stream.writestr(name, value)
    return result.getvalue()


class AdversarialArchiveReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = archive.read_zip((archive.DEFAULT_FIXTURE / 'run.zip').read_bytes())
        cls.request = json.loads(cls.members['request.json'])

    def reject_member_json(self, name, mutate):
        members = self.members.copy()
        value = json.loads(members[name])
        mutate(value)
        members[name] = encoded(value)
        reseal_inventory(members)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            archive.verify_completed(members, self.request)

    def test_real_public_packet_verifies_without_mutation(self):
        before = {p.name: p.read_bytes() for p in archive.DEFAULT_FIXTURE.iterdir()}
        result = archive.verify_fixture()
        self.assertIs(result['exact_recorded_result_match'], True)
        self.assertIs(result['hardware_attestation'], False)
        self.assertIs(result['new_inference_executed'], False)
        self.assertIsNone(result['failed']['warm_solved'])
        self.assertEqual(result['run']['warm_boards'], 4)
        self.assertEqual(before, {p.name: p.read_bytes() for p in archive.DEFAULT_FIXTURE.iterdir()})

    def test_boolean_raw_token_is_not_a_digit(self):
        self.reject_member_json('result.json',
                                lambda x: x['outcomes'][2]['raw_ids'].__setitem__(0, True))

    def test_wrong_separator_is_rejected(self):
        self.reject_member_json('result.json',
                                lambda x: x['outcomes'][2]['raw_ids'].__setitem__(9, 1))

    def test_clue_breaking_output_cannot_keep_success_metrics(self):
        self.reject_member_json('result.json',
                                lambda x: x['outcomes'][3]['raw_ids'].__setitem__(0, 9))

    def test_numeric_one_is_not_a_boolean_metric(self):
        self.reject_member_json('result.json',
                                lambda x: x['outcomes'][2]['metrics'].__setitem__('solved', 1))

    def test_gold_codec_corruption_is_checked_separately(self):
        self.reject_member_json('result.json',
                                lambda x: x['reconstructions'][0]['raw_ids'].__setitem__(0, 2))

    def test_invalid_latent_code_is_rejected(self):
        for value in (True, -1, 16, 0.5):
            with self.subTest(value=value):
                self.reject_member_json('result.json',
                                        lambda x: x['outcomes'][2]['latent_ids'].__setitem__(0, value))

    def test_warmup_cannot_be_counted_as_an_extra_warm_board(self):
        self.reject_member_json('result.json',
                                lambda x: x['outcomes'][1].__setitem__('phase', 'warm'))

    def test_incomplete_outcome_inventory_is_rejected(self):
        self.reject_member_json('result.json', lambda x: x['outcomes'].pop())

    def test_all_progress_records_are_required(self):
        members = self.members.copy()
        members.pop('progress/progress-005.json')
        with self.assertRaises(ValueError):
            archive.verify_completed(reseal_inventory(members), self.request)

    def test_progress_output_disagreement_is_rejected(self):
        self.reject_member_json('progress/progress-003.json',
                                lambda x: x['outcome']['raw_ids'].__setitem__(0, 2))

    def test_progress_load_disagreement_is_rejected(self):
        self.reject_member_json('progress/progress-000.json',
                                lambda x: x.__setitem__('load_seconds', 0))

    def test_progress_metric_type_must_match_final_result(self):
        self.reject_member_json('progress/progress-003.json',
                                lambda x: x['outcome']['metrics'].__setitem__('solved', 1))

    def test_nonfinite_negative_and_boolean_durations_are_rejected(self):
        for value in (True, -1, float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value):
                self.reject_member_json('timings.json',
                                        lambda x: x.__setitem__('load_seconds', value))

    def test_boolean_memory_and_excess_allocator_memory_are_rejected(self):
        for value in (True, -1, self.request['protocol']['max_allocator_bytes'] + 1):
            with self.subTest(value=value):
                self.reject_member_json('timings.json',
                                        lambda x: x.__setitem__('max_reserved_bytes', value))

    def test_component_timing_mismatch_is_rejected(self):
        self.reject_member_json('timings.json',
                                lambda x: x['measurements'][2].__setitem__('total_seconds', 1))

    def test_false_return_code_is_not_integer_zero(self):
        self.reject_member_json('status.json', lambda x: x.__setitem__('returncode', False))

    def test_true_return_code_is_not_integer_one_in_failure(self):
        manifest = json.loads((archive.DEFAULT_FIXTURE / 'manifest.json').read_bytes())
        members = archive.read_zip(
            (archive.DEFAULT_FIXTURE / 'failed-run.public-derived.zip').read_bytes())
        derivation = manifest['failure_derivation']
        members['status.json'] = encoded({'returncode': True, 'state': 'failed'})
        inventory = json.loads(members['original-inventory.json'])
        inventory['status.json'] = archive.digest(members['status.json'])
        members['original-inventory.json'] = encoded(inventory)
        for name in ('status.json', 'inventory.json'):
            row = derivation['members'][name]
            data = members[row['derived_member']]
            row['original_sha256'] = row['derived_sha256'] = archive.digest(data)
            row['original_size'] = row['derived_size'] = len(data)
        with self.assertRaises(ValueError):
            archive.verify_failure(members, derivation, json.loads(members['request.json']))

    def test_missing_loader_diagnostics_is_rejected(self):
        members = self.members.copy()
        members['worker.log'] = b''
        with self.assertRaises(ValueError):
            archive.verify_completed(reseal_inventory(members), self.request)

    def test_duplicate_named_loader_diagnostic_is_rejected(self):
        members = self.members.copy()
        lines = members['worker.log'].splitlines(keepends=True)
        positions = [i for i, line in enumerate(lines) if line.startswith(b'{"loader_contract"')]
        lines[positions[1]] = lines[positions[0]]
        members['worker.log'] = b''.join(lines)
        with self.assertRaises(ValueError):
            archive.verify_completed(reseal_inventory(members), self.request)

    def test_duplicate_json_key_is_rejected(self):
        members = self.members.copy()
        members['status.json'] = b'{"returncode":1,"returncode":0,"state":"completed"}\n'
        with self.assertRaises(ValueError):
            archive.verify_completed(reseal_inventory(members), self.request)

    def test_unexpected_checkpoint_member_is_rejected(self):
        members = self.members.copy()
        members['generator.pt'] = b'not a real checkpoint'
        with self.assertRaises(ValueError):
            archive.verify_completed(reseal_inventory(members), self.request)

    def test_zip_traversal_and_noncanonical_names_are_rejected(self):
        for name in ('../outside', '/absolute', 'a/../b', 'a//b', './name',
                     'C:/name', 'a\\b', 'name\npart'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.read_zip(zip_bytes([(name, b'payload')]))

    def test_zip_duplicate_and_casefold_collision_are_rejected(self):
        for names in (('same', 'same'), ('Name', 'name')):
            with self.subTest(names=names), self.assertRaises(ValueError):
                archive.read_zip(zip_bytes([(name, b'payload') for name in names]))

    def test_zip_raw_nul_truncation_is_rejected(self):
        data = zip_bytes([('bad?member', b'payload')]).replace(b'bad?member', b'bad\0member')
        with self.assertRaises(ValueError):
            archive.read_zip(data)

    def test_zip_symlink_and_special_members_are_rejected(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFDIR, stat.S_IFSOCK):
            item = zipfile.ZipInfo('special')
            item.create_system = 3
            item.external_attr = (mode | 0o644) << 16
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                archive.read_zip(zip_bytes([(item, b'payload')]))

    def test_zip_cardinality_and_decompression_limits_are_rejected(self):
        for entries in ([('m' + str(i), b'x') for i in range(101)],
                        [('large', b'x' * (1024 * 1024 + 1))]):
            with self.assertRaises(ValueError):
                archive.read_zip(zip_bytes(entries))

    def test_public_file_change_requires_external_commitment(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'fixture'
            shutil.copytree(archive.DEFAULT_FIXTURE, destination)
            receipt = destination / 'replay-equality.json'
            value = json.loads(receipt.read_bytes())
            value['exact_result_match'] = False
            receipt.write_bytes(encoded(value))
            with self.assertRaises(ValueError):
                archive.verify_fixture(destination)

    def test_manifest_cannot_be_self_certified(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'fixture'
            shutil.copytree(archive.DEFAULT_FIXTURE, destination)
            manifest = destination / 'manifest.json'
            value = json.loads(manifest.read_bytes())
            value['failure_derivation']['rejected_tensor_identified_in_original'] = True
            manifest.write_bytes(encoded(value))
            with self.assertRaises(ValueError):
                archive.verify_fixture(destination)

    def test_public_symlink_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'fixture'
            shutil.copytree(archive.DEFAULT_FIXTURE, destination)
            payload = destination / 'run.zip'
            payload.unlink()
            payload.symlink_to(archive.DEFAULT_FIXTURE / 'run.zip')
            with self.assertRaises(ValueError):
                archive.verify_fixture(destination)


if __name__ == '__main__':
    unittest.main()
