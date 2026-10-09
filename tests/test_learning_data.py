import copy
from pathlib import Path
import unittest

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.learning_data import dataset, audit_bytes, validate_admission, source_receipt
from trinite.foundations_data import dataset as parent_dataset

ROOT = Path(__file__).resolve().parents[1]


class LearningDataTests(unittest.TestCase):
    def test_retained_admission_and_unchanged_facts_families_and_splits(self):
        for workload in ('arithmetic', 'fold', 'lattice'):
            raw, manifest = dataset(workload); admitted = audit_bytes(raw, manifest)
            directory = ROOT/'fixtures/learning-data-v1'/workload
            self.assertEqual(raw, (directory/'dataset.json').read_bytes())
            self.assertEqual(manifest, (directory/'manifest.json').read_bytes())
            original = {r['example_identity']: r for r in parse_json(parent_dataset(workload)[0])['examples']}
            rows = parse_json(raw)['examples']; self.assertEqual(len(rows), len(original))
            revision = identity(json_bytes(source_receipt()))
            for row in rows:
                old = original[row['ordering_identity']]
                for key in ('text', 'prompt', 'answer', 'formal', 'split', 'task', 'family_id'):
                    self.assertEqual(row[key], old[key])
                self.assertEqual(row['generator_revision'], revision)
                self.assertEqual(row['seed'], 31)
                self.assertEqual(row['transformation_lineage']['parent_example_identity'], old['example_identity'])
                self.assertEqual(row['example_identity'], identity(json_bytes({k:v for k,v in row.items() if k!='example_identity'})))
            validate_admission(admitted['admission'], workload)

    def test_refreshed_hashes_cannot_admit_missing_or_false_lineage(self):
        raw, manifest = dataset('arithmetic')
        for field in ('generator_revision', 'seed', 'transformation_lineage', 'ordering_identity'):
            record = parse_json(raw); record['examples'][0].pop(field)
            row = record['examples'][0]
            row['example_identity'] = identity(json_bytes({k:v for k,v in row.items() if k!='example_identity'}))
            forged = json_bytes(record); admission = parse_json(manifest); admission['dataset_identity']=identity(forged)
            with self.subTest(field=field), self.assertRaises(ContractError):
                audit_bytes(forged, json_bytes(admission))

    def test_contradictory_admission_evidence_is_rejected(self):
        record = parse_json(dataset('fold')[1])['admission']
        for key, value in (('outcome','rejected'),('author',''),('reviewed_on','not-a-date'),
                           ('rights_basis','copied copyrighted prose'),('scope','all future text'),
                           ('supporting_reference',{})):
            altered = copy.deepcopy(record); altered[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_admission(altered, 'fold')
        with self.assertRaises(ContractError): audit_bytes(raw := b'{}', raw)


if __name__ == '__main__':
    unittest.main()
