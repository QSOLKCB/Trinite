from collections import defaultdict
from pathlib import Path
import unittest

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.curriculum_data import dataset, audit_bytes, projections, source_receipt
from trinite.curriculum_oracle import solve, parse_prompt
from trinite.learning_data import dataset as parent_dataset

ROOT = Path(__file__).resolve().parents[1]


class ScalarCurriculumTests(unittest.TestCase):
    def test_independent_labels_lineage_and_unchanged_family_splits(self):
        for workload in ('arithmetic', 'fold', 'lattice'):
            raw, manifest = dataset(workload); record = audit_bytes(raw, manifest)
            parent_raw, parent_manifest = parent_dataset(workload)
            parents = {r['example_identity']: r for r in parse_json(parent_raw)['examples']}
            splits = defaultdict(set); children = defaultdict(set)
            self.assertEqual(record['parent_dataset_identity'], identity(parent_raw))
            self.assertEqual(record['parent_manifest_identity'], identity(parent_manifest))
            for row in parse_json(raw)['examples']:
                parent = parents[row['transformation_lineage']['parent_example_identity']]
                self.assertEqual((row['family_id'], row['split'], row['carrier']),
                                 (parent['family_id'], parent['split'], parent['carrier']))
                self.assertEqual(row['generator_revision'], identity(json_bytes(source_receipt())))
                self.assertEqual(solve(workload, row['formal']), row['answer'])
                self.assertEqual(parse_prompt(workload, row['prompt']), row['formal'])
                self.assertEqual(row['example_identity'], identity(json_bytes({k:v for k,v in row.items() if k!='example_identity'})))
                splits[row['family_id']].add(row['split']); children[parent['example_identity']].add(row['task'])
                if workload == 'arithmetic' or (workload == 'lattice' and parent['task'] != 'traversal'):
                    for key in ('formal', 'answer', 'prompt', 'text'): self.assertEqual(row[key], parent[key])
            self.assertTrue(all(len(v) == 1 for v in splits.values()))
            self.assertEqual(set(children), set(parents))
            for key, parent in parents.items():
                self.assertEqual(children[key], {spec['task'] for spec, _ in projections(workload, parent)})

    def test_frozen_preparation_manifests(self):
        for workload in ('arithmetic', 'fold', 'lattice'):
            raw, manifest = dataset(workload)
            retained = ROOT/'fixtures/scalar-curriculum-v1'/workload/'manifest.json'
            self.assertEqual(manifest, retained.read_bytes())
            self.assertEqual(identity(raw), parse_json(manifest)['dataset_identity'])

    def test_refreshed_hashes_do_not_admit_changed_labels_families_or_rights(self):
        raw, manifest = dataset('lattice')
        for key, value in (('answer', '999'), ('split', 'test'), ('family_id', 'invented'),
                           ('generator_revision', 'sha256:'+'0'*64)):
            data = parse_json(raw); row = data['examples'][0]; row[key] = value
            row['example_identity'] = identity(json_bytes({k:v for k,v in row.items() if k!='example_identity'}))
            forged = json_bytes(data); changed = parse_json(manifest); changed['dataset_identity'] = identity(forged)
            with self.subTest(key=key), self.assertRaises(ContractError): audit_bytes(forged, json_bytes(changed))
        changed = parse_json(manifest); changed['admission']['rights_basis'] = 'blanket permission'
        with self.assertRaises(ContractError): audit_bytes(raw, json_bytes(changed))

    def test_independent_scalar_oracles_and_parser_bounds(self):
        self.assertEqual(solve('fold', {'task':'sum', 'values':[-2,1,2]}), '1')
        self.assertEqual(solve('fold', {'task':'squares', 'values':[-2,1,2]}), '9')
        for i in range(27):
            values = [solve('lattice', {'task':task, 'index':i}) for task in
                      ('traversal-value','traversal-x','traversal-y','traversal-z')]
            self.assertEqual(int(values[0]), 9*int(values[1])+3*int(values[2])+int(values[3]))
        for workload, prompt in (('fold','S:-0,1='),('fold','Q:3,1='),('fold','C:1='),
                                 ('lattice','X:27='),('lattice','X:01='),('fold','X:1,2=')):
            with self.subTest(prompt=prompt), self.assertRaises(ContractError): parse_prompt(workload,prompt)


if __name__ == '__main__': unittest.main()
