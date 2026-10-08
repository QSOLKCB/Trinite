from pathlib import Path
import shutil
import tempfile
import unittest

from trinite.comparison_data import dataset, audit_bytes, numeric_items
from trinite.contracts import ContractError, json_bytes, parse_json
from trinite.qec_source import oracle, verify_pin

ROOT = Path(__file__).resolve().parents[1]


class QECSourceTests(unittest.TestCase):
    def test_retained_numeric_fixtures_match_exact_source_and_admission(self):
        for kind in ('qutrit', 'ququart'):
            root = ROOT/'fixtures/qec-v1'/kind
            self.assertEqual((root/'dataset.json').read_bytes(), dataset(kind)[0])
            self.assertEqual((root/'manifest.json').read_bytes(), dataset(kind)[1])

    def test_pinned_source_and_tampering(self):
        report = verify_pin()
        self.assertEqual(len(report['files']), 9)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'package'
            shutil.copytree(ROOT/'src/trinite/_qec', target/'_qec')
            for name in ('qec-pin.json', 'QEC-LICENSE.txt'):
                shutil.copyfile(ROOT/'src/trinite'/name, target/name)
            verify_pin(target)
            path = target/'_qec/qutrit/exact.py'
            path.write_bytes(path.read_bytes()+b'#changed\n')
            with self.assertRaises(ContractError): verify_pin(target)

    def test_every_certified_error_passes_two_independent_checks(self):
        self.assertEqual(len(numeric_items('qutrit')), 40)
        self.assertEqual(len(numeric_items('ququart')), 75)
        with self.assertRaises(ContractError): numeric_items('unknown')

    def test_unknown_syndromes_reject_outside_bounded_tables(self):
        from itertools import product
        for kind, radix, width, expected in (('qutrit', 3, 4, 41), ('ququart', 2, 8, 76)):
            code, decoder, _ = oracle(kind)
            accepted, rejected = 0, 0
            for syndrome in product(range(radix), repeat=width):
                try:
                    correction = decoder.decode(syndrome)
                except ValueError:
                    rejected += 1
                else:
                    self.assertLessEqual(correction.weight, 1)
                    accepted += 1
            self.assertEqual(accepted, expected)
            self.assertGreater(rejected, 0)

    def test_all_carriers_tasks_and_error_variants_stay_in_support_families(self):
        for kind, counts in (('qutrit', (96, 32, 32)), ('ququart', (180, 60, 60))):
            raw, manifest = dataset(kind); report = audit_bytes(raw, manifest)
            self.assertEqual(tuple(report['counts'][s] for s in ('train', 'validation', 'test')), counts)
            examples = parse_json(raw, canonical=True)['examples']
            families = {}
            problems = {}
            for e in examples:
                families.setdefault(e['family_id'], set()).add(e['split'])
                problems.setdefault(e['formal_identity'], []).append(e)
            self.assertEqual(len(families), 5)
            self.assertTrue(all(len(v) == 1 for v in families.values()))
            for variants in problems.values():
                self.assertEqual(len(variants), 4)
                self.assertEqual(len({e['split'] for e in variants}), 1)
                self.assertEqual({e['task'] for e in variants}, {'syndrome', 'correction'})
                self.assertEqual({e['carrier'] for e in variants}, {'infix', 'fields'})

    def test_contradictory_admission_and_changed_labels_never_pass(self):
        raw, manifest = dataset('qutrit')
        for field, value in (('outcome', 'rejected'), ('rights_basis', 'copied copyrighted prose'),
                             ('scope', 'all internet text'), ('reviewed_on', 'not-a-date'), ('author', '')):
            altered = parse_json(manifest)
            altered['admission'][field] = value
            with self.assertRaises(ContractError): audit_bytes(raw, json_bytes(altered))
        for field in ('answer', 'prompt', 'family_id', 'split'):
            altered = parse_json(raw)
            altered['examples'][0][field] = 'altered'
            with self.assertRaises(ContractError): audit_bytes(json_bytes(altered), manifest)


if __name__ == '__main__': unittest.main()
