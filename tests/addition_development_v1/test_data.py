"""Exact finite admission/scheduling conformance, not a target model experiment."""
from collections import Counter
import copy
import unittest

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.tokenizer import ByteTokenizer
from addition_development_v1 import data
from addition_development_v1.protocol import ARMS, STRATA, PROTOCOL, cells


class AdditionDataTests(unittest.TestCase):
    def test_complete_unique_finite_domain_and_counts(self):
        raw, manifest = data.dataset()
        receipt = data.audit_bytes(raw, manifest)
        rows = parse_json(raw)['examples']
        self.assertEqual(len(rows), 13*13*2)
        self.assertEqual(receipt['counts'], {'all_unique': 338, 'control': 20,
            'candidate': 40, 'familiar-answer': 36, 'novel-answer': 124, 'range-length': 138})
        self.assertEqual(receipt['family_counts'], {'control': 5, 'candidate': 10,
            'familiar-answer': 9, 'novel-answer': 36, 'range-length': 36})
        self.assertEqual(len({r['example_identity'] for r in rows}), 338)
        self.assertEqual(len({r['prompt'] for r in rows}), 338)
        self.assertEqual(receipt['dataset_identity'], identity(raw))
        self.assertEqual(receipt['range_length_answer_coverage'], {
            'familiar-answer': {'rows': 20, 'families': 5},
            'novel-answer': {'rows': 118, 'families': 31}})

    def test_exact_labels_exhaustive_independent_direct_sum(self):
        rows = parse_json(data.dataset()[0])['examples']
        for row in rows:
            a, b = row['formal']['a'], row['formal']['b']
            self.assertEqual(row['answer'], str(a+b))
            self.assertEqual(data.parse_prompt(row['prompt']), row['formal'])
            core = {k: v for k, v in row.items() if k != 'example_identity'}
            self.assertEqual(row['example_identity'], identity(json_bytes(core)))

    def test_swap_and_carrier_transforms_never_cross_family_split(self):
        families = {}
        for row in parse_json(data.dataset()[0])['examples']:
            key = tuple(sorted((row['formal']['a'], row['formal']['b'])))
            state = (row['family_id'], row['split'], row['stratum'], tuple(row['training_arms']))
            self.assertEqual(families.setdefault(key, state), state)
        self.assertEqual(len(families), 91)

    def test_shared_development_excludes_union_of_both_training_arms(self):
        development = sum((data.examples(s) for s in STRATA), [])
        for arm in ARMS:
            train = data.examples(arm)
            for key in ('family_id', 'semantic_identity', 'example_identity', 'prompt'):
                self.assertFalse({r[key] for r in train} & {r[key] for r in development})
        control = {r['example_identity'] for r in data.examples('control')}
        candidate = {r['example_identity'] for r in data.examples('candidate')}
        self.assertLess(control, candidate)

    def test_strata_are_exhaustive_disjoint_and_labelled_exploratory(self):
        arms = data.examples('candidate')
        groups = [data.examples(s) for s in STRATA]
        identities = [r['example_identity'] for r in arms + sum(groups, [])]
        self.assertEqual(len(set(identities)), 338)
        self.assertEqual(len(identities), 338)
        for row in data.examples('familiar-answer'):
            self.assertIn(int(row['answer']), data.SUMS)
            self.assertLess(max(row['formal']['a'], row['formal']['b']), 10)
        for row in data.examples('novel-answer'):
            self.assertNotIn(int(row['answer']), data.SUMS)
            self.assertLess(max(row['formal']['a'], row['formal']['b']), 10)
        for row in data.examples('range-length'):
            self.assertGreaterEqual(max(row['formal']['a'], row['formal']['b']), 10)
            self.assertEqual(row['coverage']['answer_seen_in_both_training_arms'], int(row['answer']) in data.SUMS)
        self.assertFalse(PROTOCOL['fresh_confirmation'])
        self.assertNotIn('confirmation', data.STRATA)

    def test_unique_training_diagnostics_not_repeated_visits(self):
        for arm, unique in (('control', 20), ('candidate', 40)):
            rows = data.examples(arm)
            self.assertEqual(len(rows), unique)
            self.assertEqual(len({r['example_identity'] for r in rows}), unique)
            cycle = data.schedule(arm)
            self.assertEqual(len(cycle), 40)
            self.assertEqual(len({r['example_identity'] for r in cycle}), unique)
            self.assertTrue(all(r['split'] == 'train' for r in cycle))
            counts = Counter(r['example_identity'] for r in cycle)
            self.assertEqual(set(counts.values()), {2 if arm == 'control' else 1})

    def test_cycle_answer_carrier_length_and_operand_marginals_match(self):
        report = data.validate_matching()
        self.assertEqual(report['control']['cycle'], report['candidate']['cycle'])
        cycle = report['control']['cycle']
        self.assertEqual(cycle['answer_histogram'], {'5': 8, '7': 8, '9': 8, '11': 8, '13': 8})
        self.assertEqual(cycle['left_operand_histogram'], {str(v): 4 for v in range(10)})
        self.assertEqual(cycle['right_operand_histogram'], {str(v): 4 for v in range(10)})
        self.assertEqual(cycle['carrier_histogram'], {'infix': 20, 'fields': 20})

    def test_every_matched_slot_and_batch_has_same_target_input_exposure(self):
        left, right = data.schedule('control'), data.schedule('candidate')
        def slot(row):
            e = ByteTokenizer(64).encode(row['text'], answer_start=len(row['prompt'].encode()))
            return row['answer'], row['carrier'], len(row['prompt']), len(e.input_ids), sum(e.loss_mask)
        self.assertEqual([slot(r) for r in left], [slot(r) for r in right])
        for start in range(0, 40, 8):
            for index in (2, 3, 4):
                self.assertEqual(sum(slot(r)[index] for r in left[start:start+8]),
                                 sum(slot(r)[index] for r in right[start:start+8]))

    def test_full_update_budget_is_exact_complete_cycles(self):
        for arm in ARMS:
            receipt = data.exposure(arm)
            self.assertEqual(receipt['updates'], 4000)
            self.assertEqual(receipt['scheduled_visits'], 32000)
            self.assertEqual(receipt['cycles'], 800)
            self.assertEqual(receipt['scheduled_target_tokens'], 76800)
            self.assertEqual(receipt['scheduled_model_input_tokens'], 284800)
            self.assertEqual(receipt['cycle_identity'], identity(json_bytes(receipt['cycle_example_identities'])))

    def test_no_model_boundary_in_data_or_protocol_imports(self):
        import ast
        forbidden = {'torch', 'numpy', 'trinite.model', 'trinite.training', 'requests', 'urllib', 'socket'}
        for path in ('scripts/addition_development_v1/data.py', 'scripts/addition_development_v1/protocol.py'):
            tree = ast.parse((data.ROOT/path).read_text())
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): names.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom): names.append(node.module or '')
            self.assertFalse(set(names) & forbidden)

    def test_sources_bind_generator_protocol_and_shared_tokenizer(self):
        sources = data.source_receipt()
        for name, expected in sources.items():
            self.assertEqual(expected, identity((data.ROOT/name).read_bytes()))
        self.assertIn('scripts/addition_development_v1/data.py', sources)
        self.assertIn('scripts/addition_development_v1/protocol.py', sources)
        self.assertIn('src/trinite/tokenizer.py', sources)

    def test_data_calls_do_not_return_shared_mutable_rows(self):
        a = data.examples('control')
        a[0]['answer'] = '999'
        self.assertNotEqual(a, data.examples('control'))
        b = data.schedule('control')
        b[0]['answer'] = '999'
        self.assertNotEqual(b, data.schedule('control'))

    def test_unknown_arm_stratum_and_prompt_rejected(self):
        for value in ('train', 'test', 'fresh-confirmation', '', None, 0):
            with self.subTest(value=value):
                with self.assertRaises(ContractError): data.examples(value)
                with self.assertRaises(ContractError): data.schedule(value)
        for prompt in ('A:01,2=', 'A:-1,2=', 'A:13,0=', 'M:1,2=', 'A(1,2)=', 'A:1,2= ', 'A:١,2=', ''):
            with self.subTest(prompt=prompt), self.assertRaises(ContractError): data.parse_prompt(prompt)

    def test_invalid_formal_types_keys_bounds_rejected(self):
        for formal in ({'task': 'multiply', 'a': 1, 'b': 2},
                       {'task': 'add', 'a': True, 'b': 2},
                       {'task': 'add', 'a': 1, 'b': -1},
                       {'task': 'add', 'a': 13, 'b': 0},
                       {'task': 'add', 'a': 1, 'b': 2, 'answer': '3'}):
            with self.subTest(formal=formal), self.assertRaises(ContractError): data.oracle(formal)
        with self.assertRaises(ContractError): data._row(1, 2, 'unknown')

    def test_mutated_labels_text_and_membership_cannot_be_resealed(self):
        raw, manifest = data.dataset()
        for key, value in (('answer', '999'), ('text', 'A:0,0=999'),
                           ('split', 'train'), ('training_arms', ['control', 'candidate']),
                           ('family_id', 'fake')):
            rows = parse_json(raw)
            target = next(r for r in rows['examples'] if r['split'] == 'test')
            target[key] = value
            target['example_identity'] = identity(json_bytes({k: v for k, v in target.items() if k != 'example_identity'}))
            changed = json_bytes(rows)
            receipt = parse_json(manifest)
            receipt['dataset_identity'] = identity(changed)
            with self.subTest(key=key), self.assertRaises(ContractError):
                data.audit_bytes(changed, json_bytes(receipt))

    def test_rights_policy_or_source_hash_cannot_be_resealed(self):
        raw, manifest = data.dataset()
        for change in ('rights', 'source', 'counts'):
            receipt = parse_json(manifest)
            if change == 'rights': receipt['admission']['rights_basis'] = 'public internet'
            elif change == 'source': receipt['sources']['src/trinite/tokenizer.py'] = 'sha256:'+'0'*64
            else: receipt['counts']['control'] = 40
            with self.subTest(change=change), self.assertRaises(ContractError):
                data.audit_bytes(raw, json_bytes(receipt))

    def test_missing_duplicate_or_reordered_rows_rejected(self):
        raw, manifest = data.dataset()
        for mode in ('missing', 'duplicate', 'reordered'):
            d = parse_json(raw)
            if mode == 'missing': d['examples'].pop()
            elif mode == 'duplicate': d['examples'].append(copy.deepcopy(d['examples'][0]))
            else: d['examples'].reverse()
            with self.subTest(mode=mode), self.assertRaises(ContractError):
                data.audit_bytes(json_bytes(d), manifest)

    def test_noncanonical_oversized_and_nonbyte_admission_rejected(self):
        raw, manifest = data.dataset()
        for bad in (raw+b' ', b'{}', 'not bytes', b'x'*(1024*1024+1)):
            with self.assertRaises(ContractError): data.audit_bytes(bad, manifest)
        with self.assertRaises(ContractError): data.audit_bytes(raw, b'x'*(64*1024+1))

    def test_all_cells_and_claim_gates_are_frozen(self):
        self.assertEqual(cells(), ((0,'control'), (0,'candidate'), (1,'control'),
                                  (1,'candidate'), (2,'control'), (2,'candidate')))
        for name in ('fresh_confirmation', 'learning_adequate', 'phase5_ready', 'release_ready'):
            self.assertFalse(PROTOCOL[name])
        self.assertEqual(PROTOCOL['milestones'], [0,1000,2000,3000,4000])
        self.assertEqual(PROTOCOL['geometry_loss'], '0')
        self.assertEqual(PROTOCOL['cache_mode'], 'unsupported')


if __name__ == '__main__':
    unittest.main()
