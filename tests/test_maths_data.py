from collections import defaultdict
import unittest
from trinite.contracts import ContractError, json_bytes, parse_json
from trinite.maths_data import dataset, audit_bytes, label, specs, family, render, orbit
from trinite.maths_oracle import solve, parse_prompt, TASKS


class MathsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.manifest = dataset(); cls.data = parse_json(cls.raw); cls.receipt = parse_json(cls.manifest)

    def test_full_domain_labels_and_rendering_independently_checked(self):
        self.assertEqual(len(self.data['examples']), 3154)
        for spec in specs():
            self.assertEqual(label(spec), solve(spec))
            for carrier in ('infix', 'fields'): self.assertEqual(parse_prompt(render(spec, carrier)), spec)

    def test_every_task_split_and_correlated_carriers_stay_together(self):
        splits = defaultdict(set); parents = defaultdict(list)
        for e in self.data['examples']:
            splits[e['family_id']].add(e['split']); parents[e['semantic_identity']].append(e)
        self.assertTrue(all(len(v) == 1 for v in splits.values()))
        self.assertTrue(all(len(v) == 2 and len({e['split'] for e in v}) == 1 for v in parents.values()))
        self.assertEqual(set(self.receipt['task_counts']), set(TASKS))
        self.assertTrue(all(all(c[s] for s in ('train','validation','test')) for c in self.receipt['task_counts'].values()))

    def test_matrix_orbits_tasks_rhs_and_set_relabellings_share_families(self):
        a = [1, 1, 0, -1]
        for b in ([0,-1,1,1], [1,0,1,-1], [-1,-1,0,1]): self.assertEqual(orbit(a), orbit(b))
        anchor = family({'task':'determinant','values':a})
        for task in ('dot','matvec-0','solve-1'):
            self.assertEqual(anchor, family({'task':task,'values':a+([0,1] if task != 'dot' else [])}))
        self.assertEqual(family({'task':'union-count','values':[1,3]}), family({'task':'intersection-count','values':[4,6]}))

    def test_admission_label_prompt_and_split_tampering_rejected(self):
        self.assertEqual(audit_bytes(self.raw,self.manifest),self.receipt)
        for field, value in (('answer','99'),('prompt','UC:0,0='),('split','test')):
            data = parse_json(self.raw); data['examples'][0][field] = value
            if json_bytes(data) != self.raw:
                with self.assertRaises(ContractError): audit_bytes(json_bytes(data),self.manifest)
        manifest = parse_json(self.manifest); manifest['admission']['rights_basis'] = 'internet textbook'
        with self.assertRaises(ContractError): audit_bytes(self.raw,json_bytes(manifest))

    def test_known_rational_singular_boundary_and_rejections(self):
        for spec, expected in (({'task':'solve-0','values':[1,1,1,-1,1,0]}, '1/2'),
                               ({'task':'solve-1','values':[1,1,1,-1,1,0]}, '1/2'),
                               ({'task':'solve-0','values':[0,0,0,0,1,1]}, 'ERR'),
                               ({'task':'choose','values':[8,4]}, '70'),
                               ({'task':'choose','values':[0,-1]}, '0')):
            self.assertEqual(label(spec),expected);self.assertEqual(solve(spec),expected)
        for prompt in ('DT:01,0,0,1=', 'LS0:1,0,0,1,1=', 'UC:8,0=', 'ZZ:0=', 'UC:-0,1=', 'UC:True,0='):
            with self.assertRaises(ContractError): parse_prompt(prompt)
        with self.assertRaises(ContractError): solve({'task':'choose','values':[True,0]})
