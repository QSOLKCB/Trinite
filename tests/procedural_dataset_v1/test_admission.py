"""Real three-module admission plus adversarial artifact and boundary contracts."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import queue_evaluation_v1 as gate
from procedural_dataset_v1 import admission as a, cases, transport
from trinite.tokenizer import BOS, EOS


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='procedural-admission-test-')
        cls.base = Path(cls.temp.name) / 'base'
        train_requests = {(row['module'], cases.canonical(row['request']))
                          for row in a.inventory(17) if row['split'] == 'train'}
        cls.privileged_calls = []
        actual_run = transport.run
        def checked(module, operation, request):
            if operation != 'observe':
                key = (module, cases.canonical(request))
                if key not in train_requests:
                    raise AssertionError('held-out native evaluation during admission')
                cls.privileged_calls.append(key)
            return actual_run(module, operation, request)
        with mock.patch.object(transport, 'run', side_effect=checked):
            cls.anchor = a.prepare(cls.base)
        cls.raw = {name: (cls.base / name).read_bytes() for name in a.FILES}
        cls.values = {name: gate.decode(data) for name, data in cls.raw.items()}
        cls.regenerated = (cls.values['requests.json'], cls.values['public.json'],
                           cls.values['manifest.json']['audit'], cls.values['train-targets.json'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='procedural-admission-case-')
        self.path = Path(self.scratch.name) / 'bundle'
        shutil.copytree(self.base, self.path)

    def tearDown(self):
        self.scratch.cleanup()

    def reseal(self, filename, mutate):
        for name, raw in self.raw.items():
            (self.path / name).write_bytes(raw)
        values = copy.deepcopy(self.values)
        mutate(values[filename])
        raw = gate.encode(values[filename])
        (self.path / filename).write_bytes(raw)
        manifest = values['manifest.json']
        if filename != 'manifest.json':
            manifest['files'][filename] = gate.digest(raw)
        raw_manifest = gate.encode(manifest)
        (self.path / 'manifest.json').write_bytes(raw_manifest)
        return gate.digest(raw_manifest)

    def reject(self, anchor=None):
        # The reference is produced by actual modules in setUpClass. Avoid hundreds
        # of redundant native starts for artifact-only corruption; the independent
        # successful verification test below reruns every required native oracle.
        def regenerated(seed):
            gate.integer(seed, 0, 65535)
            return copy.deepcopy(self.regenerated)
        with mock.patch.object(a, 'materialize', side_effect=regenerated):
            with self.assertRaises((ValueError, OSError)):
                a.verify(self.path, anchor or self.anchor)

    def test_actual_read_only_regeneration_and_train_export(self):
        before = {name: (self.path / name).read_bytes() for name in a.FILES}
        exported = a.training_rows(self.path, self.anchor)  # actual full native verification
        actual = self.values
        self.assertEqual(exported, actual['train-targets.json'])
        self.assertEqual(before, {name: (self.path / name).read_bytes() for name in a.FILES})
        train_ids = {row['id'] for row in actual['requests.json'] if row['split'] == 'train'}
        self.assertEqual({row['id'] for row in exported}, train_ids)
        self.assertEqual(len(exported), len(train_ids))
        self.assertEqual({r['module'] for r in actual['requests.json'] if r['split'] == 'train'},
                         {'ternary', 'tombstone', 'context'})
        self.assertFalse(a.PROTOCOL['execution_enabled'])

    def test_resealed_label_and_token_tampering(self):
        for field, value in [('answer', 'forged'), ('tokens', [0]),
                             ('attention_mask', [0]), ('loss_mask', [1]), ('id', 'foreign')]:
            with self.subTest(field=field):
                anchor = self.reseal('train-targets.json', lambda v, k=field, x=value: v[0].__setitem__(k, x))
                self.reject(anchor)

    def test_resealed_provenance_inventory_and_counts(self):
        mutations = [
            ('admission.json', lambda v: v.__setitem__('rights_basis', 'all internet data')),
            ('admission.json', lambda v: v.__setitem__('outcome', 'rejected')),
            ('attempt.json', lambda v: v.__setitem__('seed', 18)),
            ('requests.json', lambda v: v[0].__setitem__('split', 'test')),
            ('requests.json', lambda v: v[0].__setitem__('seed', 18)),
            ('requests.json', lambda v: v[0].__setitem__('id', '../../foreign')),
            ('requests.json', lambda v: v.append(v[0])),
            ('public.json', lambda v: v[0].__setitem__('target', 1)),
            ('public.json', lambda v: v.pop()),
            ('manifest.json', lambda v: v['audit'].__setitem__('families', 999)),
            ('manifest.json', lambda v: v['files'].__setitem__('../outside.json', '0'*64)),
            ('manifest.json', lambda v: v['protocol'].__setitem__('execution_enabled', True)),
            ('manifest.json', lambda v: v['sources'].__setitem__('src/trinite/tokenizer.py', '0'*64)),
            ('manifest.json', lambda v: v['binaries']['context'].__setitem__('binary_sha256', '0'*64)),
            ('manifest.json', lambda v: v.__setitem__('seed', True)),
            ('manifest.json', lambda v: v.__setitem__('extra', 1)),
            ('manifest.json', lambda v: v.pop('families')),
        ]
        for filename, mutation in mutations:
            with self.subTest(filename=filename, mutation=mutation):
                self.reject(self.reseal(filename, mutation))

    def test_external_anchor_required(self):
        self.reject('0'*64)
        self.reject('../manifest.json')

    def test_invalid_json_and_noncanonical_bytes(self):
        original = self.raw['manifest.json']
        for raw in [original[:-2], b'{"schema":1,"schema":2}', b'{}',
                    original.replace(b'"seed":17', b'"seed":17.0'),
                    b' ' + original, original.replace(b'"seed":17', b'"seed":-0')]:
            with self.subTest(raw=raw[:60]):
                (self.path / 'manifest.json').write_bytes(raw)
                self.reject(gate.digest(raw))

    def test_missing_extra_symlink_hardlink_and_oversize(self):
        path = self.path / 'public.json'
        path.unlink()
        self.reject()
        path.write_bytes(self.raw['public.json'])
        extra = self.path / 'extra.json'
        extra.write_bytes(b'{}')
        self.reject()
        extra.unlink()
        path.unlink()
        path.symlink_to(self.base / 'public.json')
        self.reject()
        path.unlink()
        os.link(self.base / 'public.json', path)
        self.reject()
        path.unlink()
        path.write_bytes(b' ' * (a.LIMIT + 1))
        self.reject()

    def test_failures_preserved_and_fresh_destination_only(self):
        destination = Path(self.scratch.name) / 'failure'
        evidence = {'exit_code': 3, 'failure': 'timeout', 'stdout_b64': '', 'stderr_b64': ''}
        with mock.patch.object(a, 'observed', side_effect=gate.ProcessError(evidence)):
            with self.assertRaises(gate.ProcessError):
                a.prepare(destination)
        failure = gate.decode((destination / 'failure.json').read_bytes())
        self.assertFalse(failure['success'])
        self.assertEqual(failure['process'], evidence)
        self.assertEqual(failure['context']['row'], a.inventory(17)[0])
        self.assertEqual(failure['context']['operation'], 'observe')
        attempt = gate.decode((destination / 'attempt.json').read_bytes())
        self.assertEqual(attempt, {'seed': 17, 'protocol': a.PROTOCOL,
                                  'sources': a.sources(), 'binaries': transport.identities()})
        self.assertEqual(failure['seed'], 17)
        self.assertFalse((destination / 'manifest.json').exists())
        with self.assertRaises(OSError):
            a.prepare(destination)
        self.assertEqual(gate.decode((destination / 'failure.json').read_bytes()), failure)

    def test_no_heldout_privileged_calls(self):
        train_requests = {(row['module'], cases.canonical(row['request']))
                          for row in self.values['requests.json'] if row['split'] == 'train'}
        self.assertEqual(set(self.privileged_calls), train_requests)
        self.assertEqual(len(self.privileged_calls), len(train_requests))

    def test_public_schema_and_inference_no_metadata_or_target(self):
        for row, observation in zip(self.values['requests.json'], self.values['public.json']):
            module = row['module']
            with mock.patch.object(a, 'target', side_effect=AssertionError('target leaked')):
                tokens = a.public_only(observation, module)
            self.assertEqual(tokens[0], BOS)
            self.assertNotIn(EOS, tokens)
            self.assertEqual(bytes(tokens[1:]).decode('ascii'), a.prompt(module, observation))
            for forbidden in ('seed', 'family', 'split', 'target', 'receipt', 'source_path'):
                with self.assertRaises(ValueError):
                    a.public_only(dict(observation, **{forbidden: 1}), module)

    def test_byte_budget_exact_boundary_and_loss_mask(self):
        observation = dict(schema=cases.OBSERVATION_SCHEMAS['context'], contexts=8,
                           symbols=8, meanings=2, table=[[1]*8 for _ in range(8)],
                           context_mask=[True]*2+[False]*6, symbol_mask=[True]*8, query=0)
        prompt = a.prompt('context', observation)
        self.assertEqual(len(prompt.encode('ascii')), 252)
        encoding = a.supervised('context', observation, '-1')
        self.assertEqual(len(encoding.input_ids), 256)
        self.assertEqual(encoding.input_ids[-1], EOS)
        boundary = 1 + len(prompt.encode('ascii'))
        self.assertTrue(all(x == 0 for x in encoding.loss_mask[:boundary]))
        self.assertTrue(all(x == 1 for x in encoding.loss_mask[boundary:]))
        overlength = dict(observation, context_mask=[True]+[False]*7)
        with self.assertRaises(ValueError):
            a.supervised('context', overlength, '-1')
        with self.assertRaises(ValueError):
            a.supervised('context', observation, '000')  # task-specific answer cap

    def test_seed_and_duplicate_prefix_guards(self):
        for seed in (-1, 65536, True, 1.0, '17'):
            with self.assertRaises(ValueError):
                a.inventory(seed)
        rows, public, _, _ = copy.deepcopy(self.regenerated)
        with self.assertRaises(ValueError):
            a.audit(rows + rows[:1], public + public[:1])
        indices = [i for i, r in enumerate(rows) if r['module'] == 'context']
        i = indices[0]
        j = next(j for j in indices if public[j]['query'] != public[i]['query'] and
                 {k: v for k, v in public[j].items() if k != 'query'} ==
                 {k: v for k, v in public[i].items() if k != 'query'})
        rows[j]['family'] = 'forged-family'
        with self.assertRaises(ValueError):
            a.audit(rows, public)


if __name__ == '__main__':
    unittest.main()
