"""Engineering-only real native zero-step coverage; no historical run replay.

Optimizer construction below is confined to native snapshot fixture preparation.
No optimizer step/backward runs, and diagnostic execution denies construction too.
"""
import ast
from copy import deepcopy
from contextlib import ExitStack
import io
import json
import math
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch
import warnings
import zipfile

import torch
from safetensors.torch import load, save
import forward_checkpoint_comparison_v1 as diagnostic
from trinite.contracts import identity, json_bytes
from trinite.model import Decoder
from trinite.training import TrainingState, create_optimizer, evaluate, model_identity
from trinite.comparison_checkpoint import tensor_payload
from context_symbol_learning_v1.spec import plan


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.inputs = diagnostic.pins()
        cls.rows = cls.inputs['rows']
        cls.config = plan(17, True)
        cls.original = Decoder(cls.config.model, lane='dense', seed=17)
        # Exercise the authoritative native safe snapshot writer at zero updates.
        state = TrainingState(cls.config, 'dense', cls.original,
            create_optimizer(cls.original, cls.config), {'train':cls.rows},
            identity(json_bytes(cls.rows)), 'engineering-only', model_identity(cls.original),
            evaluate(cls.original, cls.rows, 8))
        with (patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward')),
              patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('update'))):
            cls.payload = tensor_payload(state)
        cls.metadata = dict(schema='trinite.context-symbol-checkpoint.v1', binding='engineering-only',
            sources={}, environment={'engineering':'zero-step native snapshot'}, plan=cls.config.to_dict(),
            dataset_identity=state.dataset_identity, initialization_identity=state.initialization_identity,
            initial_train_loss=state.initial_train_loss, model_identity=model_identity(cls.original),
            step=0, cursor=0, target_tokens=0, history=[], train_diagnostics=[])
        cls.raw = json_bytes(cls.metadata)

    def model(self, payload=None, metadata=None):
        return diagnostic.load_model(self.payload if payload is None else payload,
            self.raw if metadata is None else json_bytes(metadata), self.config.model.to_dict())[0]

    def test_native_zero_step_forward_no_training_or_hidden_oracle(self):
        with ExitStack() as stack:
            for target in ('torch.optim.AdamW.__init__', 'torch.Tensor.backward',
                           'torch.autograd.backward', 'trinite.training.create_optimizer',
                           'context_symbol_learning_v1.numerical.state_for',
                           'context_symbol_learning_v1.numerical.restore',
                           'context_symbol_learning_v1.numerical.feasibility_kernel',
                           'context_symbol_learning_v1.data.inventory',
                           'procedural_dataset_v1.cases.expected_target'):
                stack.enter_context(patch(target, side_effect=AssertionError(target)))
            model = self.model()
            before = diagnostic.tensor_inventory(model)
            reports = diagnostic.forward_rows(model, self.rows)
            self.assertEqual(before, diagnostic.tensor_inventory(model))
        self.assertEqual(before, diagnostic.tensor_inventory(self.original))
        self.assertEqual(len(reports), 8)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertIs(model.output_weight, model.token_embedding)
        for row, report in zip(self.rows, reports):
            target = row['ids'][row['loss'].index(1):]
            self.assertEqual(report['exact'], report['greedy_tokens'] == target)
            self.assertGreaterEqual(len(report['greedy_tokens']), 1)
            self.assertLessEqual(len(report['greedy_tokens']), 3)
            if 257 in report['greedy_tokens']:
                self.assertEqual(report['greedy_tokens'][-1], 257)
            self.assertEqual([p['target'] for p in report['teacher_forced']], target)
            for p in report['teacher_forced']:
                self.assertNotEqual(p['target'], p['competitor'])
                self.assertTrue(math.isfinite(p['margin']))
                self.assertEqual(p['margin'], p['correct_logit'] - p['competitor_logit'])
                self.assertEqual(float.fromhex(p['correct_logit_hex']), p['correct_logit'])

    def test_native_logit_parity_and_metadata_identity(self):
        model = self.model()
        ids = torch.tensor([self.rows[0]['ids'][:-1]], dtype=torch.int64)
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(ids, mask).logits, self.original(ids, mask).logits))
        meta = deepcopy(self.metadata)
        meta['model_identity'] = 'sha256:' + '0'*64
        with self.assertRaisesRegex(ValueError, 'model identity'):
            self.model(metadata=meta)

    def test_safetensors_duplicate_header_and_header_bound(self):
        header = b'{"duplicate":{},"duplicate":{}}'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.model(len(header).to_bytes(8, 'little') + header)
        with self.assertRaises(ValueError):
            self.model((65537).to_bytes(8, 'little') + b' ' * 65537)

    def test_foreign_preloaded_source_rejected(self):
        foreign = types.ModuleType('trinite.model')
        foreign.__file__ = '/tmp/not-the-pinned-decoder.py'
        with patch.dict(sys.modules, {'trinite.model':foreign}), self.assertRaises(ValueError):
            diagnostic.bind_imports()

    def test_source_contract_and_eight_exposed_rows(self):
        self.assertEqual(self.inputs['prior']['model_config'], self.inputs['current']['model_config'])
        self.assertEqual(len(self.rows), 8)
        self.assertEqual([r['answer'] for r in self.rows], ['-1']*3 + ['0']*2 + ['1']*3)
        source = Path(diagnostic.__file__).read_text()
        tree = ast.parse(source)
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(n and ('training' in n or 'checkpoint' in n or 'oracle' in n) for n in imports))
        self.assertNotIn('torch.load(', source)
        self.assertNotIn('pickle', source.replace('not pickle', ''))

    def test_strict_json_duplicate_nonfinite_and_bound(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
                    b'{"x":1e999}', b' ' * ((1 << 20) + 1)):
            with self.subTest(raw=raw[:40]), self.assertRaises((ValueError, OverflowError)):
                diagnostic.strict_json(raw)

    def test_rows_closed_and_bounded(self):
        mutations = [lambda r:r[0].update(extra=True), lambda r:r.pop(),
            lambda r:r[0]['ids'].__setitem__(0, True), lambda r:r[0]['ids'].__setitem__(1, 259),
            lambda r:r[0]['mask'].__setitem__(1, 0), lambda r:r[0]['loss'].__setitem__(1, 1),
            lambda r:r[0]['loss'].__setitem__(0, False), lambda r:r[0].update(answer='2'),
            lambda r:r[0]['ids'].__setitem__(-1, 258), lambda r:r[1].update(identity=r[0]['identity']),
            lambda r:r[0]['ids'].extend([0]*256)]
        for mutate in mutations:
            rows = deepcopy(self.rows)
            mutate(rows)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                diagnostic.validate_rows(rows)

    def test_metadata_closed_and_fixed_config(self):
        mutations = [lambda m:m.update(extra=1), lambda m:m.update(step=True),
            lambda m:m.update(step=127), lambda m:m.update(schema='wrong'),
            lambda m:m['plan']['model'].update(width=64)]
        for mutate in mutations:
            metadata = deepcopy(self.metadata)
            mutate(metadata)
            with self.assertRaises(ValueError):
                self.model(metadata=metadata)

    def test_tensor_shape_dtype_nonfinite_closed_inventory(self):
        names = list(load(self.payload))
        mutations = [lambda t:t.__setitem__(names[0], t[names[0]].flatten()[:1]),
            lambda t:t.__setitem__(names[0], t[names[0]].double()),
            lambda t:t[names[0]].flatten().__setitem__(0, float('nan')),
            lambda t:t[names[0]].flatten().__setitem__(0, float('inf')),
            lambda t:t.__setitem__('surprise', torch.zeros(1)), lambda t:t.pop(names[0])]
        for mutate in mutations:
            tensors = load(self.payload)
            mutate(tensors)
            with self.assertRaises(ValueError):
                self.model(save(tensors))
        with self.assertRaises(Exception):
            self.model(b'not safetensors')
        with self.assertRaises(ValueError):
            self.model(b'0' * ((1 << 20) + 1))

    def test_discarded_optimizer_inventory_is_checked(self):
        meta = deepcopy(self.metadata)
        meta['step'] = 128
        tensors = load(self.payload)
        for name, p in self.original.named_parameters():
            tensors['optimizer.'+name+'.step'] = torch.tensor(128.)
            tensors['optimizer.'+name+'.exp_avg'] = torch.zeros_like(p)
            tensors['optimizer.'+name+'.exp_avg_sq'] = torch.zeros_like(p)
        self.model(save(tensors), meta)
        key = next(k for k in tensors if k.endswith('.step'))
        tensors[key] = torch.tensor(127.)
        with self.assertRaises(ValueError):
            self.model(save(tensors), meta)
        tensors[key] = torch.tensor(128.)
        key = next(k for k in tensors if k.endswith('.exp_avg_sq'))
        tensors[key].flatten()[0] = -1
        with self.assertRaises(ValueError):
            self.model(save(tensors), meta)

    def test_differences_zero_and_single_parameter(self):
        a, b = self.model(), self.model()
        rows = diagnostic.differences(a, b)
        self.assertEqual(sum(r['count'] for r in rows), 24768)
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(r['max_abs'] == r['symmetric_relative_max'] == r['bitwise_different_elements'] == 0 for r in rows))
        with torch.no_grad():
            next(b.parameters()).flatten()[0] += .25
        rows = diagnostic.differences(a, b)
        self.assertEqual(sum(r['bitwise_different_elements'] for r in rows), 1)
        self.assertAlmostEqual(max(r['max_abs'] for r in rows), .25, places=7)

    def test_signed_zero_is_bitwise_difference_not_numerical_error(self):
        a, b = self.model(), self.model()
        with torch.no_grad():
            next(a.parameters()).flatten()[0] = 0.0
            next(b.parameters()).flatten()[0] = -0.0
        rows = diagnostic.differences(a, b)
        self.assertEqual(sum(r['bitwise_different_elements'] for r in rows), 1)
        self.assertTrue(all(r['max_abs'] == r['symmetric_relative_max'] == 0 for r in rows))

    def test_cli_rejects_nested_evidence_output_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence = root/'evidence'
            evidence.mkdir()
            nested = evidence/'nested'
            nested.mkdir()
            arguments = ['diagnostic', '--prior', str(evidence/'old.zip'),
                '--current', str(root/'current.zip'), '--output', str(nested/'report.json')]
            with patch.object(sys, 'argv', arguments), self.assertRaisesRegex(ValueError, 'evidence directory'):
                diagnostic.main()

    def archive(self, folder, entries):
        path = Path(folder)/'test.zip'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
                for name, raw in entries:
                    z.writestr(name, raw)
        pin = dict(payload_sha256=diagnostic.digest(path.read_bytes()), members={'model':diagnostic.digest(b'ok')})
        return path, pin

    def test_zip_hash_and_member_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            path, pin = self.archive(folder, [('model', b'ok'), ('heldout.json', b'NEVER READ')])
            original_read = zipfile.ZipFile.read
            reads = []
            def tracked_read(archive, name, *args, **kwargs):
                reads.append(name)
                self.assertNotIn('heldout', name)
                return original_read(archive, name, *args, **kwargs)
            with patch.object(zipfile.ZipFile, 'read', tracked_read):
                self.assertEqual(diagnostic.read_deposit(path, pin), {'model':b'ok'})
            self.assertEqual(reads, ['model'])
            bad = deepcopy(pin)
            bad['payload_sha256'] = '0'*64
            with self.assertRaises(ValueError):
                diagnostic.read_deposit(path, bad)
            bad = deepcopy(pin)
            bad['members']['model'] = '0'*64
            with self.assertRaises(ValueError):
                diagnostic.read_deposit(path, bad)

    def test_zip_path_duplicates_symlink_and_member_bound(self):
        for name in ('../escape', '/absolute', 'a\\b', 'a/../b', 'a//b'):
            with tempfile.TemporaryDirectory() as folder:
                path, pin = self.archive(folder, [('model', b'ok'), (name, b'bad')])
                with self.assertRaises(ValueError):
                    diagnostic.read_deposit(path, pin)
        with tempfile.TemporaryDirectory() as folder:
            path, pin = self.archive(folder, [('model', b'ok'), ('model', b'ok')])
            with self.assertRaises(ValueError):
                diagnostic.read_deposit(path, pin)
        with tempfile.TemporaryDirectory() as folder:
            info = zipfile.ZipInfo('symlink')
            info.external_attr = (0o120777 << 16)
            path, pin = self.archive(folder, [('model', b'ok'), (info, b'target')])
            with self.assertRaises(ValueError):
                diagnostic.read_deposit(path, pin)
        with tempfile.TemporaryDirectory() as folder:
            path, pin = self.archive(folder, [('model', b'x' * ((1 << 20)+1))])
            with self.assertRaises(ValueError):
                diagnostic.read_deposit(path, pin)

    def test_regular_file_resource_limits_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'tiny'
            source.write_bytes(b'abc')
            self.assertEqual(diagnostic.read_file(source, 3), b'abc')
            with self.assertRaises(ValueError):
                diagnostic.read_file(source, 2)
            link = root/'link'
            link.symlink_to(source)
            with self.assertRaises(ValueError):
                diagnostic.read_file(link, 3)
            source = root/'pins.json'
            inputs = deepcopy(self.inputs)
            first = next(iter(inputs['runtime_sources']))
            inputs['runtime_sources'][first] = '0'*64
            source.write_text(json.dumps(inputs))
            with patch.object(diagnostic, 'PINS', source), self.assertRaises(ValueError):
                diagnostic.pins()


if __name__ == '__main__':
    unittest.main()
