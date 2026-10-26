"""Tiny engineering-only producer imports; never evaluates the target checkpoint."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.budget_training import plan_for
from native_diagnostic_v1 import request, workflow
from unseen_evaluation_v1 import producer


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class ProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name) / 'packet'; cls.base.mkdir()
        cls.stack = ExitStack()
        cls.plan = replace(plan_for(0, 'extended'), steps=2, validation_every=1,
            model=ModelConfig(context_length=64, blocks=1, width=16, heads=2, feed_forward_width=32))
        cls.stack.enter_context(patch('native_diagnostic_v1.request.plan_for', return_value=cls.plan))
        cls.stack.enter_context(patch.dict(request.PROTOCOL, {'milestones': [0, 1, 2]}))
        cls.stack.enter_context(patch('native_diagnostic_v1.workflow.MILESTONES', (0, 1, 2)))
        cls.stack.enter_context(patch('unseen_evaluation_v1.producer.MILESTONES', (0, 1, 2)))
        cls.stack.enter_context(patch('unseen_evaluation_v1.producer.FINAL_STEP', 2))
        cls.raw = json_bytes(request.core(0)); cls.rid = identity(cls.raw)
        cls.cid = workflow.run(cls.base, cls.raw, cls.rid)
        cls.mid = parse_json((cls.base / 'report.json').read_bytes(), canonical=True)['final_model_identity']

    @classmethod
    def tearDownClass(cls):
        cls.stack.close(); cls.temp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'packet'; shutil.copytree(self.base, self.root)

    def inspect(self):
        return producer.inspect_packet(self.root, self.rid, self.cid, self.mid)

    def test_readonly_real_custody_and_no_optimizer_or_forward(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch('torch.optim.AdamW', side_effect=AssertionError('optimizer')), \
             patch('trinite.training.run_steps', side_effect=AssertionError('updates')), \
             patch('trinite.model.Decoder.forward', side_effect=AssertionError('forward')):
            binding = self.inspect()
            model = producer.load_model(self.root, binding)
        self.assertEqual(producer.model_identity(model), self.mid)
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(binding['producer_source'], request.sources())
        self.assertEqual(binding['checkpoint_tensor_identity'], identity((self.root / 'tensors-2.safetensors').read_bytes()))

    def test_external_identities_are_required(self):
        for index in range(3):
            ids = [self.rid, self.cid, self.mid]; ids[index] = 'sha256:' + '0' * 64
            with self.subTest(index=index), self.assertRaises(ContractError):
                producer.inspect_packet(self.root, *ids)

    def test_source_and_environment_guards_not_substituted(self):
        with patch('native_diagnostic_v1.request.sources', return_value={}):
            with self.assertRaises(ContractError): self.inspect()
        with patch('native_diagnostic_v1.request.require_environment', return_value={}):
            with self.assertRaises(ContractError): self.inspect()

    def test_symlink_and_tamper_rejected(self):
        (self.root / 'linked').symlink_to(self.base)
        with self.assertRaises(ContractError): self.inspect()
        (self.root / 'linked').unlink()
        (self.root / 'report.json').write_bytes(b'{}\n')
        with self.assertRaises(ContractError): self.inspect()

    def test_binding_extra_or_altered_field_rejected(self):
        binding = self.inspect()
        for key, value in [('unknown', True), ('seed', False), ('producer_source', {})]:
            altered = {**binding, key: value}
            with self.subTest(key=key), self.assertRaises(ContractError): producer.load_model(self.root, altered)

    def checkpoint(self, change_record=None, change_tensors=None):
        raw = (self.root / 'tensors-2.safetensors').read_bytes()
        metadata = parse_json((self.root / 'metadata-2.json').read_bytes(), canonical=True)
        if change_record: change_record(metadata)
        if change_tensors:
            tensors = producer.load(raw); change_tensors(tensors); raw = save(tensors)
        return producer._checkpoint(raw, json_bytes(metadata), parse_json(self.raw, canonical=True), self.rid, 2)

    def test_history_bool_foreign_schedule_and_source_rejected(self):
        changes = [lambda r: r.update(step=True), lambda r: r.update(cursor=True),
            lambda r: r['history'][0].update(step=True),
            lambda r: r['history'][0].update(examples=['sha256:' + '0' * 64] * 8),
            lambda r: r.update(source={}), lambda r: r.update(initial_train_loss_hex='nan')]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ContractError): self.checkpoint(change_record=change)

    def test_initial_loss_must_be_continuous_across_milestones(self):
        from trinite.observation import BundleObserver
        path = self.root / 'metadata-2.json'
        record = parse_json(path.read_bytes(), canonical=True)
        record['initial_train_loss_hex'] = (float.fromhex(record['initial_train_loss_hex']) + 1).hex()
        path.write_bytes(json_bytes(record))
        completion = parse_json((self.root / 'completion.json').read_bytes(), canonical=True)
        outputs = {name: (self.root / name).read_bytes() for name in completion['artifacts']}
        shutil.rmtree(self.root / 'provenance')
        observer = BundleObserver(self.root / 'provenance')
        raw, manifest = request.corpus()
        observer.record('native-learning-diagnostic', inputs={'request.json': self.raw,
            'dataset.json': raw, 'dataset-manifest.json': manifest}, outputs=outputs)
        verification = json_bytes(observer.finalize())
        (self.root / 'verification.json').write_bytes(verification)
        completion['artifacts'] = {name: identity(raw) for name, raw in outputs.items()}
        completion['verification_identity'] = identity(verification)
        completion['manifest_identity'] = identity((self.root / 'provenance/manifest.json').read_bytes())
        encoded = json_bytes(completion); (self.root / 'completion.json').write_bytes(encoded)
        with self.assertRaisesRegex(ContractError, 'initial-loss continuity'):
            producer.inspect_packet(self.root, self.rid, identity(encoded), self.mid)

    def test_missing_extra_nonfinite_and_bad_optimizer_slots_rejected(self):
        def nonfinite(t):
            key = next(k for k in t if k.startswith('model.')); t[key].view(-1)[0] = float('nan')
        def badslot(t):
            key = next(k for k in t if k.endswith('.step')); t[key] = torch.tensor(1.0)
        for change in [lambda t: t.pop(next(iter(t))), lambda t: t.update(extra=torch.tensor(0.)), nonfinite, badslot]:
            with self.subTest(change=change), self.assertRaises(ContractError): self.checkpoint(change_tensors=change)


if __name__ == '__main__':
    unittest.main()
