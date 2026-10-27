"""Bounded synthetic checkpoint integration; not target measurement evidence."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from trinite.comparison_checkpoint import save
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.model import Decoder
from trinite.training import model_identity
from cpu_cost_ledger_v1.controller import configure, run
from cpu_cost_ledger_v1.request import core, validate
from cpu_cost_ledger_v1.workflow import audit


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def setup_input(self, root):
        model = Decoder(ModelConfig(context_length=8, blocks=1, width=4, heads=1, feed_forward_width=8), lane='dense', seed=4)
        payload = save({'model.'+n: p.detach().clone() for n,p in model.named_parameters()})
        checkpoint = root/'model.safetensors'; checkpoint.write_bytes(payload)
        value = {'config': model.config.to_dict(), 'lane': 'dense', 'model_identity': model_identity(model),
                 'checkpoint_identity': identity(payload), 'prompts': ['', 'a'], 'repetitions': 2,
                 'output_tokens': 2, 'worker_seconds': 30}
        raw = json_bytes(core(value)); request = root/'request.json'; request.write_bytes(raw)
        return checkpoint, request, raw

    def test_fresh_subprocess_closed_bundle_and_readonly_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint, request, raw = self.setup_input(root)
            original = checkpoint.read_bytes()
            result = run(root/'attempt', request, identity(raw), checkpoint)
            self.assertEqual(result['outcome'], 'completed', result)
            files = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            receipt = audit(root/'attempt', raw, identity(raw), result['completion_identity'])
            self.assertTrue(receipt['verified']); self.assertFalse(receipt['predictions_replayed'])
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(files, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError): run(root/'attempt', request, identity(raw), checkpoint)
            with self.assertRaises(ContractError): audit(root/'attempt', raw, identity(raw), 'sha256:'+'0'*64)
            import json
            attempt_path = root/'attempt'/'attempt.json'
            attempt_raw = attempt_path.read_bytes(); attempt = json.loads(attempt_raw)
            attempt['outcome'] = 'timed_out'; attempt['completion_identity'] = None
            attempt_path.write_bytes(json_bytes(attempt))
            with self.assertRaisesRegex(ContractError, 'controller'):
                audit(root/'attempt', raw, identity(raw), result['completion_identity'])
            attempt_path.write_bytes(attempt_raw)
            (root/'attempt'/'numerical.json').write_bytes(b'{}\n')
            with self.assertRaises(ContractError): audit(root/'attempt', raw, identity(raw), result['completion_identity'])

    def test_resealed_semantic_tampering_is_rejected(self):
        import copy
        import json
        import shutil
        from trinite.observation import BundleObserver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint, request, raw = self.setup_input(root)
            result = run(root/'attempt', request, identity(raw), checkpoint)
            self.assertEqual(result['outcome'], 'completed', result)
            output = root/'attempt'
            originals = {name: json.loads((output/name).read_bytes())
                         for name in ('numerical.json', 'observations.json', 'completion.json')}
            for kind in ('parity', 'positions', 'prefix', 'memory', 'negative_timing'):
                with self.subTest(kind=kind):
                    n = copy.deepcopy(originals['numerical.json'])
                    o = copy.deepcopy(originals['observations.json'])
                    c = copy.deepcopy(originals['completion.json'])
                    if kind == 'parity':
                        n['rows'][0]['parity'] = False; o['rows'][0]['parity'] = False
                    elif kind == 'positions':
                        for record in (n['rows'][0], o['rows'][0]):
                            record['all_passes_processed_positions'] += 1
                    elif kind == 'prefix':
                        o['rows'][0]['forward_observations'][0]['prefix_identity'] = 'sha256:'+'0'*64
                    elif kind == 'memory': o['native_kv_payload_bytes'] = 0
                    else: o['rows'][0]['parity_before']['wall_ns'] = -1
                    journals = []
                    for i, row in enumerate(o['rows']):
                        encoded = json_bytes(row); (output/f'row-{i:02d}.json').write_bytes(encoded)
                        journals.append(identity(encoded))
                    o['journal_identities'] = c['journal_identities'] = journals
                    nr, ob = json_bytes(n), json_bytes(o)
                    (output/'numerical.json').write_bytes(nr); (output/'observations.json').write_bytes(ob)
                    shutil.rmtree(output/'provenance')  # only fresh synthetic test artifacts
                    observer = BundleObserver(output/'provenance', actor='trinite.cpu-cost-ledger.v1')
                    observer.record('inference-cost-observation', inputs={'request.json': raw},
                                    outputs={'numerical.json': nr, 'observations.json': ob})
                    vr = json_bytes(observer.finalize()); (output/'verification.json').write_bytes(vr)
                    c['artifacts'] = {name: identity((output/name).read_bytes()) for name in c['artifacts']}
                    c['manifest_identity'] = identity((output/'provenance/manifest.json').read_bytes())
                    completion = json_bytes(c); (output/'completion.json').write_bytes(completion)
                    attempt = json.loads((output/'attempt.json').read_bytes())
                    attempt['completion_identity'] = identity(completion)
                    (output/'attempt.json').write_bytes(json_bytes(attempt))
                    with self.assertRaises(ContractError): audit(output, raw, identity(raw), identity(completion))

    def test_request_rejects_source_and_environment_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, raw = self.setup_input(Path(directory))
            self.assertEqual(validate(raw, identity(raw))['cache_mode'], 'unsupported')
            with patch('cpu_cost_ledger_v1.request.sources', return_value={}):
                with self.assertRaises(ContractError): validate(raw, identity(raw))
            with patch('cpu_cost_ledger_v1.request.environment', return_value={}):
                with self.assertRaises(ContractError): validate(raw, identity(raw))

    def test_worker_failed_import_retains_attempt_not_completion(self):
        # Structural invalid checkpoint is admitted by its exact bytes, then rejected
        # before any model forward by the worker's authoritative safe reader.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint, request, raw = self.setup_input(root)
            import json
            value = json.loads(raw)['plan']; checkpoint.write_bytes(b'not safetensors')
            value['checkpoint_identity'] = identity(checkpoint.read_bytes())
            raw = json_bytes(core(value)); request.write_bytes(raw)
            result = run(root/'attempt', request, identity(raw), checkpoint)
            self.assertEqual(result['outcome'], 'failed'); self.assertIsNone(result['completion_identity'])
            self.assertTrue((root/'attempt'/'worker.log').is_file())
            self.assertTrue((root/'attempt'/'attempt.json').is_file())


if __name__ == '__main__': unittest.main()
