"""Synthetic structural receipts only, not real original-run admission evidence."""
import copy
import unittest

from frozen_music_math_v1 import artifacts as a
from frozen_music_math_v1 import receipt as r
from frozen_music_math_v1.protocol import CHECKPOINTS


def synthetic_original(root=None):
    """Complete-shaped test doubles; payload bytes are NOT real safetensors."""
    request_raw = b'synthetic original request; never native input'
    request_id = a.digest(request_raw)
    inventory = {}
    for name in sorted(r.inventory_names()):
        raw = request_raw if name.endswith('request.json') else ('synthetic-only ' + name).encode()
        if root is not None:
            path = root / name; path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
        inventory[name] = a.digest(raw)
    checkpoints = []
    for subset, seed, step in CHECKPOINTS:
        prefix = f'{subset}-dense-{seed}/step-{step:03d}'
        checkpoints.append({'subset': subset, 'seed': seed, 'step': step,
            'payload': prefix + '.safetensors', 'metadata': prefix + '-checkpoint.json',
            'payload_identity': inventory[prefix + '.safetensors'],
            'metadata_identity': inventory[prefix + '-checkpoint.json'],
            'model_identity': a.digest(('synthetic model ' + prefix).encode())})
    packages = {'torch': '2.8.0+cpu', 'numpy': '2.3.5', 'safetensors': '0.6.2',
        'packaging': '25.0', 'filelock': '3.32.3', 'typing-extensions': '4.16.0',
        'setuptools': '78.1.0', 'sympy': '1.14.0', 'networkx': '3.6.1', 'jinja2': '3.1.6',
        'fsspec': '2026.7.0', 'mpmath': '1.3.0', 'markupsafe': '3.0.3'}
    return {'commit': r.ORIGINAL_COMMIT, 'tree': r.ORIGINAL_TREE,
        'request_identity': request_id, 'matrix_identity': inventory['matrix.json'],
        'admission_identity': a.digest(b'synthetic admission'),
        'source': {path: a.digest(('synthetic source ' + path).encode()) for path in (
            'scripts/music_overfit_v1/numerical.py', 'src/trinite/comparison_checkpoint.py',
            'src/trinite/comparison.py')},
        'environment': {'python': '3.11.9', 'platform': 'Linux-synthetic-test-only',
            'machine': 'x86_64', 'torch': '2.8.0+cpu',
            'cpu': {'vendor_id': 'synthetic', 'model name': 'test-double', 'flags': 'synthetic'},
            'cpu_kernel_capability': 'AVX2', 'default_dtype': 'torch.float32',
            'default_device': 'cpu', 'threads': 1, 'interop_threads': 1,
            'deterministic': True, 'dtype': 'float32', 'device': 'cpu',
            'optimizer': 'AdamW-single-tensor', 'packages': packages},
        'inventory': inventory, 'checkpoints': checkpoints, 'safe_structure_verified': True,
        'optimizer_replayed': False, 'predictions_replayed': False}


class ReceiptContracts(unittest.TestCase):
    def test_complete_shape_and_actual_pinned_endpoint_names(self):
        value = synthetic_original()
        self.assertEqual(len(value['inventory']), 82)
        self.assertEqual(r.validate_original(value), value)
        self.assertEqual(len(value['checkpoints']), 8)

    def test_top_level_pin_claim_and_environment_rejections(self):
        changes = [('commit', '0'*40), ('tree', '0'*40), ('request_identity', '0'*64),
                   ('safe_structure_verified', 1), ('optimizer_replayed', True),
                   ('predictions_replayed', True), ('environment', {}), ('source', {})]
        for field, replacement in changes:
            value = synthetic_original(); value[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.validate_original(value)
        value = synthetic_original(); value['new'] = False
        with self.assertRaises(ValueError): r.validate_original(value)
        value = synthetic_original(); value['environment']['threads'] = True
        with self.assertRaises(ValueError): r.validate_original(value)
        value = synthetic_original(); value['environment']['packages']['numpy'] = 'unbound'
        with self.assertRaises(ValueError): r.validate_original(value)

    def test_sources_require_safe_paths_and_digest_types(self):
        for path in ('.', '/absolute.py', '../escape.py', 'a/../b.py', 'a//b.py', 'a\\b.py', 'a\nb.py'):
            value = synthetic_original(); value['source'][path] = a.digest(b'bad')
            with self.subTest(path=path), self.assertRaises(ValueError):
                r.validate_original(value)
        value = synthetic_original()
        value['source']['src/trinite/comparison.py'] = True
        with self.assertRaises(ValueError): r.validate_original(value)

    def test_inventory_missing_extra_unbound_and_wrong_endpoint_rejected(self):
        changes = []
        value = synthetic_original(); del value['inventory']['one-dense-17/step-008.safetensors']; changes.append(value)
        value = synthetic_original(); value['inventory']['extra'] = a.digest(b'extra'); changes.append(value)
        value = synthetic_original(); value['inventory']['matrix.json'] = a.digest(b'changed'); changes.append(value)
        value = synthetic_original(); value['inventory']['one-dense-17/request.json'] = a.digest(b'changed'); changes.append(value)
        value = synthetic_original(); value['checkpoints'].reverse(); changes.append(value)
        value = synthetic_original(); value['checkpoints'][0]['payload'] = '../other.safetensors'; changes.append(value)
        value = synthetic_original(); value['checkpoints'][0]['payload_identity'] = a.digest(b'changed'); changes.append(value)
        value = synthetic_original(); value['checkpoints'][0]['step'] = False; changes.append(value)
        value = synthetic_original(); value['checkpoints'][0]['model_identity'] = 1; changes.append(value)
        value = synthetic_original(); value['checkpoints'][1] = copy.deepcopy(value['checkpoints'][0]); changes.append(value)
        for value in changes:
            with self.assertRaises(ValueError): r.validate_original(value)

    def test_attempt_exact_authorization_schema_and_context(self):
        request_id = a.digest(b'probe request')
        value = {'schema': r.ATTEMPT_SCHEMA, 'request_identity': request_id,
                 'authorization': 'explicit-inference-flag', 'selection': r.SELECTION}
        self.assertEqual(r.validate_attempt(value, request_id), value)
        for field, replacement in [('schema', 'other'), ('request_identity', a.digest(b'other')),
                                   ('authorization', True), ('selection', 'best final')]:
            changed = dict(value); changed[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.validate_attempt(changed, request_id)
        changed = dict(value); changed['extra'] = False
        with self.assertRaises(ValueError): r.validate_attempt(changed, request_id)


if __name__ == '__main__':
    unittest.main()
