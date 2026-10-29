"""Synthetic custody mechanics only: no original weights or training execution."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.model import Decoder
from trinite.training import model_identity
from music_exposure_v1.request import core, cell_name
from music_exposure_v1.spec import CHECKPOINTS, plan_for, rows
from music_exposure_v1.reporting import summarize
from music_exposure_v1.workflow import SCOPE, CLAIMS, missing_checkpoint_files
from music_exposure_v1.controller import bound_files
from music_learning_v1.numerical import encoded
from frozen_music_eval_v1 import producer


def synthetic_checkpoint(request, request_id, seed, step):
    plan = plan_for(seed)
    model = Decoder(plan.model, lane='dense', seed=seed)
    tensors = {}
    for name, p in model.named_parameters():
        tensors['model.' + name] = p.detach().clone().contiguous()
        if step:
            tensors['optimizer.' + name + '.step'] = torch.tensor(float(step))
            tensors['optimizer.' + name + '.exp_avg'] = torch.zeros_like(p)
            tensors['optimizer.' + name + '.exp_avg_sq'] = torch.zeros_like(p)
    train = encoded(rows())
    history = [{'step': i + 1, 'loss_hex': 0.0.hex(), 'gradient_norm_hex': 0.0.hex(),
        'target_tokens': 16, 'examples': [train[(i * 8 + j) % 40]['identity'] for j in range(8)]}
        for i in range(step)]
    metadata = {'schema': 'trinite.music-exposure-checkpoint.v1', 'seed': seed,
        'request_identity': request_id, 'plan': plan.to_dict(),
        'dataset_identity': request['dataset_identity'], 'admission_identity': request['admission_identity'],
        'initialization_identity': model_identity(model), 'initial_train_loss_hex': 0.0.hex(),
        'model_identity': model_identity(model), 'step': step, 'cursor': step * 8,
        'target_tokens': step * 16, 'history': history,
        'validation': [{'step': s, 'loss_hex': 0.0.hex()} for s in range(64, step + 1, 64)]}
    return save(tensors), json_bytes(metadata)


def synthetic_packet(root):
    """Gold outputs are fabricated harness inputs, never model accuracy evidence."""
    request = core(); request_raw = json_bytes(request); request_id = identity(request_raw)
    (root / 'request.json').write_bytes(request_raw)
    records = []
    for row in rows():
        tokens = [*row['answer'].encode(), 257]
        records.append({'example_identity': row['example_identity'], 'family_id': row['family_id'],
            'task': row['task'], 'split': 'train', 'expected_tokens': tokens, 'generated_tokens': tokens,
            'correct': True, 'baseline_tokens': [55, 257], 'baseline_correct': tokens == [55, 257]})
    predictions = json_bytes({'predictions': records})
    matrix = {'schema': 'trinite.music-exposure-matrix.v1', 'request_identity': request_id,
              'cells': {}, 'outcome': 'training_fit_passed', 'both_seeds_fit': True, **CLAIMS}
    for seed in (17, 29):
        name = cell_name(seed); cell = root / name; cell.mkdir()
        (cell / 'request.json').write_bytes(request_raw)
        (cell / 'worker.log').write_bytes(b'synthetic harness only\n')
        artifacts = {}
        for step in CHECKPOINTS:
            payload, metadata = synthetic_checkpoint(request, request_id, seed, step)
            for suffix, raw in (('.safetensors', payload), ('-checkpoint.json', metadata),
                                ('-predictions.json', predictions), ('-journal.jsonl', b''.join(map(json_bytes, records)))):
                filename = f'step-{step:03d}' + suffix
                (cell / filename).write_bytes(raw); artifacts[filename] = identity(raw)
        report = {'schema': 'trinite.music-exposure-report.v1', 'seed': seed,
            'request_identity': request_id, 'admission_identity': request['admission_identity'],
            'model_identity': parse_json(metadata)['model_identity'], 'verification_scope': SCOPE,
            'step': 640, 'target_tokens': 10240, 'training_fit': True, 'update_seconds_hex': 1.0.hex(),
            'observations': {str(s): producer._score(predictions) for s in CHECKPOINTS},
            'observations_contrasts': {str(s): summarize(predictions) for s in CHECKPOINTS}, **CLAIMS}
        raw = json_bytes(report); (cell / 'report.json').write_bytes(raw); artifacts['report.json'] = identity(raw)
        completion = json_bytes({'schema': 'trinite.music-exposure-completion.v1',
            'request_identity': request_id, 'cell': name, 'artifacts': artifacts})
        (cell / 'completion.json').write_bytes(completion)
        attempt = json_bytes({'schema': 'trinite.music-exposure-attempt.v1', 'cell': name,
            'request_identity': request_id, 'outcome': 'completed', 'worker_exit': 0,
            'detail': 'synthetic harness only', 'completion_identity': identity(completion),
            'wall_seconds_hex': 1.0.hex(), 'missing_checkpoint_files': missing_checkpoint_files(cell)})
        (cell / 'attempt.json').write_bytes(attempt)
        matrix['cells'][name] = {'outcome': 'completed', 'attempt_identity': identity(attempt),
            'completion_identity': identity(completion), 'retained_files': bound_files(cell)}
    raw = json_bytes(matrix); (root / 'matrix.json').write_bytes(raw)
    return request_id, identity(raw)


class ProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    def test_full_synthetic_packet_no_optimizer_or_forward_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); request_id, matrix_id = synthetic_packet(root)
            before = producer.inventory(root)
            with patch('trinite.training.create_optimizer', side_effect=AssertionError('optimizer forbidden')), \
                 patch.object(torch.optim.AdamW, '__init__', side_effect=AssertionError('optimizer forbidden')), \
                 patch.object(Decoder, 'forward', side_effect=AssertionError('forward forbidden')):
                binding = producer.inspect_matrix(root, request_id, matrix_id)
                models = producer.load_models(root, binding)
            self.assertEqual(set(models), {17, 29})
            self.assertEqual([(c['seed'], c['step']) for c in binding['checkpoints']], [(17, 640), (29, 640)])
            self.assertEqual(before, producer.inventory(root))
            self.assertTrue(all(p.grad is None for m in models.values() for p in m.parameters()))
            with self.assertRaises(ContractError):
                producer.inspect_matrix(root, request_id, identity(b'other matrix'))
            (root / 'train-17/step-640.safetensors').write_bytes(b'not safetensors')
            with self.assertRaises(ContractError):
                producer.load_models(root, binding)

    def test_checkpoint_rejects_context_progress_slots_and_pickle(self):
        request = core(); request_id = identity(json_bytes(request))
        payload, raw = synthetic_checkpoint(request, request_id, 17, 640)
        for field, value in (('step', 64), ('seed', 29), ('target_tokens', True), ('cursor', 0),
                             ('initialization_identity', identity(b'other')), ('model_identity', identity(b'other'))):
            metadata = parse_json(raw); metadata[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                producer.checkpoint(payload, json_bytes(metadata), request, request_id, 17, 640)
        for bad in (b'\x80\x04pickle', b'', payload[:20]):
            with self.assertRaises(ContractError):
                producer.checkpoint(bad, raw, request, request_id, 17, 640)
        tensors = producer.load(payload)
        key = next(k for k in tensors if k.endswith('.exp_avg_sq'))
        tensors[key].fill_(-1)
        with self.assertRaises(ContractError):
            producer.checkpoint(save(tensors), raw, request, request_id, 17, 640)
        tensors = producer.load(payload); tensors['unrequested'] = torch.tensor(0.0)
        with self.assertRaises(ContractError):
            producer.checkpoint(save(tensors), raw, request, request_id, 17, 640)
        for seed, step in ((True, 640), (17, 1), (29, 64.0)):
            with self.assertRaises(ContractError):
                producer.checkpoint(payload, raw, request, request_id, seed, step)

    def test_resealed_complete_packet_cannot_hide_score_or_history_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); request_id, matrix_id = synthetic_packet(root)
            original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            for name, mutate in (
                ('report.json', lambda value: value['observations']['640'].update(correct=39)),
                ('step-640-checkpoint.json', lambda value: value['history'][3].update(target_tokens=15)),
            ):
                with self.subTest(name=name):
                    for path, raw in original.items(): path.write_bytes(raw)
                    cell = root / 'train-17'
                    changed = parse_json((cell / name).read_bytes()); mutate(changed)
                    changed_raw = json_bytes(changed); (cell / name).write_bytes(changed_raw)
                    completion = parse_json((cell / 'completion.json').read_bytes())
                    completion['artifacts'][name] = identity(changed_raw)
                    raw = json_bytes(completion); (cell / 'completion.json').write_bytes(raw)
                    completion_id = identity(raw)
                    attempt = parse_json((cell / 'attempt.json').read_bytes())
                    attempt['completion_identity'] = completion_id
                    raw = json_bytes(attempt); (cell / 'attempt.json').write_bytes(raw)
                    matrix = parse_json((root / 'matrix.json').read_bytes())
                    matrix['cells']['train-17'].update(completion_identity=completion_id,
                        attempt_identity=identity(raw), retained_files=bound_files(cell))
                    raw = json_bytes(matrix); (root / 'matrix.json').write_bytes(raw)
                    # Even re-binding the altered bytes cannot bypass semantic checks.
                    with self.assertRaises(ContractError):
                        producer.inspect_matrix(root, request_id, identity(raw))
                    with self.assertRaises(ContractError):
                        producer.inspect_matrix(root, request_id, matrix_id)

    def test_resealed_malformed_history_rejected(self):
        request = core(); request_id = identity(json_bytes(request))
        payload, raw = synthetic_checkpoint(request, request_id, 17, 640)
        metadata = parse_json(raw)
        metadata['history'][7]['examples'].reverse()
        with self.assertRaises(ContractError):
            producer.checkpoint(payload, json_bytes(metadata), request, request_id, 17, 640)


if __name__ == '__main__':
    unittest.main()
