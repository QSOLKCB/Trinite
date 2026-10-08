"""Frozen three-lane numerical comparison and detached evaluation orchestration."""
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import resource
import time

import torch

from .contracts import ContractError, identity, json_bytes, parse_json
from .model import Decoder, CaptureSpec, Linear
from .quantizer import quantize, quantize_four
from .tokenizer import ByteTokenizer, BOS, EOS
from .inspection import inventory, tensor_identity
from .training import require_environment, environment, model_identity, evaluate, run_steps
from .comparison_training import (PROTOCOL_IDENTITY, PROTOCOL_COMMIT, LANES, WORKLOADS, SEEDS,
                                  protocol, sources, corpus, state_for)
from .geometry import alignment, rademacher, verify_geometry_pin
from .observation import BundleObserver

def freeze_request(path):
    require_environment()
    request = {'schema': 'trinite.comparison-request.v1', 'protocol': protocol(),
               'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
               'source': sources(), 'environment': environment(),
               'geometry_pin': verify_geometry_pin(),
               'corpora': {w: {'dataset': identity(corpus(w)[0]), 'manifest': identity(corpus(w)[1])}
                           for w in WORKLOADS},
               'created_at': datetime.now(timezone.utc).isoformat(),
               'clock_assurance': 'local; source protocol commit precedes target outcomes'}
    raw = json_bytes(request)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        out.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY,
            'planned_cells': 27}


def read_request(path, expected_identity):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ContractError('comparison request requires a regular file')
    with path.open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or identity(raw) != expected_identity:
        raise ContractError('comparison request identity/size mismatch')
    r = parse_json(raw, canonical=True)
    expected = {'schema': 'trinite.comparison-request.v1', 'protocol': protocol(),
                'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
                'source': sources(), 'environment': require_environment(),
                'geometry_pin': verify_geometry_pin(),
                'corpora': {w: {'dataset': identity(corpus(w)[0]), 'manifest': identity(corpus(w)[1])}
                            for w in WORKLOADS},
                'clock_assurance': 'local; source protocol commit precedes target outcomes'}
    if type(r) is not dict or set(r) != set(expected)|{'created_at'}:
        raise ContractError('invalid comparison request fields')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('comparison request source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(r['created_at']).utcoffset() is None:
            raise ValueError('timezone required')
    except (ValueError, TypeError) as error:
        raise ContractError('invalid comparison request time') from error
    return r, raw


def greedy(model, prompt, max_tokens):
    ids = [BOS, *prompt.encode('utf-8')]
    emitted = []
    with torch.no_grad():
        for _ in range(max_tokens):
            if len(ids) > model.config.context_length:
                break
            tensor = torch.tensor([ids], dtype=torch.int64, device='cpu')
            logits = model(tensor, torch.ones_like(tensor, dtype=torch.bool)).logits
            token = int(logits[0, -1].argmax())
            emitted.append(token)
            if token == EOS:
                break
            ids.append(token)
    return emitted


def score_model(model, workload):
    examples = parse_json(corpus(workload)[0], canonical=True)['examples']
    tokenizer = ByteTokenizer(model.config.context_length)
    # One workload-wide cap, frozen from formal grammar, never a per-item answer hint.
    cap = {'formal': 2, 'qutrit': 5, 'ququart': 6}[workload]
    scored = []
    splits = {}
    for split in ('train', 'validation', 'test'):
        group = [e for e in examples if e['split'] == split]
        encoded = []
        for e in group:
            enc = tokenizer.encode(e['text'], answer_start=len(e['prompt'].encode()))
            encoded.append({'ids': enc.input_ids, 'mask': enc.attention_mask, 'loss': enc.loss_mask})
            tokens = greedy(model, e['prompt'], cap)
            expected = [*e['answer'].encode(), EOS]
            scored.append({'example_identity': e['example_identity'], 'family_id': e['family_id'],
                'task': e.get('task', 'modular-add'), 'split': split,
                'expected_tokens': expected, 'generated_tokens': tokens, 'correct': tokens == expected})
        rows = [e for e in scored if e['split'] == split]
        families = defaultdict(list); tasks = defaultdict(list)
        for row in rows:
            families[row['family_id']].append(int(row['correct']))
            tasks[row['task']].append(int(row['correct']))
        family_values = {name: (sum(v)/len(v)).hex() for name, v in sorted(families.items())}
        splits[split] = {'examples': len(rows), 'correct': sum(r['correct'] for r in rows),
            'accuracy_hex': (sum(r['correct'] for r in rows)/len(rows)).hex(),
            'family_macro_accuracy_hex': (math.fsum(float.fromhex(v) for v in family_values.values())/len(family_values)).hex(),
            'families': family_values, 'tasks': {k: (sum(v)/len(v)).hex() for k, v in sorted(tasks.items())},
            'target_loss_hex': evaluate(model, encoded, 4)}
    return splits, scored


def weight_diagnostics(model):
    results = []
    for name, module in model.named_modules():
        if not isinstance(module, Linear):
            continue
        if model.lane == 'dense':
            effective = module.weight.detach(); codes = None
        else:
            q = quantize(module.weight) if model.lane == 'ternary' else quantize_four(module.weight)
            effective, codes = q.weight.detach(), q.codes
        difference = effective.double()-module.weight.detach().double()
        results.append({'module': name, 'elements': module.weight.numel(),
            'mse_hex': float(difference.square().mean()).hex(),
            'max_error_hex': float(difference.abs().max()).hex(),
            'code_occupancy': None if codes is None else
                {str(i): int((codes == i).sum()) for i in ((-1, 0, 1) if model.lane == 'ternary' else (-3, -1, 1, 3))}})
    return results


def geometry_measure(model, workload):
    groups = defaultdict(list)
    for e in parse_json(corpus(workload)[0], canonical=True)['examples']:
        if e['split'] == 'train':
            groups[(e['family_id'], e.get('task', 'modular-add'))].append(e)
    records = []
    for (family, task), group in sorted(groups.items()):
        identities = sorted({e['semantic_identity'] for e in group})[:3]
        selected = [e for e in group if e['semantic_identity'] in identities]
        if len(selected) != 6:
            raise ContractError('geometry requires three complete carrier pairs')
        for e in sorted(selected, key=lambda e: e['example_identity']):
            ids = [BOS, *e['prompt'].encode()]
            tensor = torch.tensor([ids], dtype=torch.int64, device='cpu')
            with torch.no_grad():
                output = model(tensor, torch.ones_like(tensor, dtype=torch.bool),
                               capture=CaptureSpec(('final',), tuple(range(1, len(ids)))))
            records.append({'family': family, 'task': task, 'item': e['semantic_identity'],
                'carrier': e['carrier'], 'example_identity': e['example_identity'],
                'tensor_identity': tensor_identity(output.captures['final']),
                'points': output.captures['final'][0].tolist()})
    conditions = {}
    for condition in ('native', 'one-sided-hash-permutation', 'rademacher'):
        contrasts = []; pairs = []
        for family, task in sorted(groups):
            group = [r for r in records if (r['family'], r['task']) == (family, task)]
            same, different = [], []
            for a, b in itertools.combinations(group, 2):
                same_item = a['item'] == b['item'] and a['carrier'] != b['carrier']
                different_item = a['item'] != b['item'] and a['carrier'] == b['carrier']
                if not (same_item or different_item):
                    continue
                left, right = a['points'], b['points']
                if condition == 'one-sided-hash-permutation':
                    order = sorted(range(len(right)), key=lambda i: identity(json_bytes(
                        {'seed': 17, 'example': b['example_identity'], 'position': i})))
                    right = [right[i] for i in order]
                elif condition == 'rademacher':
                    left = rademacher(left, a['example_identity'], 17)
                    right = rademacher(right, b['example_identity'], 17)
                value = alignment(left, right)
                (same if same_item else different).append(value)
                pairs.append({'family': family, 'task': task, 'left': a['example_identity'],
                    'right': b['example_identity'], 'same_item': same_item, 'cosine_hex': value.hex()})
            if len(same) != 3 or len(different) != 6:
                raise ContractError('incomplete geometry comparison matrix')
            contrasts.append({'family': family, 'task': task,
                'contrast_hex': (math.fsum(same)/3-math.fsum(different)/6).hex()})
        values = [float.fromhex(v['contrast_hex']) for v in contrasts]
        conditions[condition] = {'mean_hex': (math.fsum(values)/len(values)).hex(),
            'min_hex': min(values).hex(), 'max_hex': max(values).hex(), 'groups': contrasts,
            'pairs': pairs, 'evidence_class': 'OBSERVATION' if condition == 'native' else 'SIMULATION'}
    return conditions, records


def inference_timing(model, workload):
    prompts = [e['prompt'] for e in parse_json(corpus(workload)[0], canonical=True)['examples']
               if e['split'] == 'test']
    def one_pass():
        with torch.no_grad():
            for prompt in prompts:
                ids = torch.tensor([[BOS, *prompt.encode()]], dtype=torch.int64, device='cpu')
                model(ids, torch.ones_like(ids, dtype=torch.bool))
    one_pass()
    samples = []
    for _ in range(3):
        clock = time.perf_counter(); one_pass(); samples.append(time.perf_counter()-clock)
    return {'whole_test_prompt_pass_seconds_hex': [v.hex() for v in samples],
            'prompts_per_pass': len(prompts), 'scope': 'full CPU float32 prompt forward; one warmup; no generation, imports or observer'}


def _write(root, name, value):
    raw = json_bytes(value)
    with (root/name).open('xb') as out:
        out.write(raw)
    return raw


def train_cell(root, workload, lane, seed, request, request_identity):
    from .comparison_checkpoint import snapshot
    _, request_raw = read_request(request, request_identity)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    try:
        clock = time.perf_counter()
        state = state_for(workload, lane, seed)
        run_steps(state)
        seconds = time.perf_counter()-clock
        payload, metadata = snapshot(state, workload, request_identity)
        (root/'tensors.safetensors').write_bytes(payload)
        (root/'metadata.json').write_bytes(metadata)
        _write(root, 'steps.json', {'steps': state.history, 'validation': state.validation})
        result = {'schema': 'trinite.comparison-training.v1', 'outcome': 'completed',
            'workload': workload, 'lane': lane, 'seed': seed, 'request_identity': request_identity,
            'initialization_identity': state.initialization_identity, 'model_identity': model_identity(state.model),
            'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity,
            'plan_identity': state.config.content_identity, 'parameter_count': state.config.model.parameter_count,
            'target_tokens': state.target_tokens, 'steps': state.step,
            'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
            'initial_train_loss_hex': state.initial_train_loss,
            'final_validation_loss_hex': state.validation[-1]['loss_hex'],
            'tensors_identity': identity(payload), 'metadata_identity': identity(metadata),
            'training_wall_seconds_hex': seconds.hex(),
            'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'rss_scope': 'fresh Linux training worker lifetime, including imports; not incremental allocation',
            'snapshot_bytes': len(payload), 'latent_float32_bytes': state.config.model.parameter_count*4}
        read_request(request, request_identity)
        _write(root, 'training.json', result)
        return result
    except Exception as error:
        _write(root, 'failure.json', {'outcome': 'failed', 'stage': 'training', 'error': type(error).__name__+': '+str(error),
                                    'workload': workload, 'lane': lane, 'seed': seed})
        raise


def evaluate_cell(root, workload, lane, seed, request, request_identity):
    from .comparison_checkpoint import restore
    _, request_raw = read_request(request, request_identity)
    root = Path(root)
    try:
        info = parse_json((root/'training.json').read_bytes(), canonical=True)
        if any(info.get(k) != v for k, v in {'workload': workload, 'lane': lane, 'seed': seed,
                                            'request_identity': request_identity, 'outcome': 'completed'}.items()):
            raise ContractError('training cell/context mismatch')
        state = restore((root/'tensors.safetensors').read_bytes(), (root/'metadata.json').read_bytes(),
            workload, lane, seed, request_identity, (info['tensors_identity'], info['metadata_identity']))
        if state.step != state.config.steps or model_identity(state.model) != info['model_identity']:
            raise ContractError('evaluation requires the elected final model')
        splits, predictions = score_model(state.model, workload)
        before = model_identity(state.model)
        geometry, captures = geometry_measure(state.model, workload)
        timing = inference_timing(state.model, workload)
        if model_identity(state.model) != before or any(p.grad is not None for p in state.model.parameters()):
            raise ContractError('evaluation mutated the model')
        _write(root, 'predictions.json', {'predictions': predictions})
        _write(root, 'geometry.json', geometry)
        _write(root, 'captures.json', {'records': [{**r,
            'points': [[float(x).hex() for x in row] for row in r['points']]} for r in captures]})
        _write(root, 'inventory.json', inventory(state.model))
        result = {'schema': 'trinite.comparison-evaluation.v1', 'outcome': 'completed',
            'workload': workload, 'lane': lane, 'seed': seed, 'model_identity': before,
            'request_identity': request_identity, 'scores': splits, 'weights': weight_diagnostics(state.model),
            'inference': timing, 'geometry': {k: v['mean_hex'] for k, v in geometry.items()},
            'evidence_scope': protocol()['evidence_scope'], 'result_status': 'inconclusive',
            'replication_status': 'not_attempted', 'quantum_execution': False, 'packed_inference': False}
        _write(root, 'evaluation.json', result)
        read_request(request, request_identity)
        data, manifest = corpus(workload)
        collector = BundleObserver(root/'provenance')
        collector.record('train-and-evaluate-comparison', inputs={'request.json': request_raw,
            'dataset.json': data, 'dataset-manifest.json': manifest}, outputs={name: (root/name).read_bytes()
                for name in ('training.json', 'steps.json', 'tensors.safetensors', 'metadata.json',
                             'evaluation.json', 'predictions.json', 'geometry.json', 'captures.json', 'inventory.json')})
        verification = collector.finalize()
        _write(root, 'verification.json', verification)
        eligible = bool(verification['integrity_verified'] and verification['manifest_scope'] == 'closed'
                        and not verification['known_missing_artifacts'] and not verification['errors'])
        _write(root, 'report.json', {'compute_outcome': 'completed', 'observer_evidence_eligible': eligible,
                                   'verification': verification, 'release_ready': False})
        if not eligible:
            raise ContractError('comparison evidence failed actual upstream verification')
        return result
    except Exception as error:
        if not (root/'failure.json').exists():
            _write(root, 'failure.json', {'outcome': 'failed', 'stage': 'evaluation',
                                       'error': type(error).__name__+': '+str(error)})
        raise


def verified_cell(directory, workload, lane, seed, request_identity):
    """Bind consumed result copies to actual verified bundle members/index."""
    from .observation import verify_observation
    verification = verify_observation(directory/'provenance')
    manifest = parse_json((directory/'provenance/manifest.json').read_bytes())
    members = {a['content_identity'] for a in manifest['core']['artifacts']}
    indexes = []
    for key in members:
        raw = (directory/'provenance/artifacts/sha256'/key[7:]).read_bytes()
        try:
            candidate = parse_json(raw)
        except ContractError:
            continue
        if type(candidate) is dict and candidate.get('schema') == 'trinite.observer-index.v1':
            indexes.append(candidate)
    if len(indexes) != 1:
        raise ContractError('comparison bundle requires one verified name index')
    names = indexes[0]['artifacts']
    if names.get('request.json') != request_identity:
        raise ContractError('cell bundle is not bound to the elected comparison request')
    results = {}
    for name in ('training.json', 'evaluation.json'):
        path = directory/name
        if path.is_symlink() or not path.is_file():
            raise ContractError('unsafe/missing cell result')
        with path.open('rb') as stream:
            raw = stream.read(1024*1024+1)
        if len(raw) > 1024*1024 or identity(raw) != names.get(name) or names[name] not in members:
            raise ContractError('cell result copy differs from verified retained evidence')
        r = parse_json(raw, canonical=True)
        if any(r.get(k) != v for k, v in {'workload': workload, 'lane': lane, 'seed': seed,
                'request_identity': request_identity, 'outcome': 'completed'}.items()):
            raise ContractError('cell result context mismatch')
        results[name] = r
    return {'training': results['training.json'], 'evaluation': results['evaluation.json'],
            'verification': verification, 'outcome': 'completed'}


def _comparison_groups(cells):
    """Build aggregates locally; publish only after all pairs and metrics validate."""
    by_key = {(c['training']['workload'], c['training']['seed'], c['training']['lane']): c for c in cells}
    for workload, seed in itertools.product(WORKLOADS, SEEDS):
        runs = [by_key[workload, seed, lane]['training'] for lane in LANES]
        for name in ('initialization_identity', 'dataset_identity', 'manifest_identity', 'plan_identity',
                     'parameter_count', 'target_tokens', 'schedule_identity', 'steps'):
            if len({r[name] for r in runs}) != 1:
                raise ContractError(f'unmatched comparison {name}: {workload}, seed {seed}')
    groups = {}
    for workload in WORKLOADS:
        groups[workload] = {}
        for lane in LANES:
            values, diffs, wall_ratios, latent_ratios = [], [], [], []
            for seed in SEEDS:
                cell, dense = by_key[workload, seed, lane], by_key[workload, seed, 'dense']
                value = float.fromhex(cell['evaluation']['scores']['test']['family_macro_accuracy_hex'])
                baseline = float.fromhex(dense['evaluation']['scores']['test']['family_macro_accuracy_hex'])
                wall = float.fromhex(cell['training']['training_wall_seconds_hex'])
                dense_wall = float.fromhex(dense['training']['training_wall_seconds_hex'])
                latent = cell['training']['latent_float32_bytes']
                dense_latent = dense['training']['latent_float32_bytes']
                if (not math.isfinite(value) or not 0 <= value <= 1
                        or not math.isfinite(baseline) or not 0 <= baseline <= 1
                        or not math.isfinite(wall) or wall <= 0
                        or not math.isfinite(dense_wall) or dense_wall <= 0
                        or type(latent) is not int or latent <= 0
                        or type(dense_latent) is not int or dense_latent <= 0):
                    raise ContractError(f'invalid comparison metrics: {workload}, seed {seed}, {lane}')
                wall_ratio, latent_ratio = wall / dense_wall, latent / dense_latent
                if not math.isfinite(wall_ratio) or not math.isfinite(latent_ratio):
                    raise ContractError('nonfinite comparison resource ratio')
                values.append(value); diffs.append(value-baseline)
                wall_ratios.append(wall_ratio); latent_ratios.append(latent_ratio)
            groups[workload][lane] = {'seed_accuracy_hex': [v.hex() for v in values],
                'mean_hex': (math.fsum(values)/len(SEEDS)).hex(),
                'min_hex': min(values).hex(), 'max_hex': max(values).hex(),
                'paired_accuracy_difference_hex': [v.hex() for v in diffs],
                'paired_training_wall_ratio_hex': [v.hex() for v in wall_ratios],
                'paired_latent_byte_ratio_hex': [v.hex() for v in latent_ratios],
                'quality_margin_each_seed': [v >= -.05 for v in diffs],
                'wall_margin_each_seed': [v <= 1.25 for v in wall_ratios],
                'latent_byte_margin_each_seed': [v <= 1 for v in latent_ratios],
                'interpretation': 'descriptive only; small visible tasks and three seeds; no significance or superiority claim'}
    return groups


def summarize(root, request, request_identity, worker_outcomes):
    read_request(request, request_identity)
    root = Path(root); cells = []; groups = {}; aggregate_errors = []
    for workload, seed, lane in itertools.product(WORKLOADS, SEEDS, LANES):
        directory = root/f'{workload}-{seed}-{lane}'
        failure = worker_outcomes.get(directory.name)
        if failure or not (directory/'report.json').is_file():
            cells.append({'workload': workload, 'seed': seed, 'lane': lane, 'outcome': 'failed',
                          'errors': failure or 'missing complete report'})
            continue
        try:
            cells.append(verified_cell(directory, workload, lane, seed, request_identity))
        except (ContractError, OSError) as error:
            cells.append({'workload': workload, 'seed': seed, 'lane': lane, 'outcome': 'failed',
                          'errors': type(error).__name__+': '+str(error)})
    if not worker_outcomes and all(c['outcome'] == 'completed' for c in cells):
        try:
            groups = _comparison_groups(cells)
        except (ContractError, KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
            # Assignment occurs only on success; no partial groups escape.
            aggregate_errors.append(type(error).__name__+': '+str(error))
    result = {'schema': 'trinite.comparison-summary.v1', 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
        'outcome': 'completed' if groups else 'failed', 'cells': cells, 'groups': groups,
        'worker_errors': worker_outcomes, 'aggregate_errors': aggregate_errors,
        'result_status': 'inconclusive', 'phase4_negative_result_preserved': True,
        'scope': protocol()['evidence_scope'], 'release_ready': False}
    _write(root, 'summary.json', result)
    return result
