"""Frozen maths/control pilot, native updates, safe tensors and fresh scoring."""
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import resource
import time

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from .maths_data import dataset as maths_dataset, audit_bytes as maths_audit
from .learning_data import dataset as legacy_dataset, audit_bytes as legacy_audit
from .maths_plan import MathsPlan
from .tokenizer import ByteTokenizer
from .model import Decoder
from .training import TrainingState, create_optimizer, evaluate, model_identity, require_environment, run_steps
from .comparison_checkpoint import tensor_payload, progress, restore_tensors
from .foundations import read_bytes, write, failure, score_examples
from .observation import BundleObserver, verify_observation

PROTOCOL_IDENTITY = 'sha256:95fb85d66f0c722e47612239e171bda800ebf0ee63e5bee8a2cc73cd6367f5ca'
PROFILES = ('control', 'expanded')
OUTPUTS = ('report.json', 'predictions.json', 'steps.json', 'tensors.safetensors', 'metadata.json')


def protocol():
    raw = read_bytes(Path(__file__).parent/'maths-protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY: raise ContractError('maths protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    package = Path(__file__).resolve().parent; root = package.parents[1]
    paths = sorted([*package.glob('*.py'), *package.glob('*.json'), package/'cpu-dependencies.lock',
                    root/'requirements.lock', root/'scripts/run_maths.py'])
    return {str(p.relative_to(root)): identity(read_bytes(p)) for p in paths}


def corpora():
    result = {}
    for kind, generate, audit in (('maths', maths_dataset, maths_audit),
                                  ('legacy', lambda: legacy_dataset('arithmetic'), legacy_audit)):
        raw, manifest = generate(); audit(raw, manifest)
        result[kind] = (raw, manifest, parse_json(raw, canonical=True)['examples'])
    return result


def cells():
    return tuple((seed, profile) for seed in (0, 1) for profile in (PROFILES if seed == 0 else tuple(reversed(PROFILES))))


def cell_name(seed, profile):
    if type(seed) is not int or type(profile) is not str or (seed, profile) not in cells():
        raise ContractError('unknown fixed maths cell')
    return f'maths-{seed}-{profile}-dense'


def state_for(seed, profile, plan=None):
    cell_name(seed, profile); require_environment()
    expected = replace(MathsPlan.from_dict(protocol()['training']), seed=seed)
    plan = expected if plan is None else plan
    if type(plan) is not MathsPlan or plan.seed != seed: raise ContractError('maths plan/seed mismatch')
    packs = corpora(); examples = packs['legacy'][2]+(packs['maths'][2] if profile == 'expanded' else [])
    tokenizer = ByteTokenizer(plan.model.context_length); rows = []
    for item in examples:
        if item['split'] != 'train': continue
        e = tokenizer.encode(item['text'], answer_start=len(item['prompt'].encode()))
        rows.append({'identity': item['example_identity'], 'ids': e.input_ids, 'mask': e.attention_mask, 'loss': e.loss_mask})
    rows.sort(key=lambda r: identity(json_bytes({'seed': seed, 'item': r['identity']})))
    if len({r['identity'] for r in rows}) != len(rows): raise ContractError('duplicate training rows')
    targets = sum(sum(rows[(i*plan.batch_size+j) % len(rows)]['loss']) for i in range(plan.steps) for j in range(plan.batch_size))
    if targets > plan.max_target_tokens: raise ContractError('maths target budget exceeded')
    data_id = identity(json_bytes({k: identity(v[0]) for k, v in packs.items() if k == 'legacy' or profile == 'expanded'}))
    manifest_id = identity(json_bytes({k: identity(v[1]) for k, v in packs.items() if k == 'legacy' or profile == 'expanded'}))
    model = Decoder(plan.model, lane='dense', seed=seed)
    return TrainingState(plan, 'dense', model, create_optimizer(model, plan), {'train': rows, 'validation': rows},
        data_id, manifest_id, model_identity(model), evaluate(model, rows, plan.batch_size))


def request_core():
    return {'schema': 'trinite.maths-request.v1', 'protocol': protocol(), 'protocol_identity': PROTOCOL_IDENTITY,
            'source': sources(), 'environment': require_environment(), 'corpora':
            {k: {'dataset_identity': identity(v[0]), 'manifest_identity': identity(v[1])} for k, v in corpora().items()}}


def freeze_request(path):
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY, 'planned_cells': len(cells())}


def read_request(path, expected_identity):
    require_identity(expected_identity); raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('maths request identity mismatch')
    r = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(r, set(expected)|{'created_at'}, 'maths request')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()): raise ContractError('maths source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(r['created_at']).utcoffset() is None: raise ValueError('timezone required')
    except (TypeError, ValueError) as error: raise ContractError('invalid request clock') from error
    return r, raw


def context(state, seed, profile, request_identity):
    require_identity(request_identity); cell_name(seed, profile)
    if type(state.config) is not MathsPlan or state.config.seed != seed or state.lane != 'dense':
        raise ContractError('maths checkpoint profile mismatch')
    return {'schema': 'trinite.maths-checkpoint.v1', 'seed': seed, 'profile': profile,
        'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
        'source': sources(), 'environment': require_environment(), 'plan': state.config.to_dict(),
        'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity, 'initial_train_loss_hex': state.initial_train_loss,
        'cycle_identity': identity(json_bytes([r['identity'] for r in state.data['train']])), 'diagnostic_view': 'train-only.v1'}


def snapshot(state, seed, profile, request_identity):
    return tensor_payload(state), json_bytes({**context(state, seed, profile, request_identity),
        'model_identity': model_identity(state.model), 'step': state.step, 'cursor': state.cursor,
        'target_tokens': state.target_tokens, 'history': state.history, 'train_diagnostics': state.validation})


def restore(payload, metadata, seed, profile, request_identity, plan=None):
    r = parse_json(metadata, canonical=True); state = state_for(seed, profile, plan)
    expected = context(state, seed, profile, request_identity)
    exact_keys(r, set(expected)|{'model_identity', 'step', 'cursor', 'target_tokens', 'history', 'train_diagnostics'}, 'maths checkpoint')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()): raise ContractError('maths checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = r['step'], r['cursor'], r['target_tokens']
    state.history, state.validation = r['history'], r['train_diagnostics']; progress(state)
    return restore_tensors(state, payload, r['model_identity'])


def diagnostics(rows):
    # Semantic pairs are looked up from admitted examples, never inferred from outputs.
    pairs = {}; mapping = {e['example_identity']: e for v in corpora().values() for e in v[2]}
    for row in rows:
        item = mapping[row['example_identity']]
        pairs.setdefault(item['semantic_identity'], []).append(row)
    return {'missing_eos': sum(not r['generated_tokens'] or r['generated_tokens'][-1] != 257 for r in rows),
            'wrong_answer_with_eos': sum(not r['correct'] and r['generated_tokens'][-1] == 257 for r in rows),
            'carrier_pairs': len(pairs),
            'carrier_disagreement': sum(len({tuple(r['generated_tokens']) for r in p}) != 1 for p in pairs.values()),
            'both_carriers_exact': sum(all(r['correct'] for r in p) for p in pairs.values())}


def report_for(state, seed, profile, request_identity, seconds_hex, rss):
    initial = Decoder(state.config.model, lane='dense', seed=seed)
    if model_identity(initial) != state.initialization_identity: raise ContractError('initial model identity mismatch')
    scores, predictions = {}, {}
    for kind, pack in corpora().items():
        for stage, model, splits in (('initial', initial, ('test',)), ('final', state.model, ('train', 'test'))):
            for split in splits:
                key = kind+'-'+stage+'-'+split
                values, rows = score_examples(model, pack[2], split, protocol()['generation_cap'])
                scores[key] = {**values, 'drift_diagnostics': diagnostics(rows)}
                predictions[key] = {'predictions': rows}
    return {**context(state, seed, profile, request_identity), 'schema': 'trinite.maths-report.v1', 'outcome': 'completed', 'model_identity': model_identity(state.model),
        'steps': state.step, 'target_tokens': state.target_tokens,
        'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
        'initial_train_loss_hex': state.initial_train_loss,
        'final_train_loss_hex': evaluate(state.model, state.data['train'], state.config.batch_size),
        'training_wall_seconds_hex': seconds_hex, 'peak_process_rss_kib': rss,
        'scores': scores, 'held_out_scored': True, 'scope': protocol()['scope']}, predictions


def train_cell(root, seed, profile, request, request_identity):
    _, request_raw = read_request(request, request_identity); root = Path(root); root.mkdir(parents=True, exist_ok=False)
    try:
        state = state_for(seed, profile); started = time.perf_counter(); run_steps(state)
        seconds = time.perf_counter()-started
        payload, metadata = snapshot(state, seed, profile, request_identity)
        (root/'tensors.safetensors').write_bytes(payload); (root/'metadata.json').write_bytes(metadata)
        write(root, 'steps.json', {'steps': state.history, 'train_diagnostics': state.validation})
        report, predictions = report_for(state, seed, profile, request_identity, seconds.hex(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(root, 'report.json', report); write(root, 'predictions.json', predictions)
        read_request(request, request_identity)
        collector = BundleObserver(root/'provenance')
        inputs = {'request.json': request_raw}
        for kind, (raw, manifest, _) in corpora().items(): inputs.update({kind+'-dataset.json': raw, kind+'-manifest.json': manifest})
        collector.record('maths-pilot', inputs=inputs, outputs={n: read_bytes(root/n) for n in OUTPUTS})
        receipt = collector.finalize(); write(root, 'verification.json', receipt); verify_observation(root/'provenance')
        return report
    except Exception as error:
        failure(root, 'maths', error); raise


def verify_cell(root, seed, profile, request, request_identity):
    _, request_raw = read_request(request, request_identity); root = Path(root)
    receipt = verify_observation(root/'provenance')
    if read_bytes(root/'verification.json') != json_bytes(receipt): raise ContractError('maths retained verification mismatch')
    # Shared closed name-index binding is provided by the upstream verified artifacts.
    artifact_root = root/'provenance/artifacts/sha256'
    indices = [parse_json(read_bytes(p), canonical=True) for p in artifact_root.iterdir()
               if p.stat().st_size < 128*1024 and read_bytes(p).startswith(b'{"artifacts":')]
    if len(indices) != 1 or indices[0].get('schema') != 'trinite.observer-index.v1': raise ContractError('missing unique maths artifact index')
    names = indices[0]['artifacts']
    manifest = parse_json(read_bytes(root/'provenance/manifest.json'), canonical=True)
    members = {a['content_identity'] for a in manifest['core']['artifacts']}
    if any(k not in members for k in names.values()): raise ContractError('maths index references unretained content')
    expected_inputs = {'request.json': request_raw}
    for kind, (raw, manifest, _) in corpora().items(): expected_inputs.update({kind+'-dataset.json': raw, kind+'-manifest.json': manifest})
    if set(names) != set(expected_inputs)|set(OUTPUTS): raise ContractError('maths observer names mismatch')
    def bound(name):
        raw = read_bytes(artifact_root/names[name].split(':')[1])
        if identity(raw) != names[name]: raise ContractError('maths bound artifact mismatch')
        return raw
    if any(bound(k) != v for k, v in expected_inputs.items()): raise ContractError('maths observer input mismatch')
    if any(bound(k) != read_bytes(root/k) for k in OUTPUTS): raise ContractError('maths retained output mismatch')
    r = parse_json(bound('report.json'), canonical=True)
    state = restore(bound('tensors.safetensors'), bound('metadata.json'), seed, profile, request_identity)
    if state.step != state.config.steps: raise ContractError('maths final checkpoint required')
    wall = float.fromhex(r['training_wall_seconds_hex'])
    if not math.isfinite(wall) or not 0 < wall <= state.config.max_seconds or wall.hex() != r['training_wall_seconds_hex']:
        raise ContractError('invalid maths wall budget')
    integer(r['peak_process_rss_kib'], 'RSS', 1, 2**63-1)
    expected, predictions = report_for(state, seed, profile, request_identity, wall.hex(), r['peak_process_rss_kib'])
    if (json_bytes(expected) != bound('report.json') or json_bytes(predictions) != bound('predictions.json')
            or bound('steps.json') != json_bytes({'steps': state.history, 'train_diagnostics': state.validation})
            or state.validation[-1]['loss_hex'] != r['final_train_loss_hex']): raise ContractError('maths fresh model/scoring/history mismatch')
    return {'seed': seed, 'profile': profile, 'outcome': 'completed', 'report': r,
            'report_identity': identity(bound('report.json')), 'verification': receipt}


def summarize(root, request, request_identity, errors, *, publish=True):
    read_request(request, request_identity); root = Path(root)
    if (type(errors) is not dict or not set(errors) <= {cell_name(*c) for c in cells()}|{'summary'}
            or any(type(v) is not str or not v.strip() for v in errors.values())): raise ContractError('invalid maths worker errors')
    if read_bytes(root/'request.json') != read_bytes(request): raise ContractError('retained maths request mismatch')
    records = []
    for seed, profile in cells():
        try:
            if cell_name(seed, profile) in errors: raise ContractError(errors[cell_name(seed, profile)])
            records.append(verify_cell(root/cell_name(seed, profile), seed, profile, request, request_identity))
        except (ContractError, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            records.append({'seed': seed, 'profile': profile, 'outcome': 'failed', 'error': type(error).__name__+': '+str(error)})
    complete = all(r['outcome'] == 'completed' for r in records) and not errors
    pairing_errors, contrasts = [], {}
    if complete:
        for seed in (0, 1):
            a, b = [next(r['report'] for r in records if r['seed'] == seed and r['profile'] == p) for p in PROFILES]
            if a['initialization_identity'] != b['initialization_identity'] or a['steps'] != b['steps']:
                pairing_errors.append('initialization/update mismatch seed '+str(seed))
            if any(a['scores'][k] != b['scores'][k] for k in ('maths-initial-test', 'legacy-initial-test')):
                pairing_errors.append('initial scoring mismatch seed '+str(seed))
            contrasts[str(seed)] = {kind: {'control_correct': a['scores'][kind+'-final-test']['correct'],
                'expanded_correct': b['scores'][kind+'-final-test']['correct'],
                'examples': a['scores'][kind+'-final-test']['examples'],
                'delta_correct': b['scores'][kind+'-final-test']['correct']-a['scores'][kind+'-final-test']['correct']}
                for kind in ('maths', 'legacy')}
    result = {'schema': 'trinite.maths-summary.v1', 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'outcome': 'completed' if complete and not pairing_errors else 'failed',
        'cells': records, 'contrasts': contrasts, 'worker_errors': errors, 'pairing_errors': pairing_errors,
        'selected_candidate': None, 'held_out_scored': complete, 'dense_learning_adequate': False,
        'format_selection_eligible': False, 'phase5_ready': False, 'release_ready': False,
        'historical_evidence_preserved': True, 'scope': protocol()['scope']}
    if publish: write(root, 'summary.json', result)
    return result
