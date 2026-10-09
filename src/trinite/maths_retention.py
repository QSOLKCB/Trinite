"""Frozen arithmetic rehearsal experiment; shared native updates and safe evidence."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import resource
import time

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from .maths_learning import corpora, diagnostics, state_for as pooled_state, sources as maths_sources
from .maths_plan import MathsPlan
from .model import Decoder
from .training import evaluate, model_identity, require_environment, run_steps
from .comparison_checkpoint import tensor_payload, progress, restore_tensors
from .foundations import read_bytes, write, failure, score_examples
from .observation import BundleObserver, verify_observation

PROTOCOL_IDENTITY = 'sha256:cf9c5928428cfd1a1dd82f515cdb44b4c10c0e6316b97aa7967f56d416c3af7d'
PROFILES = ('pooled', 'rehearsal')
OUTPUTS = ('report.json', 'predictions.json', 'steps.json', 'tensors.safetensors', 'metadata.json')


def protocol():
    raw = read_bytes(Path(__file__).parent/'maths-retention-protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY: raise ContractError('retention protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    root = Path(__file__).resolve().parents[2]
    return {**maths_sources(), 'scripts/run_maths_retention.py': identity(read_bytes(root/'scripts/run_maths_retention.py'))}


def cells():
    return tuple((seed, profile) for seed in (0, 1) for profile in (PROFILES if seed == 0 else tuple(reversed(PROFILES))))


def cell_name(seed, profile):
    if type(seed) is not int or type(profile) is not str or (seed, profile) not in cells():
        raise ContractError('unknown fixed retention cell')
    return f'maths-retention-{seed}-{profile}-dense'


def state_for(seed, profile, plan=None):
    cell_name(seed, profile)
    expected = replace(MathsPlan.from_dict(protocol()['training']), seed=seed)
    plan = expected if plan is None else plan
    if type(plan) is not MathsPlan or plan.seed != seed or plan.batch_size != 8:
        raise ContractError('retention requires MathsPlan, matching seed and eight-example batches')
    state = pooled_state(seed, 'expanded', plan)
    if profile == 'rehearsal':
        legacy_ids = {e['example_identity'] for e in corpora()['legacy'][2] if e['split'] == 'train'}
        legacy = [row for row in state.data['train'] if row['identity'] in legacy_ids]
        maths = [row for row in state.data['train'] if row['identity'] not in legacy_ids]
        # Both independent streams retain the original seeded rank. Only train rows enter the schedule.
        state.data['train'] = [stream[(step*4+j) % len(stream)]
                              for step in range(plan.steps) for stream in (legacy, maths) for j in range(4)]
    targets = sum(sum(state.data['train'][(i*8+j) % len(state.data['train'])]['loss'])
                  for i in range(plan.steps) for j in range(8))
    if targets > plan.max_target_tokens: raise ContractError('retention target budget exceeded')
    return state


def exposure(state):
    mapping = {e['example_identity']: (kind, e['task'])
               for kind, pack in corpora().items() for e in pack[2] if e['split'] == 'train'}
    visits = Counter(); targets = Counter(); inputs = Counter()
    for step in state.history:
        for key in step['examples']:
            visits[key] += 1
    encoded = {r['identity']: r for r in state.data['validation']}
    for key, count in visits.items():
        kind, task = mapping[key]
        targets[kind] += count*sum(encoded[key]['loss'])
        inputs[kind] += count*(len(encoded[key]['ids'])-1)
    result = {}
    for kind in ('legacy', 'maths'):
        ids = [key for key, item in mapping.items() if item[0] == kind]
        counts = [visits[key] for key in ids]
        result[kind] = {'unique_training_examples': len(ids), 'example_visits': sum(counts),
                       'scored_target_tokens': targets[kind], 'nonpadding_input_tokens': inputs[kind],
                       'minimum_visits': min(counts), 'maximum_visits': max(counts),
                       'visit_histogram': {str(k): v for k, v in sorted(Counter(counts).items())},
                       'tasks': {task: sum(visits[key] for key in ids if mapping[key][1] == task)
                                 for task in sorted({mapping[key][1] for key in ids})}}
    if sum(r['scored_target_tokens'] for r in result.values()) != state.target_tokens:
        raise ContractError('retention exposure target mismatch')
    return result


def training_failures(scores, kind):
    if kind not in ('legacy', 'maths') or not scores['tasks']:
        raise ContractError('retention criterion needs an admitted nonempty task set')
    threshold = protocol()['retention_threshold' if kind == 'legacy' else 'maths_threshold']
    return [task for task, value in sorted(scores['tasks'].items())
            if value['correct']*threshold['denominator'] < value['examples']*threshold['numerator']]


def request_core():
    return {'schema': 'trinite.maths-retention-request.v1', 'protocol': protocol(), 'protocol_identity': PROTOCOL_IDENTITY,
            'source': sources(), 'environment': require_environment(), 'corpora':
            {k: {'dataset_identity': identity(v[0]), 'manifest_identity': identity(v[1])} for k, v in corpora().items()}}


def freeze_request(path):
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY, 'planned_cells': len(cells())}


def read_request(path, expected_identity):
    require_identity(expected_identity); raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('retention request identity mismatch')
    r = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(r, set(expected)|{'created_at'}, 'retention request')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()): raise ContractError('retention source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(r['created_at']).utcoffset() is None: raise ValueError('timezone required')
    except (TypeError, ValueError) as error: raise ContractError('invalid request clock') from error
    return r, raw


def context(state, seed, profile, request_identity):
    require_identity(request_identity); cell_name(seed, profile)
    if type(state.config) is not MathsPlan or state.config.seed != seed or state.lane != 'dense':
        raise ContractError('retention checkpoint profile mismatch')
    return {'schema': 'trinite.maths-retention-checkpoint.v1', 'seed': seed, 'profile': profile,
        'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
        'source': sources(), 'environment': require_environment(), 'plan': state.config.to_dict(),
        'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity, 'initial_train_loss_hex': state.initial_train_loss,
        'cycle_identity': identity(json_bytes([r['identity'] for r in state.data['train']])), 'diagnostic_view': protocol()['diagnostic_view']}


def snapshot(state, seed, profile, request_identity):
    return tensor_payload(state), json_bytes({**context(state, seed, profile, request_identity),
        'model_identity': model_identity(state.model), 'step': state.step, 'cursor': state.cursor,
        'target_tokens': state.target_tokens, 'history': state.history, 'train_diagnostics': state.validation})


def restore(payload, metadata, seed, profile, request_identity, plan=None):
    r = parse_json(metadata, canonical=True); state = state_for(seed, profile, plan)
    expected = context(state, seed, profile, request_identity)
    exact_keys(r, set(expected)|{'model_identity', 'step', 'cursor', 'target_tokens', 'history', 'train_diagnostics'}, 'retention checkpoint')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()): raise ContractError('retention checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = r['step'], r['cursor'], r['target_tokens']
    state.history, state.validation = r['history'], r['train_diagnostics']; progress(state)
    return restore_tensors(state, payload, r['model_identity'])


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
    return {**context(state, seed, profile, request_identity), 'schema': 'trinite.maths-retention-report.v1', 'outcome': 'completed', 'model_identity': model_identity(state.model),
        'steps': state.step, 'target_tokens': state.target_tokens,
        'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
        'initial_train_loss_hex': state.initial_train_loss,
        'final_train_loss_hex': evaluate(state.model, state.data['validation'], state.config.batch_size),
        'training_wall_seconds_hex': seconds_hex, 'peak_process_rss_kib': rss,
        'scores': scores, 'exposure': exposure(state),
        'training_gate_failures': {k: training_failures(scores[k+'-final-train'], k) for k in ('legacy', 'maths')},
        'held_out_scored': True, 'scope': protocol()['scope']}, predictions


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
        collector.record('maths-retention-pilot', inputs=inputs, outputs={n: read_bytes(root/n) for n in OUTPUTS})
        receipt = collector.finalize(); write(root, 'verification.json', receipt); verify_observation(root/'provenance')
        return report
    except Exception as error:
        failure(root, 'maths-retention', error); raise


def verify_cell(root, seed, profile, request, request_identity):
    _, request_raw = read_request(request, request_identity); root = Path(root)
    receipt = verify_observation(root/'provenance')
    if read_bytes(root/'verification.json') != json_bytes(receipt): raise ContractError('retention retained verification mismatch')
    # Shared closed name-index binding is provided by the upstream verified artifacts.
    artifact_root = root/'provenance/artifacts/sha256'
    indices = [parse_json(read_bytes(p), canonical=True) for p in artifact_root.iterdir()
               if p.stat().st_size < 128*1024 and read_bytes(p).startswith(b'{"artifacts":')]
    if len(indices) != 1 or indices[0].get('schema') != 'trinite.observer-index.v1': raise ContractError('missing unique retention artifact index')
    names = indices[0]['artifacts']
    manifest = parse_json(read_bytes(root/'provenance/manifest.json'), canonical=True)
    members = {a['content_identity'] for a in manifest['core']['artifacts']}
    if any(k not in members for k in names.values()): raise ContractError('retention index references unretained content')
    expected_inputs = {'request.json': request_raw}
    for kind, (raw, manifest, _) in corpora().items(): expected_inputs.update({kind+'-dataset.json': raw, kind+'-manifest.json': manifest})
    if set(names) != set(expected_inputs)|set(OUTPUTS): raise ContractError('retention observer names mismatch')
    def bound(name):
        raw = read_bytes(artifact_root/names[name].split(':')[1])
        if identity(raw) != names[name]: raise ContractError('retention bound artifact mismatch')
        return raw
    if any(bound(k) != v for k, v in expected_inputs.items()): raise ContractError('retention observer input mismatch')
    if any(bound(k) != read_bytes(root/k) for k in OUTPUTS): raise ContractError('retention retained output mismatch')
    r = parse_json(bound('report.json'), canonical=True)
    state = restore(bound('tensors.safetensors'), bound('metadata.json'), seed, profile, request_identity)
    if state.step != state.config.steps: raise ContractError('retention final checkpoint required')
    wall = float.fromhex(r['training_wall_seconds_hex'])
    if not math.isfinite(wall) or not 0 < wall <= state.config.max_seconds or wall.hex() != r['training_wall_seconds_hex']:
        raise ContractError('invalid retention wall budget')
    integer(r['peak_process_rss_kib'], 'RSS', 1, 2**63-1)
    expected, predictions = report_for(state, seed, profile, request_identity, wall.hex(), r['peak_process_rss_kib'])
    if (json_bytes(expected) != bound('report.json') or json_bytes(predictions) != bound('predictions.json')
            or bound('steps.json') != json_bytes({'steps': state.history, 'train_diagnostics': state.validation})
            or state.validation[-1]['loss_hex'] != r['final_train_loss_hex']): raise ContractError('retention fresh model/scoring/history mismatch')
    return {'seed': seed, 'profile': profile, 'outcome': 'completed', 'report': r,
            'report_identity': identity(bound('report.json')), 'verification': receipt}


def summarize(root, request, request_identity, errors, *, publish=True):
    read_request(request, request_identity); root = Path(root)
    if (type(errors) is not dict or not set(errors) <= {cell_name(*c) for c in cells()}|{'summary'}
            or any(type(v) is not str or not v.strip() for v in errors.values())): raise ContractError('invalid retention worker errors')
    if read_bytes(root/'request.json') != read_bytes(request): raise ContractError('retained retention request mismatch')
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
            if (a['initialization_identity'] != b['initialization_identity'] or a['steps'] != b['steps']
                    or a['initial_train_loss_hex'] != b['initial_train_loss_hex']
                    or sum(v['example_visits'] for v in a['exposure'].values()) != sum(v['example_visits'] for v in b['exposure'].values())):
                pairing_errors.append('initialization/update mismatch seed '+str(seed))
            if any(a['scores'][k] != b['scores'][k] for k in ('maths-initial-test', 'legacy-initial-test')):
                pairing_errors.append('initial scoring mismatch seed '+str(seed))
            contrasts[str(seed)] = {kind: {split: {'pooled_correct': a['scores'][kind+'-final-'+split]['correct'],
                    'rehearsal_correct': b['scores'][kind+'-final-'+split]['correct'],
                    'examples': a['scores'][kind+'-final-'+split]['examples'],
                    'delta_correct': b['scores'][kind+'-final-'+split]['correct']-a['scores'][kind+'-final-'+split]['correct']}
                    for split in ('train', 'test')}
                for kind in ('maths', 'legacy')}
    result = {'schema': 'trinite.maths-retention-summary.v1', 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'outcome': 'completed' if complete and not pairing_errors else 'failed',
        'cells': records, 'contrasts': contrasts, 'worker_errors': errors, 'pairing_errors': pairing_errors,
        'selected_candidate': None, 'held_out_scored': complete, 'dense_learning_adequate': False,
        'format_selection_eligible': False, 'phase5_ready': False, 'release_ready': False,
        'historical_evidence_preserved': True, 'scope': protocol()['scope']}
    if publish: write(root, 'summary.json', result)
    return result
