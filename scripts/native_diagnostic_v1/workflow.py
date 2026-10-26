"""Train-only observer around the one authoritative native update loop."""
import time
import math
import resource
from pathlib import Path

from trinite.budget_training import state_for
from trinite.budget_plan import BudgetPlan
from trinite.budget_checkpoint import snapshot_state, restore_state
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from trinite.observation import BundleObserver, verify_observation
from trinite.training import run_steps, model_identity
from .request import (PROTOCOL, MILESTONES, read, validate, core, corpus, training_examples)
from .artifacts import Store
from .mechanics import preflight, validate_preflight
from .measurements import measure


def write(store, outputs, name, value, *, raw=False):
    data = value if raw else json_bytes(value)
    store.write(name, data)
    outputs[name] = data


def run(root, request_raw, request_identity, expected_directory=None):
    """Worker only. Parent enforces a single wall deadline for the whole call."""
    request = validate(request_raw, request_identity)
    store = Store(root, expected_directory); outputs = {}
    started = time.monotonic(); update_seconds = 0.; timings = []
    try:
        store.write('request.json', request_raw)
        checks = preflight()
        write(store, outputs, 'preflight.json', checks)
        validate_preflight(checks)
        state = state_for('arithmetic', request['seed'], 'extended', BudgetPlan.from_dict(request['plan']))
        examples = training_examples()
        initial = model_identity(state.model)
        for milestone in MILESTONES:
            if time.monotonic() - started > PROTOCOL['worker_seconds']:
                raise ContractError('whole diagnostic worker deadline exceeded')
            if milestone:
                before = time.monotonic()
                run_steps(state, stop_after=milestone)
                update_seconds += time.monotonic() - before
                if update_seconds > PROTOCOL['update_seconds_total']:
                    raise ContractError('cumulative update budget exceeded')
            before = time.monotonic()
            diagnostic = measure(state.model, examples, PROTOCOL['generation_cap'])
            timings.append({'step': milestone, 'diagnostic_seconds_hex': (time.monotonic()-before).hex()})
            payload, metadata = snapshot_state(state, 'arithmetic', 'extended', request_identity)
            write(store, outputs, f'diagnostics-{milestone}.json', diagnostic)
            write(store, outputs, f'tensors-{milestone}.safetensors', payload, raw=True)
            write(store, outputs, f'metadata-{milestone}.json', metadata, raw=True)
        validate(request_raw, request_identity)
        report = {'schema': 'trinite.native-diagnostic-report.v1', 'outcome': 'completed',
                  'request_identity': request_identity, 'seed': request['seed'],
                  'milestones': list(MILESTONES), 'initialization_identity': initial,
                  'final_model_identity': model_identity(state.model), 'steps': state.step,
                  'target_tokens': state.target_tokens, 'parameter_count': state.config.model.parameter_count,
                  'cache_mode': 'unsupported', 'held_out_scored': False,
                  'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
                  'verification_scope': 'custody, request binding and safe checkpoint structure; not full optimizer replay'}
        write(store, outputs, 'report.json', report)
        # Non-replay observations deliberately do not enter model acceptance.
        write(store, outputs, 'resources.json', {'schema':'trinite.native-diagnostic-resources.v1',
              'scope':'whole worker lifetime RSS includes imports and preflight; not a hard memory cap',
              'wall_seconds_hex':(time.monotonic()-started).hex(),
              'update_seconds_hex':update_seconds.hex(), 'diagnostics':timings,
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'kv_payload_bytes':None, 'kv_unavailable_reason':'native backend has no reusable cache'})
        raw, manifest = corpus()
        store.check()
        # Linux-only workflow: keep provenance attached to the pinned attempt inode.
        provenance = Path('/proc/self/fd')/str(store.fd)/'provenance'
        observer = BundleObserver(provenance)
        observer.record('native-learning-diagnostic', inputs={'request.json':request_raw,
            'dataset.json':raw, 'dataset-manifest.json':manifest}, outputs=outputs)
        verification = observer.finalize()
        store.check()
        if not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']:
            raise ContractError('native diagnostic custody did not close')
        store.write('verification.json', json_bytes(verification))
        # Revalidate after every operation and before publishing completion.
        validate(request_raw, request_identity)
        completion = {'schema':'trinite.native-diagnostic-completion.v1',
            'request_identity':request_identity, 'artifacts':{name:identity(data) for name,data in outputs.items()},
            'manifest_identity':identity(read(provenance/'manifest.json')),
            'verification_identity':identity(json_bytes(verification))}
        encoded = json_bytes(completion)
        store.write('completion.json', encoded)
        return identity(encoded)
    finally:
        store.close()


def audit(root, request_raw, request_identity, expected_identity):
    """Read-only verification of retained bytes; does not re-run target training."""
    root = Path(root)
    request = validate(request_raw, request_identity)
    require_identity(expected_identity)
    retained_bytes = 0
    def retained(name):
        nonlocal retained_bytes
        raw = read(root/name)
        retained_bytes += len(raw)
        if retained_bytes > PROTOCOL['max_retained_payload_bytes']:
            raise ContractError('retained diagnostic payload budget exceeded')
        return raw
    completion_raw = retained('completion.json')
    if identity(completion_raw) != expected_identity:
        raise ContractError('completion identity mismatch')
    completion = parse_json(completion_raw, canonical=True)
    exact_keys(completion, {'schema','request_identity','artifacts','manifest_identity','verification_identity'}, 'completion')
    if completion['schema'] != 'trinite.native-diagnostic-completion.v1' or completion['request_identity'] != request_identity:
        raise ContractError('completion request/schema mismatch')
    if retained('request.json') != request_raw:
        raise ContractError('retained request differs')
    names = {'preflight.json','report.json','resources.json'} | {
        f'{kind}-{step}.{suffix}' for step in MILESTONES
        for kind,suffix in (('diagnostics','json'),('tensors','safetensors'),('metadata','json'))}
    exact_keys(completion['artifacts'], names, 'completed artifacts')
    bound = {}
    for name, expected in completion['artifacts'].items():
        require_identity(expected)
        raw = retained(name)
        if identity(raw) != expected:
            raise ContractError('retained artifact differs: '+name)
        bound[name] = raw
    verification_raw = retained('verification.json')
    if (identity(verification_raw) != completion['verification_identity']
            or identity(read(root/'provenance/manifest.json')) != completion['manifest_identity']):
        raise ContractError('retained verification/manifest differs')
    actual_verification = verify_observation(root/'provenance')
    if json_bytes(actual_verification) != verification_raw:
        raise ContractError('retained upstream verification report differs')
    manifest = parse_json(read(root/'provenance/manifest.json'), canonical=True)
    indexes = []
    for item in manifest['core']['artifacts']:
        key = item['content_identity']
        raw = read(root/'provenance/artifacts/sha256'/key[7:])
        try: value = parse_json(raw, canonical=True)
        except ContractError: continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1':
            indexes.append(value['artifacts'])
    raw, admitted = corpus()
    expected_names = {**completion['artifacts'], 'request.json': request_identity,
                      'dataset.json': identity(raw), 'dataset-manifest.json': identity(admitted)}
    if indexes != [expected_names]:
        raise ContractError('upstream name index does not bind every requested artifact')
    from .measurements import validate_report
    examples = training_examples()
    states = []
    for step in MILESTONES:
        payload, metadata = bound[f'tensors-{step}.safetensors'], bound[f'metadata-{step}.json']
        state = restore_state(payload, metadata, 'arithmetic', request['seed'], 'extended',
                              request_identity, (identity(payload), identity(metadata)), BudgetPlan.from_dict(request['plan']))
        if state.step != step:
            raise ContractError('checkpoint milestone mismatch')
        states.append(state)
        diagnostic = parse_json(bound[f'diagnostics-{step}.json'], canonical=True)
        validate_report(diagnostic, examples, PROTOCOL['generation_cap'])
        if (diagnostic['model_identity'] != model_identity(state.model)
                or any(row['context_length'] != state.config.model.context_length for row in diagnostic['rows'])):
            raise ContractError('diagnostics/checkpoint model or context mismatch')
    if model_identity(states[0].model) != states[0].initialization_identity:
        raise ContractError('zero-step checkpoint differs from seeded initialization')
    report = parse_json(bound['report.json'], canonical=True)
    expected_report = {'schema': 'trinite.native-diagnostic-report.v1', 'outcome': 'completed',
        'request_identity':request_identity, 'seed':request['seed'], 'milestones':list(MILESTONES),
        'initialization_identity':model_identity(states[0].model),
        'final_model_identity':model_identity(states[-1].model), 'steps':states[-1].step,
        'target_tokens':states[-1].target_tokens, 'parameter_count':states[-1].config.model.parameter_count,
        'cache_mode':'unsupported', 'held_out_scored':False, 'learning_adequate':False,
        'phase5_ready':False, 'release_ready':False,
        'verification_scope':'custody, request binding and safe checkpoint structure; not full optimizer replay'}
    if json_bytes(report) != json_bytes(expected_report):
        raise ContractError('diagnostic report/checkpoint context mismatch')
    preflight_record = parse_json(bound['preflight.json'], canonical=True)
    validate_preflight(preflight_record)
    validate_resources(parse_json(bound['resources.json'], canonical=True))
    validate(request_raw, request_identity)
    return {'schema':'trinite.native-diagnostic-audit.v1', 'verified':True,
            'completion_identity':expected_identity, 'optimizer_replayed':False,
            'diagnostic_predictions_replayed':False, 'held_out_scored':False,
            'learning_adequate':False}


def validate_resources(record):
    exact_keys(record, {'schema','scope','wall_seconds_hex','update_seconds_hex','diagnostics',
        'peak_rss_kib','kv_payload_bytes','kv_unavailable_reason'}, 'resource observations')
    expected = {'schema':'trinite.native-diagnostic-resources.v1',
        'scope':'whole worker lifetime RSS includes imports and preflight; not a hard memory cap',
        'kv_payload_bytes':None, 'kv_unavailable_reason':'native backend has no reusable cache'}
    if any(json_bytes(record[k]) != json_bytes(v) for k,v in expected.items()):
        raise ContractError('resource observation context mismatch')
    def seconds(raw):
        if type(raw) is not str: raise ContractError('resource seconds require hex string')
        try: value = float.fromhex(raw)
        except (ValueError, OverflowError) as error: raise ContractError('invalid resource seconds') from error
        if not math.isfinite(value) or value < 0 or value.hex() != raw:
            raise ContractError('resource seconds must be canonical finite nonnegative')
        return value
    wall = seconds(record['wall_seconds_hex']); update = seconds(record['update_seconds_hex'])
    if update > PROTOCOL['update_seconds_total'] or wall > PROTOCOL['worker_seconds'] or update > wall:
        raise ContractError('resource observations exceed frozen completion bounds')
    if type(record['peak_rss_kib']) is not int or not 0 < record['peak_rss_kib'] < 2**53:
        raise ContractError('resource RSS must be a positive integer observation')
    rows = record['diagnostics']
    if type(rows) is not list or len(rows) != len(MILESTONES):
        raise ContractError('resource milestone count mismatch')
    total = update
    for row, step in zip(rows, MILESTONES):
        exact_keys(row, {'step','diagnostic_seconds_hex'}, 'resource milestone')
        if type(row['step']) is not int or row['step'] != step:
            raise ContractError('resource milestone mismatch')
        total += seconds(row['diagnostic_seconds_hex'])
    if total > wall: raise ContractError('resource observations exceed whole measured wall time')
