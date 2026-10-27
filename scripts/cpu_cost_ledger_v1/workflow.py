"""Fresh artifacts and read-only integrity checks, never optimizer replay."""
from pathlib import Path

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from trinite.observation import BundleObserver, verify_observation
from native_diagnostic_v1.artifacts import Store
from unseen_evaluation_v1.packet import inventory, unchanged
from .request import LIMITS, read, validate

ARTIFACTS = ('numerical.json', 'observations.json', 'verification.json')


def write(store, name, raw):
    if len(raw) > LIMITS['artifact_bytes']:
        raise ContractError('ledger artifact exceeds frozen bound')
    store.write(name, raw)


def run(root, request_raw, request_identity, checkpoint, expected_directory, startup, startup_start):
    from .measure import clock, elapsed, load_model, measure
    start = clock(); request = validate(request_raw, request_identity); validation_time = elapsed(start)
    value = request['plan']; store = Store(root, expected_directory)
    try:
        start = clock(); model, checkpoint_bytes = load_model(checkpoint, value); loading_time = elapsed(start)
        journal = []
        def retain(row):
            # Exclusive bounded per-row journals survive parity failures/timeouts.
            raw = json_bytes(row)
            write(store, f'row-{len(journal):02d}.json', raw)
            journal.append(identity(raw))
        start = clock(); measured = measure(model, value, retain); measurement_time = elapsed(start)
        numerical_rows = [{key: row[key] for key in ('prompt_index', 'repetition', 'prompt_binding',
            'before', 'measured', 'after', 'parity', 'all_passes_forward_calls', 'all_passes_processed_positions')}
            for row in measured['rows']]
        numerical = {'schema': 'trinite.cpu-cost-ledger-numerical.v1', 'request_identity': request_identity,
            'model_identity': value['model_identity'], 'checkpoint_identity': value['checkpoint_identity'],
            'rows': numerical_rows, 'all_passes_forward_calls': measured['all_passes_forward_calls'],
            'all_passes_processed_positions': measured['all_passes_processed_positions'],
            'training_updates': 0, 'cache_mode': 'unsupported', 'output_parity': True}
        nr = json_bytes(numerical); write(store, 'numerical.json', nr)
        observations = {'schema': 'trinite.cpu-cost-ledger-observations.v1', 'request_identity': request_identity,
            'startup_imports_and_configuration': startup, 'request_validation_audit': validation_time,
            'checkpoint_read_safe_decode_construct_copy_validate_hash': loading_time,
            'measurement_and_journal_writes': measurement_time, 'worker_through_measurement': elapsed(startup_start),
            'checkpoint_artifact_bytes': checkpoint_bytes,
            'scopes': 'forward timers include native input/state validation and math; exclude tokenization/tensor creation/hash/finite audit; instrumented end-to-end includes them; all CPU is this process, not machine/tree',
            'journal_identities': journal, **measured}
        ob = json_bytes(observations); write(store, 'observations.json', ob)
        start = clock()
        observer = BundleObserver(Path(root)/'provenance', actor='trinite.cpu-cost-ledger.v1')
        observer.record('inference-cost-observation', inputs={'request.json': request_raw},
                        outputs={'numerical.json': nr, 'observations.json': ob})
        verification = observer.finalize()
        vr = json_bytes(verification); write(store, 'verification.json', vr)
        audit_cost = elapsed(start)
        if not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']:
            raise ContractError('ledger provenance failed')
        # Rebind sources/environment and exact read-only checkpoint after all observations.
        validate(request_raw, request_identity)
        if identity(read(checkpoint, LIMITS['checkpoint_bytes'])) != value['checkpoint_identity']:
            raise ContractError('checkpoint changed during measurement')
        completion = {'schema': 'trinite.cpu-cost-ledger-completion.v1', 'request_identity': request_identity,
            'artifacts': {name: identity(read(Path(root)/name)) for name in ARTIFACTS},
            'manifest_identity': identity(read(Path(root)/'provenance/manifest.json')),
            'provenance_creation_and_verification_audit': audit_cost, 'journal_identities': journal,
            'scope': 'closed observation bundle; external model-only identity binding, not producer custody or numerical replay'}
        raw = json_bytes(completion); write(store, 'completion.json', raw)
        return identity(raw)
    finally:
        store.close()


def audit(root, request_raw, request_identity, expected):
    """Verify exact externally bound observation bytes and logical work, not timings/replay."""
    request = validate(request_raw, request_identity); value = request['plan']; bounds = request['bounds']
    root = Path(root); snapshot = inventory(root); cr = read(root/'completion.json')
    if read(root/'request.json') != request_raw:
        raise ContractError('retained request differs')
    if identity(cr) != expected:
        raise ContractError('external completion identity differs')
    c = parse_json(cr, canonical=True)
    exact_keys(c, {'schema', 'request_identity', 'artifacts', 'manifest_identity',
        'provenance_creation_and_verification_audit', 'journal_identities', 'scope'}, 'completion')
    if c['schema'] != 'trinite.cpu-cost-ledger-completion.v1' or c['request_identity'] != request_identity:
        raise ContractError('completion context differs')
    exact_keys(c['artifacts'], set(ARTIFACTS), 'artifacts')
    raw = {name: read(root/name) for name in ARTIFACTS}
    if any(identity(raw[name]) != key for name, key in c['artifacts'].items()):
        raise ContractError('artifact identity differs')
    if identity(read(root/'provenance/manifest.json')) != c['manifest_identity']:
        raise ContractError('manifest identity differs')
    if json_bytes(verify_observation(root/'provenance')) != raw['verification.json']:
        raise ContractError('closed observation verification differs')
    # Bind actual retained bundle payloads, not merely an unrelated valid bundle.
    for payload in (request_raw, raw['numerical.json'], raw['observations.json']):
        if read(root/'provenance/artifacts/sha256'/identity(payload)[7:]) != payload:
            raise ContractError('bundle payload differs')
    manifest = parse_json(read(root/'provenance/manifest.json'), canonical=True)
    indexes = []
    for item in manifest['core']['artifacts']:
        payload = read(root/'provenance/artifacts/sha256'/item['content_identity'][7:])
        try: record = parse_json(payload, canonical=True)
        except ContractError: continue
        if type(record) is dict and record.get('schema') == 'trinite.observer-index.v1':
            indexes.append(record['artifacts'])
    if indexes != [{'request.json': request_identity, 'numerical.json': c['artifacts']['numerical.json'],
                    'observations.json': c['artifacts']['observations.json']}]:
        raise ContractError('closed observer index differs')
    n = parse_json(raw['numerical.json'], canonical=True); o = parse_json(raw['observations.json'], canonical=True)
    if (o['schema'] != 'trinite.cpu-cost-ledger-observations.v1' or
            n['schema'] != 'trinite.cpu-cost-ledger-numerical.v1' or
            n['request_identity'] != request_identity or o['request_identity'] != request_identity or
            n['checkpoint_identity'] != value['checkpoint_identity'] or n['model_identity'] != value['model_identity'] or
            n['training_updates'] != 0 or n['cache_mode'] != 'unsupported' or n['output_parity'] is not True):
        raise ContractError('numerical context differs')
    count = len(value['prompts']) * value['repetitions']
    if len(n['rows']) != count or len(o['rows']) != count or len(c['journal_identities']) != count:
        raise ContractError('missing expected repetitions')
    top = {Path(p).parts[0] for p in snapshot['members'] if p != '.'}
    required = set(ARTIFACTS) | {'request.json', 'completion.json', 'provenance', 'worker.log'} | {f'row-{i:02d}.json' for i in range(count)}
    if top not in (required, required | {'attempt.json'}):
        raise ContractError('attempt membership differs')
    if 'attempt.json' in top:
        attempt = parse_json(read(root/'attempt.json'), canonical=True)
        if (attempt.get('schema') != 'trinite.cpu-cost-ledger-attempt.v1' or
                attempt.get('request_identity') != request_identity or attempt.get('outcome') != 'completed' or
                type(attempt.get('worker_exit')) is not int or attempt['worker_exit'] != 0 or
                attempt.get('completion_identity') != expected):
            raise ContractError('controller did not certify successful bounded completion')
    if (o['journal_identities'] != c['journal_identities'] or o['cache_mode'] != 'unsupported'
            or o['model_identity'] != value['model_identity'] or o['training_updates'] != 0
            or o['speed_claim'] is not False or o['learning_adequate'] is not False
            or any(o[k] is not None for k in ('native_kv_payload_bytes', 'memory_ceiling_bytes',
                'transient_forward_peak_bytes', 'sampled_process_tree_rss_bytes', 'runtime_overhead_bytes'))
            or o['unique_resident_parameter_storage_bytes'] != bounds['predicted_unique_parameter_bytes']
            or type(o['checkpoint_artifact_bytes']) is not int or not 1 <= o['checkpoint_artifact_bytes'] <= LIMITS['checkpoint_bytes']
            or type(o['worker_lifetime_peak_rss_bytes']) is not int or o['worker_lifetime_peak_rss_bytes'] < 0):
        raise ContractError('observation memory/count/claim scope differs')
    def timing(record):
        exact_keys(record, {'wall_ns', 'process_cpu_ns'}, 'timing')
        if any(type(v) is not int or v < 0 for v in record.values()):
            raise ContractError('invalid timing observation')
    for key in ('startup_imports_and_configuration', 'request_validation_audit',
                'checkpoint_read_safe_decode_construct_copy_validate_hash', 'measurement_and_journal_writes',
                'identity_audit_before', 'identity_audit_after', 'worker_through_measurement'):
        timing(o[key])
    timing(c['provenance_creation_and_verification_audit'])
    calls = positions = 0
    for index, (row, observation) in enumerate(zip(n['rows'], o['rows'])):
        prompt_index, repetition = divmod(index, value['repetitions'])
        for key in ('instrumented_end_to_end', 'parity_before', 'parity_after', 'instrumentation_audit'):
            timing(observation[key])
        if observation['first_parity_before_generation_in_worker'] is not (index == 0):
            raise ContractError('first-generation scope differs')
        prompt = value['prompts'][prompt_index]; length = 1+len(prompt.encode())
        if (row['prompt_index'], row['repetition']) != (prompt_index, repetition) or row['prompt_binding'] != bounds['prompt_bindings'][prompt_index]:
            raise ContractError('row/prompt ordering differs')
        if any(observation[k] != v for k, v in row.items()):
            raise ContractError('observations/numerical rows differ')
        if not row['parity'] or not row['before'] == row['measured'] == row['after']:
            raise ContractError('output parity differs')
        prediction = row['measured']; tokens = prediction['tokens']; cap = value['output_tokens']
        if (type(tokens) is not list or not 1 <= len(tokens) <= cap
                or any(type(t) is not int or not 0 <= t <= 258 for t in tokens)
                or 257 in tokens[:-1]):
            raise ContractError('invalid token sequence')
        from .measure import prediction as expected_prediction
        if prediction != expected_prediction(tokens, cap) or prediction['stop_reason'] == 'context_cap':
            raise ContractError('invalid decoding stop/count')
        work = 3*len(tokens); processed = 3*(len(tokens)*length + len(tokens)*(len(tokens)-1)//2)
        if row['all_passes_forward_calls'] != work or row['all_passes_processed_positions'] != processed:
            raise ContractError('logical compute count differs')
        calls += work; positions += processed
        steps = observation['forward_observations']
        if len(steps) != len(tokens): raise ContractError('forward observation count differs')
        for i, step in enumerate(steps):
            prefix = [256, *prompt.encode(), *tokens[:i]]
            expected_step = {'checkpoint_identity': value['checkpoint_identity'],
                'prefix_identity': identity(json_bytes(prefix)),
                'mask_identity': identity(json_bytes([[True]*len(prefix)])),
                'absolute_positions': list(range(len(prefix))), 'input_tokens': len(prefix),
                'phase': 'prompt_forward' if i == 0 else 'decode_full_prefix'}
            if any(step[k] != v for k, v in expected_step.items()):
                raise ContractError('checkpoint/prefix/position binding differs')
            if any(type(step[k]) is not int or step[k] < 0 for k in ('wall_ns', 'process_cpu_ns')):
                raise ContractError('invalid timing observation')
        jr = read(root/f'row-{index:02d}.json')
        if identity(jr) != c['journal_identities'][index] or jr != json_bytes(observation):
            raise ContractError('row journal differs')
    for record in (n, o):
        if record['all_passes_forward_calls'] != calls or record['all_passes_processed_positions'] != positions:
            raise ContractError('aggregate logical work differs')
    if calls > bounds['maximum_forward_calls'] or positions > bounds['maximum_processed_positions']:
        raise ContractError('compute budget exceeded')
    unchanged(root, snapshot)
    return {'verified': True, 'completion_identity': expected, 'output_parity_recorded': True,
            'predictions_replayed': False, 'optimizer_replayed': False,
            'timings_are_observations': True, 'learning_adequate': False}
