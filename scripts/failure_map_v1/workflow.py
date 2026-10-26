"""Frozen inference worker and non-replay evidence audit, with original scores intact."""
from pathlib import Path
import resource
import time

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from trinite.observation import BundleObserver, verify_observation
from native_diagnostic_v1.artifacts import Store
from native_diagnostic_v1.measurements import _hex
from unseen_evaluation_v1.packet import inventory, unchanged
from addition_development_v1.data import dataset, audit_bytes, examples
from addition_development_v1.request import cell_name
from addition_development_v1.protocol import STRATA
from addition_development_v1.numerical import check_score
from .protocol import PROTOCOL, CLAIMS
from .request import read, validate, separate
from frozen_token_diagnostic_v1.producer import load_cell, snapshot_matrix, unchanged_matrix
from frozen_token_diagnostic_v1 import producer as producer_contract
from .measurements import measure, validate_report

RESOURCE_SCOPE = 'wall time after initial validation through measurements; RSS process high-water includes imports; finalization excluded; not hard RSS or CPU-time cap'
RAW_SCOPE = 'unchanged original scores, compared with new prompt-only predictions; audit checks retained equality, not fresh inference'


def chunk_names():
    n = PROTOCOL['progress_chunk_rows']
    return {f'progress-{i:03d}.json' for i in range((PROTOCOL['rows_per_cell']+n-1)//n)}


def artifact_names():
    return {'report.json', 'resources.json', 'progress-index.json', 'measurements.json',
            'original-training.json'} | chunk_names() | {f'original-raw-{s}.json' for s in STRATA}


def report(request_identity, binding, arm, seed):
    return {'schema': 'trinite.failure-map-report.v1', 'outcome': 'completed',
            'request_identity': request_identity, 'cell': cell_name(arm, seed),
            'producer_matrix_identity': binding['producer_matrix_identity'],
            'model_identity': binding['model_identities'][cell_name(arm, seed)],
            'checkpoint_step': 4000, 'training_updates': 0, 'cache_mode': 'unsupported',
            'target_inference_executed': True,
            'historical_raw_predictions_compared': len(examples(arm))+sum(len(examples(s)) for s in STRATA),
            'conditioning': PROTOCOL['conditioning'], 'raw_score_scope': RAW_SCOPE,
            'audit_scope': 'source/input/custody/safe structure and retained metric consistency only; not numerical replay',
            **CLAIMS}


def original_training_bytes(producer, binding, arm, seed):
    """Read the training artifact through its already externally bound completion."""
    name = cell_name(arm, seed)
    completion_raw = read(Path(producer)/name/'completion.json')
    if identity(completion_raw) != binding['cells'][name]['completion_identity']:
        raise ContractError('producer completion changed before anchor read')
    completion = parse_json(completion_raw, canonical=True)
    filename = f'training-{producer_contract.STEPS}.json'
    raw = read(Path(producer)/name/filename)
    if identity(raw) != completion['artifacts'][filename]:
        raise ContractError('producer training anchor identity differs')
    return raw


def run(root, request_raw, request_identity, producer, arm, seed, expected_directory):
    separate(root, producer)
    request = validate(request_raw, request_identity, producer)
    before = snapshot_matrix(producer)
    store = Store(root, expected_directory); outputs = {}; started = time.monotonic()
    def write(name, value, raw=False):
        encoded = value if raw else json_bytes(value)
        store.write(name, encoded); outputs[name] = encoded
    try:
        store.write('request.json', request_raw)
        model, original = load_cell(producer, request['producer'], arm, seed)
        expected_model = request['producer']['model_identities'][cell_name(arm, seed)]
        original_training = original_training_bytes(producer, request['producer'], arm, seed)
        check_score(parse_json(original_training, canonical=True), arm, 'train', expected_model)
        unchanged_matrix(producer, before)
        pending = []; chunk = 0
        def flush():
            nonlocal chunk
            if pending:
                write(f'progress-{chunk:03d}.json', list(pending))
                pending.clear(); chunk += 1
        def on_row(row):
            if time.monotonic()-started > PROTOCOL['worker_seconds']:
                raise ContractError('failure-map worker deadline exceeded')
            pending.append(row)
            if len(pending) == PROTOCOL['progress_chunk_rows']: flush()
        result = measure(model, arm, seed, original, original_training, progress=on_row); flush()
        validate_report(result, arm, seed, expected_model, original, original_training)
        write('measurements.json', result)
        write('original-training.json', original_training, raw=True)
        for stratum in STRATA:
            check_score(parse_json(original[stratum], canonical=True), arm, stratum, expected_model)
            write(f'original-raw-{stratum}.json', original[stratum], raw=True)
        write('progress-index.json', {n: identity(b) for n, b in outputs.items() if n.startswith('progress-')})
        if result['summary']['reproducibility'].get('mismatch', 0):
            raise ContractError('reproducibility/plumbing mismatch; observations retained, not a new learning failure')
        write('resources.json', {'schema': 'trinite.failure-map-resources.v1',
            'wall_seconds_hex': (time.monotonic()-started).hex(),
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope': RESOURCE_SCOPE, 'kv_payload_bytes': None,
            'forward_calls_upper_bound': PROTOCOL['max_forward_calls_per_cell']})
        write('report.json', report(request_identity, request['producer'], arm, seed))
        resources(parse_json(outputs['resources.json'], canonical=True))
        validate(request_raw, request_identity, producer); unchanged_matrix(producer, before)
        raw, admitted = dataset(); audit_bytes(raw, admitted)
        store.check(); provenance = Path('/proc/self/fd')/str(store.fd)/'provenance'
        observer = BundleObserver(provenance)
        observer.record('frozen-addition-exhaustive-failure-map', inputs={'request.json': request_raw,
            'dataset.json': raw, 'admission.json': admitted}, outputs=outputs)
        verification = observer.finalize(); store.check()
        if not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']:
            raise ContractError('failure-map custody did not close')
        store.write('verification.json', json_bytes(verification))
        validate(request_raw, request_identity, producer); unchanged_matrix(producer, before)
        encoded = json_bytes({'schema': 'trinite.failure-map-completion.v1',
            'request_identity': request_identity, 'cell': cell_name(arm, seed),
            'artifacts': {name: identity(data) for name, data in outputs.items()},
            'manifest_identity': identity(read(provenance/'manifest.json')),
            'verification_identity': identity(json_bytes(verification))})
        store.write('completion.json', encoded)
        return identity(encoded)
    finally:
        store.close()


def resources(record):
    exact_keys(record, {'schema','wall_seconds_hex','peak_rss_kib','scope','kv_payload_bytes','forward_calls_upper_bound'}, 'failure-map resources')
    _hex(record['wall_seconds_hex'], 'wall time', 0, PROTOCOL['worker_seconds'])
    if (record['schema'] != 'trinite.failure-map-resources.v1'
            or type(record['peak_rss_kib']) is not int or record['peak_rss_kib'] <= 0
            or record['scope'] != RESOURCE_SCOPE or record['kv_payload_bytes'] is not None
            or type(record['forward_calls_upper_bound']) is not int
            or record['forward_calls_upper_bound'] != PROTOCOL['max_forward_calls_per_cell']):
        raise ContractError('invalid failure-map resources/scope')


def audit(root, request_raw, request_identity, producer, arm, seed, expected_identity):
    """Read-only audit of retained predictions and comparisons; no fresh inference replay."""
    root = Path(root); before = inventory(root); separate(root, producer)
    request = validate(request_raw, request_identity, producer); require_identity(expected_identity)
    producer_before = snapshot_matrix(producer)
    raw = read(root/'completion.json')
    if identity(raw) != expected_identity: raise ContractError('failure-map completion identity differs')
    completion = parse_json(raw, canonical=True)
    exact_keys(completion, {'schema','request_identity','cell','artifacts','manifest_identity','verification_identity'}, 'failure-map completion')
    if (completion['schema'] != 'trinite.failure-map-completion.v1'
            or completion['request_identity'] != request_identity or completion['cell'] != cell_name(arm, seed)
            or read(root/'request.json') != request_raw):
        raise ContractError('failure-map completion context differs')
    exact_keys(completion['artifacts'], artifact_names(), 'failure-map artifacts')
    top = {Path(p).parts[0] for p in before['members'] if p != '.'}
    allowed = artifact_names() | {'request.json','completion.json','verification.json','provenance','worker.log','attempt.json'}
    if not top <= allowed or not (allowed-{'worker.log','attempt.json'}) <= top:
        raise ContractError('failure-map completed directory membership differs')
    bound = {name: read(root/name) for name in artifact_names()}
    if sum(map(len, bound.values())) > PROTOCOL['max_retained_payload_bytes']:
        raise ContractError('failure-map retained payload bound exceeded')
    for name, data in bound.items():
        if identity(data) != completion['artifacts'][name]:
            raise ContractError('failure-map retained artifact identity differs: '+name)
    if bound['report.json'] != json_bytes(report(request_identity, request['producer'], arm, seed)):
        raise ContractError('failure-map report context differs')
    resources(parse_json(bound['resources.json'], canonical=True))
    wanted_index = {n: completion['artifacts'][n] for n in chunk_names()}
    if bound['progress-index.json'] != json_bytes(wanted_index):
        raise ContractError('failure-map progress index differs')
    model_id = request['producer']['model_identities'][cell_name(arm, seed)]
    original = {s: bound[f'original-raw-{s}.json'] for s in STRATA}
    for stratum in STRATA:
        if read(Path(producer)/cell_name(arm, seed)/f'development-{stratum}.json') != original[stratum]:
            raise ContractError('original raw scores changed or were substituted')
        check_score(parse_json(original[stratum], canonical=True), arm, stratum, model_id)
    original_training = bound['original-training.json']
    if original_training_bytes(producer, request['producer'], arm, seed) != original_training:
        raise ContractError('original training anchors changed or were substituted')
    check_score(parse_json(original_training, canonical=True), arm, 'train', model_id)
    measurements = parse_json(bound['measurements.json'], canonical=True)
    summary = validate_report(measurements, arm, seed, model_id, original, original_training)
    if summary['reproducibility'].get('mismatch', 0):
        raise ContractError('completed failure-map contains reproducibility/plumbing mismatch')
    n = PROTOCOL['progress_chunk_rows']
    for i in range((PROTOCOL['rows_per_cell']+n-1)//n):
        if bound[f'progress-{i:03d}.json'] != json_bytes(measurements['rows'][i*n:(i+1)*n]):
            raise ContractError('failure-map progress rows differ from final measurements')
    vr = read(root/'verification.json'); manifest_raw = read(root/'provenance/manifest.json')
    if identity(vr) != completion['verification_identity'] or identity(manifest_raw) != completion['manifest_identity']:
        raise ContractError('failure-map custody identity differs')
    verified = verify_observation(root/'provenance')
    if json_bytes(verified) != vr or not verified['integrity_verified'] or verified['errors'] or verified['known_missing_artifacts']:
        raise ContractError('failure-map custody verification failed')
    indexes = []
    for item in parse_json(manifest_raw, canonical=True)['core']['artifacts']:
        payload = read(root/'provenance/artifacts/sha256'/item['content_identity'][7:])
        try: value = parse_json(payload, canonical=True)
        except ContractError: continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1': indexes.append(value['artifacts'])
    raw, admitted = dataset(); audit_bytes(raw, admitted)
    wanted = {**completion['artifacts'], 'request.json': request_identity,
              'dataset.json': identity(raw), 'admission.json': identity(admitted)}
    if indexes != [wanted]: raise ContractError('failure-map custody named inputs differ')
    validate(request_raw, request_identity, producer)
    unchanged(root, before); unchanged_matrix(producer, producer_before)
    return {'verified': True, 'completion_identity': expected_identity, 'cell': cell_name(arm, seed),
            'summary': summary, **CLAIMS}
