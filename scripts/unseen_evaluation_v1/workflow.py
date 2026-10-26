"""Inference-only worker and read-only evidence consistency audit."""
from pathlib import Path
import resource
import time

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from trinite.observation import BundleObserver, verify_observation
from native_diagnostic_v1.artifacts import Store
from native_diagnostic_v1.request import corpus
from .request import PROTOCOL, read, validate, separate
from .producer import load_model
from .packet import inventory, unchanged
from .measurements import measure, validate_report

CLAIMS = {'optimizer_replayed': False, 'predictions_replayed': False,
          'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
          'fresh_confirmation': False}


def run(root, request_raw, request_identity, producer, expected_directory):
    separate(root, producer)
    request = validate(request_raw, request_identity, producer)
    store = Store(root, expected_directory)
    outputs = {}
    started = time.monotonic()
    def write(name, value):
        raw = json_bytes(value)
        store.write(name, raw); outputs[name] = raw
    try:
        store.write('request.json', request_raw)
        model = load_model(producer, request['producer'])
        progress_index = 0
        def retain_row(row, targets):
            nonlocal progress_index
            write(f'row-{progress_index:04d}.json', {'row': row, 'targets': targets})
            progress_index += 1
        measurements = measure(model, on_row=retain_row)
        validate_report(measurements)
        if measurements['model_identity'] != request['producer']['model_identity']:
            raise ContractError('evaluated model identity differs from frozen producer')
        write('row-index.json', {name: identity(raw) for name, raw in outputs.items()})
        write('measurements.json', measurements)
        write('resources.json', {
            'schema': 'trinite.unseen-evaluation-resources.v1',
            'wall_seconds_hex': (time.monotonic()-started).hex(),
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope': 'wall time after initial validation through measurements; RSS process high-water at that sample includes imports and completed validation; finalization excluded; no hard memory or CPU-time cap',
            'kv_payload_bytes': None, 'kv_unavailable_reason': 'native backend has no reusable cache'})
        validate(request_raw, request_identity, producer)
        write('report.json', report(request_identity, request['producer']))
        dataset, manifest = corpus()
        store.check()
        provenance = Path('/proc/self/fd')/str(store.fd)/'provenance'
        observer = BundleObserver(provenance)
        observer.record('frozen-checkpoint-historical-heldout', inputs={
            'request.json': request_raw, 'dataset.json': dataset, 'dataset-manifest.json': manifest},
            outputs={name: raw for name, raw in outputs.items() if not (name.startswith('row-') and name != 'row-index.json')})
        verification = observer.finalize()
        if not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']:
            raise ContractError('unseen evaluation custody did not close')
        store.write('verification.json', json_bytes(verification))
        validate(request_raw, request_identity, producer)
        completion = {'schema': 'trinite.unseen-evaluation-completion.v1',
            'request_identity': request_identity,
            'artifacts': {name: identity(raw) for name, raw in outputs.items()},
            'manifest_identity': identity(read(provenance/'manifest.json')),
            'verification_identity': identity(json_bytes(verification))}
        encoded = json_bytes(completion)
        store.write('completion.json', encoded)
        return identity(encoded)
    finally:
        store.close()


def report(request_identity, binding):
    return {'schema': 'trinite.unseen-evaluation-report.v1', 'outcome': 'completed',
            'request_identity': request_identity, 'producer': binding,
            'exposure': PROTOCOL['exposure'], 'partitions': PROTOCOL['partitions'],
            'training_updates': 0, 'cache_mode': 'unsupported', **CLAIMS,
            'audit_scope': 'source/input/custody/structure and retained metric consistency; no prediction or optimizer replay'}


def resources(record):
    exact_keys(record, {'schema', 'wall_seconds_hex', 'peak_rss_kib', 'scope',
                       'kv_payload_bytes', 'kv_unavailable_reason'}, 'evaluation resources')
    from native_diagnostic_v1.measurements import _hex
    _hex(record['wall_seconds_hex'], 'worker wall time', 0, PROTOCOL['worker_seconds'])
    if (type(record['peak_rss_kib']) is not int or record['peak_rss_kib'] <= 0
            or record['schema'] != 'trinite.unseen-evaluation-resources.v1'
            or record['scope'] != 'wall time after initial validation through measurements; RSS process high-water at that sample includes imports and completed validation; finalization excluded; no hard memory or CPU-time cap'
            or record['kv_payload_bytes'] is not None
            or record['kv_unavailable_reason'] != 'native backend has no reusable cache'):
        raise ContractError('invalid evaluation resource observations')


def audit(root, request_raw, request_identity, producer, expected_identity, *, include_measurements=False):
    """No forward calls: retained logits-derived values are not independently replayed."""
    root = Path(root)
    snapshot = inventory(root)
    separate(root, producer)
    request = validate(request_raw, request_identity, producer)
    require_identity(expected_identity)
    used = 0
    def retained(name):
        nonlocal used
        raw = read(root/name)
        used += len(raw)
        if used > PROTOCOL['max_retained_payload_bytes']:
            raise ContractError('evaluation retained payload exceeds bound')
        return raw
    completion_raw = retained('completion.json')
    if identity(completion_raw) != expected_identity:
        raise ContractError('evaluation completion identity mismatch')
    completion = parse_json(completion_raw, canonical=True)
    exact_keys(completion, {'schema','request_identity','artifacts','manifest_identity','verification_identity'}, 'evaluation completion')
    if completion['schema'] != 'trinite.unseen-evaluation-completion.v1' or completion['request_identity'] != request_identity:
        raise ContractError('evaluation completion request/schema mismatch')
    if retained('request.json') != request_raw:
        raise ContractError('retained evaluation request differs')
    exact_keys(completion['artifacts'], {'measurements.json','report.json','resources.json','row-index.json'} |
               {f'row-{i:04d}.json' for i in range(128)}, 'evaluation artifacts')
    bound = {}
    for name, expected in completion['artifacts'].items():
        require_identity(expected)
        raw = retained(name)
        if identity(raw) != expected:
            raise ContractError('evaluation artifact identity differs: '+name)
        bound[name] = parse_json(raw, canonical=True)
    if json_bytes(bound['report.json']) != json_bytes(report(request_identity, request['producer'])):
        raise ContractError('evaluation report context mismatch')
    validate_report(bound['measurements.json'])
    if json_bytes(bound['row-index.json']) != json_bytes({name: digest for name, digest in completion['artifacts'].items()
            if name.startswith('row-') and name != 'row-index.json'}):
        raise ContractError('retained row index differs from completion')
    measurements = bound['measurements.json']
    for index, row in enumerate(measurements['rows']):
        expected = {'row': row, 'targets': [t for t in measurements['targets']
                    if t['example_identity'] == row['example_identity']]}
        if json_bytes(bound[f'row-{index:04d}.json']) != json_bytes(expected):
            raise ContractError('retained per-row progress differs from final measurements')
    if bound['measurements.json']['model_identity'] != request['producer']['model_identity']:
        raise ContractError('retained evaluation model differs')
    resources(bound['resources.json'])
    verification_raw = retained('verification.json')
    manifest_raw = read(root/'provenance/manifest.json')
    if identity(verification_raw) != completion['verification_identity'] or identity(manifest_raw) != completion['manifest_identity']:
        raise ContractError('evaluation custody identities differ')
    actual = verify_observation(root/'provenance')
    if (json_bytes(actual) != verification_raw or not actual['integrity_verified']
            or actual['errors'] or actual['known_missing_artifacts']):
        raise ContractError('evaluation custody verification failed')
    manifest = parse_json(manifest_raw, canonical=True)
    indexes = []
    for item in manifest['core']['artifacts']:
        raw = read(root/'provenance/artifacts/sha256'/item['content_identity'][7:])
        try:
            value = parse_json(raw, canonical=True)
        except ContractError:
            continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1':
            indexes.append(value['artifacts'])
    dataset, admitted = corpus()
    expected_names = {**{name: digest for name, digest in completion['artifacts'].items()
                      if not (name.startswith('row-') and name != 'row-index.json')}, 'request.json': request_identity,
                      'dataset.json': identity(dataset), 'dataset-manifest.json': identity(admitted)}
    if indexes != [expected_names]:
        raise ContractError('evaluation custody name index differs')
    validate(request_raw, request_identity, producer)
    unchanged(root, snapshot)
    receipt = {'schema': 'trinite.unseen-evaluation-audit.v1', 'verified': True,
            'completion_identity': expected_identity, **CLAIMS,
            'scope': 'source/input/custody/structure and retained metric consistency only'}
    return (receipt, measurements) if include_measurements else receipt
