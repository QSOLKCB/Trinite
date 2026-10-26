"""Independent source/environment and externally committed producer binding."""
from pathlib import Path
import sys

from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from addition_development_v1.request import ROOT, read
from frozen_token_diagnostic_v1.request import sources as original_sources
from addition_development_v1.data import dataset, audit_bytes, examples
from addition_development_v1.protocol import STRATA
from .protocol import PROTOCOL


def sources():
    original = original_sources()
    paths = list((ROOT/'scripts/failure_map_v1').glob('*.py'))
    paths += [ROOT/'scripts/run_failure_map.py', ROOT/'docs/FAILURE-MAP.md']
    allowed = {p.resolve() for p in paths}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] == 'failure_map_v1':
            file = getattr(module, '__file__', None)
            if file is None or Path(file).resolve() not in allowed:
                raise ContractError('failure-map import outside source-bound checkout')
    return {**original, **{p.relative_to(ROOT).as_posix(): identity(read(p)) for p in sorted(paths)}}


def core(producer, request_identity, matrix_identity, model_identities):
    from trinite.training import require_environment
    from frozen_token_diagnostic_v1.producer import inspect_matrix
    raw, admitted = dataset(); audit_bytes(raw, admitted)
    binding = inspect_matrix(producer, request_identity, matrix_identity, model_identities)
    return {'schema': 'trinite.failure-map-request.v1',
            'protocol': PROTOCOL, 'protocol_identity': identity(json_bytes(PROTOCOL)),
            'producer': binding, 'source': sources(), 'environment': require_environment(),
            'dataset_identity': identity(raw), 'admission_identity': identity(admitted),
            'strata': {s: identity(json_bytes(examples(s))) for s in STRATA}}


def validate(raw, expected_identity, producer):
    require_identity(expected_identity)
    if type(raw) is not bytes or len(raw)>1024*1024 or identity(raw)!=expected_identity:
        raise ContractError('failure-map request identity/size differs')
    record = parse_json(raw, canonical=True)
    if type(record) is not dict or type(record.get('producer')) is not dict:
        raise ContractError('failure-map producer binding required')
    b = record['producer']
    try:
        expected = core(producer, b['producer_request_identity'], b['producer_matrix_identity'], b['model_identities'])
    except KeyError as error:
        raise ContractError('incomplete failure-map producer binding') from error
    if raw != json_bytes(expected):
        raise ContractError('failure-map source/environment/producer/protocol differs')
    return record


def separate(destination, producer):
    destination, producer = Path(destination).resolve(), Path(producer).resolve()
    if destination.is_relative_to(producer) or producer.is_relative_to(destination):
        raise ContractError('failure-map output/request overlaps immutable producer')
