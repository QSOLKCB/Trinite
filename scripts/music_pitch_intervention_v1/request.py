"""Freeze exact source, environment and the complete fixed replacement study."""
import sys
from pathlib import Path
from trinite.contracts import identity, json_bytes, parse_json, require_identity
from frozen_music_eval_v1.artifacts import publish, disjoint
from .bootstrap import ROOT, read, check
from .spec import PROTOCOL, CELLS, cell_name, dataset, plan_for, schedule_receipt


def sources():
    value = check(ROOT)
    allowed = {str(ROOT / p) for p in value['files']}
    for name, module in tuple(sys.modules.items()):
        if (name == 'trinite' or name.startswith('trinite.')
                or name.split('.')[0].endswith('_v1') or name.startswith('provenance_')):
            origin = getattr(module, '__file__', None)
            if origin is None or str(Path(origin).absolute()) not in allowed:
                raise ValueError('foreign or unbound study module: ' + name)
    return value


def core():
    source = sources()
    from trinite.training import require_environment
    raw, admission = dataset()
    return {'schema': 'trinite.music-pitch-intervention-request.v1', 'source': source,
            'environment': require_environment(), 'protocol': PROTOCOL,
            'dataset_identity': identity(raw), 'admission_identity': identity(admission),
            'cells': {cell_name(arm, seed): {'arm': arm, 'seed': seed,
                'plan': plan_for(seed).to_dict(), 'schedule': schedule_receipt(arm)}
                for arm, seed in CELLS}}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw) > 4 * 1024 * 1024 or identity(raw) != expected:
        raise ValueError('study request bytes/identity differ')
    result = parse_json(raw, canonical=True)
    if raw != json_bytes(core()):
        raise ValueError('study source/environment/admission/protocol differs')
    return result


def freeze(destination):
    destination = disjoint(destination, ROOT)
    raw = json_bytes(core())
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('study request exceeds bound')
    publish(destination, raw)
    return {'request_identity': identity(raw), 'training_performed': False,
            'target_inference_performed': False}
