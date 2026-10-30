"""Exact source/environment freeze; no model or optimizer construction."""
import sys
from pathlib import Path
from trinite.contracts import identity, json_bytes, parse_json, require_identity
from frozen_music_eval_v1.artifacts import publish, disjoint
from .bootstrap import ROOT, read, check
from .spec import PROTOCOL, BACKENDS, dataset, plan_for, schedule_receipt


def sources():
    value = check(ROOT)
    allowed = {str(ROOT/p) for p in value['files']}
    for name,module in tuple(sys.modules.items()):
        if name == 'trinite' or name.startswith('trinite.') or name.split('.')[0].endswith(('_v1', '_v2')) or name.startswith('provenance_'):
            if str(Path(getattr(module,'__file__','')).absolute()) not in allowed:
                raise ValueError('foreign or unbound trial module: '+name)
    return value


def core():
    source = sources()
    from .runtime import require_environment
    raw,receipt = dataset()
    return {'schema':'trinite.music-device-trial-request.v2','source':source,
        'environment':require_environment('cuda'), 'protocol':PROTOCOL,
        'dataset_identity':identity(raw),'admission_identity':identity(receipt),
        'cells':{b:{'backend':b,'seed':17,'arm':'control','plan':plan_for(17).to_dict(),
                    'schedule':schedule_receipt('control')} for b in BACKENDS}}


def validate(raw,expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw)>4194304 or identity(raw)!=expected:
        raise ValueError('trial request bytes/identity differ')
    value=parse_json(raw,canonical=True)
    if raw!=json_bytes(core()):raise ValueError('trial source/environment/protocol differs')
    return value


def freeze(destination):
    destination=disjoint(destination,ROOT)
    raw=json_bytes(core())
    if len(raw)>4194304:raise ValueError('trial request exceeds bound')
    publish(destination,raw)
    return {'request_identity':identity(raw),'target_training_performed':False,'model_executed':False}
