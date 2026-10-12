"""Lossless budget evidence publication after fresh complete replay."""
from pathlib import Path
import zipfile

from ...contracts import ContractError, identity, json_bytes, parse_json
from .. import evidence
from . import experiment, spec


def archive(path, root, request, request_identity, verification, verification_identity):
    path=evidence.safe_path(path); root=evidence.safe_path(root).resolve()
    if path.resolve()==root or root in path.resolve().parents:
        raise ContractError('archive must be outside the run')
    result=experiment.summarize(root,request,request_identity,verification,verification_identity,tier='replay')
    if result['outcome']!='completed': raise ContractError('archive requires complete replay matrix')
    if evidence.read_bytes(root/'summary-replay.json')!=json_bytes(result):
        raise ContractError('archive summary differs from fresh replay')
    if evidence.read_bytes(root/'verification-request.json')!=evidence.read_bytes(verification):
        raise ContractError('archive external verification copy differs')
    members={}
    for p in sorted(root.rglob('*')):
        evidence.safe_path(p)
        if p.is_file(): members['run/'+p.relative_to(root).as_posix()]=evidence.read_bytes(p)
    repo=Path(__file__).resolve().parents[4]
    req=parse_json(evidence.read_bytes(request),canonical=True)
    for name, expected in req['source'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts: raise ContractError('unsafe source member')
        raw=evidence.read_bytes(repo/p)
        if identity(raw)!=expected: raise ContractError('archive producer source mismatch')
        members['source/'+p.as_posix()]=raw
    members['external/request.json']=evidence.read_bytes(request)
    members['external/verification-request.json']=evidence.read_bytes(verification)
    members['audit/replay.json']=json_bytes(result)
    if len(members)>2048 or sum(map(len,members.values()))>512*1024*1024:
        raise ContractError('archive member budget exceeded')
    inventory={'schema':'trinite.retention-budget-archive.v1','request_identity':request_identity,
               'verification_request_identity':verification_identity,
               'members':{n:{'identity':identity(raw),'bytes':len(raw)} for n,raw in sorted(members.items())}}
    members['inventory.json']=json_bytes(inventory)
    v=parse_json(members['external/verification-request.json'],canonical=True)
    expected={**{'run/'+n:key for n,key in v['root_artifacts'].items()},
              **{'run/'+cell+'/'+n:key for cell,artifacts in v['cells'].items() for n,key in artifacts.items()}}
    if any(identity(members[n])!=key for n,key in expected.items()):
        raise ContractError('archive bytes differ from external commitments')
    for cell,artifacts in v['cells'].items():
        evidence.check_commitments(root/cell,artifacts,evidence.ARTIFACTS)
        evidence.read_bound(root/cell,spec.inputs(members['external/request.json'],spec.corpora()))
    evidence.check_commitments(root,v['root_artifacts'],evidence.ROOT_ARTIFACTS)
    experiment.read_request(request,request_identity)
    # Exclusive creation; the inventory is intentionally not self-hashed.
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(members.items()): z.writestr(name,raw)
    with zipfile.ZipFile(path) as z:
        if set(z.namelist())!=set(members) or len(z.namelist())!=len(members):
            raise ContractError('archive member coverage changed')
        if any(z.read(n)!=raw for n,raw in members.items()): raise ContractError('archive roundtrip changed bytes')
    return {'archive_identity':identity(evidence.read_bytes(path,512*1024*1024)),
            'inventory_identity':identity(members['inventory.json']),'members':len(inventory['members'])}
