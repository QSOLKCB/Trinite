"""Explicit local reference study inputs and immutable source context."""
import hashlib
import importlib.metadata
import math
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .curriculum_data import dataset, audit_bytes

ROOT = Path(__file__).resolve().parents[2]
NATIVE_REQUEST = 'sha256:f75f368f04963336de3c32f70e3c28592b4ae010f2c6e7c3acd6ab2ac016b70d'
PROTOCOL_COMMIT = '5ea3c7dea35f98ef276113c7c57dd26c9b31429c'
CONDITIONS = ('native', 'stock', 'modelfile')
FILES = ('reference_inputs.py', 'reference_capture.py', 'reference_geometry.py',
         'reference_evidence.py', 'reference-geometry-protocol.json', 'reference-model-pin.json')


def read(path, limit=32*1024*1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ContractError('missing or unsafe input: '+str(path))
    with path.open('rb') as stream: raw = stream.read(limit+1)
    if len(raw) > limit: raise ContractError('input byte budget')
    return raw


def settings():
    raw = read(ROOT/'src/trinite/reference-geometry-protocol.json')
    if identity(raw) != 'sha256:ae8c391ad3f2deaa61618cd13db2e0032f80139ac854a49c5c8f8f079d2436ce':
        raise ContractError('reference protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    # Bind the entire executable package closure plus elected launcher/oracle/locks.
    paths = sorted((ROOT/'src').rglob('*.py'))
    paths += [ROOT/'src/trinite'/n for n in FILES if n.endswith('.json')]
    paths += [ROOT/p for p in ('requirements.lock', 'requirements.reference.lock',
                              'scripts/run_reference_geometry.py', 'scripts/reference_math_oracle.py')]
    paths += [ROOT/'src/trinite'/n for n in ('provenance-pin.json','cpu-dependencies.lock')]
    return {p.relative_to(ROOT).as_posix(): identity(read(p)) for p in paths}


def probes():
    raw, manifest = dataset('arithmetic'); audit_bytes(raw, manifest)
    rows = [r for r in parse_json(raw, canonical=True)['examples'] if r['split'] == 'train']
    grouped = {}
    for r in rows:
        grouped.setdefault(r['family_id'], {}).setdefault(r['task'], {}).setdefault(r['semantic_identity'], {})[r['carrier']] = r
    tasks, carriers = settings()['tasks'], settings()['carriers']
    eligible = [f for f,g in sorted(grouped.items()) if all(any(all(c in pair for c in carriers)
                for pair in g.get(t,{}).values()) for t in tasks)]
    if len(eligible) < 2: raise ContractError('insufficient paired probe families')
    selected = []
    for f in eligible[:2]:
        for t in tasks:
            pair = next(p for _,p in sorted(grouped[f][t].items()) if all(c in p for c in carriers))
            selected.extend(pair[c] for c in carriers)
    if len(selected) != 16 or any(not r['prompt'].isascii() or not r['prompt'] for r in selected):
        raise ContractError('unsupported probe selection')
    return selected, {'dataset_identity':identity(raw),'admission_identity':identity(manifest),
                      'exposure':'native visible arithmetic training; external unknown'}


def anchors(prompt):
    if not prompt or not prompt.isascii(): raise ContractError('v1 anchors require nonempty ASCII')
    return sorted({math.ceil(len(prompt)*q/4) for q in (1,2,3,4)})


def modelfile(raw):
    try: text = raw.decode('utf-8')
    except UnicodeError as error: raise ContractError('Modelfile UTF8') from error
    lines = text.splitlines()
    if len(lines) != 2 or lines[0] != 'FROM ./model' or lines[1] != 'SYSTEM '+settings()['system']:
        raise ContractError('only frozen FROM ./model plus SYSTEM profile is supported')
    return settings()['system']


def asset_manifest(directory):
    raw = read(ROOT/'src/trinite/reference-model-pin.json')
    if identity(raw) != 'sha256:bf3f6827b750e3b0fb8d2a9bde8649b0bf274e5758a9cc76b1883aa85e9d586c':
        raise ContractError('unsupported reference model pin')
    pin = parse_json(raw, canonical=True)
    actual = []
    for entry in pin['files']:
        p = Path(directory)/entry['path']
        if p.is_symlink() or not p.is_file(): raise ContractError('missing model asset '+entry['path'])
        sha = hashlib.sha256(); blob = hashlib.sha1(f"blob {entry['bytes']}\0".encode()); size = 0
        with p.open('rb') as stream:
            while chunk := stream.read(1024*1024):
                size += len(chunk); sha.update(chunk); blob.update(chunk)
        if size != entry['bytes'] or (sha.hexdigest() != entry['sha256'] if entry['sha256'] else blob.hexdigest() != entry['git_blob']):
            raise ContractError('model asset identity mismatch '+entry['path'])
        actual.append({**entry,'content_identity':'sha256:'+sha.hexdigest()})
    if any(p.name.endswith('.py') for p in Path(directory).iterdir()):
        raise ContractError('remote model Python is forbidden')
    return {**pin,'files':actual}


def native_inputs(producer, run):
    request = read(Path(run)/'request.json')
    if identity(request) != NATIVE_REQUEST: raise ContractError('unsupported native training request')
    record = parse_json(request, canonical=True)
    for name, expected in record['source'].items():
        path = Path(producer)/name if name.startswith('scripts/') else Path(producer)/'src/trinite'/name
        if name.startswith('qec/'): path = Path(producer)/'src/trinite'/name[4:]
        if identity(read(path)) != expected: raise ContractError('native producer source mismatch '+name)
    cell = Path(run)/'arithmetic-0-dense'
    return {'request_identity':NATIVE_REQUEST,'source':record['source'],
            'tensors_identity':identity(read(cell/'tensors.safetensors')),
            'metadata_identity':identity(read(cell/'metadata.json'))}


def environment():
    import platform
    versions = {n:importlib.metadata.version(n) for n in ('torch','numpy','safetensors','transformers','tokenizers','huggingface-hub')}
    if versions != {'torch':'2.8.0+cpu','numpy':'2.3.5','safetensors':'0.6.2',
                    'transformers':'4.51.3','tokenizers':'0.21.4','huggingface-hub':'0.30.2'}:
        raise ContractError('unsupported reference backend versions')
    return {'python':platform.python_version(),'platform':platform.platform(),'versions':versions,
            'threads':1,'dtype':'float32','device':'cpu','deterministic':True,'attention':'eager','cache':False}


def freeze(directory, model, producer, native_run):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    selected, admission = probes()
    mf = ('FROM ./model\nSYSTEM '+settings()['system']+'\n').encode()
    request = {'schema':'trinite.reference-request.v1','protocol':settings(), 'protocol_commit':PROTOCOL_COMMIT,
               'source':sources(),'environment':environment(),'probes':selected,'admission':admission,
               'assets':asset_manifest(model),'native':native_inputs(producer,native_run),
               'modelfile_identity':identity(mf)}
    (directory/'request.json').write_bytes(json_bytes(request)); (directory/'Modelfile').write_bytes(mf)
    # Locations are explicitly replaceable transport hints, never scientific identities.
    (directory/'locations.json').write_bytes(json_bytes({'model':str(Path(model).resolve()),
             'producer':str(Path(producer).resolve()),'native_run':str(Path(native_run).resolve())}))
    return identity(json_bytes(request))


def validate(directory, *, inputs=True):
    directory = Path(directory); request = parse_json(read(directory/'request.json'),canonical=True)
    selected, admission = probes()
    if (request['schema'] != 'trinite.reference-request.v1' or request['source'] != sources()
            or request['protocol'] != settings() or request['protocol_commit'] != PROTOCOL_COMMIT
            or request['probes'] != selected or request['admission'] != admission):
        raise ContractError('reference request context mismatch')
    mf = read(directory/'Modelfile'); modelfile(mf)
    if identity(mf) != request['modelfile_identity']: raise ContractError('Modelfile identity mismatch')
    locations = parse_json(read(directory/'locations.json'),canonical=True)
    if inputs and (request['assets'] != asset_manifest(locations['model'])
            or request['native'] != native_inputs(locations['producer'],locations['native_run'])
            or request['environment'] != environment()):
        raise ContractError('reference input/environment mismatch')
    return request, locations
