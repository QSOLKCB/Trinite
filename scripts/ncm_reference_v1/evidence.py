"""Bounded, read-only artifact verification. Not hardware attestation."""
import io
import json
import math
import re
import zipfile
from .core import digest, json_bytes, request, score, source_bytes

LIMIT = 4 * 1024 * 1024


def pack(members):
    inventory = {name: digest(data) for name, data in members.items()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted({**members, 'inventory.json': json_bytes(inventory)}.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return stream.getvalue()


def unpack(data, expected):
    if len(data) > LIMIT or digest(data) != expected:
        raise ValueError('external archive SHA-256 mismatch or size cap')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [x.filename for x in infos]
        if len(names) > 100 or len(set(names)) != len(names) or sum(x.file_size for x in infos) > LIMIT:
            raise ValueError('archive cardinality/expanded-size violation')
        if any(n.startswith('/') or '..' in n.split('/') or '\\' in n for n in names):
            raise ValueError('unsafe archive member name')
        members = {x.filename: archive.read(x) for x in infos}
    inventory = json.loads(members.pop('inventory.json'))
    if inventory != {n: digest(b) for n, b in members.items()}:
        raise ValueError('inventory mismatch')
    return members


def verify_result(result, req):
    if result.get('schema') != 'ncm-reference-result-v1' or result.get('evidence_class') != 'actual-local-cuda-inference':
        raise ValueError('wrong result schema/class')
    cases = {item['id']: item for item in req['inputs']}
    expected = [('cold', req['inputs'][0]['id']), ('warmup', req['inputs'][0]['id'])]
    expected += [('warm', x['id']) for x in req['inputs']]
    if [(x['phase'], x['id']) for x in result['outcomes']] != expected:
        raise ValueError('incomplete, duplicate or reordered outcomes')
    for row in result['outcomes']:
        codes = row['latent_ids']
        if len(codes) != 322 or any(type(x) is not int or not 0 <= x < 16 for x in codes):
            raise ValueError('invalid latent inventory')
    if [x['id'] for x in result['reconstructions']] != list(cases):
        raise ValueError('incomplete reconstruction inventory')
    for row in result['outcomes'] + result['reconstructions']:
        ids = row['raw_ids']
        if not isinstance(ids, list) or len(ids) != 89 or any(type(x) is not int or not 0 <= x < 12 for x in ids):
            raise ValueError('invalid raw output record')
        case = cases[row['id']]
        if json_bytes(row['metrics']) != json_bytes(score(ids, case['clues'], case['target'])):
            raise ValueError('metrics mismatch')
    env = result['environment']
    required = {'versions','python','platform','torch_cuda','cudnn','device','capability','total_memory','torch_build','driver','installed_packages'}
    if set(env) != required or any(not isinstance(env[k],str) or not env[k] for k in ('python','platform','torch_cuda','device','torch_build','driver')):
        raise ValueError('incomplete environment identity')
    if not env['python'].startswith('3.11.') or env['torch_cuda'] != '12.8' or type(env['cudnn']) is not int:
        raise ValueError('wrong environment identity')
    if not isinstance(env['installed_packages'],dict) or not env['installed_packages']:
        raise ValueError('missing installed package inventory')
    from .runtime import VERSIONS
    installed = {re.sub(r'[-_.]+', '-', name).lower(): value for name,value in env['installed_packages'].items()}
    if any(installed.get(re.sub(r'[-_.]+', '-', name).lower()) != version for name,version in VERSIONS.items()):
        raise ValueError('installed inventory contradicts pinned dependencies')
    if env['versions'] != VERSIONS or env['capability'] != [12, 0] or 'RTX 5060 Ti' not in env['device']:
        raise ValueError('wrong declared runtime')
    if type(env['total_memory']) is not int or env['total_memory'] < req['protocol']['minimum_total_bytes']:
        raise ValueError('wrong declared memory')


def verify_timings(timings, req):
    expected = [('cold', req['inputs'][0]['id']), ('warmup', req['inputs'][0]['id'])]
    expected += [('warm', x['id']) for x in req['inputs']]
    expected += [('gold-codec-diagnostic', x['id']) for x in req['inputs']]
    if [(x['phase'], x['id']) for x in timings['measurements']] != expected:
        raise ValueError('timing inventory mismatch')
    if timings['memory_scope'] != 'PyTorch allocator, not whole-device VRAM':
        raise ValueError('incorrect memory scope')
    if timings['max_allocated_bytes'] > timings['max_reserved_bytes']:
        raise ValueError('invalid allocated/reserved relation')
    for row in timings['measurements']:
        if 'total_seconds' not in row:
            raise ValueError('missing total timing')
        for key, value in row.items():
            if key.endswith('_seconds') and (type(value) not in (int,float) or not math.isfinite(value) or value < 0):
                raise ValueError('invalid elapsed time')
        if row['phase'] != 'gold-codec-diagnostic':
            subtotal = sum(row[k] for k in ('prepare_seconds','generator_seconds','codec_and_transfer_seconds'))
            if not math.isclose(subtotal, row['total_seconds'], abs_tol=1e-8):
                raise ValueError('timing sum mismatch')
    for key in ('max_allocated_bytes', 'max_reserved_bytes'):
        if type(timings[key]) is not int or not 0 <= timings[key] <= req['protocol']['max_allocator_bytes']:
            raise ValueError('memory budget mismatch')
    if type(timings['load_seconds']) not in (float,int) or not math.isfinite(timings['load_seconds']) or timings['load_seconds'] < 0:
        raise ValueError('invalid model-load timing')


def verify(data, expected):
    members = unpack(data, expected)
    req = json.loads(members['request.json'])
    if json_bytes(req) != json_bytes(request()):
        raise ValueError('request does not match current frozen source/protocol/formal inputs')
    source = source_bytes()
    archived = {name[7:]: value for name,value in members.items() if name.startswith('source/')}
    if archived != source:
        raise ValueError('source inventory mismatch')
    status = json.loads(members['status.json'])
    allowed = {'request.json', 'status.json', 'worker.log', 'result.json', 'timings.json'} | {'source/'+n for n in source}
    extras = set(members)-allowed
    if any(not n.startswith('progress/') or not n.endswith('.json') for n in extras):
        raise ValueError('unexpected evidence member')
    if status['state'] != 'completed':
        if status['state'] not in ('failed','timeout','interrupted'):
            raise ValueError('unknown failure state')
        return dict(state=status['state'], structural_integrity=True, inference_verified=False)
    if status['returncode'] != 0:
        raise ValueError('nonzero completed worker')
    result = json.loads(members['result.json'])
    verify_result(result, req)
    verify_timings(json.loads(members['timings.json']), req)
    return dict(state='completed', structural_integrity=True, inference_verified=False,
                metrics_recomputed=True, warm_solved=sum(x['metrics']['solved'] for x in result['outcomes'] if x['phase']=='warm'),
                samples=4, limitation='Producer evidence, not hardware attestation or numerical replay')
