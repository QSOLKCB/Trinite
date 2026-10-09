"""Closed condition/derived evidence, read-only retained receipt verification."""
import math
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json, exact_keys
from .reference_inputs import read, validate, anchors, CONDITIONS
from .reference_geometry import decode_vector, cka, distances, cosine_distances, hex_matrix
from .observation import BundleObserver, verify_observation


def audit(raw, request, condition):
    record = parse_json(raw,canonical=True)
    exact_keys(record,{'schema','condition','request_identity','model','rng_identity','layers','rows'},'reference capture')
    if (record['schema'] != 'trinite.reference-capture.v1' or record['condition'] != condition
            or record['request_identity'] != identity(json_bytes(request)) or len(record['rows']) != 16):
        raise ContractError('capture context mismatch')
    for row,probe in zip(record['rows'],request['probes']):
        exact_keys(row,{'example_identity','anchors','generation_input','output','exact_visible_answer'},'capture row')
        if row['example_identity'] != probe['example_identity']:
            raise ContractError('capture item alignment mismatch')
        if [a['endpoint'] for a in row['anchors']] != anchors(probe['prompt']):
            raise ContractError('capture anchor mismatch')
        for a in row['anchors']:
            exact_keys(a,{'endpoint','encoding','vectors'},'anchor')
            exact_keys(a['vectors'],{'block.0','final'},'layers')
            for v in a['vectors'].values(): decode_vector(v)
            e = a['encoding']; exact_keys(e,{'rendered','input_ids','offsets','user_span','selected_position'},'encoding')
            if (not 1 <= len(e['input_ids']) <= 512 or len(e['offsets']) != len(e['input_ids'])
                    or type(e['selected_position']) is not int or not 0 <= e['selected_position'] < len(e['input_ids'])):
                raise ContractError('token alignment budget')
            start,end = e['user_span']
            if e['rendered'][start:end] != probe['prompt'][:a['endpoint']]:
                raise ContractError('user-prefix alignment mismatch')
        out = row['output']; exact_keys(out,{'token_ids','text','utf8_hex','stop_reason'},'generation')
        if len(out['token_ids']) > 32 or row['exact_visible_answer'] != (out['text'] == probe['answer']):
            raise ContractError('generation scoring/budget mismatch')
        if out['text'] is not None and out['utf8_hex'] != out['text'].encode('utf8').hex():
            raise ContractError('generation byte mismatch')
    return record


def retain(path, inputs, outputs, operation):
    observer = BundleObserver(Path(path)/'provenance')
    observer.record(operation,inputs=inputs,outputs=outputs,evidence_class='DERIVED')
    receipt = observer.finalize()
    (Path(path)/'verification.json').write_bytes(json_bytes(receipt))
    verify_retained(path,inputs,outputs)


def verify_retained(path, inputs, outputs):
    path = Path(path); bundle = path/'provenance'
    result = verify_observation(bundle)
    if read(path/'verification.json') != json_bytes(result):
        raise ContractError('missing or changed retained verification receipt')
    manifest = parse_json(read(bundle/'manifest.json'),canonical=True)
    # Independent name index binds exact elected input/output membership.
    found = []
    for p in (bundle/'artifacts/sha256').iterdir():
        raw = read(p)
        if raw.startswith(b'{'):
            try: value = parse_json(raw,canonical=True)
            except ContractError: continue
            if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1': found.append(value)
    wanted = {n:identity(v) for n,v in {**inputs,**outputs}.items()}
    if len(found) != 1 or found[0]['artifacts'] != wanted:
        raise ContractError('retained condition membership mismatch')
    for key in wanted.values():
        if identity(read(bundle/'artifacts/sha256'/key.split(':')[1])) != key:
            raise ContractError('retained artifact identity mismatch')
    return identity(json_bytes(manifest))


def analyze(directory, oracle):
    directory = Path(directory); request,_ = validate(directory,inputs=False)
    records = {c:audit(read(directory/c/'capture.json'),request,c) for c in CONDITIONS}
    matrices = {c:{layer:[decode_vector(r['anchors'][-1]['vectors'][layer]) for r in rec['rows']]
                   for layer in ('block.0','final')} for c,rec in records.items()}
    pairs = (('native','stock'),('native','modelfile'),('stock','modelfile'))
    comparisons = {}
    diagnostics = {}
    for layer in ('block.0','final'):
        for a,b in pairs:
            x,y = matrices[a][layer],matrices[b][layer]; value = cka(x,y); rdm = distances(x)
            comparisons[a+':'+b+':'+layer] = {'cka_hex':None if value is None else value.hex(),
                'oracle':oracle.verify(x,y,value,rdm)}
        for c in CONDITIONS:
            x = matrices[c][layer]; rdm = distances(x)
            paths = []
            for row in records[c]['rows']:
                points = [decode_vector(a['vectors'][layer]) for a in row['anchors']]
                paths.append(math.fsum(math.sqrt(math.fsum((a-b)**2 for a,b in zip(p,q)))
                                      for p,q in zip(points,points[1:])).hex())
            # Every condition RDM is checked, including the third condition in pair order.
            oracle.verify(x,x,cka(x,x),rdm)
            diagnostics[c+':'+layer] = {'squared_euclidean_hex':hex_matrix(rdm),
                'cosine_distance_hex':hex_matrix(cosine_distances(x)),
                'paired_carrier_squared_hex':[rdm[i][i+1].hex() for i in range(0,16,2)],
                'prefix_path_length_hex':paths}
    return {'schema':'trinite.reference-analysis.v1','request_identity':identity(json_bytes(request)),
            'primary':'final layer; full prompt last user-content token; linear CKA',
            'comparisons':comparisons,'diagnostics':diagnostics,
            'exact_visible_answers':{c:sum(r['exact_visible_answer'] for r in records[c]['rows']) for c in CONDITIONS},
            'probes':16,'learning_gate':'unchanged blocked','claims':request['protocol']['claims']}


def publish(directory, oracle):
    directory = Path(directory); request,_ = validate(directory)
    request_raw = read(directory/'request.json'); mf = read(directory/'Modelfile')
    inputs = {'request.json':request_raw,'Modelfile':mf}
    manifests = {}
    for c in CONDITIONS:
        raw = read(directory/c/'capture.json'); audit(raw,request,c)
        replay = parse_json(read(directory/c/'replay.json'),canonical=True)
        if replay != {'capture_identity':identity(raw),'fresh_process_exact_match':True}:
            raise ContractError('fresh capture replay did not pass')
        outputs = {'capture.json':raw,'replay.json':json_bytes(replay),
                   'worker.log':read(directory/c/'worker.log'),'replay.log':read(directory/c/'replay.log')}
        retain(directory/c,inputs,outputs,'reference.capture.'+c)
        manifests[c] = read(directory/c/'provenance/manifest.json')
    analysis = json_bytes(analyze(directory,oracle)); (directory/'analysis.json').write_bytes(analysis)
    derived = directory/'comparison'; derived.mkdir()
    retain(derived,{**inputs,**{c+'/manifest.json':v for c,v in manifests.items()}},
           {'analysis.json':analysis},'reference.geometry.compare')
    verify(directory,oracle)


def verify(directory, oracle):
    directory = Path(directory); request,_ = validate(directory,inputs=False)
    if {p.name for p in directory.iterdir()} != {'request.json','Modelfile','locations.json','analysis.json','comparison',*CONDITIONS}:
        raise ContractError('unexpected or missing study file')
    inputs = {'request.json':read(directory/'request.json'),'Modelfile':read(directory/'Modelfile')}
    manifests = {}
    for c in CONDITIONS:
        path = directory/c
        if {p.name for p in path.iterdir()} != {'capture.json','replay.json','worker.log','replay.log','provenance','verification.json'}:
            raise ContractError('condition directory membership mismatch')
        raw = read(path/'capture.json'); audit(raw,request,c)
        replay_raw = read(path/'replay.json')
        if parse_json(replay_raw,canonical=True) != {'capture_identity':identity(raw),'fresh_process_exact_match':True}:
            raise ContractError('replay receipt mismatch')
        verify_retained(path,inputs,{'capture.json':raw,'replay.json':replay_raw,
                                   'worker.log':read(path/'worker.log'),'replay.log':read(path/'replay.log')})
        manifests[c] = read(path/'provenance/manifest.json')
    derived = directory/'comparison'
    if {p.name for p in derived.iterdir()} != {'provenance','verification.json'}:
        raise ContractError('comparison directory membership mismatch')
    analysis = json_bytes(analyze(directory,oracle))
    if analysis != read(directory/'analysis.json'): raise ContractError('retained analysis differs from recomputation')
    verify_retained(derived,{**inputs,**{c+'/manifest.json':v for c,v in manifests.items()}},{'analysis.json':analysis})
    return {'verified':True,'request_identity':identity(inputs['request.json']),
            'analysis_identity':identity(analysis),'fresh_replay':'retained receipt; use --replay to execute again'}
