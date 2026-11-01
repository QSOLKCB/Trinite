"""Zero-update preparation, exact source/runtime freeze and owner declaration."""
from io import BytesIO
from pathlib import Path
import platform
import sys
import time
import zipfile

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import anchored_json, inventory, read_json, retain_failure, write_json
from trinite.contracts import json_bytes
from . import data, numerical, oracle
from .spec import AUTHORIZATION, PROTOCOL, ROOT, clean_main, sources


def runtime():
    import torch
    import safetensors._safetensors_rust as safe
    files = dict(python=Path(sys.executable).resolve(),torch_extension=Path(torch._C.__file__),
        torch_cpu=Path(torch.__file__).parent/'lib/libtorch_cpu.so',
        c10=Path(torch.__file__).parent/'lib/libc10.so',safetensors=Path(safe.__file__))
    return dict(environment=numerical.configure(),python_implementation=platform.python_implementation(),
        executable_hashes={k:gate.digest(gate.read(p,512<<20,allow_links=True)) for k,p in files.items()},
        oracle_sha256=oracle.binary_identity(),dependency_lock_sha256=gate.digest(gate.read(ROOT/'requirements.lock')))


def archive(source):
    stream = BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,sha in source.items():
            raw = gate.read(ROOT/name,4<<20)
            gate.require(gate.digest(raw) == sha, 'source changed during archival')
            info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            z.writestr(info,raw,compress_type=zipfile.ZIP_DEFLATED)
    return stream.getvalue()


def verify_archive(raw, source):
    with zipfile.ZipFile(BytesIO(raw)) as z:
        gate.require(z.namelist() == list(source), 'source archive member inventory mismatch')
        for info in z.infolist():
            gate.require(info.file_size <= 4<<20 and gate.digest(z.read(info)) == source[info.filename],
                         'archived source bytes mismatch')


def prepare_content(log=None):
    rows = data.inventory()
    stats = data.audit(rows)
    # Observe native public facts for every row; evaluate native labels ONLY
    # for train. No validation/test target is materialized here.
    public = []
    for row in rows:
        public.append(dict(id=row['id'],observation=oracle.observe(row['request'],log)))
    train = data.training_rows(rows,lambda r:oracle.target(r,log))
    return dict(rows=rows,public=public,train=train,audit=stats,
        schedule={n:numerical.schedule(train,s,f) for n,s,f in
                  [('fit-17',17,True),('main-17',17,False),('main-29',29,False)]})


def feasibility(out):
    destination = gate.fresh(out)
    try:
        start = time.monotonic()
        git,source = clean_main(),sources()
        from queue_learning_v1.artifacts import append_json
        gate.write(destination/'oracle-processes.jsonl',b'')
        content = prepare_content(lambda v:append_json(destination/'oracle-processes.jsonl',v))
        kernel = numerical.feasibility_kernel()
        elapsed = time.monotonic()-start
        setup = max(0.,elapsed-sum(float.fromhex(x) for x in kernel['kernel_seconds_hex'])
                    -float.fromhex(kernel['diagnostic_io_seconds_hex']))
        whole = float.fromhex(kernel['whole_projection_hex'])+setup*4
        gate.require(whole < 900, 'full feasibility projection exceeds attempt limit')
        proof = dict(schema='trinite.context-symbol-feasibility.v1',git=git,sources=source,
            protocol=PROTOCOL,runtime=runtime(),audit=content['audit'],schedule=content['schedule'],
            kernel=kernel,whole_projection_hex=whole.hex(),setup_seconds_hex=setup.hex(),optimizer_updates=0)
        gate.require(sources() == source and clean_main() == git, 'feasibility source changed')
        write_json(destination/'feasibility.json',proof)
    except Exception as error:
        retain_failure(destination,error)
        if isinstance(error,gate.ProcessError):
            write_json(destination/'oracle-process-failure.json',error.evidence)
        raise


def successful(receipt):
    return receipt['exit_code'] == 0 and receipt['failure'] is None


def validate_feasibility(proof, source, git, rt):
    gate.require(proof['schema']=='trinite.context-symbol-feasibility.v1' and proof['optimizer_updates']==0
        and proof['sources']==source and proof['git']==git and proof['runtime']==rt
        and proof['protocol']==PROTOCOL and proof['kernel']['parameters']==24768
        and proof['kernel']['optimizer_updates']==0 and proof['kernel']['gradient_calls']==3
        and proof['kernel']['environment']==rt['environment'],
        'fresh matching zero-update feasibility required')
    supervisor = proof['supervisor']
    gate.require(successful(supervisor) and 0<float.fromhex(supervisor['wall_seconds_hex'])<900
        and supervisor['limits']==dict(rss_bytes_per_process=1<<30,aggregate_rss_bytes=2<<30,address_space_bytes_per_process=4<<30)
        and 0<supervisor['peak_process_rss_bytes']<1<<30
        and 0<supervisor['peak_aggregate_rss_bytes']<2<<30, 'successful supervised feasibility required')
    kernel=proof['kernel']
    timings=[float.fromhex(x) for x in kernel['kernel_seconds_hex']]
    gate.require(len(timings)==3 and all(0<t<120 for t in timings), 'invalid engineering timings')
    estimate=max(timings)*256*4
    overhead=float.fromhex(kernel['diagnostic_io_seconds_hex'])
    setup=float.fromhex(proof['setup_seconds_hex'])
    whole=float.fromhex(proof['whole_projection_hex'])
    gate.require(0<estimate<120 and overhead>0 and setup>=0 and
        estimate==float.fromhex(kernel['update_projection_hex']) and
        estimate+overhead*4==float.fromhex(kernel['whole_projection_hex']) and
        whole==estimate+overhead*4+setup*4 and whole<900, 'feasibility budgets differ')


def freeze(feasibility_file, feasibility_sha, out, anchors):
    feasibility_file,out,anchors=gate.disjoint(feasibility_file,out,anchors)
    gate.require(ROOT not in out.parents and ROOT not in anchors.parents, 'freeze/independent anchors outside checkout required')
    git,source,rt=clean_main(),sources(),runtime()
    proof=anchored_json(feasibility_file,feasibility_sha)
    validate_feasibility(proof,source,git,rt)
    destination=gate.fresh(out)
    try:
        from queue_learning_v1.artifacts import append_json
        gate.write(destination/'oracle-processes.jsonl',b'')
        content=prepare_content(lambda v:append_json(destination/'oracle-processes.jsonl',v))
        gate.require(content['audit']==proof['audit'] and content['schedule']==proof['schedule'], 'feasibility data changed')
        for name,value in [('protocol.json',PROTOCOL),('data.json',content),
                           ('admission.json',data.rights(content['rows'],source)),('feasibility.json',proof)]:
            write_json(destination/name,value)
        gate.write(destination/'source.zip',archive(source))
        manifest=dict(schema='trinite.context-symbol-freeze.v1',producer_sha=git['head'],git=git,
            sources=source,runtime=rt,files=inventory(destination))
        gate.require(clean_main()==git and sources()==source and runtime()==rt, 'freeze context changed')
        sha=write_json(destination/'manifest.json',manifest)
        anchor_sha=write_json(anchors,dict(schema='trinite.context-symbol-preparation-anchor.v1',freeze_sha256=sha,
            source_archive_sha256=manifest['files']['source.zip'],producer_sha=git['head']))
        return dict(producer_sha=git['head'],freeze_sha256=sha,anchors_sha256=anchor_sha,
                    supervised_token_ceiling=12232,optimizer_updates_ceiling=640)
    except Exception as error:
        retain_failure(destination,error)
        if isinstance(error,gate.ProcessError):
            write_json(destination/'oracle-process-failure.json',error.evidence)
        raise


def context(frozen,freeze_sha,anchors,anchors_sha,live=True,scores_only=False):
    gate.require(not (live and scores_only),'portable score verification cannot authorize execution')
    frozen,anchors=gate.disjoint(frozen,anchors)
    manifest=anchored_json(frozen/'manifest.json',freeze_sha)
    gate.keys(manifest,('schema','producer_sha','git','sources','runtime','files'))
    gate.keys(manifest['git'],('head','main','clean'))
    gate.require(manifest['schema']=='trinite.context-symbol-freeze.v1'
        and manifest['git']['clean'] is True
        and manifest['producer_sha']==manifest['git']['head']==manifest['git']['main'], 'unknown freeze/producer context')
    anchor=anchored_json(anchors,anchors_sha)
    gate.require(anchor==dict(schema='trinite.context-symbol-preparation-anchor.v1',freeze_sha256=freeze_sha,
        source_archive_sha256=manifest['files']['source.zip'],producer_sha=manifest['producer_sha']), 'preparation anchor context')
    expected={k:v for k,v in inventory(frozen).items() if k!='manifest.json'}
    gate.require(expected==manifest['files'] and set(expected)=={'protocol.json','data.json','admission.json','feasibility.json','source.zip','oracle-processes.jsonl'},
        'freeze inventory changed')
    gate.require(manifest['sources']==sources() and read_json(frozen/'protocol.json')==PROTOCOL, 'frozen source/protocol mismatch')
    verify_archive(gate.read(frozen/'source.zip',32<<20),manifest['sources'])
    content=read_json(frozen/'data.json')
    gate.keys(content,('rows','public','train','audit','schedule'))
    gate.require(content['rows']==data.inventory() and content['audit']==data.audit(content['rows']), 'data/split regeneration failed')
    expected_public=[dict(id=r['id'],observation=data.observation(r)) for r in content['rows']]
    gate.require(json_bytes(content['public'])==json_bytes(expected_public), 'public facts changed')
    numerical.check_rows(content['train'])
    gate.require(content['schedule']=={n:numerical.schedule(content['train'],s,f) for n,s,f in
        [('fit-17',17,True),('main-17',17,False),('main-29',29,False)]}, 'schedule changed')
    gate.require(read_json(frozen/'admission.json')==data.rights(content['rows'],sources()), 'source admission changed')
    rt=manifest['runtime'] if scores_only else runtime()
    gate.require(rt==manifest['runtime'], 'frozen runtime/executable changed')
    validate_feasibility(read_json(frozen/'feasibility.json'),sources(),manifest['git'],rt)
    if live:
        gate.require(clean_main()==manifest['git'], 'execution source differs from clean merged freeze')
    return manifest,content


def authorize(manifest,freeze_sha,approval):
    expected=dict(schema='trinite.context-symbol-owner-approval.v1',approved_by='Trent Slade',
        producer_sha=manifest['producer_sha'],freeze_sha256=freeze_sha,reviewed_merged_green=True,
        authorization=AUTHORIZATION,supervised_token_ceiling=12232,optimizer_updates_ceiling=640)
    gate.require(json_bytes(read_json(approval,16384))==json_bytes(expected), 'explicit owner approval of exact producer/freeze/schedule required')
