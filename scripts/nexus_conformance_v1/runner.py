# SPDX-License-Identifier: MPL-2.0
"""Explicit CPU donor build/run; no downloads and no cached success."""
import json
import os
from pathlib import Path
import resource
import platform
import sys
import shutil
import subprocess
import tempfile
import time
from . import spec

LIMIT = 16 * 1024 * 1024
UPSTREAM = spec.ROOT/'fixtures/nexus-donor-v1/upstream'
ADAPTERS = spec.ROOT/'scripts/nexus_conformance_v1/adapters'


def publish(path, raw):
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication of result + status.
    finally:
        temporary.unlink(missing_ok=True)


def check_sources():
    lock = json.loads((UPSTREAM.parent/'source-lock.json').read_bytes())
    if set(lock) != {p.relative_to(UPSTREAM).as_posix() for p in UPSTREAM.rglob('*') if p.is_file()}:
        raise ValueError('donor source inventory mismatch')
    for name, entry in lock.items():
        expected_revision = ('a9486c34eebd395c8060f8b2b03885eff10115c6' if name.startswith('formal/')
                             else '24c7b5bb8b158d84e08b10ccc8fdd16b6889b7e4')
        if entry['revision'] != expected_revision:
            raise ValueError('donor revision role mismatch')
        raw = (UPSTREAM/name).read_bytes()
        if spec.identity(raw) != entry['sha256']:
            raise ValueError('donor source mismatch: '+name)
        import hashlib
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != entry['git_blob']:
            raise ValueError('donor Git blob mismatch: '+name)
    return lock


def execute(req, folder, rustc, lean):
    if sys.platform != 'linux':
        raise ValueError('this conformance runner requires Linux (RSS units and pinned toolchains)')
    check_sources()
    folder.mkdir(parents=False, exist_ok=False)
    publish(folder/'request.json',spec.canonical(req))
    publish(folder/'input.txt',spec.input_bytes(req['cases']))
    commands=[]
    started=time.monotonic()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    outcome={'schema':'trinite.nexus-conformance-result.v1','request_identity':spec.identity(spec.canonical(req)),
             'status':'failed','commands':commands,
             'environment':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},'claim':'exact finite CPU donor conformance only'}
    def run(argv, name, cwd, stdin=None, env=None, timeout=120):
        record={'argv':argv,'log_prefix':name,'timeout_seconds':timeout}
        commands.append(record)
        with (folder/(name+'.stdout')).open('xb') as out, (folder/(name+'.stderr')).open('xb') as err:
            try:
                result=subprocess.run(argv,cwd=cwd,input=stdin,stdout=out,stderr=err,env=env,timeout=timeout,check=False)
                record['exit_code']=result.returncode
            except subprocess.TimeoutExpired:
                record['timed_out']=True
                raise ValueError(name+' timed out') from None
        if result.returncode:
            raise ValueError(name+' exited '+str(result.returncode))
        if any((folder/(name+suffix)).stat().st_size>LIMIT for suffix in ('.stdout','.stderr')):
            raise ValueError(name+' retained log exceeds post-run acceptance size bound')
        return (folder/(name+'.stdout')).read_bytes()
    try:
        if spec.canonical(req) != spec.canonical(spec.request()):
            raise ValueError('request no longer matches sources/workload')
        rust_version=run([rustc,'--version'],'rust-version',folder).decode().strip()
        lean_version=run([lean,'--version'],'lean-version',folder).decode().strip()
        if not rust_version.startswith('rustc 1.90.0 ') or not lean_version.startswith('Lean (version 4.19.0,'):
            raise ValueError('requires Rust 1.90.0 and Lean 4.19.0')
        outcome['toolchains']={'rustc':rust_version,'lean':lean_version}
        work=folder/'build'; work.mkdir()
        (work/'ve').mkdir(); (work/'Nexus').mkdir()
        for file in ('graph.rs','lattice.rs','mod.rs'):
            raw=(UPSTREAM/'src/ve'/file).read_bytes()
            if file == 'mod.rs':
                marker=b'pub mod atlas;\n'
                if raw.count(marker)!=1:
                    raise ValueError('atlas exclusion must match exactly once')
                raw=raw.replace(marker,b'')
            if file == 'lattice.rs':
                raw+=b'\n'+(ADAPTERS/'lattice_adapter.rs').read_bytes()
            (work/'ve'/file).write_bytes(raw)
        shutil.copyfile(UPSTREAM/'src/error.rs',work/'error.rs')
        shutil.copyfile(ADAPTERS/'main.rs',work/'main.rs')
        for file in ('Graph','Lattice','Runtime'):
            shutil.copyfile(UPSTREAM/f'formal/Nexus/{file}.lean',work/f'Nexus/{file}.lean')
        shutil.copyfile(ADAPTERS/'Parity.lean',work/'Parity.lean')
        outcome['compiled_sources']={p.relative_to(work).as_posix():spec.identity(p.read_bytes())
                                     for p in sorted(work.rglob('*')) if p.is_file()}
        run([rustc,'--edition=2021','-Awarnings','main.rs','-o','rust-parity'],'rust-build',work)
        env=dict(os.environ); env['LEAN_PATH']=str(work)
        for file in ('Graph','Lattice','Runtime'):
            run([lean,'-o',f'Nexus/{file}.olean',f'Nexus/{file}.lean'],'lean-build-'+file,work,env=env)
        incoming=(folder/'input.txt').read_bytes()
        rust=run([str(work/'rust-parity')],'rust-output',work,incoming)
        lean_raw=run([lean,'--run','Parity.lean'],'lean-output',work,incoming,env)
        expected=spec.expected_bytes(req['cases'])
        publish(folder/'python-output.txt',expected)
        if rust != expected or lean_raw != expected:
            raise ValueError('full-state Rust/Lean/Python exact parity mismatch')
        rows=list(spec.items(req))
        # Per-item receipts bind the original generated item, with live parity outcome.
        receipts=[{'request_identity':outcome['request_identity'],'output_identity':spec.identity(expected),
                   'item':row,'verifier_outcome':'passed: live Rust + Lean full states; Python both roundtrips and exact query projection'} for row in rows]
        item_bytes=b''.join(spec.canonical(x) for x in receipts)
        publish(folder/'items.jsonl',item_bytes)
        if spec.canonical(req)!=spec.canonical(spec.request()):
            raise ValueError('source changed during conformance')
        outcome.update(status='passed',cases=len(req['cases']),items=len(rows),plan_tuples=len(req['plan_coverage']),
                       full_state_rows=4*len(req['cases']), output_identity=spec.identity(expected),
                       item_records_identity=spec.identity(item_bytes),split=spec.SPLIT)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        outcome['error']=str(error)[:1000]
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    outcome['resources']={'elapsed_seconds_hex':(time.monotonic()-started).hex(),
                          'child_peak_rss_kib':after.ru_maxrss,'child_peak_rss_before_kib':before.ru_maxrss,
                          'scope':'Linux RUSAGE_CHILDREN cumulative high-water including compilers and donor subprocesses; not a per-model memory benchmark'}
    outcome['artifacts']={p.name:spec.identity(p.read_bytes()) for p in sorted(folder.iterdir()) if p.is_file()}
    # Single durable file combines report and controller outcome. Missing result is incomplete,
    # never a success; a failed publication retains request/input/build/log files for diagnosis.
    publish(folder/'result.json',spec.canonical(outcome))
    return outcome
