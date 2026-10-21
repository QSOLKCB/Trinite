"""One non-target, native-opaque Needle compatibility attempt. No scored producer."""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ("scripts/run_tiny_compatibility_v2.py", "scripts/tiny_compatibility_v2.py",
                "scripts/tiny_needle_worker_v2.py", "scripts/tiny_output_v1.py",
                "src/trinite/contracts.py")
PROMPT = "Return the integer in field target. Record: left=5678, target=1234, right=9012."
ANSWER_SCHEMA = {"type":"object", "properties":{"answer":{"type":"integer"}},
                 "required":["answer"], "additionalProperties":False}
TOOL = {"name":"answer", "description":"Return the answer.", "parameters":ANSWER_SCHEMA}
LIMIT = 65536
STARTUP_SECONDS = 120
CALL_SECONDS = 120
CLEANUP_SECONDS = 5
RESERVATION = ".trinite-needle-compatibility-v2-attempt"
SDK_PINS = {'needle/__init__.py': {'bytes': 25742, 'identity': 'sha256:541d316cb214f50ffd4d62a4269bf0e5408433b042b1b5152c0ddfe6a24b35f4'}, 'needle/_telemetry.py': {'bytes': 2608, 'identity': 'sha256:28cc691445c91700b4b894ecc797085e9985d0bd7c113da2539fce74fd0612cc'}, 'needle/_worker.py': {'bytes': 10567, 'identity': 'sha256:514e6f42974d841728c71af320af0156860b12c4b7b9308b25a1ea1d30a59780'}, 'needle/agent/__init__.py': {'bytes': 0, 'identity': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}, 'needle/agent/fetch.py': {'bytes': 11934, 'identity': 'sha256:e8396fcedd5ae21953b564fc9a74e36e5c99550a05b5bac1bc887dcb5ab32230'}, 'needle/agent/tools.py': {'bytes': 7927, 'identity': 'sha256:22333e9845bdd19247d3cca7407bbfabb6bebd3f1a515deffedc50b44c1e428a'}, 'needle/agent/whistle.py': {'bytes': 22908, 'identity': 'sha256:88935e847203c2fa2fe66efac331457c4dc5342c21a34f0e68d85c766bcc7769'}}
ASSET_PINS = {
    "evidence/needle-config.json":{"bytes":1273,"identity":"sha256:d65e8e076d42b26ed473313e4890dfd8397274a238fd6c1715fff539b464c04e"},
    "evidence/needle-native-header.h.txt":{"bytes":4528,"identity":"sha256:a27dac4b3239eead91d124b7f7dbce5495c13019903fd817b68c16d39b055a99"},
    "evidence/needle-porting.txt":{"bytes":14649,"identity":"sha256:20fe217b9c9662ff03f6d497c4499704248265744e0d4b487c3efe245086393b"},
    "needle3.cact": {"bytes":35335380,"identity":"sha256:c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38"},
    "libneedle3.so": {"bytes":1590584,"identity":"sha256:87476ff70e80334d7c766dbf236b72f42876072df22014c6ec8a9b087b82b260"},
    "tokenizer-embedded.raw": {"bytes":109140,"identity":"sha256:9f839cd44027ea9203738897da2bac572a5cc77aec7fceacd4391fc412a3249f"},
    **{"sdk/"+k:v for k,v in SDK_PINS.items()},
}
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES":"", "NEEDLE_TELEMETRY":"0", "DO_NOT_TRACK":"1",
    "HF_HUB_DISABLE_TELEMETRY":"1", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
    "LANG":"C.UTF-8", "LC_ALL":"C.UTF-8", "TZ":"UTC", "PATH":"/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE":"1", "PYTHONNOUSERSITE":"1", "OMP_NUM_THREADS":"16",
    "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1"}

class ContractError(ValueError):
    pass

def identity(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=True,
                       allow_nan=False)+"\n").encode()

def sdk_serialization(value):
    """Faithful JSON representation; nonfinite SDK values remain visible to rejection."""
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=True)+"\n").encode("utf-8")

def parse(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:
                raise ContractError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ContractError("nonfinite JSON")))
    if encoded(value) != raw:
        raise ContractError("canonical JSON required")
    return value

def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ContractError("schema keys mismatch")

def record(path, raw):
    return {"path":path,"bytes":len(raw),"identity":identity(raw)}

def safe_path(root, name):
    if type(name) is not str or not name or "\\" in name:
        raise ContractError("invalid relative artifact path")
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in (".","..","") for x in name.split("/")) or str(p) != name:
        raise ContractError("unsafe artifact path")
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise ContractError("symlink asset root component")
    if root.is_symlink() or not root.is_dir():
        raise ContractError("asset root must be a real directory")
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError("symlink artifact component")
    if not current.is_file():
        raise ContractError("regular artifact required")
    if not stat.S_ISREG(current.stat().st_mode):
        raise ContractError("nonregular artifact")
    return current

def file_record(root, name):
    path = safe_path(root, name)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk); size += len(chunk)
    return {"path":name,"bytes":size,"identity":"sha256:"+digest.hexdigest()}

def validate_records(records):
    if type(records) is not list:
        raise ContractError("artifact list required")
    names = []
    for item in records:
        keys(item, ("path","bytes","identity"))
        name = item["path"]
        if type(name) is not str or not name or name.startswith("/") or "\\" in name or any(x in ("", ".", "..") for x in name.split("/")):
            raise ContractError("unsafe artifact path")
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ContractError("invalid byte count")
        ident = item["identity"]
        if type(ident) is not str or len(ident)!=71 or not ident.startswith("sha256:") or any(c not in "0123456789abcdef" for c in ident[7:]):
            raise ContractError("invalid identity")
        names.append(name)
    if names != sorted(set(names)):
        raise ContractError("duplicate or unordered artifact paths")
    return records

def check_records(root, records):
    validate_records(records)
    actual = [file_record(root, item["path"]) for item in records]
    if actual != records:
        raise ContractError("artifact identity changed")
    return actual

def sources():
    return [file_record(ROOT,p) for p in sorted(SOURCE_PATHS)]

def profile():
    reason = "No verified control/default in the pinned native API; not matched to v1."
    evidence = ["evidence/needle-config.json", "evidence/needle-native-header.h.txt", "evidence/needle-porting.txt", "sdk/needle/__init__.py", "sdk/needle/_worker.py"]
    unknown = {"value":None,"reason":reason,"evidence":evidence}
    return {"schema":"trinite.tiny-native-opaque-profile.v2", "target_readiness":False,
        "model":"Cactus-Compute/needle3", "model_revision":"d6c7d6a2e9655505e0f8e23339356dd1705900eb",
        "sdk_commit":"b4fc11b04f1852888d375c8c6dfbaca44572d014", "sdk_version":"3.2.1",
        "native_distribution":"3.3.3", "model_config_version":"3.1.0",
        "pairing":"empirical_probe_only_not_vendor_supported_pairing",
        "thinking":{"value":"native_controlled","disable_control":None,"evidence":evidence},
        "sampling":unknown,"gating":unknown,"penalties":unknown,
        "requested_api_max_new_tokens":256,"native_token_accounting":unknown,
        "artifact_layer_count":20,"sdk_depth_control":None,"matched_decoding":False,
        "startup_seconds":STARTUP_SECONDS,"call_seconds":CALL_SECONDS,"cleanup_seconds":CLEANUP_SECONDS,
        "response_bytes":LIMIT,"diagnostic_bytes":LIMIT,"attempts":1,
        "capture":"complete_sdk_serialization_not_native_bytes", "builtin_repair":"native_not_observable_per_call"}

def launch_template():
    return {"schema":"trinite.tiny-launch.v2", "method":"complete", "prompt":PROMPT,
        "constructor":{"tools":[TOOL],"system":"","auto_date":False,"stateless":True,
                       "generation":3,"buffer_size":LIMIT,"tool_index_path":None},
        "max_new_tokens":256,"weights":"needle3.cact","library":"libneedle3.so",
        "sdk":"sdk", "executable_functions":[],"environment":ENVIRONMENT,
        "worker":"scripts/tiny_needle_worker_v2.py", "flags":["-I","-S","-B"],
        "nested_python":"fixed_shim_-I_-S_-B", "target_readiness":False}

def interpreter_record():
    executable = Path(sys.executable).resolve()
    return {"path":str(executable),"bytes":executable.stat().st_size,
            "identity":identity(executable.read_bytes()),"version":sys.version,
            "platform":sys.platform, "shell":{"path":"/bin/sh","identity":identity(Path("/bin/sh").read_bytes())}, "system_dynamic_dependencies":{"value":None,
                "reason":"OS loader/stdlib shared libraries are not a closed binary attestation"}}

def freeze(asset_root):
    expected = [{"path":p,**v} for p,v in sorted(ASSET_PINS.items())]
    check_records(asset_root, expected)
    return {"schema":"trinite.tiny-compatibility-request.v2", "profile":profile(),
            "launch":launch_template(),"assets":expected,"sources":sources(),
            "python":interpreter_record(),"target_readiness":False}

def validate_request(request, execution=False):
    keys(request,("schema","profile","launch","assets","sources","python","target_readiness"))
    expected = {"schema":"trinite.tiny-compatibility-request.v2", "profile":profile(),
                "launch":launch_template(),"assets":[{"path":p,**v} for p,v in sorted(ASSET_PINS.items())],
                "sources":sources(),"python":request["python"],"target_readiness":False}
    keys(request["python"],("path","bytes","identity","version","platform","shell","system_dynamic_dependencies"))
    python=request["python"]
    if not isinstance(python["path"],str) or not Path(python["path"]).is_absolute():
        raise ContractError("absolute interpreter path required")
    validate_records([{ "path":"python", "bytes":python["bytes"], "identity":python["identity"] }])
    keys(python["shell"],("path","identity"))
    validate_records([{"path":"shell","bytes":0,"identity":python["shell"]["identity"]}])
    if python["shell"]["path"]!="/bin/sh":
        raise ContractError("fixed shell required")
    if python["system_dynamic_dependencies"]!={"value":None,"reason":"OS loader/stdlib shared libraries are not a closed binary attestation"}:
        raise ContractError("overstated interpreter dependency closure")
    if type(python["version"]) is not str or not python["version"] or python["platform"]!="linux":
        raise ContractError("Linux interpreter identity required")
    if execution and python!=interpreter_record():
        raise ContractError("interpreter identity mismatch")
    if encoded(request) != encoded(expected):
        raise ContractError("request/source/asset/launch/profile/interpreter mismatch")
    return request

def publish(path, raw):
    # Same fsynced temporary + atomic no-clobber link pattern as run_tiny_output.py.
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(name,path)
        descriptor = os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        os.unlink(name)

def shim_bytes(python, worker):
    return ("#!/bin/sh\n[ \"$#\" -eq 2 ] && [ \"$1\" = "+shlex.quote(worker)+
            " ] && [ \"$2\" = --child ] || exit 64\nexec "+shlex.quote(python)+
            " -I -S -B "+shlex.quote(worker)+" --child\n").encode()

def assess(raw, completed):
    # Compile the unchanged, request-bound authoritative scorer, not captured code.
    module = types.ModuleType("tiny_output_v1")
    module.__file__ = str(ROOT/"scripts/tiny_output_v1.py")
    exec(compile(Path(module.__file__).read_bytes(),module.__file__,"exec"),module.__dict__)
    strict = {"accepted":False,"answer":None,"correct":False,"reason":None}
    try:
        if len(raw)>4096:
            raise ContractError("accepted response exceeds 4096 bytes")
        answer = module.score_payload("needle3",raw)
        strict.update(accepted=True,answer=answer,correct=answer==1234)
    except (ValueError,TypeError,RecursionError) as error:
        strict["reason"] = str(error)[:512]
    return {"capture_completed":completed,"payload_diagnostic":strict,
            "attempt_accepted":completed and strict["accepted"],
            "attempt_correct":completed and strict["correct"],"target_readiness":False,
            "compatibility_scope":"ABI/load/capture only; answer correctness is not a compatibility gate"}

def supervise(command, environment, cwd, retention=None):
    """Bounded pipes and independent startup/call/cleanup clocks, process-group kill."""
    buffers = {"stdout":bytearray(),"stderr":bytearray()}
    pending = bytearray()
    events = []
    response = b""
    status = "runtime_error"
    start = time.monotonic()
    deadline = start + STARTUP_SECONDS
    call_start = call_end = None
    cleanup_deadline = None
    error_detail = None
    retained={}
    if retention is not None:
        retained={"stdout":open(retention/"worker.stdout.bin","xb"),
                  "stderr":open(retention/"runtime.stderr.bin","xb")}
    try:
        process = subprocess.Popen(command,cwd=cwd,env=environment,stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    except BaseException:
        for stream in retained.values(): stream.close()
        raise
    selector = selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ,"stdout")
    selector.register(process.stderr,selectors.EVENT_READ,"stderr")
    failure = None
    try:
        while selector.get_map():
            if time.monotonic() >= deadline - (min(0.05,CLEANUP_SECONDS/2) if call_end is not None else 0):
                failure = "cleanup_timeout" if call_end is not None else ("call_timeout" if call_start is not None else "startup_timeout")
                break
            ready = selector.select(min(0.05,max(0,deadline-time.monotonic())))
            for key,_ in ready:
                chunk = os.read(key.fileobj.fileno(),8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                channel = key.data
                cap = 2*LIMIT if channel=="stdout" else LIMIT
                remaining = cap-len(buffers[channel])
                buffers[channel].extend(chunk[:remaining])
                if channel in retained:
                    retained[channel].write(chunk[:remaining]); retained[channel].flush()
                    os.fsync(retained[channel].fileno())
                if len(chunk)>remaining:
                    failure = channel+"_overflow"
                    break
                if channel=="stdout":
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line,_,tail = pending.partition(b"\n")
                        pending = bytearray(tail)
                        if time.monotonic() >= deadline:
                            failure = "cleanup_timeout" if call_end is not None else ("call_timeout" if call_start is not None else "startup_timeout")
                            break
                        event = parse(bytes(line)+b"\n")
                        stage = event.get("event") if type(event) is dict else None
                        if stage=="ready" and not events:
                            keys(event,("event",))
                            # Start the external call clock BEFORE allowing native complete().
                            call_start=time.monotonic();deadline=call_start+CALL_SECONDS
                        elif stage=="call" and events==["ready"]:
                            keys(event,("event",))
                            # The ready/GO handshake already started the clock.
                            pass
                        elif stage in ("result","overflow") and events==["ready","call"]:
                            keys(event,("event","response"))
                            response=base64.b64decode(event["response"],validate=True)
                            if len(response)>LIMIT:
                                raise ContractError("response overflow")
                            observed_end=time.monotonic()
                            if observed_end >= deadline:
                                response=b"";failure="call_timeout"
                                break
                            call_end=observed_end; deadline=call_end+CLEANUP_SECONDS
                            cleanup_deadline=deadline
                            if stage=="overflow": failure="response_overflow"
                        elif stage=="error":
                            keys(event,("event","message"))
                            failure="worker_error"
                        else:
                            raise ContractError("invalid worker event order")
                        events.append(stage)
                        if stage=="ready":
                            process.stdin.write(b"GO\n");process.stdin.flush()
                if failure:
                    break
            if failure:
                break
        if failure is None:
            process.wait(timeout=max(0.001,deadline-time.monotonic()))
            if pending:
                failure="partial_worker_event"
            elif process.returncode != 0:
                failure="worker_exit"
            elif events != ["ready","call","result"]:
                failure="incomplete_capture"
            else:
                status="completed"
    except (ValueError,OSError,subprocess.TimeoutExpired) as error:
        failure="supervisor_error"
        error_detail=str(error)[:256]
    finally:
        if cleanup_deadline is None:
            cleanup_deadline=time.monotonic()+CLEANUP_SECONDS
        # One cleanup deadline, never reset after timeout. Kill the entire session.
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        try: process.wait(timeout=max(0,cleanup_deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            process.poll()
            if process.returncode is None: failure="cleanup_failed"
        for stream in retained.values(): stream.close()
        selector.close()
        process.stdout.close(); process.stderr.close();process.stdin.close()
    if failure:
        status=failure
    return {"status":status,"error_detail":error_detail,"load_completed":"ready" in events,"call_started":call_start is not None,
            "capture_completed":status=="completed", "events":events,
            "response_base64":base64.b64encode(response).decode(),
            "stdout_base64":base64.b64encode(buffers["stdout"]).decode(),
            "diagnostics_base64":base64.b64encode(buffers["stderr"]).decode(),
            "returncode":process.returncode,
            "call_seconds":None if call_start is None or call_end is None else (call_end-call_start).hex(),
            "call_seconds_unavailable_reason":None if call_start is not None and call_end is not None else "call did not complete",
            "generated_tokens":{"value":None,"reason":"SDK return does not establish native generated-token accounting"},
            "memory_bytes":{"value":None,"reason":"not measured in this bounded compatibility probe"}}


def stage_files(source_root, records, target_root):
    """Copy then rehash exact files; execute only the materialized private tree."""
    check_records(source_root,records)
    for item in records:
        source=safe_path(source_root,item["path"])
        target=target_root/item["path"]
        target.parent.mkdir(parents=True,exist_ok=True)
        # These inputs are bounded by the exact frozen byte counts.
        with source.open("rb") as stream:
            raw=stream.read(item["bytes"]+1)
        if record(item["path"],raw)!=item:
            raise ContractError("asset changed during materialization")
        publish(target,raw)
    return check_records(target_root,records)


def probe(request_raw, request_identity, asset_root, output, allow_native=False):
    if not allow_native:
        raise ContractError("explicit --allow-native-execution required")
    if sys.platform!="linux":
        raise ContractError("pinned Linux CPU native artifact required")
    if identity(request_raw)!=request_identity:
        raise ContractError("request identity mismatch")
    request=validate_request(parse(request_raw),execution=True)
    asset_root=Path(asset_root).absolute()
    check_records(asset_root,request["assets"])
    output=Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ContractError("output already exists")
    attempt=asset_root/RESERVATION
    # Fixed reservation, not an output-selected UUID. Never remove after failure.
    attempt.mkdir(mode=0o700)
    fd=os.open(asset_root,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    try:
        publish(attempt/"request.json",request_raw)
        publish(attempt/"reservation.json",encoded({"request_identity":request_identity,
            "scope":"one attempt in this asset root; copied roots are not globally prevented",
            "native_execution_opt_in":True,"target_readiness":False}))
        materialized_assets=stage_files(asset_root,request["assets"],attempt/"assets")
        materialized_sources=stage_files(ROOT,request["sources"],attempt/"source")
        shim=shim_bytes(request["python"]["path"],str(attempt/"assets/sdk/needle/_worker.py"))
        publish(attempt/"python-isolated",shim)
        (attempt/"python-isolated").chmod(0o500)
        environment={**ENVIRONMENT,"NEEDLE3_LIB_PATH":str(attempt/"assets/libneedle3.so"),
                     "HOME":str(attempt),"TMPDIR":str(attempt)}
        command=[request["python"]["path"],"-I","-S","-B",
                 str(attempt/"source/scripts/tiny_needle_worker_v2.py"),str(attempt)]
        invocation={"argv":command,"cwd":str(attempt),"environment":environment,
            "nested_argv":[str(attempt/"python-isolated"),str(attempt/"assets/sdk/needle/_worker.py"),"--child"],
            "shim":record("python-isolated",shim),"shell":request["python"]["shell"],
            "materialized_assets":materialized_assets,"materialized_sources":materialized_sources,
            "network":"no requested network/download path; not an OS network sandbox",
            "dynamic_native_dependencies":{"value":None,"reason":"system loader closure not attested"}}
        publish(attempt/"invocation.json",encoded(invocation))
        capture=supervise(command,environment,attempt,retention=attempt)
        raw=base64.b64decode(capture["response_base64"],validate=True)
        publish(attempt/"sdk_serialization.bin",raw)
        publish(attempt/"capture.json",encoded(capture))
        packet={"schema":"trinite.tiny-compatibility-result.v2","request":request,
            "request_identity":request_identity,"invocation":invocation,"capture":capture,
            "assessment":assess(raw,capture["capture_completed"]),"target_readiness":False}
        completed=encoded(packet)
        publish(attempt/"prepared-result.json",completed)
        publish(output,completed)
        return completed
    except BaseException as error:
        try:
            publish(attempt/"failure.json",encoded({"error":str(error)[:1024],
                "target_readiness":False,"retry":"forbidden; preserve reservation and partial artifacts"}))
        except OSError:
            pass
        raise


def verify(packet):
    keys(packet,("schema","request","request_identity","invocation","capture","assessment","target_readiness"))
    request=validate_request(packet["request"])
    if packet["schema"]!="trinite.tiny-compatibility-result.v2" or packet["target_readiness"] is not False or identity(encoded(request))!=packet["request_identity"]:
        raise ContractError("result/request identity mismatch")
    invocation=packet["invocation"]
    keys(invocation,("argv","cwd","environment","nested_argv","shim","shell","materialized_assets","materialized_sources","network","dynamic_native_dependencies"))
    if invocation["materialized_assets"]!=request["assets"] or invocation["materialized_sources"]!=request["sources"]:
        raise ContractError("materialized source/asset identity mismatch")
    cwd=invocation["cwd"]
    if type(cwd) is not str or not Path(cwd).is_absolute() or Path(cwd).name!=RESERVATION:
        raise ContractError("invalid invocation directory")
    expected_argv=[request["python"]["path"],"-I","-S","-B",cwd+"/source/scripts/tiny_needle_worker_v2.py",cwd]
    expected_env={**ENVIRONMENT,"NEEDLE3_LIB_PATH":cwd+"/assets/libneedle3.so","HOME":cwd,"TMPDIR":cwd}
    if invocation["argv"]!=expected_argv or invocation["environment"]!=expected_env or invocation["nested_argv"]!=[cwd+"/python-isolated",cwd+"/assets/sdk/needle/_worker.py","--child"]:
        raise ContractError("actual invocation override")
    if invocation["shim"]!=record("python-isolated",shim_bytes(request["python"]["path"],cwd+"/assets/sdk/needle/_worker.py")):
        raise ContractError("shim identity mismatch")
    if invocation["shell"]!=request["python"]["shell"] or invocation["network"]!="no requested network/download path; not an OS network sandbox" or invocation["dynamic_native_dependencies"]!={"value":None,"reason":"system loader closure not attested"}:
        raise ContractError("unbound shell or overstated environment claims")
    capture=packet["capture"]
    keys(capture,("status","error_detail","load_completed","call_started","capture_completed","events","response_base64","stdout_base64","diagnostics_base64","returncode","call_seconds","call_seconds_unavailable_reason","generated_tokens","memory_bytes"))
    raw=base64.b64decode(capture["response_base64"],validate=True)
    stdout=base64.b64decode(capture["stdout_base64"],validate=True)
    diagnostics=base64.b64decode(capture["diagnostics_base64"],validate=True)
    if len(raw)>LIMIT or len(stdout)>2*LIMIT or len(diagnostics)>LIMIT:
        raise ContractError("capture exceeds frozen bound")
    statuses={"completed","startup_timeout","call_timeout","cleanup_timeout","worker_error",
        "stdout_overflow","stderr_overflow","response_overflow","supervisor_error","partial_worker_event",
        "worker_exit","incomplete_capture","cleanup_failed"}
    if type(capture["status"]) is not str or capture["status"] not in statuses:
        raise ContractError("unknown capture status")
    if capture["error_detail"] is not None and (type(capture["error_detail"]) is not str or len(capture["error_detail"])>256):
        raise ContractError("invalid error detail")
    if (capture["status"]=="supervisor_error") != (capture["error_detail"] is not None):
        raise ContractError("error detail mismatch")
    if capture["returncode"] is not None and type(capture["returncode"]) is not int:
        raise ContractError("invalid process return code")
    if capture["returncode"] is None and capture["status"]!="cleanup_failed":
        raise ContractError("missing process return code")
    # Reconstruct all fully retained protocol events, including negative results.
    observed=[]; observed_response=b""; malformed=False
    lines=stdout.split(b"\n")
    for line in lines[:-1]:
        try:
            event=parse(line+b"\n")
            kind=event.get("event") if type(event) is dict else None
            if kind=="ready" and not observed:
                keys(event,("event",))
            elif kind=="call" and observed==["ready"]:
                keys(event,("event",))
            elif kind in ("result","overflow") and observed==["ready","call"]:
                keys(event,("event","response"))
                observed_response=base64.b64decode(event["response"],validate=True)
                if len(observed_response)>LIMIT: raise ContractError("response bound")
            elif kind=="error" and "error" not in observed:
                keys(event,("event","message"))
                if type(event["message"]) is not str or len(event["message"])>1024: raise ContractError("error message")
            else:
                raise ContractError("event order")
            observed.append(kind)
        except (ValueError,TypeError):
            malformed=True;break
    # A deadline can expire just before a queued event is parsed. Such trailing
    # retained bytes are evidence, never an extra accepted event or response.
    claimed=capture["events"]
    if type(claimed) is not list or claimed!=observed[:len(claimed)]:
        raise ContractError("failure event prefix mismatch")
    if claimed!=observed and capture["status"] not in {"startup_timeout","call_timeout","cleanup_timeout","cleanup_failed"}:
        raise ContractError("unaccounted complete event")
    expected_response=observed_response if any(k in claimed for k in ("result","overflow")) else b""
    if raw!=expected_response:
        raise ContractError("response/transport mismatch")
    for key,expected in (("load_completed","ready" in claimed),("call_started","ready" in claimed),("capture_completed",capture["status"]=="completed")):
        if capture[key] is not expected: raise ContractError("capture state mismatch")
    completed=capture["status"]=="completed"
    timed=any(k in claimed for k in ("result","overflow"))
    if timed:
        if type(capture["call_seconds"]) is not str or not 0<=float.fromhex(capture["call_seconds"])<=CALL_SECONDS or capture["call_seconds_unavailable_reason"] is not None:
            raise ContractError("call timing mismatch")
    elif capture["call_seconds"] is not None or capture["call_seconds_unavailable_reason"]!="call did not complete":
        raise ContractError("invented call duration")
    status=capture["status"]
    if completed and (stdout!=b"".join(encoded(e) for e in [{"event":"ready"},{"event":"call"},{"event":"result","response":capture["response_base64"]}]) or claimed!=["ready","call","result"] or capture["returncode"]!=0):
        raise ContractError("completed capture transport mismatch")
    if status=="startup_timeout" and "ready" in claimed or status=="call_timeout" and ("ready" not in claimed or timed) or status=="cleanup_timeout" and not timed:
        raise ContractError("timeout stage mismatch")
    if status=="worker_error" and "error" not in claimed or status=="response_overflow" and "overflow" not in claimed:
        raise ContractError("worker failure mismatch")
    if status=="stderr_overflow" and len(diagnostics)!=LIMIT or status=="stdout_overflow" and len(stdout)!=2*LIMIT:
        raise ContractError("overflow size mismatch")
    if status=="partial_worker_event" and not lines[-1] or status=="worker_exit" and capture["returncode"]==0:
        raise ContractError("exit/partial mismatch")
    if status=="incomplete_capture" and (claimed==["ready","call","result"] or lines[-1] or malformed or capture["returncode"]!=0):
        raise ContractError("incomplete capture mismatch")
    if status=="response_overflow" and len(raw)!=LIMIT:
        raise ContractError("overflow prefix mismatch")
    if packet["assessment"]!=assess(raw,completed):
        raise ContractError("assessment mismatch")
    if capture["generated_tokens"]!={"value":None,"reason":"SDK return does not establish native generated-token accounting"} or capture["memory_bytes"]!={"value":None,"reason":"not measured in this bounded compatibility probe"}:
        raise ContractError("invented measurements")
    return {"verified":True,"target_readiness":False,"scope":"read_only_source_and_capture_consistency_not_inference_replay_or_hardware_attestation"}
