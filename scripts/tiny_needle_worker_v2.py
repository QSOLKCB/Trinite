"""Fixed isolated worker. Only the explicit probe path invokes this local native code."""
import sys
if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
    raise SystemExit("worker requires -I -S -B")
import base64
import os
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
module=types.ModuleType("tiny_compatibility_v2")
module.__file__=str(ROOT/"scripts/tiny_compatibility_v2.py")
exec(compile(Path(module.__file__).read_bytes(),module.__file__,"exec"),module.__dict__)


def main():
    if len(sys.argv)!=2:
        raise SystemExit("fixed attempt directory required")
    attempt=Path(sys.argv[1])
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),"wb",buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def emit(value):
        protocol.write(module.encoded(value))
    model=None
    try:
        if not attempt.is_absolute() or attempt.name!=module.RESERVATION or ROOT!=attempt/"source":
            raise module.ContractError("worker must execute from the reserved private source tree")
        reservation=module.parse((attempt/"reservation.json").read_bytes())
        request=module.validate_request(module.parse((attempt/"request.json").read_bytes()),execution=True)
        if reservation!={"request_identity":module.identity(module.encoded(request)),
            "scope":"one attempt in this asset root; copied roots are not globally prevented",
            "native_execution_opt_in":True,"target_readiness":False}:
            raise module.ContractError("native execution reservation mismatch")
        module.check_records(attempt/"assets",request["assets"])
        module.check_records(ROOT,request["sources"])
        shim=attempt/"python-isolated"
        if shim.is_symlink() or shim.read_bytes()!=module.shim_bytes(request["python"]["path"],str(attempt/"assets/sdk/needle/_worker.py")):
            raise module.ContractError("nested interpreter shim mismatch")
        expected={**module.ENVIRONMENT,"NEEDLE3_LIB_PATH":str(attempt/"assets/libneedle3.so"),
                  "HOME":str(attempt),"TMPDIR":str(attempt)}
        # CPython can add LC_CTYPE during startup; LANG/LC_ALL remain explicitly frozen.
        actual=dict(os.environ)
        if actual.get("LC_CTYPE")=="C.UTF-8" and "LC_CTYPE" not in expected:
            actual.pop("LC_CTYPE")
        if actual!=expected:
            raise module.ContractError("worker environment mismatch")
        sys.executable=str(shim)
        sys.path.insert(0,str(attempt/"assets/sdk"))
        # No external SDK is imported except here, after all materialized pins pass.
        import needle
        if needle.__version__!="3.2.1":
            raise module.ContractError("SDK version mismatch")
        model=needle.Needle(weights=str(attempt/"assets/needle3.cact"),
                            **request["launch"]["constructor"])
        emit({"event":"ready"})
        if sys.stdin.buffer.readline(4)!=b"GO\n":
            raise module.ContractError("missing supervisor GO handshake")
        emit({"event":"call"})
        result=model.complete(module.PROMPT,max_new_tokens=256)
        raw=module.sdk_serialization(result)
        if len(raw)>module.LIMIT:
            # Retain the bounded serialization prefix without treating it as complete.
            emit({"event":"overflow","response":base64.b64encode(raw[:module.LIMIT]).decode()})
            return 1
        emit({"event":"result","response":base64.b64encode(raw).decode()})
        model.close()
        model=None
        return 0
    except Exception as error:
        emit({"event":"error","message":str(error)[:1024]})
        return 1
    finally:
        protocol.close()
        # Never make a second complete() call or execute generated tool functions.
        # The supervisor owns bounded cleanup of this process and its entire session.


if __name__=="__main__":
    raise SystemExit(main())
