"""Model-free contracts. Synthetic process children import only the standard library."""
import base64
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("tiny_compatibility_v2",ROOT/"scripts/tiny_compatibility_v2.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def capture(raw=b'{"type":"call","function_calls":[{"name":"answer","arguments":{"answer":1234}}]}\n'):
    events=[{"event":"ready"},{"event":"call"},{"event":"result","response":base64.b64encode(raw).decode()}]
    return {"status":"completed","error_detail":None,"load_completed":True,"call_started":True,"capture_completed":True,
        "events":["ready","call","result"],"response_base64":base64.b64encode(raw).decode(),
        "stdout_base64":base64.b64encode(b"".join(v.encoded(x) for x in events)).decode(),
        "diagnostics_base64":"","returncode":0,"call_seconds":(0.01).hex(),"call_seconds_unavailable_reason":None,
        "generated_tokens":{"value":None,"reason":"SDK return does not establish native generated-token accounting"},
        "memory_bytes":{"value":None,"reason":"not measured in this bounded compatibility probe"}}


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.assets=self.root/"assets";self.assets.mkdir()
        (self.assets/"sample").write_bytes(b"synthetic inert bytes")
        self.pins=patch.object(v,"ASSET_PINS",{"sample":{k:x for k,x in v.file_record(self.assets,"sample").items() if k!="path"}})
        self.pins.start();self.addCleanup(self.pins.stop)

    def frozen(self):
        return v.freeze(self.assets)

    def fake_run(self,request=None,captured=None):
        request=self.frozen() if request is None else request
        raw=v.encoded(request)
        with patch.object(v,"supervise",return_value=capture() if captured is None else captured) as process:
            result=v.probe(raw,v.identity(raw),self.assets,self.root/"result.json",True)
        self.assertEqual(process.call_count,1)
        return v.parse(result)

    def test_canonical_bindings_from_unrelated_cwd(self):
        old=os.getcwd()
        try:
            os.chdir(self.root)
            self.assertEqual(self.frozen()["assets"],[v.file_record(self.assets,"sample")])
        finally: os.chdir(old)

    def test_asset_rejections(self):
        valid=self.frozen()["assets"]
        for name in ("/tmp/x","../sample","./sample","a//sample","a/../sample","a\\sample"):
            bad=copy.deepcopy(valid);bad[0]["path"]=name
            with self.assertRaises(v.ContractError):v.check_records(self.assets,bad)
        with self.assertRaises(v.ContractError):v.check_records(self.assets,valid+valid)
        (self.assets/"link").symlink_to(self.assets/"sample")
        with self.assertRaises(v.ContractError):v.file_record(self.assets,"link")
        (self.assets/"dir").mkdir()
        with self.assertRaises(v.ContractError):v.file_record(self.assets,"dir")
        (self.assets/"sample").write_bytes(b"changed")
        with self.assertRaises(v.ContractError):v.check_records(self.assets,valid)

    def test_empty_sdk_file_allowed(self):
        (self.assets/"empty.py").write_bytes(b"")
        rec=v.file_record(self.assets,"empty.py")
        self.assertEqual(rec["bytes"],0)
        v.check_records(self.assets,[rec])
        self.assertEqual(v.SDK_PINS["needle/agent/__init__.py"]["bytes"],0)

    def test_opaque_profile_is_not_matched(self):
        request=self.frozen();v.validate_request(request)
        self.assertIsNone(request["profile"]["sampling"]["value"])
        self.assertFalse(request["profile"]["matched_decoding"])
        self.assertFalse(request["target_readiness"])
        self.assertEqual(request["profile"]["thinking"]["value"],"native_controlled")
        self.assertEqual(request["launch"]["method"],"complete")
        self.assertEqual(request["launch"]["constructor"]["tools"],[v.TOOL])

    def test_forged_readiness_overrides_schema_and_sources(self):
        original=self.frozen()
        mutations=[lambda r:r.update(target_readiness=True),
            lambda r:r["profile"].update(matched_decoding=True),
            lambda r:r["profile"].update(temperature=0),
            lambda r:r["launch"].update(prompt="target row"),
            lambda r:r["launch"].update(max_new_tokens=1024),
            lambda r:r["launch"]["constructor"].update(system="override"),
            lambda r:r.update(schema="v1"),
            lambda r:r["sources"][0].update(identity="sha256:"+"0"*64),
            lambda r:r["assets"][0].update(identity="sha256:"+"0"*64)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                request=copy.deepcopy(original);mutate(request)
                with self.assertRaises(v.ContractError):v.validate_request(request)

    def test_explicit_opt_in_and_request_identity(self):
        raw=v.encoded(self.frozen())
        with patch.object(v,"supervise") as process:
            with self.assertRaises(v.ContractError):v.probe(raw,v.identity(raw),self.assets,self.root/"result",False)
            with self.assertRaises(v.ContractError):v.probe(raw,"sha256:"+"0"*64,self.assets,self.root/"result",True)
            process.assert_not_called()
        self.assertFalse((self.assets/v.RESERVATION).exists())

    def test_packet_verification_and_materialized_files(self):
        packet=self.fake_run()
        self.assertTrue(v.verify(packet)["verified"])
        attempt=self.assets/v.RESERVATION
        self.assertTrue((attempt/"reservation.json").exists())
        v.check_records(attempt/"source",packet["request"]["sources"])
        v.check_records(attempt/"assets",packet["request"]["assets"])
        self.assertFalse(packet["target_readiness"])
        self.assertTrue(packet["assessment"]["payload_diagnostic"]["correct"])
        for mutate in (lambda p:p["invocation"]["materialized_assets"][0].update(bytes=1),
                       lambda p:p["invocation"]["environment"].update(NEEDLE3_LIB_PATH="/other.so"),
                       lambda p:p["invocation"]["argv"].append("override"),
                       lambda p:p["assessment"].update(target_readiness=True),
                       lambda p:p["capture"].update(stdout_base64=""),
                       lambda p:p["capture"].update(generated_tokens={"value":0,"reason":None})):
            bad=copy.deepcopy(packet);mutate(bad)
            with self.assertRaises(v.ContractError):v.verify(bad)

    def test_quality_is_separate_and_metadata_preserved(self):
        for payload in ({"type":"call","function_calls":[{"name":"answer","arguments":{"answer":999}}]},
                        {"type":"call","unknown_metadata":{"keep":[1,2]},"suppressed_calls":[{"x":1}]},
                        {"type":"error","error":"retained"}):
            raw=v.encoded(payload);result=v.assess(raw,True)
            self.assertTrue(result["capture_completed"])
            self.assertFalse(result["payload_diagnostic"]["correct"])
            self.assertFalse(result["target_readiness"])
            self.assertEqual(v.parse(raw),payload)
        failed=v.assess(b'{"type":"call","function_calls":[{"name":"answer","arguments":{"answer":1234}}]}',False)
        self.assertTrue(failed["payload_diagnostic"]["correct"])
        self.assertFalse(failed["attempt_correct"])
        self.assertFalse(failed["attempt_accepted"])
        nonfinite=b'{"type":"call","confidence":NaN,"function_calls":[{"name":"answer","arguments":{"answer":1234}}]}'
        self.assertFalse(v.assess(nonfinite,True)["payload_diagnostic"]["accepted"])

    def test_failed_receipt_rejects_forged_state_and_claims(self):
        c=capture();c.update(status="worker_exit",capture_completed=False,returncode=2)
        packet=self.fake_run(captured=c)
        self.assertTrue(v.verify(packet)["verified"])
        self.assertFalse(packet["assessment"]["attempt_correct"])
        mutations=(lambda p:p["capture"].update(status="invented"),
            lambda p:p["capture"].update(status="incomplete_capture",returncode=0),
            lambda p:p["capture"].update(events=["invented"]),
            lambda p:p["capture"].update(returncode=True),
            lambda p:p["capture"].update(call_started=False),
            lambda p:p["capture"].update(call_seconds="nan"),
            lambda p:p["invocation"].update(shell="fabricated"),
            lambda p:p["invocation"].update(network="fully sandboxed"),
            lambda p:p["invocation"].update(dynamic_native_dependencies="all attested"))
        for mutation in mutations:
            bad=copy.deepcopy(packet);mutation(bad)
            with self.assertRaises((v.ContractError,ValueError)):v.verify(bad)

    def test_sdk_serialization_preserves_nonfinite_and_unknown_fields(self):
        value={"confidence":float("nan"),"unknown":{"suppressed":[1]},"error":"retained"}
        raw=v.sdk_serialization(value)
        self.assertIn(b'NaN',raw)
        self.assertIn(b'"suppressed":[1]',raw)
        self.assertFalse(v.assess(raw,True)["payload_diagnostic"]["accepted"])
        surrogate=v.sdk_serialization({"unknown":"\ud800"})
        self.assertIn(b'\\ud800',surrogate)
        self.assertFalse(v.assess(surrogate,True)["payload_diagnostic"]["accepted"])

    def test_reservation_blocks_repeat_even_different_output(self):
        packet=self.fake_run();raw=v.encoded(packet["request"])
        with patch.object(v,"supervise") as process:
            with self.assertRaises(FileExistsError):v.probe(raw,v.identity(raw),self.assets,self.root/"another",True)
            process.assert_not_called()

    def test_publication_crash_preserves_attempt_and_forbids_retry(self):
        request=self.frozen();raw=v.encoded(request);publish=v.publish
        def fail_output(path,data):
            if Path(path)==self.root/"result":raise OSError("synthetic publication crash")
            return publish(path,data)
        with patch.object(v,"supervise",return_value=capture()) as process,patch.object(v,"publish",side_effect=fail_output):
            with self.assertRaises(OSError):v.probe(raw,v.identity(raw),self.assets,self.root/"result",True)
            self.assertEqual(process.call_count,1)
        attempt=self.assets/v.RESERVATION
        self.assertTrue((attempt/"prepared-result.json").exists())
        self.assertTrue((attempt/"failure.json").exists())
        with patch.object(v,"supervise") as process:
            with self.assertRaises(FileExistsError):v.probe(raw,v.identity(raw),self.assets,self.root/"second",True)
            process.assert_not_called()

    def test_failure_retained_without_retry(self):
        request=self.frozen();raw=v.encoded(request)
        with patch.object(v,"supervise",side_effect=OSError("synthetic spawn failure")) as process:
            with self.assertRaises(OSError):v.probe(raw,v.identity(raw),self.assets,self.root/"result",True)
            self.assertEqual(process.call_count,1)
        self.assertTrue((self.assets/v.RESERVATION/"failure.json").exists())

    def test_atomic_no_clobber(self):
        p=self.root/"immutable";v.publish(p,b"first")
        with self.assertRaises(FileExistsError):v.publish(p,b"second")
        self.assertEqual(p.read_bytes(),b"first")

    def test_direct_worker_rejects_unreserved_source(self):
        result=subprocess.run([sys.executable,"-I","-S","-B",str(ROOT/"scripts/tiny_needle_worker_v2.py"),str(self.assets/v.RESERVATION)],capture_output=True)
        self.assertEqual(result.returncode,1)
        self.assertIn(b"worker must execute from the reserved private source tree",result.stdout)

    def test_shim_rejects_nonfixed_arguments(self):
        shim=self.root/"shim";worker=self.root/"synthetic.py"
        worker.write_text("import sys; print(int(sys.flags.isolated), int(sys.flags.no_site), sys.dont_write_bytecode)\n")
        shim.write_bytes(v.shim_bytes(sys.executable,str(worker)));shim.chmod(0o500)
        result=subprocess.run([str(shim),str(worker),"--child"],capture_output=True,check=True)
        self.assertEqual(result.stdout,b"1 1 True\n")
        self.assertEqual(subprocess.run([str(shim),"-c","raise Exception()"],capture_output=True).returncode,64)


class Supervisor(unittest.TestCase):
    def run_child(self,body,**bounds):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);script=root/"synthetic.py"
            script.write_text("import sys,time,json,base64\ndef emit(x):\n print(json.dumps(x,sort_keys=True,separators=(',',':')),flush=True)\n if x.get('event')=='ready': sys.stdin.buffer.readline(4)\n"+body)
            with patch.multiple(v,STARTUP_SECONDS=bounds.get("startup",2),CALL_SECONDS=bounds.get("call",2),CLEANUP_SECONDS=bounds.get("cleanup",1)):
                result=v.supervise([sys.executable,"-I","-S","-B",str(script)],dict(v.ENVIRONMENT),root,root)
            self.assertEqual((root/"worker.stdout.bin").read_bytes(),base64.b64decode(result["stdout_base64"]))
            self.assertEqual((root/"runtime.stderr.bin").read_bytes(),base64.b64decode(result["diagnostics_base64"]))
            return result

    def test_completed(self):
        result=self.run_child("emit({'event':'ready'});emit({'event':'call'});emit({'event':'result','response':base64.b64encode(b'{}').decode()})\n")
        self.assertEqual(result["status"],"completed")
        self.assertEqual(base64.b64decode(result["response_base64"]),b"{}")

    def test_startup_call_and_cleanup_deadlines(self):
        self.assertEqual(self.run_child("time.sleep(5)\n",startup=.1)["status"],"startup_timeout")
        self.assertEqual(self.run_child("emit({'event':'ready'});emit({'event':'call'});time.sleep(5)\n",call=.1)["status"],"call_timeout")
        r=self.run_child("emit({'event':'ready'});emit({'event':'call'});emit({'event':'result','response':'e30='});time.sleep(5)\n",cleanup=.1)
        self.assertEqual(r["status"],"cleanup_timeout");self.assertFalse(r["capture_completed"])
        self.assertEqual(base64.b64decode(r["response_base64"]),b"{}")

    def test_partial_and_error_preserved(self):
        r=self.run_child("sys.stdout.write('partial');sys.stdout.flush();sys.stderr.write('failure');sys.exit(3)\n")
        self.assertEqual(r["status"],"partial_worker_event")
        self.assertEqual(base64.b64decode(r["stdout_base64"]),b"partial")
        self.assertEqual(base64.b64decode(r["diagnostics_base64"]),b"failure")
        self.assertFalse(r["capture_completed"])

    def test_overflow(self):
        r=self.run_child("sys.stderr.write('x'*70000);sys.stderr.flush();time.sleep(5)\n")
        self.assertEqual(r["status"],"stderr_overflow")
        self.assertEqual(len(base64.b64decode(r["diagnostics_base64"])),v.LIMIT)

    def test_wrong_event_order(self):
        r=self.run_child("emit({'event':'result','response':'e30='})\n")
        self.assertTrue(r["status"].startswith("supervisor_error"))


if __name__=="__main__":unittest.main()
