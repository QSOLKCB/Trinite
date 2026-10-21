"""Synthetic contracts only: no checkpoint, engine, model call or GPU."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tiny_output_tests', ROOT/'scripts/tiny_output_v1.py')
m = importlib.util.module_from_spec(spec)
# Direct source compilation, no project bytecode lookup.
exec(compile(Path(spec.origin).read_bytes(), spec.origin, 'exec'), m.__dict__)


def manifest():
    result = {}
    for name in m.MODELS:
        digest = 'sha256:'+'a'*64
        artifacts = [{'path':name+'/'+role, 'role':role, 'bytes':1,
            'identity':('sha256:'+m.PROFILES[name]['weights_sha256'] if role == 'weights' else digest)}
            for role in ('weights','runtime','tokenizer','chat_template','compatibility_receipt','launch_config')]
        result[name] = {'profile':copy.deepcopy(m.PROFILES[name]),
            'runtime_revision':m.PROFILES[name].get('runtime_revision', 'b'*40),
            'settings':{'max_new_tokens':256,'temperature':'0','fresh_state':True,
                'network':False,'telemetry':False,'thinking':False,'auto_date':False,
                'tools_executed':False,'depth':20 if name == 'needle3' else None},
            'artifacts':artifacts, 'hardware':'synthetic test double', 'compatibility_receipt':digest,'launch_config_identity':digest}
    return result


def capture(request):
    attempts = []
    for model in m.MODELS:
        for row in request['rows']:
            answer = {'answer':row['answer']}
            if model == 'needle3':
                answer = {'type':'call','function_calls':[{'name':'answer','arguments':answer}],
                          'confidence':0.94, 'success':True, 'error':None, 'error_code':None}
            raw = json.dumps(answer).encode()
            attempts.append({'model':model,'slot':row['slot'],'status':'completed',
                'raw_base64':base64.b64encode(raw).decode(), 'channel':m.PROFILES[model]['channel'],
                'elapsed_seconds_hex':float(1).hex(), 'diagnostic':'',
                'generated_tokens':{'value':None,'scope':'runtime count', 'unavailable_reason':'test double'},
                'memory_bytes':{'value':1024,'scope':'synthetic per-call process peak RSS bytes', 'unavailable_reason':None}})
    return {'schema':'trinite.tiny-output-capture.v1','request_identity':m.identity(m.json_bytes(request)), 'attempts':attempts}


class TinyOutputTests(unittest.TestCase):
    def setUp(self):
        self.request = m.request(manifest())
        self.capture = capture(self.request)

    def test_positive_and_family_denominators(self):
        packet = m.seal(self.request, self.capture)
        self.assertEqual(m.verify(packet), packet)
        self.assertEqual([x['correct'] for x in packet['report']['summaries']], [16]*4)
        self.assertEqual([x['denominator'] for x in packet['report']['summaries']], [16]*4)

    def test_rows_provenance_and_exact_oracle(self):
        self.assertEqual(len(m.rows()), 32)
        for row in m.rows():
            self.assertIn('generator_revision', row)
            self.assertIn('verifier', row)
            if row['family'] == 'modular_addition':
                v = row['inputs']; self.assertEqual(row['answer'], (v['a']+v['b']) % v['n'])

    def test_suppression_refusal_and_error(self):
        good = {'type':'call','function_calls':[{'name':'answer','arguments':{'answer':3}}]}
        for change in ({'suppressed_calls':[{'answer':3}]}, {'type':'refuse'}, {'success':False},
                       {'error':'failed'}, {'refused':True}, {'confidence':True}, {'confidence':0.01}, {'validation':{'ungrounded':['answer']}},
                       {'validation':{'negation':True}}, {'function_calls':[]}):
            with self.subTest(change=change), self.assertRaises(m.ContractError):
                m.score_payload('needle3', json.dumps(good | change).encode())

    def test_strict_answers(self):
        for raw in (b'{"answer":true}', b'{"answer":1.0}', b'{"answer":1,"answer":2}',
                    b'```json\n{"answer":1}\n```', b'{"answer":1,"extra":2}', b'{"answer":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(m.ContractError):
                m.score_payload('bonsai2-2b', raw)

    def test_wrong_valid_answer_and_baselines(self):
        self.capture['attempts'][32]['raw_base64'] = base64.b64encode(b'{"answer":999}').decode()
        result = m.report(self.request,self.capture)
        self.assertTrue(result['scores'][32]['format_valid'])
        self.assertFalse(result['scores'][32]['correct'])
        self.assertEqual(result['summaries'][2]['format_valid'],16)
        self.assertEqual(result['summaries'][2]['correct'],15)
        self.assertEqual(result['summaries'][2]['best_constant_oracle_correct'],1)
        self.assertEqual(result['summaries'][3]['constant_zero_correct'],4)

    def test_timeout_correct_payload_not_scored(self):
        self.capture['attempts'][0]['status'] = 'timeout'
        result = m.report(self.request, self.capture)
        self.assertFalse(result['scores'][0]['correct'])
        self.assertEqual(result['summaries'][0]['denominator'],16)
        self.assertEqual(result['summaries'][0]['correct'],15)

    def test_missing_duplicate_reordered_slots(self):
        for action in ('missing','duplicate','reorder'):
            c = copy.deepcopy(self.capture)
            if action == 'missing': c['attempts'].pop()
            elif action == 'duplicate': c['attempts'][1] = c['attempts'][0]
            else: c['attempts'].reverse()
            with self.subTest(action=action), self.assertRaises(m.ContractError):
                m.report(self.request,c)

    def test_binding_and_bool_int_tampering(self):
        self.request['rows'][1]['answer'] = True
        with self.assertRaises(m.ContractError): m.validate_request(self.request)
        request = m.request(manifest())
        packet = m.seal(request,capture(request))
        packet['report']['scores'][0]['correct'] = 1
        with self.assertRaises(m.ContractError): m.verify(packet)

    def test_bad_measurements_and_limits(self):
        for key,value in [('elapsed_seconds_hex','inf'), ('elapsed_seconds_hex','0x1p+999999'), ('elapsed_seconds_hex',float(121).hex()),
                          ('raw_base64','%%%'), ('slot',False), ('channel','raw_model_tokens')]:
            c = copy.deepcopy(self.capture); c['attempts'][0][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(m.ContractError):
                m.report(self.request,c)
        c = copy.deepcopy(self.capture)
        c['attempts'][0]['generated_tokens']['unavailable_reason'] = None
        with self.assertRaises(m.ContractError): m.report(self.request,c)

    def test_unsupported_settings_and_pins(self):
        for key,value in [('network',True),('telemetry',True),('thinking',True),('depth',True)]:
            x = manifest(); x['needle3']['settings'][key] = value
            with self.subTest(key=key), self.assertRaises(m.ContractError): m.request(x)
        x = manifest(); x['bonsai2-2b']['runtime_revision'] = 'a'*40
        with self.assertRaises(m.ContractError): m.request(x)

    def test_asset_checker_reads_not_executes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(m.ContractError): m.check_assets(manifest(),folder)

    def test_cli_seal_verify_retention_and_no_clobber(self):
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            request = folder/'request.json'; capture = folder/'capture.json'; output = folder/'packet.json'
            request.write_bytes(m.json_bytes(self.request)); capture.write_bytes(m.json_bytes(self.capture))
            cmd = [sys.executable, '-I', '-S', str(ROOT/'scripts/run_tiny_output.py')]
            args = ['seal','--request',str(request),'--request-identity',m.identity(request.read_bytes()),
                    '--capture',str(capture),'--output',str(output)]
            result = subprocess.run(cmd+args,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            original = output.read_bytes()
            result = subprocess.run(cmd+['verify','--packet',str(output),'--packet-identity',m.identity(original)],capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(subprocess.run(cmd+args,capture_output=True).returncode,1)
            self.assertEqual(output.read_bytes(),original)
            self.capture['attempts'][0]['elapsed_seconds_hex'] = '0x1p+999999'
            capture.write_bytes(m.json_bytes(self.capture))
            args[-1] = str(folder/'overflow.json')
            self.assertEqual(subprocess.run(cmd+args,capture_output=True).returncode,1)
            overflow = list(folder.glob('overflow.json.attempt-*'))[0]
            self.assertTrue((overflow/'failure.json').is_file())
            capture.write_bytes(b'not json')
            args[-1] = str(folder/'failed.json')
            self.assertEqual(subprocess.run(cmd+args,capture_output=True).returncode,1)
            self.assertFalse((folder/'failed.json').exists())
            attempt = list(folder.glob('failed.json.attempt-*'))[0]
            self.assertEqual((attempt/'capture.json').read_bytes(),b'not json')
            self.assertTrue((attempt/'failure.json').is_file())

    def test_schema_never_contains_per_row_gold(self):
        schema = m.PROTOCOL['answer_schema']
        self.assertEqual(schema['properties']['answer'], {'type':'integer'})
        self.assertNotIn('enum',m.json_bytes(schema).decode())


if __name__ == '__main__':
    unittest.main()
