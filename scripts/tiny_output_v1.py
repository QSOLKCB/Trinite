"""Prospective external model-plus-harness comparison; never executes a model."""
import base64
import hashlib
import math
import json
from pathlib import Path, PurePosixPath
import re
import types

ROOT = Path(__file__).resolve().parents[1]
# Compile the authoritative contract source directly, never a project .pyc.
_contract = types.ModuleType('tiny_output_contracts')
_contract.__file__ = str(ROOT/'src/trinite/contracts.py')
import sys
sys.modules[_contract.__name__] = _contract
exec(compile(Path(_contract.__file__).read_bytes(), _contract.__file__, 'exec'), _contract.__dict__)
ContractError = _contract.ContractError
identity, json_bytes, parse_json = _contract.identity, _contract.json_bytes, _contract.parse_json
exact_keys, integer, require_identity = _contract.exact_keys, _contract.integer, _contract.require_identity

MODELS = ('needle3', 'bonsai2-2b')
PROFILES = {
    'needle3': {'model': 'Cactus-Compute/needle3',
        'revision': 'd6c7d6a2e9655505e0f8e23339356dd1705900eb',
        'weights_sha256': 'c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38',
        'channel': 'sdk_serialization', 'builtin_repair': 'enabled_not_observable_per_call',
        'minimum_reported_confidence': '0.1',
        'precision': 'mixed_2_4bit_weights_8bit_KV_activations_not_Trinite_ternary',
        'wrapper': 'complete_only_inert_answer_function_auto_date_false_stateless_true',
        'compatibility': 'unverified_engine_3.3.3_vs_model_config_3.1.0_requires_local_preflight'},
    'bonsai2-2b': {'model': 'prism-ml/Ternary-Bonsai-2-2B-gguf',
        'revision': '683f7be2629342a50987627b40881e54761a2430',
        'weights_sha256': 'd5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928',
        'channel': 'final_text', 'builtin_repair': 'none_in_declared_profile',
        'precision': 'PQ2_0_rotated_ternary_matrices_FP16_scales_F32_auxiliary',
        'wrapper': 'text_chat_enable_thinking_false_no_vision_no_tools',
        'runtime_revision': 'e3de715222b2667c54a267441159a5729f11155e'},
}
PROTOCOL = {'schema': 'trinite.tiny-output-protocol.v1',
    'scope': 'external_model_plus_harness_output_only_not_admitted_not_training',
    'calls_per_model': 32, 'families': ['extraction', 'modular_addition'],
    'sampling': 'greedy_experimental_not_vendor_recommended_defaults',
    'max_new_tokens': 256, 'timeout_seconds': 120,
    'retained_bytes_per_call': 65536, 'accepted_bytes_per_call': 4096,
    'answer_schema': {'type': 'object', 'properties': {'answer': {'type': 'integer'}},
                      'required': ['answer'], 'additionalProperties': False},
    'policy': 'one_call_per_slot_fresh_state_no_examples_retrieval_execution_retry_refill_or_additional_repair',
    'comparison': 'different_tokenizers_engines_precision_not_equal_compute_or_causal_size_effect',
    'gpt_oss': 'historical_proposals_context_only_not_matched_answer_baseline'}


def rows():
    result = []
    values = (0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 17, 31, 59, 60, 99, 359)
    for i, value in enumerate(values):
        fields = [('left', 701+i), ('target', value), ('right', 901+i)]
        if i % 2:
            fields.reverse()
        result.append({'family': 'extraction', 'inputs': dict(fields),
            'prompt': 'Return the integer in field target. Record: '+', '.join(f'{k}={v}' for k,v in fields),
            'answer': value})
    for n in (5, 7, 11, 13):
        for j, target in enumerate((0, 1, n//2, n-1)):
            a = (j*j+2*j+1) % n
            b = (target-a) % n
            value = a+b
            while value >= n:
                value -= n
            if value != (a+b) % n:
                raise ContractError('independent oracle disagreement')
            result.append({'family': 'modular_addition', 'inputs': {'n': n, 'a': a, 'b': b},
                'prompt': f'Return (a+b) modulo n for a={a}, b={b}, n={n}.', 'answer': value})
    for i, row in enumerate(result):
        row.update({'slot': i, 'source': 'trinite.public-formal.tiny-output.v1',
            'generator_revision': 'trinite.tiny-output-generator.v1',
            'split': 'public_exploratory_no_training', 'seed': None,
            'procedure': 'fixed_enumeration_in_source', 'verifier': 'exact_integer_oracle_pass'})
    return result


def source_map():
    names = ('scripts/tiny_output_v1.py', 'scripts/run_tiny_output.py',
             'src/trinite/contracts.py', 'tests/test_tiny_output.py',
             'docs/TINY-OUTPUT.md', '.github/workflows/tiny-output.yml', 'AGENTS.md')
    return {name: identity((ROOT/name).read_bytes()) for name in names}


def text(value, label, limit=4096):
    if type(value) is not str or not value or len(value.encode('utf-8')) > limit:
        raise ContractError(label+' requires bounded nonempty text')
    return value


def validate_manifest(manifest):
    exact_keys(manifest, set(MODELS), 'both model manifests')
    for model, item in manifest.items():
        exact_keys(item, {'profile', 'runtime_revision', 'settings', 'artifacts', 'hardware',
                          'compatibility_receipt', 'launch_config_identity'}, 'model manifest')
        if item['profile'] != PROFILES[model]:
            raise ContractError('profile must match pinned specification')
        if not re.fullmatch('[0-9a-f]{40}', text(item['runtime_revision'], 'runtime revision')):
            raise ContractError('immutable runtime source revision required')
        if model == 'bonsai2-2b' and item['runtime_revision'] != PROFILES[model]['runtime_revision']:
            raise ContractError('Prism runtime revision differs')
        text(item['hardware'], 'hardware')
        require_identity(item['compatibility_receipt'])
        require_identity(item['launch_config_identity'])
        # This is a source-bound preflight declaration, not a compatibility attestation.
        settings = item['settings']
        exact_keys(settings, {'max_new_tokens','temperature','fresh_state','network','telemetry',
                              'thinking','auto_date','tools_executed','depth'}, 'settings')
        required = {'max_new_tokens':256, 'temperature':'0', 'fresh_state':True,
                    'network':False, 'telemetry':False, 'thinking':False,
                    'auto_date':False, 'tools_executed':False}
        if any(type(settings[k]) is not type(v) or settings[k] != v for k,v in required.items()):
            raise ContractError('unsupported or changed frozen settings')
        if model == 'needle3':
            integer(settings['depth'], 'Needle depth', 2, 20)
        elif settings['depth'] is not None:
            raise ContractError('Bonsai depth must be null')
        artifacts = item['artifacts']
        if type(artifacts) is not list or not 6 <= len(artifacts) <= 128:
            raise ContractError('6..128 explicit local artifacts required')
        seen = set(); roles = set(); total = 0
        for entry in artifacts:
            exact_keys(entry, {'path','role','bytes','identity'}, 'artifact')
            name = text(entry['path'], 'path')
            path = PurePosixPath(name)
            if path.is_absolute() or str(path) != name or '..' in path.parts or '\\' in name or name in seen:
                raise ContractError('unique canonical contained path required')
            if entry['role'] not in {'weights','runtime','tokenizer','chat_template','compatibility_receipt','launch_config'}:
                raise ContractError('unknown role')
            integer(entry['bytes'], 'artifact bytes', 1, 4*1024**3)
            require_identity(entry['identity'])
            if entry['role'] == 'weights' and entry['identity'] != 'sha256:'+PROFILES[model]['weights_sha256']:
                raise ContractError('unapproved weights')
            if entry['role'] == 'compatibility_receipt' and entry['identity'] != item['compatibility_receipt']:
                raise ContractError('compatibility receipt mismatch')
            if entry['role'] == 'launch_config' and entry['identity'] != item['launch_config_identity']:
                raise ContractError('launch configuration mismatch')
            seen.add(name); roles.add(entry['role']); total += entry['bytes']
        if len(roles) != 6 or total > 8*1024**3:
            raise ContractError('all six artifact roles and total bound required')
    return manifest


def check_assets(manifest, root):
    validate_manifest(manifest)
    root = Path(root).resolve(strict=True)
    for item in manifest.values():
        for entry in item['artifacts']:
            path = root
            for part in PurePosixPath(entry['path']).parts:
                path = path/part
                if path.is_symlink():
                    raise ContractError('symlinks forbidden')
            if not path.is_file():
                raise ContractError('regular local artifact required')
            digest = hashlib.sha256(); count = 0
            with path.open('rb') as stream:
                while chunk := stream.read(1024*1024):
                    count += len(chunk)
                    if count > entry['bytes']:
                        raise ContractError('artifact size exceeded')
                    digest.update(chunk)
            if count != entry['bytes'] or 'sha256:'+digest.hexdigest() != entry['identity']:
                raise ContractError('local artifact mismatch')


def request(manifest):
    validate_manifest(manifest)
    return {'schema': 'trinite.tiny-output-request.v1', 'protocol': PROTOCOL,
            'models': manifest, 'rows': rows(), 'sources': source_map()}


def validate_request(value):
    exact_keys(value, {'schema','protocol','models','rows','sources'}, 'request')
    if json_bytes(value) != json_bytes(request(value['models'])):
        raise ContractError('request differs from current frozen producer')
    return value


def measurement(value, label):
    exact_keys(value, {'value','scope','unavailable_reason'}, label)
    text(value['scope'], label+' scope')
    if value['value'] is None:
        text(value['unavailable_reason'], label+' unavailable reason')
    else:
        integer(value['value'], label, 0, 2**53-1)
        if value['unavailable_reason'] is not None:
            raise ContractError('measured value cannot have unavailable reason')


def score_payload(model, raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError('duplicate response key')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ContractError('nonfinite response value')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractError('invalid response JSON') from error
    try:
        json.dumps(value, allow_nan=False)
    except ValueError as error:
        raise ContractError('nonfinite response metadata') from error
    if model == 'needle3':
        if type(value) is not dict or value.get('type') != 'call':
            raise ContractError('Needle did not return a call')
        allowed = {'type','success','error','error_code','function_calls','reasoning',
                   'confidence','prefill_tps','decode_tps','suppressed_calls','validation'}
        if set(value) - allowed:
            raise ContractError('unknown Needle envelope fields require prospective contract review')
        for key in ('confidence','prefill_tps','decode_tps'):
            if key in value and (type(value[key]) not in (int,float) or value[key] < 0):
                raise ContractError('invalid Needle numeric metadata')
        if 'confidence' in value and not 0.1 <= value['confidence'] <= 1:
            raise ContractError('invalid Needle confidence')
        if 'reasoning' in value and type(value['reasoning']) is not str:
            raise ContractError('invalid Needle reasoning metadata')
        if 'success' in value and value['success'] is not True:
            raise ContractError('Needle success not true')
        if any(value.get(k) is not None for k in ('error', 'error_code')):
            raise ContractError('Needle reported an error')
        if value.get('suppressed_calls', []) != []:
            raise ContractError('Needle suppressed calls are not accepted')
        validation = value.get('validation', {})
        if type(validation) is not dict or set(validation) - {'ungrounded', 'negation'}:
            raise ContractError('unknown Needle validation fields')
        if validation.get('ungrounded', []) != [] or validation.get('negation', False) is not False:
            raise ContractError('Needle grounding or negation rejection')
        calls = value.get('function_calls')
        if type(calls) is not list or len(calls) != 1:
            raise ContractError('exactly one active Needle call required')
        exact_keys(calls[0], {'name','arguments'}, 'Needle call')
        if calls[0]['name'] != 'answer':
            raise ContractError('unexpected Needle function')
        value = calls[0]['arguments']
    exact_keys(value, {'answer'}, 'answer')
    integer(value['answer'], 'answer', -2**31, 2**31-1)
    return value['answer']


def report(frozen, capture):
    validate_request(frozen)
    exact_keys(capture, {'schema','request_identity','attempts'}, 'capture')
    if capture['schema'] != 'trinite.tiny-output-capture.v1' or capture['request_identity'] != identity(json_bytes(frozen)):
        raise ContractError('capture request binding mismatch')
    attempts = capture['attempts']
    if type(attempts) is not list or len(attempts) != 64:
        raise ContractError('all 64 ordered slots required; preserve incomplete capture without scoring')
    scores = []; summaries = []
    for index, attempt in enumerate(attempts):
        exact_keys(attempt, {'model','slot','status','raw_base64','channel','elapsed_seconds_hex',
                            'generated_tokens','memory_bytes','diagnostic'}, 'attempt')
        model = MODELS[index//32]; row = frozen['rows'][index%32]
        if attempt['model'] != model or type(attempt['slot']) is not int or attempt['slot'] != index%32:
            raise ContractError('reordered, missing or duplicated attempt')
        if attempt['status'] not in ('completed','timeout','runtime_error','output_limit'):
            raise ContractError('unknown status')
        if attempt['channel'] != PROFILES[model]['channel']:
            raise ContractError('capture channel differs from frozen profile')
        if type(attempt['diagnostic']) is not str or len(attempt['diagnostic'].encode()) > 4096:
            raise ContractError('diagnostic must be bounded text')
        try:
            elapsed = float.fromhex(attempt['elapsed_seconds_hex'])
            raw = base64.b64decode(attempt['raw_base64'], validate=True)
        except (ValueError, TypeError, OverflowError) as error:
            raise ContractError('invalid time or base64') from error
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed > 125:
            raise ContractError('invalid per-call wall time; timeout plus five seconds cleanup bound')
        if base64.b64encode(raw).decode() != attempt['raw_base64'] or len(raw) > 65536:
            raise ContractError('noncanonical or oversized capture')
        measurement(attempt['generated_tokens'], 'generated tokens')
        measurement(attempt['memory_bytes'], 'memory bytes')
        tokens = attempt['generated_tokens']['value']
        if tokens is not None and tokens > 256:
            raise ContractError('output token bound exceeded')
        valid = False; answer = None; reason = attempt['status']
        if attempt['status'] == 'completed':
            if elapsed > 120:
                raise ContractError('completed call exceeds timeout')
            try:
                if len(raw) > 4096:
                    raise ContractError('accepted response byte bound exceeded')
                answer = score_payload(model, raw)
                valid = True; reason = 'valid'
            except ContractError as error:
                reason = str(error)[:256]
        scores.append({'model':model, 'slot':row['slot'], 'family':row['family'],
                       'format_valid':valid, 'correct':valid and answer == row['answer'],
                       'answer':answer, 'reason':reason, 'raw_identity':identity(raw)})
    for model in MODELS:
        for family in PROTOCOL['families']:
            group = [s for s in scores if s['model'] == model and s['family'] == family]
            gold = [r['answer'] for r in frozen['rows'] if r['family'] == family]
            summaries.append({'model':model, 'family':family, 'denominator':16,
                'format_valid':sum(s['format_valid'] for s in group),
                'correct':sum(s['correct'] for s in group),
                'constant_zero_correct':gold.count(0),
                'best_constant_oracle_correct':max(gold.count(v) for v in set(gold))})
    return {'schema':'trinite.tiny-output-report.v1', 'request_identity':identity(json_bytes(frozen)),
            'capture_identity':identity(json_bytes(capture)), 'scores':scores, 'summaries':summaries,
            'claim':'self_reported_capture_conformance_not_execution_attestation_or_student_benefit'}


def seal(frozen, capture):
    return {'schema':'trinite.tiny-output-packet.v1', 'request':frozen,
            'capture':capture, 'report':report(frozen, capture)}


def verify(packet):
    exact_keys(packet, {'schema','request','capture','report'}, 'packet')
    if json_bytes(packet) != json_bytes(seal(packet['request'], packet['capture'])):
        raise ContractError('packet differs from recomputed report')
    return packet
