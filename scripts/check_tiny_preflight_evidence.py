"""Read-only retained preflight audit; never execute captured adapters or models."""
import sys
if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
        and sys.flags.no_user_site and sys.flags.safe_path):
    raise SystemExit('requires python -I -S -B scripts/check_tiny_preflight_evidence.py')
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import types
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'fixtures/tiny-preflight-local-20261021'
COMMITMENT = 'sha256:9b15d4c6ad8eb15c4079809092d2741ba045e95a5e72da9f9d239c8e13cd1fc6'
LIMIT = 4 * 1024 * 1024
TOTAL = 24 * 1024 * 1024
PREFIX = 'logs/bonsai2-2b.smoke-result.json.attempt-d4e4b8004587469a96db64068ad0b4a5/'
SOURCE = {
    'scripts/tiny_output_v1.py': '2830070999d296be909e8729ab0a62008501df93093626fc40bba5db7fd6ca55',
    'src/trinite/contracts.py': '36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def same(a, b, message):
    require(canonical(a) == canonical(b), message)


def safe_path(name):
    require(type(name) is str and bool(name), 'member name required')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name
            and '\\' not in name, 'unsafe path')
    return Path(*p.parts)


def read(path, limit=LIMIT):
    require(path.is_file() and not path.is_symlink(), 'regular file required')
    with path.open('rb') as f:
        raw = f.read(limit + 1)
    require(len(raw) <= limit, 'size bound exceeded')
    return raw


def inventory(folder):
    raw = read(folder / 'commitment.json')
    require(digest(raw) == COMMITMENT, 'commitment mismatch')
    p = json.loads(raw)
    require(canonical(p) == raw, 'noncanonical commitment')
    require(p['schema'] == 'trinite.tiny-preflight-local-evidence.v1', 'schema')
    same(p['original_zip'], {'identity': 'sha256:2b06fcd8685fd022da26d19837517c6f5affc10c63e4ce4bcc203913b140f31d',
                            'bytes': 4685694, 'members': 1488}, 'original archive pin')
    require(p['producer_commit'] == 'bb6758b69143f829dfdb4b44ece17170f81e9093'
            and p['producer_tree'] == 'ead80f22a3255d5669daeb7f407c57dc90aad4ac', 'producer pin')
    require(p['added_members'] == p['public_transformations'] == [], 'unexpected transformations')
    require({x.name for x in folder.iterdir()} == {'commitment.json', 'objects.zip', '.gitattributes'}, 'fixture inventory')
    require(read(folder / '.gitattributes') == b'objects.zip binary\ncommitment.json text eol=lf\n', 'binary rule')
    data = read(folder / 'objects.zip')
    require(digest(data) == p['objects_archive']['identity'] and len(data) == p['objects_archive']['bytes'], 'archive identity')
    expected = {r['stored_path'] for r in p['members'].values()}
    objects = {}; total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        require(len(z.namelist()) == len(set(z.namelist())) == p['objects_archive']['unique_objects']
                and set(z.namelist()) == expected, 'object inventory')
        for info in z.infolist():
            safe_path(info.filename)
            require(not info.is_dir() and info.file_size <= LIMIT and not info.flag_bits & 1
                    and info.external_attr >> 16 == 0o100644, 'object type/size')
            total += info.file_size
            require(total <= TOTAL, 'expanded size bound')
            with z.open(info) as stream:
                b = stream.read(LIMIT + 1)
            require(len(b) == info.file_size, 'expanded size mismatch')
            objects[info.filename] = b
    members = {}; total = 0
    for name, rec in p['members'].items():
        safe_path(name)
        require(rec['stored_path'] == 'objects/' + rec['identity'][7:] + '.bin', 'object path')
        b = objects[rec['stored_path']]
        require(digest(b) == rec['identity'] and len(b) == rec['bytes'], 'member identity')
        total += len(b); require(total <= TOTAL, 'logical size bound')
        members[name] = b
    require(len(members) == 1484 and len(p['excluded']) == 4, 'retention counts')
    for name, rec in p['excluded'].items():
        safe_path(name)
        require(name not in members and re.fullmatch('sha256:[0-9a-f]{64}', rec['identity'])
                and type(rec['bytes']) is int and rec['bytes'] > 0 and rec['reason'], 'exclusion')
    original = {}
    for line in members['SHA256SUMS'].decode('ascii').splitlines():
        h, name = line.split('  ', 1); safe_path(name)
        require(name not in original and re.fullmatch('[0-9a-f]{64}', h), 'original manifest record')
        original[name] = 'sha256:' + h
    require(len(original) == 1487 and set(original) | {'SHA256SUMS'} == set(members) | set(p['excluded']), 'original manifest coverage')
    for name, identity in original.items():
        require(p['members'].get(name, p['excluded'].get(name))['identity'] == identity, 'original digest mismatch')
    return p, members


def source_module(members, root):
    # Only the two independently pinned, GitHub-verified repository sources may
    # execute. Captured adapter/upstream source remains inert byte evidence.
    for name, expected in SOURCE.items():
        raw = read(ROOT / name)
        require(digest(raw) == 'sha256:' + expected and raw == members['checkout/' + name], 'authoritative source mismatch')
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    module = types.ModuleType('pinned_tiny_output_evidence')
    module.__file__ = str(root / 'scripts/tiny_output_v1.py')
    exec(compile(read(root / 'scripts/tiny_output_v1.py'), module.__file__, 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module


def sse(raw):
    chunks = []; terminal = False; done = False; tokens = None
    for block in raw.decode('utf-8').rstrip('\n').split('\n\n'):
        if not block: continue
        require(block.startswith('data: ') and '\n' not in block, 'SSE framing')
        require(not done, 'SSE after DONE')
        value = block[6:]
        if value == '[DONE]':
            require(terminal, 'SSE missing terminal stop'); done = True; continue
        event = json.loads(value)
        for choice in event['choices']:
            require(choice['index'] == 0, 'SSE choice')
            content = choice['delta'].get('content', '')
            if content is None: content = ''
            require(type(content) is str and not (terminal and content), 'SSE content after stop')
            chunks.append(content.encode('utf-8'))
            reason = choice['finish_reason']
            if reason is not None:
                require(reason == 'stop' and not terminal, 'SSE terminal'); terminal = True
        if 'usage' in event:
            require(tokens is None and terminal, 'SSE usage order')
            tokens = event['usage']['completion_tokens']
    require(done and terminal and type(tokens) is int and tokens == 30, 'SSE completion')
    return b''.join(chunks), tokens


def evidence_check(m, module):
    result = json.loads(m['logs/bonsai2-2b.smoke-result.json'])
    require(m['logs/bonsai2-2b.smoke-result.json'] == m[PREFIX + 'prepared-result.json'], 'prepared publication bytes')
    raw = m[PREFIX + 'raw.bin']
    require(raw == m[PREFIX + 'final_text.bin'] == base64.b64decode(result['record']['raw_base64'], validate=True), 'raw copies')
    require(raw == b'{\n  "target": 1234\n}\n</think>\n\n{\n  "target": 1234\n}' and len(raw) == 51, 'complete unmodified response')
    reconstructed, tokens = sse(m[PREFIX + 'transport.sse'])
    require(reconstructed == raw and result['record']['generated_tokens']['value'] == tokens, 'SSE capture equality')
    same([result['call_started'], result['cleanup_exceeded'], result['done_received'], result['exit_code'], result['record']['status']],
         [True, False, True, 0, 'completed'], 'call lifecycle')
    require(result['record']['memory_bytes']['value'] == 771403776
            and 'not device memory or exact peak' in result['record']['memory_bytes']['scope'], 'memory scope')
    require(result['record']['elapsed_seconds_hex'] == '0x1.8b94ae19c0000p-1'
            and result['load_elapsed_seconds'] == 1.0441245340043679, 'time records')
    require(module.score_payload('bonsai2-2b', b'{"answer":1234}') == 1234,
            'authoritative scorer positive control')
    try:
        module.score_payload('bonsai2-2b', raw)
    except module.ContractError as error:
        require(str(error) == 'invalid response JSON', 'scorer rejection reason')
    else:
        raise ValueError('authoritative scorer accepted smoke')
    request = json.loads(m[PREFIX + 'request-body.json'])
    same(request['chat_template_kwargs'], {'enable_thinking': False}, 'thinking setting')
    require(request['reasoning_format'] == 'none' and request['temperature'] == 0
            and request['max_tokens'] == 256 and request['cache_prompt'] is False, 'request settings')
    same(request['messages'], [{'role': 'user', 'content': 'Return the integer in field target. Record: left=5678, target=1234, right=9012.\nReturn only a JSON object with the integer field answer.'}], 'non-target prompt')
    props = json.loads(m[PREFIX + 'runtime-props.json'])
    require(props['chat_template'].encode() == m['assets/bonsai/chat-template.jinja'], 'runtime template')
    require(m[PREFIX + 'resolved-launch-config.json'] == m['assets/bonsai/launch-config.resolved.json'], 'launch copies')
    report = json.loads(m['reports/preflight-result.json'])
    same(report['attempt_counts'], {'bonsai_model_loads': 1, 'bonsai_scored': 0, 'bonsai_smoke': 1,
         'needle_model_loads': 0, 'needle_scored': 0, 'needle_smoke': 0, 'retries': 0}, 'attempt counts')
    require(report['manifest_execution_ready'] is False and report['stages']['student_training'] == 'not-run'
            and report['stages']['scored_inference'] == 'not-run' and report['evidence_gap'], 'blocked scope')
    same(report['bonsai_smoke_assessment'], {'accepted': False, 'answer': None, 'correct': False, 'reason': 'invalid response JSON'}, 'negative assessment')
    manifest_raw = m['reports/manifest.proposed.json']; manifest = json.loads(manifest_raw)
    module.validate_manifest(manifest)
    validation = json.loads(m['reports/manifest-validation.json'])
    require(validation['status'] == 'passed' and validation['execution_ready'] is False
            and validation['manifest_identity'] == digest(manifest_raw), 'structure versus readiness')
    needle = json.loads(m['assets/needle/compatibility-receipt.json'])
    same([needle['status'], needle['model_loads'], needle['smoke_calls'], needle['target_calls'], needle['runtime_executed']],
         ['blocked', 0, 0, 0, False], 'Needle blocker')
    require(len(needle['blockers']) == 4, 'Needle blocker inventory')
    require(json.loads(m['logs/bootstrap.json'])['pre_session_command']['exit_code'] == 2, 'bootstrap gap')
    source = json.loads(m['reports/source-final.json'])
    for name, h in source['source_hashes'].items():
        require(digest(m['checkout/' + name]) == 'sha256:' + h, 'captured source identity')
    # Cross-bind all retained path/identity/size references, without pretending
    # excluded model/runtime files were supplied or checked on this machine.
    def references(value):
        if isinstance(value, dict):
            if {'path', 'identity', 'bytes'} <= value.keys() and value['path'] in m:
                b = m[value['path']]
                require(digest(b) == value['identity'] and len(b) == value['bytes'], 'receipt identity')
            for v in value.values(): references(v)
        elif isinstance(value, list):
            for v in value: references(v)
    for name in ('reports/preflight-result.json', 'assets/needle/compatibility-receipt.json', 'assets/bonsai/compatibility-receipt.json', 'reports/smoke-assessment.json'):
        references(json.loads(m[name]))


def audit():
    _, m = inventory(FOLDER)
    with tempfile.TemporaryDirectory(prefix='tiny-evidence-') as temp:
        module = source_module(m, Path(temp))
        evidence_check(m, module)
    require(not {'torch', 'needle', 'transformers'} & set(sys.modules), 'runtime imported')
    return m


class FunctionalTests(unittest.TestCase):
    def test_archive_and_actual_scorer(self):
        audit()

    def test_paths(self):
        for name in ('../x', '/x', 'a/../b', 'a\\b', ''):
            with self.assertRaises(ValueError): safe_path(name)

    def test_commitment_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); (folder / 'commitment.json').write_bytes(b'{}\n')
            with self.assertRaises(ValueError): inventory(folder)

    def test_sse_truncated_or_bad_terminal(self):
        _, m = inventory(FOLDER); raw = m[PREFIX + 'transport.sse']
        for bad in (raw.replace(b'data: [DONE]\n', b''), raw.replace(b'"finish_reason":"stop"', b'"finish_reason":null'),
                    raw.replace(b'"finish_reason":"stop"', b'"finish_reason":"length"'), raw + b'\ndata: [DONE]\n\n'):
            with self.assertRaises(ValueError): sse(bad)

    def test_semantic_mutations(self):
        _, original = inventory(FOLDER)
        mutations = [('logs/bonsai2-2b.smoke-result.json', lambda v: v['record'].update(raw_base64='eA==')),
                     ('reports/preflight-result.json', lambda v: v['attempt_counts'].update(needle_scored=1)),
                     ('reports/manifest-validation.json', lambda v: v.update(execution_ready=True)),
                     (PREFIX + 'request-body.json', lambda v: v['chat_template_kwargs'].update(enable_thinking=True))]
        with tempfile.TemporaryDirectory() as temp:
            module = source_module(original, Path(temp))
            for name, mutate in mutations:
                m = dict(original); v = json.loads(m[name]); mutate(v); m[name] = canonical(v)
                with self.assertRaises(ValueError): evidence_check(m, module)
            m = dict(original); m['checkout/scripts/tiny_output_v1.py'] += b'\n'
            with self.assertRaises(ValueError): source_module(m, Path(temp))


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
    else:
        require(not sys.argv[1:], 'unknown arguments')
        audit()
        print('Verified retained preflight: Needle blocked, one Bonsai CPU format failure, zero scored calls; no inference or training executed.')
