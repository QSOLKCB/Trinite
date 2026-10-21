"""Source-bound intake under the explicit bounded teacher-origin policy."""
from pathlib import Path, PurePosixPath
import hashlib
import re
from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes

ROOT = Path(__file__).resolve().parents[2]
COUNT = 32
MODULI = (5, 7, 9)
MAX_FILE = 32 * 1024**3
MAX_TOTAL = 48 * 1024**3
ROLES = {'weights', 'tokenizer', 'runtime', 'chat_template'}
PROTOCOL = {
    'schema': 'trinite.teacher-intake-protocol.v1',
    'policy': 'trinite.teacher-origin-formal-policy.v1',
    'admission': 'SOURCE_SPECIFIC_FORMAL_ONLY', 'student_execution': 'SEPARATE_FROZEN_CPU_REQUEST',
    'attempts': COUNT, 'moduli': list(MODULI), 'minimum_unique': 16,
    'sampling': {'temperature': '0', 'top_p': '1', 'seed': 0,
                 'max_new_tokens': 512, 'reasoning_effort': 'low'},
    'response': 'final-channel UTF-8 JSON object with exactly n,a,b; no answer or prose',
    'capture': 'one fresh independent conversation per indexed slot; all attempts retained; no retries or refill',
    'accepted_response_bytes': 4096, 'retained_response_bytes': 65536,
    'duplicate_family': '(n,min(a,b),max(a,b)); first valid occurrence retained',
    'selection': 'all unique valid proposals; minimum16 is engineering feasibility, not quality',
    'limits': '32 calls, at most512 new tokens each; per-call timeout120s; no adaptive generation',
    'claims': 'formal-only scoped admission; no model-execution attestation or learning result',
}


def source_map():
    paths = list((ROOT/'scripts/teacher_intake_v1').glob('*.py'))
    paths += list((ROOT/'src/trinite').rglob('*.py'))
    paths += list((ROOT/'tests/teacher_intake_v1').glob('*.py'))
    paths += [ROOT/p for p in ('scripts/run_teacher_intake.py', 'scripts/run_teacher_student.py', 'requirements.lock', 'src/trinite/cpu-dependencies.lock',
              'src/trinite/contracts.py', 'docs/DATA.md', 'AGENTS.md', 'docs/TEACHER-INTAKE.md',
              '.github/workflows/teacher-intake.yml')]
    return {p.relative_to(ROOT).as_posix(): identity(p.read_bytes()) for p in sorted(paths)}


def text(value, label, limit=256):
    if type(value) is not str or not value or len(value.encode('utf-8')) > limit:
        raise ContractError(label+' must be bounded nonempty text')
    return value


def validate_manifest(manifest):
    exact_keys(manifest, {'model', 'revision', 'runtime', 'quantization', 'artifacts'}, 'runtime manifest')
    if manifest['model'] != 'openai/gpt-oss-20b':
        raise ContractError('only declared openai/gpt-oss-20b is in scope')
    if type(manifest['revision']) is not str or not re.fullmatch('[0-9a-f]{40}', manifest['revision']):
        raise ContractError('immutable model repository revision required')
    text(manifest['runtime'], 'runtime'); text(manifest['quantization'], 'quantization')
    files = manifest['artifacts']
    if type(files) is not list or not 4 <= len(files) <= 128:
        raise ContractError('4..128 local artifacts required')
    seen=set(); roles=set(); total=0
    for entry in files:
        exact_keys(entry, {'path','role','bytes','identity'}, 'artifact')
        name=text(entry['path'], 'artifact path')
        path=PurePosixPath(name)
        if (path.is_absolute() or str(path)!=name or any(p in ('..','.') for p in path.parts)
                or '\\' in name or name in seen):
            raise ContractError('unique canonical relative artifact path required')
        if entry['role'] not in ROLES:
            raise ContractError('unknown artifact role')
        integer(entry['bytes'], 'artifact size', 1, MAX_FILE)
        if type(entry['identity']) is not str or not re.fullmatch('sha256:[0-9a-f]{64}',entry['identity']):
            raise ContractError('artifact SHA256 required')
        seen.add(name); roles.add(entry['role']); total+=entry['bytes']
    if roles!=ROLES or total>MAX_TOTAL:
        raise ContractError('all roles and bounded total artifact size required')
    return manifest


def check_assets(manifest, root):
    """Read only declared regular files, never load or execute their contents."""
    validate_manifest(manifest)
    root=Path(root).resolve(strict=True)
    for entry in manifest['artifacts']:
        path=root
        for component in PurePosixPath(entry['path']).parts:
            path=path/component
            if path.is_symlink():
                raise ContractError('artifact symlinks are forbidden')
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ContractError('artifact must be a contained regular file')
        digest=hashlib.sha256(); size=0
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):
                size+=len(chunk)
                if size>entry['bytes']:
                    raise ContractError('artifact exceeds frozen size')
                digest.update(chunk)
        if size!=entry['bytes'] or 'sha256:'+digest.hexdigest()!=entry['identity']:
            raise ContractError('artifact bytes differ from frozen manifest')


def request(manifest):
    validate_manifest(manifest)
    return {'schema':'trinite.teacher-intake-request.v1', 'protocol':PROTOCOL,
            'source':source_map(), 'runtime_manifest':manifest,
            'runtime_evidence':'declared identity plus local-file hash check; not official-origin or execution attestation',
            'slots':[{'index':i, 'n':MODULI[i%3], 'a':(i//3)%MODULI[i%3],
                      'prompt':f'Slot {i}: Propose one modular addition input. Return only JSON with keys n,a,b and integer values. '
                               f'n must equal {MODULI[i%3]}; a must equal {(i//3)%MODULI[i%3]}; choose b with 0 <= b < n. Do not include an answer or explanation.'}
                     for i in range(COUNT)]}


def validate_request(value):
    if type(value) is not dict or 'runtime_manifest' not in value or json_bytes(value)!=json_bytes(request(value['runtime_manifest'])):
        raise ContractError('request source/protocol/runtime binding mismatch')
    return value
