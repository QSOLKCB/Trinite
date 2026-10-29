"""Immutable reports-only result authentication and scoring, never model execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/frozen-music-eval-result-v1'
HISTORICAL = ROOT / 'fixtures/music-exposure-result-20261029-v1'
LIMIT = 1024 * 1024
ORIGINAL_HASHES = {'original-attempt/attempt.json': 'sha256:1ae593877597395cf18315256b547ece687b44a7c7da2f344d4de55214e314ed', 'original-attempt/predictions.jsonl': 'sha256:b0939e4508697d4909e08acefea50ac568ae008092f1952068145c814355a56f', 'original-attempt/replay-gate.json': 'sha256:82f06bab3e8edb6bb3893cc58deb5eeac51096f9cbae6a9022622a84b0ba8198', 'original-attempt/report.json': 'sha256:634847e20f796ce045cc3f81287a9506223125cfa9a4a445d8f044a0019848a3', 'original-attempt/worker-completion.json': 'sha256:1c785da4b5013cbbbd00cfac7a5da5a9d92b21f4226200f094d7d32328100666', 'audit-receipt.json': 'sha256:9131095f48a76d0240f9442c6cdd83df073250e6b07f17a3ddca89b119976aa2', 'original-SHA256SUMS.txt': 'sha256:4b7c91f242e4a7f966703816abee8aa4896e65ff71e95e7d6976815f191e9e0d'}
FIXTURE_HASHES = {**ORIGINAL_HASHES, **{'README.md': 'sha256:42aefa202458d528dcb12dd045037a1a129e07feb3a5140d5025367239ba7f93', 'provenance-DERIVED.json': 'sha256:aa805d745a204df5941f5f0b563b2e4bbee66268c04bb24c4ff3977958947063'}}
MODULES = {'trinite': 'src/trinite/__init__.py', 'trinite.contracts': 'src/trinite/contracts.py', 'trinite.tokenizer': 'src/trinite/tokenizer.py', 'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py', 'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py', 'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py', 'music_learning_v1': 'scripts/music_learning_v1/__init__.py', 'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py', 'music_contrast_v1': 'scripts/music_contrast_v1/__init__.py', 'music_contrast_v1.spec': 'scripts/music_contrast_v1/spec.py', 'music_contrast_v1.reporting': 'scripts/music_contrast_v1/reporting.py', 'frozen_music_eval_v1': 'scripts/frozen_music_eval_v1/__init__.py', 'frozen_music_eval_v1.protocol': 'scripts/frozen_music_eval_v1/protocol.py', 'frozen_music_eval_v1.artifacts': 'scripts/frozen_music_eval_v1/artifacts.py', 'frozen_music_eval_v1.receipt': 'scripts/frozen_music_eval_v1/receipt.py', 'frozen_music_eval_v1.workflow': 'scripts/frozen_music_eval_v1/workflow.py'}
SOURCES = {'src/trinite/__init__.py': 'sha256:029d147c8a741f0cecffc90611d77c6a4047a019863d86ca40992c630a3524b6', 'src/trinite/contracts.py': 'sha256:36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24', 'src/trinite/tokenizer.py': 'sha256:fa87f79f667da8cabeeae0e999add9f420102efbfb746a0c0b275b8703d845bb', 'scripts/symbolic_music_v1/__init__.py': 'sha256:aea01a9d1bd6ae591665b9fcc3658189091a162565851d4c92579731f82405db', 'scripts/symbolic_music_v1/core.py': 'sha256:9c7446c8d275dbc02fe195eac7a660130cf70a6943fa31428c0d4c7f49bba6ff', 'scripts/symbolic_music_v1/oracle.py': 'sha256:3b6c3e00a2caca2a4d2e12d6ab93fda912197d638cec220175a8073b4ee09af0', 'scripts/music_learning_v1/__init__.py': 'sha256:352fce75f453ef87602ff6b451800c6eb765fa2edd2c79d5428013b6f05b5c9a', 'scripts/music_learning_v1/codec.py': 'sha256:5236fab68f2ff11ff4248d1187ddd0db5797138784665eeca4323ff9c192fa0b', 'scripts/music_contrast_v1/__init__.py': 'sha256:3d67f788b234aba0de63cfe5368e9a9dac7d8d8220ecfc5120b5ce708694428d', 'scripts/music_contrast_v1/spec.py': 'sha256:34f051fe47feb48baff3ca636730d9348682812e2ffb4bd3e2a4c07b78cb3317', 'scripts/music_contrast_v1/reporting.py': 'sha256:5f15f9231e3b57e32410a8cd2eb7d66f49565fa13661a680c8327d03fba2b846', 'scripts/frozen_music_eval_v1/__init__.py': 'sha256:24e16706fad053362bd8270c99db1d6220c11560cb61813290956cd23efb9f20', 'scripts/frozen_music_eval_v1/protocol.py': 'sha256:919f5b6a083b437a340b3263d83c98692f070e633a29408bc1ab88e695e321cd', 'scripts/frozen_music_eval_v1/artifacts.py': 'sha256:9d8da0e97e48d94a5372daf250c3ba2ecd46dc81ad783e3c3d847c39446ed462', 'scripts/frozen_music_eval_v1/receipt.py': 'sha256:97f37922eedebd6f1323f03c0123187734b2379b447dac783e155574847cb29b', 'scripts/frozen_music_eval_v1/workflow.py': 'sha256:c7b05d44b0c6815dc586059bd8004da6273887527d4d592849f068b291233f5d'}
HISTORICAL_HASHES = {'train-17--step-640-predictions.json': 'sha256:503c0f780351647c1a127c7cd4516d6f7f7309853cfa69b9c2a8d7c57f9a35c1', 'train-29--step-640-predictions.json': 'sha256:503c0f780351647c1a127c7cd4516d6f7f7309853cfa69b9c2a8d7c57f9a35c1'}


def digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def read(path):
    """Bounded no-follow read with ancestor and observed mutation rejection."""
    path = Path(os.path.abspath(path))
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in path.parents):
        raise ValueError('unsafe ancestor')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT:
        raise ValueError('bounded regular file required')
    signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if signature(os.fstat(fd)) != signature(before):
            raise ValueError('file replaced at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(LIMIT + 1)
        if (len(raw) > LIMIT or signature(os.fstat(fd)) != signature(before)
                or signature(path.lstat()) != signature(before)):
            raise ValueError('file changed during read')
        return raw
    finally:
        os.close(fd)


def sources():
    """Authenticate the entire stdlib-only scorer closure before project imports.

    The interpreter, stdlib and this verifier are trusted. Reject ordinary
    filesystem/package-resolution shadows, not arbitrary runtime monkeypatches.
    """
    for name, relative in MODULES.items():
        path = ROOT / relative
        if digest(read(path)) != SOURCES[relative]:
            raise ValueError('frozen scoring source changed')
        package = path.parent
        if (package/'__pycache__').exists():
            raise ValueError('scorer import cache forbidden')
        stem = path.stem
        if path.name != '__init__.py' and (package/stem).exists():
            raise ValueError('scorer package shadow forbidden')
        for member in package.iterdir():
            if member.stem == stem and member.suffix in ('.pyc','.pyo','.so','.pyd'):
                raise ValueError('scorer extension/cache shadow forbidden')
            if member.name.startswith(stem+'.') and member.suffix in ('.so','.pyd'):
                raise ValueError('scorer extension shadow forbidden')
    for name, relative in MODULES.items():
        path = ROOT / relative
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module,'__file__','')).absolute() != path:
            raise ValueError('foreign scoring module')
    # Parent packages are listed before children; all package initializers have
    # been authenticated above before find_spec may execute them.
    for name, relative in MODULES.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin != str(ROOT/relative):
            raise ValueError('foreign scoring import resolution')


def snapshot(fixture):
    """Exact bounded file/directory membership, no links or special files."""
    fixture = Path(os.path.abspath(fixture))
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in fixture.parents):
        raise ValueError('unsafe ancestor')
    if not stat.S_ISDIR(fixture.lstat().st_mode):
        raise ValueError('non-link fixture directory required')
    found_files, found_dirs = set(), set()
    for current, directories, files in os.walk(fixture, followlinks=False):
        for name in directories:
            p = Path(current) / name
            if not stat.S_ISDIR(p.lstat().st_mode):
                raise ValueError('non-link fixture directory required')
            relative = p.relative_to(fixture).as_posix()
            if relative != 'original-attempt':
                raise ValueError('fixture directory membership differs')
            found_dirs.add(relative)
        for name in files:
            found_files.add((Path(current) / name).relative_to(fixture).as_posix())
            if len(found_files) > len(FIXTURE_HASHES):
                raise ValueError('fixture membership differs')
    if found_files != set(FIXTURE_HASHES) or found_dirs != {'original-attempt'}:
        raise ValueError('fixture membership differs')
    return {name: read(fixture / name) for name in FIXTURE_HASHES}


def historical_predictions():
    result = {}
    for name, expected in HISTORICAL_HASHES.items():
        raw = read(HISTORICAL / name)
        if digest(raw) != expected:
            raise ValueError('historical training predictions changed')
        result[name] = raw
    return result


def score(before):
    """Recompute retained journal scores with the frozen original scorer."""
    sources()
    from frozen_music_eval_v1.artifacts import encoded, parse
    from frozen_music_eval_v1.protocol import score_records
    from frozen_music_eval_v1.workflow import validate_completion, replay_gate
    from frozen_music_eval_v1.receipt import validate_attempt
    sources()
    records = [parse(line) for line in before['original-attempt/predictions.jsonl'].splitlines(keepends=True)]
    summary = score_records(records)
    historical = historical_predictions()
    for seed in (17, 29):
        old = parse(historical[f'train-{seed}--step-640-predictions.json'])['predictions']
        retained = {r['example_identity']: r['generated_tokens'] for r in old}
        replay = {r['example_identity']: r['generated_tokens'] for r in records
                  if r['stage'] == 'train_replay' and r['seed'] == seed}
        if encoded(replay) != encoded(retained):
            raise ValueError('historical raw training prediction parity differs')
    completion = parse(before['original-attempt/worker-completion.json'])
    validate_completion(completion, records)
    if before['original-attempt/replay-gate.json'] != encoded(replay_gate()):
        raise ValueError('retained replay gate differs')
    attempt = parse(before['original-attempt/attempt.json'])
    report = parse(before['original-attempt/report.json'])
    validate_attempt(attempt, report['request_identity'])
    files = {name: digest(before['original-attempt/' + name]) for name in
             ('attempt.json', 'predictions.jsonl', 'replay-gate.json', 'worker-completion.json')}
    # Omitted request/log bytes are not recreated or independently verified.
    files.update({'request.json': report['request_identity'], 'worker.log': digest(b'')})
    expected = {'schema': 'trinite.frozen-music-eval-report.v1',
        'request_identity': attempt['request_identity'], 'outcome': 'completed',
        'failure': None, 'original_inventory_unchanged': True, 'files': files,
        'summary': summary, 'worker_completion': completion, 'fresh_confirmation': False,
        'causal_transfer_established': False, 'optimizer_replayed': False,
        'learning_adequate': False, 'release_ready': False}
    if encoded(report) != encoded(expected):
        raise ValueError('retained report/score/completion consistency differs')
    audit = {'outcome': 'completed', 'report_identity': digest(before['original-attempt/report.json']),
        'scope': 'read-only report/source/score consistency; no tensor or inference replay',
        'target_inference_performed': False, 'verified': True}
    if before['audit-receipt.json'] != encoded(audit):
        raise ValueError('supplied audit receipt differs')
    manifest = {}
    for line in before['original-SHA256SUMS.txt'].decode('ascii').splitlines():
        checksum, name = line.split('  ', 1)
        if name in manifest:
            raise ValueError('duplicate original manifest member')
        manifest[name] = 'sha256:' + checksum
    for name in ORIGINAL_HASHES:
        if name != 'original-SHA256SUMS.txt' and manifest.get(name) != digest(before[name]):
            raise ValueError('original handoff manifest link differs')
    if historical_predictions() != historical:
        raise ValueError('historical fixture changed during scoring')
    sources()
    return summary


def verify(fixture=FIXTURE):
    before = snapshot(fixture)
    if {name: digest(raw) for name, raw in before.items()} != FIXTURE_HASHES:
        raise ValueError('preserved fixture bytes changed')
    summary = score(before)
    from frozen_music_eval_v1.artifacts import parse, encoded
    provenance = parse(before['provenance-DERIVED.json'])
    links = {'supplied_original_files': {n: digest(before[n]) for n in ORIGINAL_HASHES if n != 'original-SHA256SUMS.txt'},
        'report_identity': digest(before['original-attempt/report.json']),
        'request_identity': parse(before['original-attempt/attempt.json'])['request_identity'],
        'supplied_manifest_sha256': digest(before['original-SHA256SUMS.txt'])[7:],
        'evaluation_commit': '99a5896dd893f9d338e6c2c81c49dba29fbecbb1',
        'evaluation_tree': '9a91ca23c458801bf124fb721f5572b3309b64c8',
        'full_original_custody_verified': False, 'inference_replayed': False, 'optimizer_replayed': False}
    if any(encoded(provenance[k]) != encoded(v) for k, v in links.items()):
        raise ValueError('derived provenance links or scope differ')
    if snapshot(fixture) != before:
        raise ValueError('fixture changed during verification')
    sources()
    return {'schema': 'trinite.frozen-music-eval-retained-check.v1', 'verified': True,
        'scope': 'supplied byte integrity, historical training prediction parity and retained-token score consistency only',
        'execution_outcome_in_supplied_report': 'completed',
        'interpretation': 'negative heldout-development result; descriptive, no predeclared accuracy threshold',
        'decodes_in_journal': summary['decodes'], 'generated_tokens': summary['forward_calls'],
        'historical_training_predictions_equal': 80, 'heldout_structural_parents': 2,
        'canonical_exact_per_seed': {s: summary['by_seed'][s]['heldout']['canonical_primary']['exact_plus_eos'] for s in ('17','29')},
        'original_raw_custody_verified': False, 'actual_execution_attested': False,
        'inference_replayed': False, 'optimizer_replayed': False,
        'fresh_confirmation': False, 'learning_adequate': False, 'release_ready': False}


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(), sort_keys=True))
    except (ValueError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
