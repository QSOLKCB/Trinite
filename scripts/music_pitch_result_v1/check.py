"""Immutable reports-only pitch-result authentication and scoring; no model execution."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/music-pitch-result-v1'
LIMIT = 1024 * 1024
ORIGINAL_HASHES = {'original-run/audit.stdout': 'sha256:b37d33ed9aa9d8d5144af5cf9ad9cd07927f40db1016cf51fee3fe647641f4ef', 'original-run/control-17/attempt.json': 'sha256:ae8458ebd95eb8ef692620accb87200a692fe099ebc96597f0e29ca131094565', 'original-run/control-17/completion.json': 'sha256:7cd61f2876554b450c4a2fc46b6752d6f12b2af13a87236fab559d6c63bb8590', 'original-run/control-17/diagnostic-predictions.jsonl': 'sha256:a5e73cd8c5ed4998576ecb6a4ff520dea3aa2c26f48dec282fefd0cfc08ec6c2', 'original-run/control-17/report.json': 'sha256:7f877e1d9a2bafe11c231fcbbfba46bfe0fd10c58e6877c28976c7a2481d3c3e', 'original-run/control-29/attempt.json': 'sha256:96c8ae1c856415276dd3112a9396fed3a4bce95dd8237e0420c22ba95e74504e', 'original-run/control-29/completion.json': 'sha256:f823320643d0be291bd67b41386424f8b06753c2441c675c4041e2fd4f705966', 'original-run/control-29/diagnostic-predictions.jsonl': 'sha256:e00d7ae5673385bbb332ee60a2ca59d64d37c1a40e5b886e0e0335fa4150afc6', 'original-run/control-29/report.json': 'sha256:810b22359b9486bdd17bb011828fe9875d4971f2940b00f347cdadfb15aa9db0', 'original-run/development/attempt.json': 'sha256:2c99badb918654b6f184edd11e31021dcbea26d0ed49daf7851641ca7c505ac4', 'original-run/development/completion.json': 'sha256:4cb8877e6ba05d5959c56f6f1ca8cd253114ef1b68a732ec424ccbde6b0569e1', 'original-run/development/predictions.jsonl': 'sha256:9ad0182bed9593e0c92f464aa84f4d8f929e0402b0f2a2c22328e90ccb65b76d', 'original-run/development/report.json': 'sha256:bc2c09fc9552db509b8695c7a3745ed46192a3023d1645041646a470d70a6571', 'original-run/development-gate.json': 'sha256:3bd77afe8c845b4fd282f6dfbdebabe859d0bbd714cabf22c07e7579d85cc889', 'original-run/matrix.json': 'sha256:59c92c9217a35b2b5a5594fa9ba839a903807398b30788a1c286095f4b69b6db', 'original-run/pitch-17/attempt.json': 'sha256:474c8546d4a63011fb60390bd213f2ba955dea2d70148c73c572da08d697f1f1', 'original-run/pitch-17/completion.json': 'sha256:b820a9feaeb3dd049d9a6b00d6f4077ef3373dc2675c2561b5f650a5c4358313', 'original-run/pitch-17/diagnostic-predictions.jsonl': 'sha256:9e636fdfcc6cbdd8551537ec8647db9bf5c4d8d2980d3ab630cc04164e97977c', 'original-run/pitch-17/report.json': 'sha256:5817673adf8b7531e55a621c730a2700d094de27ef391139a0b0f8437c6648b3', 'original-run/pitch-29/attempt.json': 'sha256:b6acfb66cc99f46c516f394a0a302f93c02d52012defe942f442cac63a338e95', 'original-run/pitch-29/completion.json': 'sha256:7fffc336b348014bb7e07c52480f1d44ef36a4b5fb607c8b83a32c3fefeb4434', 'original-run/pitch-29/diagnostic-predictions.jsonl': 'sha256:f542052ddfcf0982b19d1083ff547e2d054ce2c38115798e31068ece0bd31f34', 'original-run/pitch-29/report.json': 'sha256:b70d62895a235b6cce1167b1cdb3f835a33703217af9b419c36e7fa2eabb2007', 'source-SHA256SUMS': 'sha256:452afdfdeb7b5ab83304dc03d509315064672191d59d7fddea7b7f05f8ec0366'}
FIXTURE_HASHES = {'README.md': 'sha256:98d7557c5a98ad77408638f69e779836c90eea6b534665b94e158542e0a046ae', 'original-run/audit.stdout': 'sha256:b37d33ed9aa9d8d5144af5cf9ad9cd07927f40db1016cf51fee3fe647641f4ef', 'original-run/control-17/attempt.json': 'sha256:ae8458ebd95eb8ef692620accb87200a692fe099ebc96597f0e29ca131094565', 'original-run/control-17/completion.json': 'sha256:7cd61f2876554b450c4a2fc46b6752d6f12b2af13a87236fab559d6c63bb8590', 'original-run/control-17/diagnostic-predictions.jsonl': 'sha256:a5e73cd8c5ed4998576ecb6a4ff520dea3aa2c26f48dec282fefd0cfc08ec6c2', 'original-run/control-17/report.json': 'sha256:7f877e1d9a2bafe11c231fcbbfba46bfe0fd10c58e6877c28976c7a2481d3c3e', 'original-run/control-29/attempt.json': 'sha256:96c8ae1c856415276dd3112a9396fed3a4bce95dd8237e0420c22ba95e74504e', 'original-run/control-29/completion.json': 'sha256:f823320643d0be291bd67b41386424f8b06753c2441c675c4041e2fd4f705966', 'original-run/control-29/diagnostic-predictions.jsonl': 'sha256:e00d7ae5673385bbb332ee60a2ca59d64d37c1a40e5b886e0e0335fa4150afc6', 'original-run/control-29/report.json': 'sha256:810b22359b9486bdd17bb011828fe9875d4971f2940b00f347cdadfb15aa9db0', 'original-run/development/attempt.json': 'sha256:2c99badb918654b6f184edd11e31021dcbea26d0ed49daf7851641ca7c505ac4', 'original-run/development/completion.json': 'sha256:4cb8877e6ba05d5959c56f6f1ca8cd253114ef1b68a732ec424ccbde6b0569e1', 'original-run/development/predictions.jsonl': 'sha256:9ad0182bed9593e0c92f464aa84f4d8f929e0402b0f2a2c22328e90ccb65b76d', 'original-run/development/report.json': 'sha256:bc2c09fc9552db509b8695c7a3745ed46192a3023d1645041646a470d70a6571', 'original-run/development-gate.json': 'sha256:3bd77afe8c845b4fd282f6dfbdebabe859d0bbd714cabf22c07e7579d85cc889', 'original-run/matrix.json': 'sha256:59c92c9217a35b2b5a5594fa9ba839a903807398b30788a1c286095f4b69b6db', 'original-run/pitch-17/attempt.json': 'sha256:474c8546d4a63011fb60390bd213f2ba955dea2d70148c73c572da08d697f1f1', 'original-run/pitch-17/completion.json': 'sha256:b820a9feaeb3dd049d9a6b00d6f4077ef3373dc2675c2561b5f650a5c4358313', 'original-run/pitch-17/diagnostic-predictions.jsonl': 'sha256:9e636fdfcc6cbdd8551537ec8647db9bf5c4d8d2980d3ab630cc04164e97977c', 'original-run/pitch-17/report.json': 'sha256:5817673adf8b7531e55a621c730a2700d094de27ef391139a0b0f8437c6648b3', 'original-run/pitch-29/attempt.json': 'sha256:b6acfb66cc99f46c516f394a0a302f93c02d52012defe942f442cac63a338e95', 'original-run/pitch-29/completion.json': 'sha256:7fffc336b348014bb7e07c52480f1d44ef36a4b5fb607c8b83a32c3fefeb4434', 'original-run/pitch-29/diagnostic-predictions.jsonl': 'sha256:f542052ddfcf0982b19d1083ff547e2d054ce2c38115798e31068ece0bd31f34', 'original-run/pitch-29/report.json': 'sha256:b70d62895a235b6cce1167b1cdb3f835a33703217af9b419c36e7fa2eabb2007', 'provenance-DERIVED.json': 'sha256:ca93a6802e384224a887b475078badc1dd2ba704b3cbd28822439694281daab7', 'source-SHA256SUMS': 'sha256:452afdfdeb7b5ab83304dc03d509315064672191d59d7fddea7b7f05f8ec0366'}
MODULES = {'trinite': 'src/trinite/__init__.py', 'trinite.contracts': 'src/trinite/contracts.py', 'trinite.foundations_plan': 'src/trinite/foundations_plan.py', 'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py', 'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py', 'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py', 'music_learning_v1': 'scripts/music_learning_v1/__init__.py', 'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py', 'music_learning_v1.numerical': 'scripts/music_learning_v1/numerical.py', 'music_contrast_v1': 'scripts/music_contrast_v1/__init__.py', 'music_contrast_v1.spec': 'scripts/music_contrast_v1/spec.py', 'music_contrast_v1.reporting': 'scripts/music_contrast_v1/reporting.py', 'music_exposure_v1': 'scripts/music_exposure_v1/__init__.py', 'music_exposure_v1.spec': 'scripts/music_exposure_v1/spec.py', 'music_pitch_intervention_v1': 'scripts/music_pitch_intervention_v1/__init__.py', 'music_pitch_intervention_v1.spec': 'scripts/music_pitch_intervention_v1/spec.py', 'music_pitch_intervention_v1.bootstrap': 'scripts/music_pitch_intervention_v1/bootstrap.py', 'music_pitch_intervention_v1.reporting': 'scripts/music_pitch_intervention_v1/reporting.py', 'trinite.tokenizer': 'src/trinite/tokenizer.py'}
SOURCES = {'src/trinite/__init__.py': 'sha256:029d147c8a741f0cecffc90611d77c6a4047a019863d86ca40992c630a3524b6', 'src/trinite/contracts.py': 'sha256:36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24', 'src/trinite/foundations_plan.py': 'sha256:f4a4cee055b97e6aa1217340b1a103cd561caeaca01e6b358e71eaedbab8a8d7', 'scripts/symbolic_music_v1/__init__.py': 'sha256:aea01a9d1bd6ae591665b9fcc3658189091a162565851d4c92579731f82405db', 'scripts/symbolic_music_v1/core.py': 'sha256:9c7446c8d275dbc02fe195eac7a660130cf70a6943fa31428c0d4c7f49bba6ff', 'scripts/symbolic_music_v1/oracle.py': 'sha256:3b6c3e00a2caca2a4d2e12d6ab93fda912197d638cec220175a8073b4ee09af0', 'scripts/music_learning_v1/__init__.py': 'sha256:352fce75f453ef87602ff6b451800c6eb765fa2edd2c79d5428013b6f05b5c9a', 'scripts/music_learning_v1/codec.py': 'sha256:5236fab68f2ff11ff4248d1187ddd0db5797138784665eeca4323ff9c192fa0b', 'scripts/music_learning_v1/numerical.py': 'sha256:214fa9dfd4490cedb7a55489331db1ad550b8b5dfdf5021294a117d470f20813', 'scripts/music_contrast_v1/__init__.py': 'sha256:3d67f788b234aba0de63cfe5368e9a9dac7d8d8220ecfc5120b5ce708694428d', 'scripts/music_contrast_v1/spec.py': 'sha256:34f051fe47feb48baff3ca636730d9348682812e2ffb4bd3e2a4c07b78cb3317', 'scripts/music_contrast_v1/reporting.py': 'sha256:5f15f9231e3b57e32410a8cd2eb7d66f49565fa13661a680c8327d03fba2b846', 'scripts/music_exposure_v1/__init__.py': 'sha256:36ac2634a27e9076d885362c988335f154cd071ae167b6e7c8d1a481df9f5c6d', 'scripts/music_exposure_v1/spec.py': 'sha256:c80b16e68ab85ebdfe06a40e406941b3e19ae5596ca7945babac9df8e35448d9', 'scripts/music_pitch_intervention_v1/__init__.py': 'sha256:83debd77b5034b913e86fa01026b14109ea58c90bf2ad9153ffae2ebe711a98d', 'scripts/music_pitch_intervention_v1/spec.py': 'sha256:c6228050f2f24f47404b7bae747b6578f81d5c368a9db419accaabfd69529855', 'scripts/music_pitch_intervention_v1/bootstrap.py': 'sha256:e515872f9d841feb9218043f561ff1a8216172b6d26f1fdb615b8506eb2d9c4e', 'scripts/music_pitch_intervention_v1/reporting.py': 'sha256:a2c7b95a5b5cfbaac7da0a912af6d83ceb1a714eeb0c466c7a9bb0dda23c4bf9', 'src/trinite/tokenizer.py': 'sha256:fa87f79f667da8cabeeae0e999add9f420102efbfb746a0c0b275b8703d845bb'}
HISTORICAL_HASHES = {'fixtures/frozen-music-eval-result-v1/original-attempt/predictions.jsonl': 'sha256:b0939e4508697d4909e08acefea50ac568ae008092f1952068145c814355a56f'}
CELLS = ('control-17', 'pitch-17', 'pitch-29', 'control-29')
REQUEST_ID = 'sha256:f1bbfa44e642fa523979da586069d18564a3c3319a8b5c47f864baa1b0fdf198'
MATRIX_ID = 'sha256:59c92c9217a35b2b5a5594fa9ba839a903807398b30788a1c286095f4b69b6db'

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
    """Exact bounded membership; refuse links, special files and extra payloads."""
    fixture = Path(os.path.abspath(fixture))
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in fixture.parents):
        raise ValueError('unsafe ancestor')
    if not stat.S_ISDIR(fixture.lstat().st_mode):
        raise ValueError('non-link fixture directory required')
    expected_dirs = {p.as_posix() for name in FIXTURE_HASHES
                     for p in Path(name).parents if p != Path('.')}
    found_files, found_dirs = set(), set()
    for current, directories, files in os.walk(fixture, followlinks=False):
        for name in directories:
            p = Path(current) / name
            if not stat.S_ISDIR(p.lstat().st_mode):
                raise ValueError('non-link fixture directory required')
            relative = p.relative_to(fixture).as_posix()
            if relative not in expected_dirs:
                raise ValueError('fixture directory membership differs')
            found_dirs.add(relative)
        for name in files:
            found_files.add((Path(current) / name).relative_to(fixture).as_posix())
            if len(found_files) > len(FIXTURE_HASHES):
                raise ValueError('fixture membership differs')
    if found_files != set(FIXTURE_HASHES) or found_dirs != expected_dirs:
        raise ValueError('fixture membership differs')
    return {name: read(fixture / name) for name in FIXTURE_HASHES}


def historical_predictions():
    result = {name: read(ROOT/name) for name in HISTORICAL_HASHES}
    if {name: digest(raw) for name, raw in result.items()} != HISTORICAL_HASHES:
        raise ValueError('historical development journal changed')
    return result


def score(before):
    """Check available bindings and recompute scores; never load omitted tensors."""
    sources()
    from trinite.contracts import parse_json, json_bytes, exact_keys
    from music_pitch_intervention_v1.reporting import matrix_summary, score_cell
    sources()
    historical = historical_predictions()
    def parse(name):
        return parse_json(before['original-run/'+name], canonical=True)
    def raw_id(name):
        return digest(before['original-run/'+name])
    def same(left, right, label):
        if json_bytes(left) != json_bytes(right):
            raise ValueError(label+' differs')
    def records(name):
        return [parse_json(line, canonical=True) for line in
                before['original-run/'+name].splitlines(keepends=True)]
    matrix, gate, audit = parse('matrix.json'), parse('development-gate.json'), parse('audit.stdout')
    if raw_id('matrix.json') != MATRIX_ID or matrix['request_identity'] != REQUEST_ID:
        raise ValueError('original matrix or owner request link differs')
    claims = {'fresh_confirmation': False, 'learning_adequate': False,
              'phase5_ready': False, 'release_ready': False}
    scope = 'source-bound bytes, safe tensor/progress structure and retained scores; no inference or optimizer replay'
    exact_keys(matrix, {'schema','request_identity','outcome','cells','development','gate_identity','failure','summary','verification_scope','files',*claims}, 'matrix')
    same({k:matrix[k] for k in claims}, claims, 'matrix claims')
    if (matrix['schema'] != 'trinite.music-pitch-matrix.v1' or matrix['outcome'] != 'completed'
            or matrix['failure'] is not None or matrix['verification_scope'] != scope):
        raise ValueError('retained matrix context differs')
    exact_keys(matrix['cells'], set(CELLS), 'matrix cells')
    present = {n.removeprefix('original-run/'):digest(b) for n,b in before.items()
               if n.startswith('original-run/') and n not in ('original-run/matrix.json','original-run/audit.stdout')}
    omitted = {cell+'/'+name for cell in CELLS for name in
               ('diagnostic-predictions.json','final-checkpoint.json','final.safetensors','request.json','worker.log')}
    omitted.update(('request.json','development/worker.log'))
    if set(matrix['files']) != set(present) | omitted:
        raise ValueError('matrix available/omitted membership differs')
    same({n:matrix['files'][n] for n in present}, present, 'available matrix bindings')
    diagnostics = []
    initializations = {}
    for cell in CELLS:
        arm, seed = cell.split('-'); seed = int(seed)
        journal = records(cell+'/diagnostic-predictions.jsonl')
        diagnostics.extend(journal)
        summary = score_cell(journal, arm, seed, 'diagnostic')
        report, completion = parse(cell+'/report.json'), parse(cell+'/completion.json')
        exact_keys(report, {'arm','seed','request_identity','admission_identity','step','target_tokens',
            'initialization_identity','model_identity','update_seconds_hex','diagnostic_summary','diagnostic_decodes',
            'diagnostic_forward_calls','schema','cell','diagnostics','verification_scope',*claims}, 'cell report')
        same(report['diagnostics'], summary, 'cell diagnostic score')
        same(report['diagnostic_summary'], summary, 'cell native diagnostic score')
        same({k:report[k] for k in claims}, claims, 'cell claims')
        if (report['schema'] != 'trinite.music-pitch-cell.v1' or report['cell'] != cell
                or report['arm'] != arm or type(report['seed']) is not int or report['seed'] != seed
                or report['request_identity'] != REQUEST_ID or report['verification_scope'] != scope
                or type(report['step']) is not int or report['step'] != 640
                or type(report['target_tokens']) is not int or report['target_tokens'] != 10240
                or report['diagnostic_decodes'] != 80
                or report['diagnostic_forward_calls'] != sum(len(r['generated_tokens']) for r in journal)):
            raise ValueError('cell report context/accounting differs')
        seconds = float.fromhex(report['update_seconds_hex'])
        if not math.isfinite(seconds) or not 0 <= seconds <= 120 or seconds.hex() != report['update_seconds_hex']:
            raise ValueError('recorded update duration outside ceiling')
        initializations[cell] = report['initialization_identity']
        same(completion, {'schema':'trinite.music-pitch-completion.v1','request_identity':REQUEST_ID,
                         'cell':cell,'artifacts':{n:matrix['files'][cell+'/'+n] for n in
                         ('diagnostic-predictions.json','diagnostic-predictions.jsonl','final-checkpoint.json','final.safetensors','report.json')}}, 'cell completion')
        for name in ('report.json','diagnostic-predictions.jsonl'):
            if raw_id(cell+'/'+name) != completion['artifacts'][name]:
                raise ValueError('available completion artifact differs')
    for seed in (17,29):
        if initializations['control-'+str(seed)] != initializations['pitch-'+str(seed)]:
            raise ValueError('reported paired initialization identity differs')
    same(gate, {'schema':'trinite.music-pitch-development-gate.v1','request_identity':REQUEST_ID,
                'cells':{cell:raw_id(cell+'/completion.json') for cell in CELLS},
                'selection':'all four safe completed final640 cells; no training-fit gate'}, 'development gate')
    if raw_id('development-gate.json') != matrix['gate_identity']:
        raise ValueError('matrix gate link differs')
    for cell in (*CELLS, 'development'):
        attempt = parse(cell+'/attempt.json')
        record = matrix['development'] if cell == 'development' else matrix['cells'][cell]
        same(record, {'outcome':'completed','completion_identity':raw_id(cell+'/completion.json'),
                      'attempt_identity':raw_id(cell+'/attempt.json'),
                      'files':{n.removeprefix(cell+'/'):h for n,h in matrix['files'].items() if n.startswith(cell+'/')}}, 'matrix attempt links')
        exact_keys(attempt, {'schema','action','cell','request_identity','outcome','worker_exit','completion_identity','detail','wall_seconds_hex'}, 'attempt')
        if (attempt['schema'] != 'trinite.music-pitch-attempt.v1' or attempt['cell'] != cell
                or attempt['action'] != ('_development' if cell == 'development' else '_train')
                or attempt['request_identity'] != REQUEST_ID or attempt['outcome'] != 'completed'
                or type(attempt['worker_exit']) is not int or attempt['worker_exit'] != 0
                or attempt['completion_identity'] != record['completion_identity']
                or type(attempt['detail']) is not str):
            raise ValueError('retained attempt context differs')
        seconds = float.fromhex(attempt['wall_seconds_hex'])
        if not math.isfinite(seconds) or seconds < 0 or seconds.hex() != attempt['wall_seconds_hex']:
            raise ValueError('invalid recorded wall duration')
    development_records = records('development/predictions.jsonl')
    summary = matrix_summary(diagnostics, development_records)
    same(matrix['summary'], summary, 'matrix retained score')
    report = parse('development/report.json')
    same(report, {'schema':'trinite.music-pitch-development.v1','request_identity':REQUEST_ID,
        'gate_identity':matrix['gate_identity'],'summary':summary,
        'generation':{'decodes':480,'forward_calls':sum(len(r['generated_tokens']) for r in development_records),
                      'model_identities_unchanged':True,'training_performed':False},
        'input_files_unchanged':True,'model_states_unchanged':True,'verification_scope':scope,**claims},
        'development report/accounting')
    same(parse('development/completion.json'), {
        'schema':'trinite.music-pitch-development-completion.v1','request_identity':REQUEST_ID,
        'gate_identity':matrix['gate_identity'],'artifacts':{
            n:raw_id('development/'+n) for n in ('predictions.jsonl','report.json')}}, 'development completion')
    if (report['request_identity'] != REQUEST_ID or report['gate_identity'] != matrix['gate_identity']
            or report['schema'] != 'trinite.music-pitch-development.v1'
            or report['verification_scope'] != scope):
        raise ValueError('development context differs')
    same({k:report[k] for k in claims}, claims, 'development claims')
    same(audit, {'verified':True,'outcome':'completed','matrix_identity':MATRIX_ID,
                 'scope':scope,'inference_replayed':False,'optimizer_replayed':False,**claims}, 'supplied audit receipt')
    manifest = {}
    for line in before['source-SHA256SUMS'].decode('ascii').splitlines():
        checksum, name = line.split('  ', 1)
        if name in manifest:
            raise ValueError('duplicate original handoff manifest member')
        manifest[name] = 'sha256:'+checksum
    for name in ORIGINAL_HASHES:
        if name != 'source-SHA256SUMS' and manifest.get(name) != digest(before[name]):
            raise ValueError('original handoff manifest link differs')
    provenance = parse_json(before['provenance-DERIVED.json'], canonical=True)
    same(provenance['included_original_files'], sorted(n.removeprefix('original-run/') for n in ORIGINAL_HASHES if n.startswith('original-run/')), 'derived original inventory')
    same(provenance['matrix_bound_present'], sorted(present), 'derived present inventory')
    same(provenance['matrix_bound_omitted'], sorted(omitted), 'derived omitted inventory')
    same({k:provenance[k] for k in ('schema','producer_commit','producer_tree','source_archive_sha256')},
         {'schema':'trinite.music-pitch-result-provenance.v1',
          'producer_commit':'6258ecbeea2dd5553ebb73619f17432dd83c5ad6',
          'producer_tree':'6ade3340409344cccc2712962f4651580fb8f255',
          'source_archive_sha256':'d7ea7dd7c223cdfe8332a5a1c037c9a6bd9656413fe992206292c085722f38df'},
         'derived source provenance')
    if (provenance['original_matrix_identity'] != MATRIX_ID
            or provenance['original_request_identity_owner_receipt'] != REQUEST_ID):
        raise ValueError('derived identities differ')
    for key in ('actual_execution_attested','full_run_custody','inference_replayed','optimizer_replayed','fresh_confirmation'):
        if provenance[key] is not False:
            raise ValueError('derived verification scope differs')
    if historical_predictions() != historical:
        raise ValueError('historical journal changed during scoring')
    sources()
    return summary


def verify(fixture=FIXTURE):
    before = snapshot(fixture)
    if {name:digest(raw) for name,raw in before.items()} != FIXTURE_HASHES:
        raise ValueError('preserved fixture bytes changed')
    summary = score(before)
    if snapshot(fixture) != before:
        raise ValueError('fixture changed during verification')
    sources()
    return {'schema':'trinite.music-pitch-retained-check.v1','verified':True,
        'scope':'supplied byte integrity, available links and retained-token score consistency only',
        'execution_outcome_in_supplied_report':'completed',
        'interpretation':'mixed/negative development result; descriptive, no accuracy pass threshold',
        'decodes_in_journal':summary['decodes'],'generated_tokens':summary['decode_forwards'],
        'available_matrix_bound_files':21,'omitted_matrix_bound_files':22,
        'development_structural_parents':2,
        'canonical_exact_by_cell':{cell:summary['development']['by_cell'][cell]['canonical_primary']['exact_plus_eos'] for cell in CELLS},
        'original_raw_custody_verified':False,'actual_execution_attested':False,
        'inference_replayed':False,'optimizer_replayed':False,'fresh_confirmation':False,
        'learning_adequate':False,'release_ready':False}


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(), sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
