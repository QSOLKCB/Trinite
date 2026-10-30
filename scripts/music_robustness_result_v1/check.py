"""Reports-only v2 CPU repeat and standalone device-freeze error preservation."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/music-robustness-result-v1'
LIMIT = 1024 * 1024
ORIGINAL_HASHES = {'cpu/audit.stdout': 'sha256:28e853ff7f24c3e209de406969e19e09fa924acb4f992aa51ed0d3683cf2f825', 'cpu/control-17/attempt.json': 'sha256:b2256480e0fe4a26f1346d369adb2554ad5dc09f31ba70439d95af242405dc58', 'cpu/control-17/completion.json': 'sha256:ffa7a6a81511da13f20e9ef8fe7024002b1ee4e816d308826756c79aa1e8f8a3', 'cpu/control-17/diagnostic-predictions.jsonl': 'sha256:83589d201a2ed03121569d955723a4c5122b666d997811eda061651259c8f4b0', 'cpu/control-17/report.json': 'sha256:84b1b5af3f970dddc26800290e8ac092d76214dc08dda55b49377102aefd15a3', 'cpu/control-29/attempt.json': 'sha256:ed63ee0d2270f340f51e488ffb39f7b64bb11aecb434f4244d6de714b2215dfa', 'cpu/control-29/completion.json': 'sha256:4c26914930ab2a570693e246c59cc258fedbc837c847e57a6efb71022c38f1da', 'cpu/control-29/diagnostic-predictions.jsonl': 'sha256:40c42c1bf4af61393591348b45ab5b78e74a1d916bc6960192b5e4516904cb68', 'cpu/control-29/report.json': 'sha256:125e240b67ffd9a082923719fdf3afb05ca58965d9f5eaee6625ff452b75b825', 'cpu/development/attempt.json': 'sha256:b206151cf45190fdc11b93b01d0f1c488a2638b71e6a3c886429e4cb51bd05b9', 'cpu/development/completion.json': 'sha256:47e690f476f17e6bbe4913e15383e59ef41d2a410bdd22125bc15e514df70210', 'cpu/development/predictions.jsonl': 'sha256:6f5b4d236d1f8535e34cae27d4ea60066dde529a6cd2fed3b37d97ce4325ea88', 'cpu/development/report.json': 'sha256:c98552a43aa6af52f7fb312bd60f65bdd81e419139866744f991a31b166cb312', 'cpu/development-gate.json': 'sha256:860ad49f9bddbe2581404e3daa9881e683527968de93ddc082070df3e6ee1daa', 'cpu/matrix.json': 'sha256:3aefd8026194a557cef3dcbee380e61dbb81f50ffb42dfc264900bf0ea0921e2', 'cpu/swap-17/attempt.json': 'sha256:ebc385b0d7fcd4feb637d4f684cdc8d9fa82888a14080d8214b411ca317b7e29', 'cpu/swap-17/completion.json': 'sha256:94bd4d53cf1aa0edb938c66c760f92bc0d39664c3e4fc2d469e2065acacacc3c', 'cpu/swap-17/diagnostic-predictions.jsonl': 'sha256:652b51f2b8e7b68858dfc7c4082301d278b3bfb9854c6fcdcf21ee6d5eafd692', 'cpu/swap-17/report.json': 'sha256:9889b6b94fbfd29485f724f552e554d35d0a470ef992a4c49ef302713cff8e4f', 'cpu/swap-29/attempt.json': 'sha256:63a03ac3dac30fc69a0068eceb56e8eaa20a162836566473e7cbdd8ba4a1a0e9', 'cpu/swap-29/completion.json': 'sha256:f67e3957775dca6f4fee1ee63224b6078901d9f3e2f762811b183fdf68509ce5', 'cpu/swap-29/diagnostic-predictions.jsonl': 'sha256:dc106535e2c2b5493bf8ffb684ad82282ca760e437064484cd48ccbab9ffd283', 'cpu/swap-29/report.json': 'sha256:01c4024cfe84d86c40f55c05fff006ecf266af4d3ad212dc6f6b42d0dc7ca467', 'device/freeze.stderr': 'sha256:86c9c26aeeb44f30172caa707e36a0e998152c69689113127d236fb679d27ce6', 'source-SHA256SUMS': 'sha256:e8b93d7eba92d75f0f5969195c5d4cd610fc7de4b035d857c56513bd51f782c2'}
FIXTURE_HASHES = {'README.md': 'sha256:b473ff5e6aeb59cb9a76b9936a3dafe1b50d581e1afce3a909a0e86cf7091c56', 'cpu/audit.stdout': 'sha256:28e853ff7f24c3e209de406969e19e09fa924acb4f992aa51ed0d3683cf2f825', 'cpu/control-17/attempt.json': 'sha256:b2256480e0fe4a26f1346d369adb2554ad5dc09f31ba70439d95af242405dc58', 'cpu/control-17/completion.json': 'sha256:ffa7a6a81511da13f20e9ef8fe7024002b1ee4e816d308826756c79aa1e8f8a3', 'cpu/control-17/diagnostic-predictions.jsonl': 'sha256:83589d201a2ed03121569d955723a4c5122b666d997811eda061651259c8f4b0', 'cpu/control-17/report.json': 'sha256:84b1b5af3f970dddc26800290e8ac092d76214dc08dda55b49377102aefd15a3', 'cpu/control-29/attempt.json': 'sha256:ed63ee0d2270f340f51e488ffb39f7b64bb11aecb434f4244d6de714b2215dfa', 'cpu/control-29/completion.json': 'sha256:4c26914930ab2a570693e246c59cc258fedbc837c847e57a6efb71022c38f1da', 'cpu/control-29/diagnostic-predictions.jsonl': 'sha256:40c42c1bf4af61393591348b45ab5b78e74a1d916bc6960192b5e4516904cb68', 'cpu/control-29/report.json': 'sha256:125e240b67ffd9a082923719fdf3afb05ca58965d9f5eaee6625ff452b75b825', 'cpu/development/attempt.json': 'sha256:b206151cf45190fdc11b93b01d0f1c488a2638b71e6a3c886429e4cb51bd05b9', 'cpu/development/completion.json': 'sha256:47e690f476f17e6bbe4913e15383e59ef41d2a410bdd22125bc15e514df70210', 'cpu/development/predictions.jsonl': 'sha256:6f5b4d236d1f8535e34cae27d4ea60066dde529a6cd2fed3b37d97ce4325ea88', 'cpu/development/report.json': 'sha256:c98552a43aa6af52f7fb312bd60f65bdd81e419139866744f991a31b166cb312', 'cpu/development-gate.json': 'sha256:860ad49f9bddbe2581404e3daa9881e683527968de93ddc082070df3e6ee1daa', 'cpu/matrix.json': 'sha256:3aefd8026194a557cef3dcbee380e61dbb81f50ffb42dfc264900bf0ea0921e2', 'cpu/swap-17/attempt.json': 'sha256:ebc385b0d7fcd4feb637d4f684cdc8d9fa82888a14080d8214b411ca317b7e29', 'cpu/swap-17/completion.json': 'sha256:94bd4d53cf1aa0edb938c66c760f92bc0d39664c3e4fc2d469e2065acacacc3c', 'cpu/swap-17/diagnostic-predictions.jsonl': 'sha256:652b51f2b8e7b68858dfc7c4082301d278b3bfb9854c6fcdcf21ee6d5eafd692', 'cpu/swap-17/report.json': 'sha256:9889b6b94fbfd29485f724f552e554d35d0a470ef992a4c49ef302713cff8e4f', 'cpu/swap-29/attempt.json': 'sha256:63a03ac3dac30fc69a0068eceb56e8eaa20a162836566473e7cbdd8ba4a1a0e9', 'cpu/swap-29/completion.json': 'sha256:f67e3957775dca6f4fee1ee63224b6078901d9f3e2f762811b183fdf68509ce5', 'cpu/swap-29/diagnostic-predictions.jsonl': 'sha256:dc106535e2c2b5493bf8ffb684ad82282ca760e437064484cd48ccbab9ffd283', 'cpu/swap-29/report.json': 'sha256:01c4024cfe84d86c40f55c05fff006ecf266af4d3ad212dc6f6b42d0dc7ca467', 'device/freeze.stderr': 'sha256:86c9c26aeeb44f30172caa707e36a0e998152c69689113127d236fb679d27ce6', 'provenance-DERIVED.json': 'sha256:2563ca5d7a966b7ef2bb13e8e8c7882291483597bd716d43c8ae884a8cd2885b', 'source-SHA256SUMS': 'sha256:e8b93d7eba92d75f0f5969195c5d4cd610fc7de4b035d857c56513bd51f782c2'}
MODULES = {'trinite': 'src/trinite/__init__.py', 'trinite.contracts': 'src/trinite/contracts.py', 'trinite.foundations_plan': 'src/trinite/foundations_plan.py', 'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py', 'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py', 'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py', 'music_learning_v1': 'scripts/music_learning_v1/__init__.py', 'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py', 'music_learning_v1.numerical': 'scripts/music_learning_v1/numerical.py', 'music_contrast_v1': 'scripts/music_contrast_v1/__init__.py', 'music_contrast_v1.spec': 'scripts/music_contrast_v1/spec.py', 'music_contrast_v1.reporting': 'scripts/music_contrast_v1/reporting.py', 'music_exposure_v1': 'scripts/music_exposure_v1/__init__.py', 'music_exposure_v1.spec': 'scripts/music_exposure_v1/spec.py', 'trinite.tokenizer': 'src/trinite/tokenizer.py', 'music_placement_v1': 'scripts/music_placement_v1/__init__.py', 'music_placement_v1.spec': 'scripts/music_placement_v1/spec.py', 'music_placement_v1.reporting': 'scripts/music_placement_v1/reporting.py'}
SOURCES = {'src/trinite/__init__.py': 'sha256:029d147c8a741f0cecffc90611d77c6a4047a019863d86ca40992c630a3524b6', 'src/trinite/contracts.py': 'sha256:36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24', 'src/trinite/foundations_plan.py': 'sha256:f4a4cee055b97e6aa1217340b1a103cd561caeaca01e6b358e71eaedbab8a8d7', 'scripts/symbolic_music_v1/__init__.py': 'sha256:aea01a9d1bd6ae591665b9fcc3658189091a162565851d4c92579731f82405db', 'scripts/symbolic_music_v1/core.py': 'sha256:9c7446c8d275dbc02fe195eac7a660130cf70a6943fa31428c0d4c7f49bba6ff', 'scripts/symbolic_music_v1/oracle.py': 'sha256:3b6c3e00a2caca2a4d2e12d6ab93fda912197d638cec220175a8073b4ee09af0', 'scripts/music_learning_v1/__init__.py': 'sha256:352fce75f453ef87602ff6b451800c6eb765fa2edd2c79d5428013b6f05b5c9a', 'scripts/music_learning_v1/codec.py': 'sha256:5236fab68f2ff11ff4248d1187ddd0db5797138784665eeca4323ff9c192fa0b', 'scripts/music_learning_v1/numerical.py': 'sha256:214fa9dfd4490cedb7a55489331db1ad550b8b5dfdf5021294a117d470f20813', 'scripts/music_contrast_v1/__init__.py': 'sha256:3d67f788b234aba0de63cfe5368e9a9dac7d8d8220ecfc5120b5ce708694428d', 'scripts/music_contrast_v1/spec.py': 'sha256:34f051fe47feb48baff3ca636730d9348682812e2ffb4bd3e2a4c07b78cb3317', 'scripts/music_contrast_v1/reporting.py': 'sha256:5f15f9231e3b57e32410a8cd2eb7d66f49565fa13661a680c8327d03fba2b846', 'scripts/music_exposure_v1/__init__.py': 'sha256:36ac2634a27e9076d885362c988335f154cd071ae167b6e7c8d1a481df9f5c6d', 'scripts/music_exposure_v1/spec.py': 'sha256:c80b16e68ab85ebdfe06a40e406941b3e19ae5596ca7945babac9df8e35448d9', 'src/trinite/tokenizer.py': 'sha256:fa87f79f667da8cabeeae0e999add9f420102efbfb746a0c0b275b8703d845bb', 'scripts/music_placement_v1/__init__.py': 'sha256:03a38e3bfe0523473faf6de5fba8f80afb3ce561cfadf12631300b3d5a1b562e', 'scripts/music_placement_v1/spec.py': 'sha256:60660075dac0062ff334bac0db577cb539108f136b77a0ad927a00af8d3eef33', 'scripts/music_placement_v1/reporting.py': 'sha256:4bcfd7cddfe9f92cc2133117ec853588834a5895e0b62697c77f20e5655fa3ab'}
REFERENCE_SOURCES = {'scripts/music_placement_v2/controller.py': 'sha256:653778295d54fa210411b1e16676e503da6f80d3542cd55ba153a244c11513e1', 'scripts/music_placement_v2/workflow.py': 'sha256:82cf0b9512b23703aa4d581ed0b29cc301478aec903491dc35347a4cbbc56b20', 'scripts/music_placement_v2/reporting.py': 'sha256:14fcccc74c5c6af5c42c0e5a9e46c1212b9fd31ddef58722e205d420f6e78787', 'scripts/music_placement_v2/spec.py': 'sha256:fe9af81a1bc511f6f7fdb093c650acb221b7d229c3b69bfcb802a5e55293f30e', 'scripts/music_device_trial_v2/controller.py': 'sha256:4c879da4b99320a3b899ec28ba2e36457a569f662fcfa5579153b45ca56f930b', 'scripts/music_device_trial_v2/runtime.py': 'sha256:07cf257954b7c0ff99728264e9b635c45815ad3941b73d1a3621123715884ae8'}
HISTORICAL_HASHES = {'fixtures/music-placement-result-v1/cpu/matrix.json': 'sha256:a25c8b1a2a0b66bb536c5fb02c7ed2fa359bdb62845b36fd445eb4b1bf21896e', 'fixtures/music-placement-result-v1/cpu/development/predictions.jsonl': 'sha256:6f5b4d236d1f8535e34cae27d4ea60066dde529a6cd2fed3b37d97ce4325ea88', 'fixtures/music-placement-result-v1/cpu/control-17/diagnostic-predictions.jsonl': 'sha256:83589d201a2ed03121569d955723a4c5122b666d997811eda061651259c8f4b0', 'fixtures/music-placement-result-v1/cpu/swap-17/diagnostic-predictions.jsonl': 'sha256:652b51f2b8e7b68858dfc7c4082301d278b3bfb9854c6fcdcf21ee6d5eafd692', 'fixtures/music-placement-result-v1/cpu/swap-29/diagnostic-predictions.jsonl': 'sha256:dc106535e2c2b5493bf8ffb684ad82282ca760e437064484cd48ccbab9ffd283', 'fixtures/music-placement-result-v1/cpu/control-29/diagnostic-predictions.jsonl': 'sha256:40c42c1bf4af61393591348b45ab5b78e74a1d916bc6960192b5e4516904cb68'}
CELLS = ('control-17','swap-17','swap-29','control-29')
REQUEST_ID = 'sha256:e5476fa5dd0393578767a928a54bad62ca72cc06bf9f6a7a4f342d6332b4f3fa'
MATRIX_ID = 'sha256:3aefd8026194a557cef3dcbee380e61dbb81f50ffb42dfc264900bf0ea0921e2'


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
    for relative, expected in REFERENCE_SOURCES.items():
        if digest(read(ROOT / relative)) != expected:
            raise ValueError('frozen timing reference source changed')
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


def score(before):
    """Check available bindings and recompute scores; never load omitted tensors."""
    sources()
    from trinite.contracts import parse_json, json_bytes, exact_keys
    from music_placement_v1.reporting import matrix_summary, score_cell
    sources()
    historical = historical_predictions()
    def parse(name):
        return parse_json(before['cpu/'+name], canonical=True)
    def raw_id(name):
        return digest(before['cpu/'+name])
    def same(left, right, label):
        if json_bytes(left) != json_bytes(right):
            raise ValueError(label+' differs')
    def records(name):
        return [parse_json(line, canonical=True) for line in
                before['cpu/'+name].splitlines(keepends=True)]
    matrix, gate, audit = parse('matrix.json'), parse('development-gate.json'), parse('audit.stdout')
    if raw_id('matrix.json') != MATRIX_ID or matrix['request_identity'] != REQUEST_ID:
        raise ValueError('original matrix or owner request link differs')
    claims = {'fresh_confirmation': False, 'learning_adequate': False,
              'phase5_ready': False, 'release_ready': False}
    scope = 'source-bound bytes, safe tensor/progress structure and retained scores; no inference or optimizer replay'
    exact_keys(matrix, {'schema','request_identity','outcome','cells','development','gate_identity','failure','summary','verification_scope','files',*claims}, 'matrix')
    same({k:matrix[k] for k in claims}, claims, 'matrix claims')
    if (matrix['schema'] != 'trinite.music-placement-matrix.v2' or matrix['outcome'] != 'completed'
            or matrix['failure'] is not None or matrix['verification_scope'] != scope):
        raise ValueError('retained matrix context differs')
    exact_keys(matrix['cells'], set(CELLS), 'matrix cells')
    present = {n.removeprefix('cpu/'):digest(b) for n,b in before.items()
               if n.startswith('cpu/') and n not in ('cpu/matrix.json','cpu/audit.stdout')}
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
        if (report['schema'] != 'trinite.music-placement-cell.v2' or report['cell'] != cell
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
        same(completion, {'schema':'trinite.music-placement-completion.v2','request_identity':REQUEST_ID,
                         'cell':cell,'artifacts':{n:matrix['files'][cell+'/'+n] for n in
                         ('diagnostic-predictions.json','diagnostic-predictions.jsonl','final-checkpoint.json','final.safetensors','report.json')}}, 'cell completion')
        for name in ('report.json','diagnostic-predictions.jsonl'):
            if raw_id(cell+'/'+name) != completion['artifacts'][name]:
                raise ValueError('available completion artifact differs')
    for seed in (17,29):
        if initializations['control-'+str(seed)] != initializations['swap-'+str(seed)]:
            raise ValueError('reported paired initialization identity differs')
    same(gate, {'schema':'trinite.music-placement-development-gate.v2','request_identity':REQUEST_ID,
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
        if (attempt['schema'] != 'trinite.music-placement-attempt.v2' or attempt['cell'] != cell
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
    same(report, {'schema':'trinite.music-placement-development.v2','request_identity':REQUEST_ID,
        'gate_identity':matrix['gate_identity'],'summary':summary,
        'generation':{'decodes':480,'forward_calls':sum(len(r['generated_tokens']) for r in development_records),
                      'model_identities_unchanged':True,'training_performed':False},
        'input_files_unchanged':True,'model_states_unchanged':True,'verification_scope':scope,**claims},
        'development report/accounting')
    same(parse('development/completion.json'), {
        'schema':'trinite.music-placement-development-completion.v2','request_identity':REQUEST_ID,
        'gate_identity':matrix['gate_identity'],'artifacts':{
            n:raw_id('development/'+n) for n in ('predictions.jsonl','report.json')}}, 'development completion')
    if (report['request_identity'] != REQUEST_ID or report['gate_identity'] != matrix['gate_identity']
            or report['schema'] != 'trinite.music-placement-development.v2'
            or report['verification_scope'] != scope):
        raise ValueError('development context differs')
    same({k:report[k] for k in claims}, claims, 'development claims')
    same(audit, {'verified':True,'outcome':'completed','matrix_identity':MATRIX_ID,
                 'scope':scope,'inference_replayed':False,'optimizer_replayed':False,**claims}, 'supplied audit receipt')
    # Authenticate standalone stderr as supplied bytes/schema only. It is NOT a
    # source-bound request or proof of execution stage, hardware, or no attempts.
    stderr = parse_json(before['device/freeze.stderr'], canonical=True)
    same(stderr, {'detail':'both trial cells require the pinned CUDA wheel and one available GPU',
                  'outcome':'unsupported','type':'UnsupportedRuntime'}, 'standalone device stderr')
    historical_root = 'fixtures/music-placement-result-v1/cpu/'
    for name in ('development/predictions.jsonl', *(c+'/diagnostic-predictions.jsonl' for c in CELLS)):
        if before['cpu/'+name] != historical[historical_root+name]:
            raise ValueError('historical raw journal differs')
    prior_matrix = parse_json(historical[historical_root+'matrix.json'], canonical=True)
    same(summary, prior_matrix['summary'], 'historical full CPU summary')
    manifest = {}
    for line in before['source-SHA256SUMS'].decode('ascii').splitlines():
        checksum, name = line.split('  ', 1)
        if name in manifest:raise ValueError('duplicate manifest member')
        manifest[name] = 'sha256:'+checksum
    for name in ORIGINAL_HASHES:
        if name != 'source-SHA256SUMS' and manifest.get(name) != digest(before[name]):
            raise ValueError('original handoff manifest link differs')
    check_provenance(parse_json(before['provenance-DERIVED.json'], canonical=True), present, omitted, same)
    if historical_predictions() != historical:raise ValueError('historical fixture changed during verification')
    sources()
    return summary


def historical_predictions():
    result={name:read(ROOT/name) for name in HISTORICAL_HASHES}
    if {name:digest(raw) for name,raw in result.items()}!=HISTORICAL_HASHES:
        raise ValueError('historical CPU fixture changed')
    return result


def check_provenance(provenance,present,omitted,same):
    expected={'schema':'trinite.music-robustness-result-provenance.v1',
        'classification':'DERIVED reports-only robustness repeat; overall incomplete, not a new scientific confirmation',
        'producer_commit':'c3b10965f8bfcc7042e6a9a601e02f5febb694c3','producer_tree':'ec12756812a3f05efdb8a37b1f94c2b2cf63f7b3',
        'source_archive_sha256':'b0120051acd014f115b780816df7edaea40770b9e40faacb39039cc786f46108',
        'included_original_files':sorted(n for n in ORIGINAL_HASHES if n!='source-SHA256SUMS'),
        'omitted_handoff_files':['HANDOFF.md','INVENTORY.txt','REPORT.md'],
        'cpu':{'matrix_bound_present':sorted(present),'matrix_bound_omitted':sorted(omitted),
               'original_matrix_identity':MATRIX_ID,'original_request_identity_owner_receipt':REQUEST_ID},
        'overall_outcome_in_owner_handoff':'incomplete','cpu_outcome_in_supplied_report':'completed',
        'device_outcome_in_supplied_stderr':'unsupported',
        'device_receipt_scope':'standalone supplied freeze stderr; no request, environment or execution binding',
        'device_performance_comparison_available':False,'allocator_policy_validated':False,
        'actual_execution_attested':False,'full_run_custody':False,'hardware_attested':False,'timing_attested':False,
        'inference_replayed':False,'optimizer_replayed':False,'fresh_confirmation':False}
    same(provenance,expected,'derived provenance and scope')


def verify(fixture=FIXTURE):
    before=snapshot(fixture)
    if {name:digest(raw) for name,raw in before.items()}!=FIXTURE_HASHES:
        raise ValueError('preserved fixture bytes changed')
    summary=score(before)
    if snapshot(fixture)!=before:raise ValueError('fixture changed during verification')
    sources()
    return {'schema':'trinite.music-robustness-retained-check.v1','verified':True,
        'scope':'supplied byte integrity, available CPU links, retained-token scoring and historical raw-byte parity; standalone device error text only',
        'overall_outcome':'incomplete','cpu_outcome_in_supplied_report':'completed',
        'device_error_in_supplied_stderr':{'outcome':'unsupported','type':'UnsupportedRuntime'},
        'interpretation':'v2 CPU repeat of v1 development evidence; device lane unsupported in supplied stderr, not a GPU result',
        'decodes_in_journal':summary['decodes'],'generated_tokens':summary['decode_forwards'],
        'available_matrix_bound_files':21,'omitted_matrix_bound_files':22,'development_structural_parents':2,
        'historical_raw_records_equal':800,'historical_summary_equal':True,
        'canonical_exact_by_cell':{c:summary['development']['by_cell'][c]['canonical_primary']['exact_plus_eos'] for c in CELLS},
        'original_raw_custody_verified':False,'actual_execution_attested':False,'hardware_attested':False,'timing_attested':False,
        'device_invocation_attested':False,'device_error_source_bound':False,'allocator_fix_validated':False,
        'partial_custody_failure_path_validated':False,
        'other_attempts_absence_attested':False,'device_performance_comparison_available':False,
        'inference_replayed':False,'optimizer_replayed':False,'fresh_confirmation':False,'learning_adequate':False,'release_ready':False}


def main():
    try:
        if len(sys.argv)!=1:raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(),sort_keys=True))
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(str(error),file=sys.stderr);return 1
    return 0
