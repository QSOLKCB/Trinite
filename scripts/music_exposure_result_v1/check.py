"""Authenticate supplied report tokens and rescore only; no execution or closed custody."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/music-exposure-result-20261029-v1'
HISTORICAL = ROOT / 'fixtures/music-contrast-result-20261028-v1'
LIMIT = 1024 * 1024
FIXTURE_HASHES = {'README.md': 'sha256:671d04f0e70e823cf56feb3e5f8adb2a503fad56a263375663b10f46ac0fae5f', 'provenance.json': 'sha256:53558b164ffa7a8e77ae2d297890b3743dd238beefc0bbadc41e992ccd5c9e5b', 'summary.json': 'sha256:18cfff95b2680ed1b14601aa7984c3f192e75aae5762dac582cf8d49c0ce78c2', 'train-17--step-000-journal.jsonl': 'sha256:354497c03b6e5a6e2f498b0e5cf0b1a5c101908cb4b65b7fc2fdd8908ccbbd8d', 'train-17--step-000-predictions.json': 'sha256:1d90af54fe7d157ce3d329dac2dd131075c1e0c99064de8072979f73e711cd7a', 'train-17--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587', 'train-17--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1', 'train-17--step-640-journal.jsonl': 'sha256:4a1a05c03980040af21cc4f4c23459aedac0b282a0f17c9bc6bd25d62c31d5df', 'train-17--step-640-predictions.json': 'sha256:503c0f780351647c1a127c7cd4516d6f7f7309853cfa69b9c2a8d7c57f9a35c1', 'train-29--step-000-journal.jsonl': 'sha256:e3a0527b4b3fe6ef8f89118af06ad61b97cc88e435c14fca181bcf4eb2a6b00a', 'train-29--step-000-predictions.json': 'sha256:e07352a75364b6070660e0e8841459d7a1be61c456e1d8bf95a9f5154b82a08a', 'train-29--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587', 'train-29--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1', 'train-29--step-640-journal.jsonl': 'sha256:4a1a05c03980040af21cc4f4c23459aedac0b282a0f17c9bc6bd25d62c31d5df', 'train-29--step-640-predictions.json': 'sha256:503c0f780351647c1a127c7cd4516d6f7f7309853cfa69b9c2a8d7c57f9a35c1'}
MODULES = {'trinite': 'src/trinite/__init__.py', 'trinite.contracts': 'src/trinite/contracts.py', 'trinite.tokenizer': 'src/trinite/tokenizer.py', 'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py', 'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py', 'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py', 'music_learning_v1': 'scripts/music_learning_v1/__init__.py', 'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py', 'music_contrast_v1': 'scripts/music_contrast_v1/__init__.py', 'music_contrast_v1.spec': 'scripts/music_contrast_v1/spec.py', 'music_contrast_v1.reporting': 'scripts/music_contrast_v1/reporting.py', 'trinite.foundations_plan': 'src/trinite/foundations_plan.py', 'music_learning_v1.numerical': 'scripts/music_learning_v1/numerical.py', 'music_exposure_v1': 'scripts/music_exposure_v1/__init__.py', 'music_exposure_v1.spec': 'scripts/music_exposure_v1/spec.py', 'music_exposure_v1.reporting': 'scripts/music_exposure_v1/reporting.py'}
SOURCES = {'src/trinite/__init__.py': 'sha256:029d147c8a741f0cecffc90611d77c6a4047a019863d86ca40992c630a3524b6', 'src/trinite/contracts.py': 'sha256:36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24', 'src/trinite/tokenizer.py': 'sha256:fa87f79f667da8cabeeae0e999add9f420102efbfb746a0c0b275b8703d845bb', 'scripts/symbolic_music_v1/__init__.py': 'sha256:aea01a9d1bd6ae591665b9fcc3658189091a162565851d4c92579731f82405db', 'scripts/symbolic_music_v1/core.py': 'sha256:9c7446c8d275dbc02fe195eac7a660130cf70a6943fa31428c0d4c7f49bba6ff', 'scripts/symbolic_music_v1/oracle.py': 'sha256:3b6c3e00a2caca2a4d2e12d6ab93fda912197d638cec220175a8073b4ee09af0', 'scripts/music_learning_v1/__init__.py': 'sha256:352fce75f453ef87602ff6b451800c6eb765fa2edd2c79d5428013b6f05b5c9a', 'scripts/music_learning_v1/codec.py': 'sha256:5236fab68f2ff11ff4248d1187ddd0db5797138784665eeca4323ff9c192fa0b', 'scripts/music_contrast_v1/__init__.py': 'sha256:3d67f788b234aba0de63cfe5368e9a9dac7d8d8220ecfc5120b5ce708694428d', 'scripts/music_contrast_v1/spec.py': 'sha256:34f051fe47feb48baff3ca636730d9348682812e2ffb4bd3e2a4c07b78cb3317', 'scripts/music_contrast_v1/reporting.py': 'sha256:5f15f9231e3b57e32410a8cd2eb7d66f49565fa13661a680c8327d03fba2b846', 'src/trinite/foundations_plan.py': 'sha256:f4a4cee055b97e6aa1217340b1a103cd561caeaca01e6b358e71eaedbab8a8d7', 'scripts/music_learning_v1/numerical.py': 'sha256:214fa9dfd4490cedb7a55489331db1ad550b8b5dfdf5021294a117d470f20813', 'scripts/music_exposure_v1/__init__.py': 'sha256:36ac2634a27e9076d885362c988335f154cd071ae167b6e7c8d1a481df9f5c6d', 'scripts/music_exposure_v1/spec.py': 'sha256:c80b16e68ab85ebdfe06a40e406941b3e19ae5596ca7945babac9df8e35448d9', 'scripts/music_exposure_v1/reporting.py': 'sha256:09c0783716ee95c8b0e7cdfbbcfa748e428ae76980ac23f7bd75df8f25763e74'}
HISTORICAL_HASHES = {'train-17--step-000-journal.jsonl': 'sha256:354497c03b6e5a6e2f498b0e5cf0b1a5c101908cb4b65b7fc2fdd8908ccbbd8d', 'train-17--step-000-predictions.json': 'sha256:1d90af54fe7d157ce3d329dac2dd131075c1e0c99064de8072979f73e711cd7a', 'train-17--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587', 'train-17--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1', 'train-29--step-000-journal.jsonl': 'sha256:e3a0527b4b3fe6ef8f89118af06ad61b97cc88e435c14fca181bcf4eb2a6b00a', 'train-29--step-000-predictions.json': 'sha256:e07352a75364b6070660e0e8841459d7a1be61c456e1d8bf95a9f5154b82a08a', 'train-29--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587', 'train-29--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1'}

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


def snapshot(fixture):
    fixture = Path(fixture).absolute()
    if not stat.S_ISDIR(fixture.lstat().st_mode):
        raise ValueError('non-link fixture directory required')
    if set(p.name for p in fixture.iterdir()) != set(FIXTURE_HASHES):
        raise ValueError('derivative fixture membership differs')
    return {name: read(fixture / name) for name in FIXTURE_HASHES}


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


def score(before):
    """Recompute the complete six-observation report with the unchanged scorer."""
    sources()
    from trinite.contracts import json_bytes, parse_json
    from music_exposure_v1.reporting import summarize
    from music_exposure_v1.spec import schedule_receipt
    sources()
    summaries = {}
    tokens = 0
    for seed in (17, 29):
        for step in ('000', '064', '640'):
            prefix = f'train-{seed}--step-{step}'
            raw = before[prefix+'-predictions.json']
            records = parse_json(raw, canonical=True)['predictions']
            journal = b''.join(json_bytes(row) for row in records)
            if journal != before[prefix+'-journal.jsonl']:
                raise ValueError('prediction/journal copies differ')
            result = summarize(raw)
            expected = {'000': (0, 0, 0), '064': (20, 0, 0), '640': (40, 10, 40)}[step]
            observed = tuple(result['all'][k] for k in ('exact_plus_eos', 'complete_groups', 'both_correct_edges'))
            if observed != expected:
                raise ValueError('retained training-fit trajectory differs')
            summaries[prefix+'-predictions.json'] = result
            tokens += sum(len(r['generated_tokens']) for r in records)
    return {'schema': 'trinite.music-exposure-retained-summary.v1',
        'outcome': 'training_fit_passed', 'both_seeds_fit': True,
        'observations': summaries,
        'declared_schedules': {str(step): schedule_receipt(step) for step in (0, 64, 640)},
        'schedule_scope': 'static regenerated protocol schedule, not observed optimizer execution',
        'heldout_model_evaluation': 'unrun; outside this train-only protocol',
        'unique_diagnostic_observations': 240, 'journal_duplicate_records': 240,
        'decode_forwards': tokens, 'fresh_confirmation': False, 'learning_adequate': False,
        'phase5_ready': False, 'release_ready': False}


def historical_parity(before):
    """Check original retained step0/64 bytes, not original model execution."""
    result = {}
    for name, expected in HISTORICAL_HASHES.items():
        raw = read(HISTORICAL/name)
        if digest(raw) != expected or before[name] != raw:
            raise ValueError('historical step0/64 byte parity differs')
        result[name] = expected
    return result


def verify(fixture=FIXTURE):
    before = snapshot(fixture)
    if {n: digest(raw) for n, raw in before.items()} != FIXTURE_HASHES:
        raise ValueError('preserved fixture bytes changed')
    parity = historical_parity(before)
    summary = score(before)
    from trinite.contracts import json_bytes, parse_json
    if json_bytes(summary) != before['summary.json']:
        raise ValueError('derived summary differs')
    provenance = parse_json(before['provenance.json'], canonical=True)
    supplied = {n: digest(raw) for n, raw in before.items() if n.startswith('train-')}
    if (provenance['supplied_prediction_journal_hashes'] != supplied
            or provenance['summary_identity'] != digest(before['summary.json'])
            or provenance['historical_step0_64_byte_parity'] != parity):
        raise ValueError('preservation provenance links differ')
    if snapshot(fixture) != before:
        raise ValueError('fixture changed during verification')
    if historical_parity(before) != parity:
        raise ValueError('historical fixture changed during verification')
    sources()
    return {'schema': 'trinite.music-exposure-report-check.v1', 'verified': True,
        'scope': 'supplied prediction/journal byte integrity, historical prediction parity and retained-token scoring only',
        'observations': 240, 'duplicate_journal_rows': 240, 'outcome': summary['outcome'],
        'final_exact_per_seed': {str(s): summary['observations'][f'train-{s}--step-640-predictions.json']['all']['exact_plus_eos'] for s in (17,29)},
        'both_seeds_fit': True, 'historical_step0_64_files_equal': 8,
        'heldout_model_evaluation': 'unrun', 'original_raw_custody_verified': False,
        'inference_replayed': False, 'optimizer_replayed': False,
        'fresh_confirmation': False, 'learning_adequate': False, 'phase5_ready': False, 'release_ready': False}


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(), sort_keys=True))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
