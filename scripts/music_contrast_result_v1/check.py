"""Check reconstructed report tokens only, never model execution or raw custody."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/music-contrast-result-20261028-v1'
LIMIT = 1024 * 1024
FIXTURE_HASHES = {'README.md': 'sha256:adc87c8d82a2f6e0611c49e76ff87f62e9d7f0bbc69048d6c7bd7ec2007884b5', 'provenance.json': 'sha256:28873b3bec92204c857abc70bb86feedd35a1de43c50dc14613c5f7eacf670b8', 'summary.json': 'sha256:efff76dc4e1f0cb2acaf65962d8324013d8cd56a8a5bafc472b468dbb923c46a', 'train-17--step-000-journal.jsonl': 'sha256:354497c03b6e5a6e2f498b0e5cf0b1a5c101908cb4b65b7fc2fdd8908ccbbd8d', 'train-17--step-000-predictions.json': 'sha256:1d90af54fe7d157ce3d329dac2dd131075c1e0c99064de8072979f73e711cd7a', 'train-17--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587', 'train-17--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1', 'train-29--step-000-journal.jsonl': 'sha256:e3a0527b4b3fe6ef8f89118af06ad61b97cc88e435c14fca181bcf4eb2a6b00a', 'train-29--step-000-predictions.json': 'sha256:e07352a75364b6070660e0e8841459d7a1be61c456e1d8bf95a9f5154b82a08a', 'train-29--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587', 'train-29--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1'}
MODULES = {'trinite': 'src/trinite/__init__.py', 'trinite.contracts': 'src/trinite/contracts.py', 'trinite.tokenizer': 'src/trinite/tokenizer.py', 'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py', 'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py', 'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py', 'music_learning_v1': 'scripts/music_learning_v1/__init__.py', 'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py', 'music_contrast_v1': 'scripts/music_contrast_v1/__init__.py', 'music_contrast_v1.spec': 'scripts/music_contrast_v1/spec.py', 'music_contrast_v1.reporting': 'scripts/music_contrast_v1/reporting.py'}
SOURCES = {'src/trinite/__init__.py': 'sha256:029d147c8a741f0cecffc90611d77c6a4047a019863d86ca40992c630a3524b6', 'src/trinite/contracts.py': 'sha256:36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24', 'src/trinite/tokenizer.py': 'sha256:fa87f79f667da8cabeeae0e999add9f420102efbfb746a0c0b275b8703d845bb', 'scripts/symbolic_music_v1/__init__.py': 'sha256:aea01a9d1bd6ae591665b9fcc3658189091a162565851d4c92579731f82405db', 'scripts/symbolic_music_v1/core.py': 'sha256:9c7446c8d275dbc02fe195eac7a660130cf70a6943fa31428c0d4c7f49bba6ff', 'scripts/symbolic_music_v1/oracle.py': 'sha256:3b6c3e00a2caca2a4d2e12d6ab93fda912197d638cec220175a8073b4ee09af0', 'scripts/music_learning_v1/__init__.py': 'sha256:352fce75f453ef87602ff6b451800c6eb765fa2edd2c79d5428013b6f05b5c9a', 'scripts/music_learning_v1/codec.py': 'sha256:5236fab68f2ff11ff4248d1187ddd0db5797138784665eeca4323ff9c192fa0b', 'scripts/music_contrast_v1/__init__.py': 'sha256:3d67f788b234aba0de63cfe5368e9a9dac7d8d8220ecfc5120b5ce708694428d', 'scripts/music_contrast_v1/spec.py': 'sha256:34f051fe47feb48baff3ca636730d9348682812e2ffb4bd3e2a4c07b78cb3317', 'scripts/music_contrast_v1/reporting.py': 'sha256:5f15f9231e3b57e32410a8cd2eb7d66f49565fa13661a680c8327d03fba2b846'}

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
    sources()
    from trinite.contracts import json_bytes, parse_json
    from music_contrast_v1.reporting import summarize
    sources()
    summaries = {}
    tokens = 0
    for seed in (17,29):
        for step in ('000','064'):
            prefix = f'train-{seed}--step-{step}'
            raw = before[prefix+'-predictions.json']
            records = parse_json(raw,canonical=True)['predictions']
            journal = b''.join(json_bytes(row) for row in records)
            if journal != before[prefix+'-journal.jsonl']:
                raise ValueError('prediction/journal copies differ')
            summaries[prefix+'-predictions.json'] = summarize(raw,'train')
            tokens += sum(len(r['generated_tokens']) for r in records)
    gate = all(summaries[f'train-{s}--step-064-predictions.json']['all']['exact_plus_eos']==40 for s in (17,29))
    # This preservation supports precisely the observed failed gate. It cannot
    # invent heldout results or turn positive replacement rows into a new run.
    if gate:
        raise ValueError('retained negative training gate unexpectedly passes')
    return {'schema':'trinite.music-contrast-retained-summary.v1',
        'outcome':'training_fit_failed','training_gate_passed':gate,
        'observations':summaries,
        'heldout':{str(s):{'outcome':'not_attempted','reason':'both-seed training-fit gate not met'} for s in (17,29)},
        'unique_diagnostic_observations':160,'journal_duplicate_records':160,
        'decode_forwards':tokens,'fresh_confirmation':False,'learning_adequate':False,
        'phase5_ready':False,'release_ready':False}


def verify(fixture=FIXTURE):
    before = snapshot(fixture)
    if {n:digest(raw) for n,raw in before.items()} != FIXTURE_HASHES:
        raise ValueError('preserved fixture bytes changed')
    summary = score(before)
    from trinite.contracts import parse_json
    if summary != parse_json(before['summary.json'],canonical=True):
        raise ValueError('derived summary differs')
    provenance = parse_json(before['provenance.json'],canonical=True)
    reconstructed = {n:digest(raw) for n,raw in before.items() if n.startswith('train-')}
    if provenance['reconstructed_prediction_hashes'] != reconstructed or provenance['summary_identity'] != digest(before['summary.json']):
        raise ValueError('preservation provenance links differ')
    if snapshot(fixture) != before:
        raise ValueError('fixture changed during verification')
    sources()
    return {'schema':'trinite.music-contrast-report-check.v1','verified':True,
        'scope':'reconstructed prediction hashes and report-token scoring only',
        'observations':160,'duplicate_journal_rows':160,'outcome':summary['outcome'],
        'final_exact_per_seed':{str(s):summary['observations'][f'train-{s}--step-064-predictions.json']['all']['exact_plus_eos'] for s in (17,29)},
        'training_gate_passed':False,'heldout_model_evaluation':'unrun',
        'original_raw_custody_verified':False,'inference_replayed':False,'optimizer_replayed':False,
        'fresh_confirmation':False,'learning_adequate':False,'phase5_ready':False,'release_ready':False}


def main():
    try:
        if len(sys.argv)!=1:
            raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(),sort_keys=True))
    except (ValueError,OSError) as error:
        print(str(error),file=sys.stderr)
        return 1
    return 0
