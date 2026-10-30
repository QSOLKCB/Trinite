"""Preserve a reproduced offline task audit and supplied CUDA observations only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/music-task-audit-result-v1'
LIMIT = 4 * 1024 * 1024
ORIGINAL_HASHES = {'music-task-audit.json': 'sha256:15217b97421c53d0d75522b43775292764b510657d002621142611c5e2caad96', 'cuda-availability.json': 'sha256:4a551d4525280c9a222c109810f4c339e295b23f476d4b27f4a1096e7f86a03a', 'music-task-audit.exit': 'sha256:9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa', 'cuda-availability.exit': 'sha256:9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa', 'source-SHA256SUMS': 'sha256:142b811d613eb6546eebd36bf97ee74360f1d32539c6e803af166d2288419bb4'}
FIXTURE_HASHES = {'README.md': 'sha256:ec1c8c0b748fb946bd0f05064b540f85951064bda9ee8a48d52fb0da0f3358b3', 'cuda-availability.exit': 'sha256:9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa', 'cuda-availability.json': 'sha256:4a551d4525280c9a222c109810f4c339e295b23f476d4b27f4a1096e7f86a03a', 'music-task-audit.exit': 'sha256:9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa', 'music-task-audit.json': 'sha256:15217b97421c53d0d75522b43775292764b510657d002621142611c5e2caad96', 'provenance-DERIVED.json': 'sha256:189814ff5708bcb2665bdd130a6a718f156a7634698101ca692c019d81a040d8', 'source-SHA256SUMS': 'sha256:142b811d613eb6546eebd36bf97ee74360f1d32539c6e803af166d2288419bb4'}
MODULES = {'music_task_audit_v1': 'scripts/music_task_audit_v1/__init__.py', 'music_task_audit_v1.check': 'scripts/music_task_audit_v1/check.py', 'music_task_audit_v1.analysis': 'scripts/music_task_audit_v1/analysis.py'}
SOURCES = {'scripts/music_task_audit_v1/__init__.py': 'sha256:a446367b088afc819f3936f0721abc3dbc250e2605c8253598bd80f9d4d3cbf6', 'scripts/music_task_audit_v1/check.py': 'sha256:9af64e0a13188b0d675d9140c0f56422aab2fcea4c9e1f665db32f06836639d8', 'scripts/music_task_audit_v1/analysis.py': 'sha256:9f81df4dadd97b1e55302f7afd2926ccce069895570d1547b676b36f14a52b4a', 'scripts/check_music_task_audit.py': 'sha256:39a95e303768b21ecba8698c7b64531080b50394b710d73e9864fb201cdf933f'}
DIAGNOSTIC_SOURCES = {'scripts/check_cuda_availability.py': 'sha256:326cf66f6034c252c63dc983234b0bd8efa20d577c0de1dd5ad2e75627431ab0', 'scripts/cuda_availability_v1/__init__.py': 'sha256:68ff8398940207e3d332f3570b17e053b58637b8fc7bfe11cb32fdf4eaf6c66b', 'scripts/cuda_availability_v1/check.py': 'sha256:0cdd6057e085e625877bed07bdf8881e29c10ee9dba229b619562501dd849f70'}


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

OMITTED_MANIFEST_HASHES = {'HANDOFF.md':'sha256:936d670052b8b1fe4b7929e37295f95bcca73ec73cee254bc39ecc69c7db6f57',
                           'REPORT.md':'sha256:6d384c333f35f09dfcd9ad561dd1a5db8910097644f82fb7dca59ab1092b56d3'}

CUDA_REPORT_ID = 'sha256:04bdd2f2a2a261be27e8fe9937562aa082143f5b06681313e7a3bce8c64ec7b0'


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def parse(raw):
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON field')
            result[key]=value
        return result
    def invalid_constant(value):raise ValueError('non-finite JSON constant')
    value=json.loads(raw,object_pairs_hook=object_pairs,parse_constant=invalid_constant)
    if canonical(value)!=raw:raise ValueError('noncanonical retained JSON')
    return value


def sources():
    """Pin historical audit before import; never import CUDA diagnostic modules."""
    for name,expected in {**SOURCES,**DIAGNOSTIC_SOURCES}.items():
        if digest(read(ROOT/name))!=expected:raise ValueError('frozen source changed')
    package=ROOT/'scripts/music_task_audit_v1'
    if package.is_symlink():raise ValueError('linked historical audit package')
    for entry in package.rglob('*'):
        if entry.is_symlink() or entry.name=='__pycache__' or entry.suffix in ('.pyc','.pyo','.so','.pyd'):
            raise ValueError('historical audit import caches/shadows forbidden')
    for name,relative in MODULES.items():
        path=ROOT/relative
        if path.name!='__init__.py' and (path.parent/path.stem).exists():
            raise ValueError('historical audit package shadow forbidden')
        loaded=sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded,'__file__','')).absolute()!=path:
            raise ValueError('foreign historical audit module')
    # Parent initializer bytes authenticated above before find_spec may import it.
    for name,relative in MODULES.items():
        spec=importlib.util.find_spec(name)
        if spec is None or spec.origin!=str(ROOT/relative):raise ValueError('foreign historical audit import resolution')


def validate_cuda(wrapper):
    """Validate supplied report semantics, not the machine or invocation."""
    expected_report={
        'schema':'trinite.cuda-availability.v1',
        'scope':'local observations only; no explanation of an earlier failure or proof of hardware health',
        'platform':'linux','cuda_visible_devices':'unset',
        'python':{'executable_symlink':True,'implementation':'cpython',
                  'interpreter_selection':'same absolute lexical executable; path withheld','version':'3.12.13','virtual_environment':True},
        'metadata':{'distributions':{'numpy':{'status':'present','version':'2.3.5'},
                    'safetensors':{'status':'present','version':'0.6.2'},'torch':{'status':'present','version':'2.8.0+cu128'}},
                    'returncode':0,'status':'completed','stderr_observed':False},
        # There is intentionally NO returncode: the utility was not executed.
        'nvidia_smi':{'status':'unsafe_executable'},
        'cuda_query':{'available':False,'compiled_cuda':'12.8','device_count':0,'returncode':0,
                     'status':'no_visible_cuda','stderr_observed':True,'torch_version':'2.8.0+cu128'},
        'limits':{'combined_output_bytes_per_query':16384,'max_devices':16,'max_queries':3,'seconds_per_query':10},
        'source_files':dict(DIAGNOSTIC_SOURCES),'driver_initialization_possible':True,
        'root_cause':'unknown','gpu_health_established':False,'hardware_attested':False,
        'model_executed':False,'repair_attempted':False,'tensor_allocation_requested':False,
        'training_performed':False,'trial_admission_established':False}
    if type(wrapper) is not dict or set(wrapper)!={'report','report_identity'}:
        raise ValueError('supplied CUDA wrapper fields differ')
    if canonical(wrapper['report'])!=canonical(expected_report):
        raise ValueError('supplied CUDA status/schema/scope differs')
    if wrapper['report_identity']!=CUDA_REPORT_ID or digest(canonical(wrapper['report']))!=wrapper['report_identity']:
        raise ValueError('supplied CUDA canonical identity differs')
    return wrapper['report']


def score(before):
    sources()
    cuda=validate_cuda(parse(before['cuda-availability.json']))
    for name in ('music-task-audit.exit','cuda-availability.exit'):
        if before[name]!=b'0\n':raise ValueError('supplied launcher exit receipt differs')
    manifest={}
    for line in before['source-SHA256SUMS'].decode('ascii').splitlines():
        checksum,name=line.split('  ',1)
        if len(checksum)!=64 or any(character not in '0123456789abcdef' for character in checksum):
            raise ValueError('invalid handoff manifest checksum')
        if name in manifest:raise ValueError('duplicate handoff manifest member')
        manifest[name]='sha256:'+checksum
    expected_names=(set(ORIGINAL_HASHES)-{'source-SHA256SUMS'})|set(OMITTED_MANIFEST_HASHES)
    if set(manifest)!=expected_names:raise ValueError('handoff manifest membership differs')
    if {name:manifest[name] for name in OMITTED_MANIFEST_HASHES}!=OMITTED_MANIFEST_HASHES:
        raise ValueError('omitted handoff narrative commitment differs')
    for name in ORIGINAL_HASHES:
        if name!='source-SHA256SUMS' and manifest.get(name)!=digest(before[name]):
            raise ValueError('original handoff manifest link differs')
    check_provenance(parse(before['provenance-DERIVED.json']))
    retained=parse(before['music-task-audit.json'])
    # The pinned historical wrapper authenticates its complete unchanged
    # source/input closure and invokes the original authoritative analysis.
    from music_task_audit_v1 import check as original
    original.sources()
    reproduced=original.verify()
    if canonical(reproduced)!=before['music-task-audit.json']:
        raise ValueError('complete offline audit report differs')
    original.sources();sources()
    return {'audit':retained,'cuda':cuda}


PROVENANCE = {'claims': {'absence_of_other_attempts_attested': False, 'actual_invocation_attested': False, 'causal_mechanism_established': False, 'fresh_confirmation': False, 'full_original_custody_verified': False, 'gpu_health_established': False, 'gpu_root_cause_established': False, 'hardware_attested': False, 'historical_namespace_equivalence_attested': False, 'learning_adequate': False, 'new_training_result': False, 'release_ready': False, 'runtime_replayed': False}, 'diagnostic_report_identity': 'sha256:04bdd2f2a2a261be27e8fe9937562aa082143f5b06681313e7a3bce8c64ec7b0', 'evidence_class': 'DERIVED', 'omitted_manifest_payloads': ['HANDOFF.md', 'REPORT.md'], 'originals': {'cuda-availability.exit': 'sha256:9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa', 'cuda-availability.json': 'sha256:4a551d4525280c9a222c109810f4c339e295b23f476d4b27f4a1096e7f86a03a', 'music-task-audit.exit': 'sha256:9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa', 'music-task-audit.json': 'sha256:15217b97421c53d0d75522b43775292764b510657d002621142611c5e2caad96'}, 'owner_archive_sha256': 'c2988c3748c8f021930a0e42ffed884ef159f1dd5e5a4bc9fa958d7a95a338b5', 'owner_reported_context': {'diagnostic_restricted_without_escalation': True, 'earlier_clone_failure_exit': 128, 'historical_namespace_equivalence_asserted': False, 'later_clone_exit': 0, 'same_failed_preparation_lexical_python': True, 'single_audit_and_explicit_cuda_query': True, 'source_acquisition_only_received_approval': True, 'source_and_environment_unchanged': True, 'status': 'owner narrative assertions; not independently established by public receipts'}, 'private_omissions': ['raw stderr', 'interpreter and environment details', 'source verification manifests', 'operational commands and failed preparation records'], 'schema': 'trinite.music-task-audit-result-provenance.v1', 'source_merge': '7dbe25f7c26f15836de584fb9a0690119d62b28c', 'source_tree': '5f43900f391d5167fefc5edceb581984deafd23d', 'unchanged_manifest': 'source-SHA256SUMS'}


def check_provenance(value):
    if canonical(value)!=canonical(PROVENANCE):
        raise ValueError('derived provenance/scope differs')


def verify(fixture=FIXTURE):
    before=snapshot(fixture)
    if {name:digest(raw) for name,raw in before.items()}!=FIXTURE_HASHES:
        raise ValueError('preserved fixture bytes changed')
    result=score(before)
    if snapshot(fixture)!=before:raise ValueError('fixture changed during verification')
    sources()
    analysis=result['audit']['analysis'];cuda=result['cuda']
    return {'schema':'trinite.music-task-audit-retained-check.v1','verified':True,
        'scope':'complete deterministic offline task-report reproduction and supplied CUDA observation consistency; no live diagnostic or model calls',
        'complete_offline_report_reproduced':True,'original_audit_exit_receipt':0,'cuda_diagnostic_exit_receipt':0,
        'cuda_report_identity':CUDA_REPORT_ID,'retained_record_occurrences':analysis['retained']['total_retained_records'],
        'retained_sections':len(analysis['retained']['sections']),'structural_development_parents':2,
        'nvidia_smi_status':cuda['nvidia_smi']['status'],'nvidia_smi_executed_in_supplied_receipt':False,
        'cuda_query_status':cuda['cuda_query']['status'],'cuda_available':cuda['cuda_query']['available'],
        'cuda_device_count':cuda['cuda_query']['device_count'],'cuda_query_stderr_observed':cuda['cuda_query']['stderr_observed'],
        'cuda_root_cause':'unknown',
        'diagnostic_interpretation':'supplied observations only; unsafe_executable means utility was skipped, not that it was malware; launcher exit zero means receipt production, not GPU availability',
        'actual_execution_attested':False,'hardware_attested':False,'runtime_attested':False,
        'historical_namespace_equivalence_attested':False,'inference_replayed':False,'training_performed':False,
        'causal_mechanism_established':False,'absence_of_other_attempts_attested':False,
        'full_original_custody_verified':False,'fresh_confirmation':False,'learning_adequate':False,'release_ready':False}


def main():
    try:
        if len(sys.argv)!=1:raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(),sort_keys=True,allow_nan=False))
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(str(error),file=sys.stderr);return 1
    return 0
