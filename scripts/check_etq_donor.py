"""Offline, read-only comparison against the real pinned SONIFICATION runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from trinite.contracts import json_bytes
from trinite.e8_reference import etq

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/e8-etq-reference'
DONOR_BLOBS = {
    'src/etq-model.mjs': '47a4bbc26012344079380c7037b9ad91d4d4e6ca',
    'src/etq-v3-canonical.mjs': '1540f59bb2913d122fb295ee428096cf4c5067f1',
    'src/etq-v3-core.mjs': '7ea74655afdf3154e967de1618e82706bf0505ee',
    'examples/etq-303.v3.canonical.json': '518e6ec67998e1d1fbb828126e599d5e3021c586',
    'LICENSE': '52d135112eeb402d23eeabbcd23513d389553ddd',
}
PRODUCER_PATHS = (
    'scripts/check_etq_donor.py', 'scripts/etq_donor_export.mjs',
    'src/trinite/__init__.py', 'src/trinite/contracts.py',
    'src/trinite/e8_reference/__init__.py', 'src/trinite/e8_reference/etq.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def byte_identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def read_json(path):
    return parse_json(path.read_bytes())


class DonorProcessError(RuntimeError):
    """Keep captured bytes available to callers and visible in CLI tracebacks."""
    def __init__(self, argv, *, returncode=None, timeout=None, stdout=None, stderr=None):
        self.argv = list(argv)
        self.returncode, self.timeout = returncode, timeout
        self.stdout, self.stderr = stdout or b'', stderr or b''
        # repr preserves non-UTF-8 bytes as escapes rather than losing diagnostics.
        super().__init__(f'donor process failed: argv={self.argv!r}; '
                         f'returncode={returncode!r}; timeout={timeout!r}\n'
                         f'stdout={self.stdout!r}\nstderr={self.stderr!r}')


def run_node(argv, *, env, cwd, timeout):
    try:
        result = subprocess.run(argv, env=env, cwd=cwd, capture_output=True,
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise DonorProcessError(argv, timeout=error.timeout,
                                stdout=error.stdout, stderr=error.stderr) from error
    if result.returncode != 0 or result.stderr:
        raise DonorProcessError(argv, returncode=result.returncode,
                                stdout=result.stdout, stderr=result.stderr)
    return result


def compare_output(raw, published):
    """Compare complete live payloads, not just their asserted commitments."""
    output = json.loads(raw)
    require(set(output) == {'commitments', 'eventDocumentCanonical', 'graph', 'registry'},
            'donor output fields differ')
    require(raw == json_bytes(output)[:-1], 'donor output is not canonical')
    expected_registry = {
        'schema': 'qsol.etq-303.site-registry/v1', 'modelId': 'ETQ-303',
        'modelVersion': '3.0.0', 'baseModel': 'ETQ-101@2.0.0',
        'siteCount': 101, 'sites': etq.site_registry(),
    }
    checks = {
        'all_seven_published_commitments':
            json_bytes(output['commitments']) == json_bytes(published) == json_bytes(etq.commitments()),
        'event_document_canonical_bytes':
            output['eventDocumentCanonical'].encode('utf-8') == etq.event_document_bytes(),
        'complete_lifted_graph': json_bytes(output['graph']) == json_bytes(etq.lifted_graph()),
        'complete_site_registry': json_bytes(output['registry']) == json_bytes(expected_registry),
    }
    for name, passed in checks.items():
        require(passed, 'donor comparison failed: ' + name)
    return checks


def compare(fixture=FIXTURE, node='node'):
    """Execute verified snapshot bytes in a temporary directory on every call."""
    prior = (fixture / 'donor-runtime-evidence-v2.json').read_bytes()
    require(prior == json_bytes(parse_json(prior)), 'historical v2 evidence is not canonical')
    blobs, actual_blobs = {}, {}
    for path, expected in DONOR_BLOBS.items():
        raw = (fixture / 'donor' / path).read_bytes()
        digest = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
        require(digest == expected, 'donor Git blob mismatch: ' + path)
        blobs[path] = raw
        actual_blobs[path] = digest
    source = read_json(fixture / 'donor-commitments.json')
    require(source['source'] == {
        'repository': 'QSOLKCB/SONIFICATION', 'commit': etq.SONIFICATION_COMMIT,
        'path': 'examples/etq-303.v3.canonical.json',
        'git_blob': DONOR_BLOBS['examples/etq-303.v3.canonical.json'], 'license': 'MPL-2.0',
    }, 'published fixture source mismatch')
    upstream = json.loads(blobs['examples/etq-303.v3.canonical.json'])['determinism']
    published = {key: upstream[key] for key in etq.commitments()}
    require(json_bytes(source['commitments']) == json_bytes(published), 'published commitments mismatch')
    producer = {path: (ROOT / path).read_bytes() for path in PRODUCER_PATHS}
    # No ambient NODE_OPTIONS preload or NODE_PATH substitution; no package acquisition.
    env = {key: value for key, value in os.environ.items() if not key.startswith('NODE_')}
    with tempfile.TemporaryDirectory(prefix='trinite-etq-') as temporary:
        work = Path(temporary)
        for path, raw in blobs.items():
            target = work / 'donor' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        driver = work / 'export.mjs'
        driver.write_bytes(producer['scripts/etq_donor_export.mjs'])
        version = run_node([node, '--version'], env=env, cwd=work,
                           timeout=10).stdout.decode().strip()
        result = run_node([node, str(driver)], env=env, cwd=work, timeout=30)
    checks = compare_output(result.stdout, published)
    checks['exact_source_blobs'] = actual_blobs == DONOR_BLOBS
    old = read_json(fixture / 'donor-runtime-comparison.json')
    require(old['schema'] == 'trinite.etq-donor-runtime-comparison.v1', 'historical schema mismatch')
    require(old['donor_commit'] == etq.SONIFICATION_COMMIT, 'historical commit mismatch')
    require(old['donor_git_blobs'] == {Path(k).name: v for k, v in DONOR_BLOBS.items() if k.endswith('.mjs')},
            'historical source blobs mismatch')
    require(json_bytes(old['checks']) == json_bytes(checks), 'historical checks mismatch')
    require(json_bytes(old['commitments']) == json_bytes(published), 'historical commitments mismatch')
    record = {
        'schema': 'trinite.etq-donor-runtime-evidence.v3',
        'node_version': version,
        'historical_evidence_v2': byte_identity(prior),
        'donor_commit': etq.SONIFICATION_COMMIT,
        'donor_files': {path: dict(byte_identity(raw), git_blob=DONOR_BLOBS[path])
                        for path, raw in blobs.items()},
        'producer_files': {path: byte_identity(raw) for path, raw in producer.items()},
        'historical_record': byte_identity((fixture / 'donor-runtime-comparison.json').read_bytes()),
        'commitments_record': byte_identity((fixture / 'donor-commitments.json').read_bytes()),
        'stdout': byte_identity(result.stdout), 'checks': checks, 'commitments': published,
    }
    return record, result.stdout, version


def verify(fixture=FIXTURE, node='node'):
    expected_raw = (fixture / 'donor-runtime-evidence-v3.json').read_bytes()
    expected = parse_json(expected_raw)
    require(expected_raw == json_bytes(expected), 'retained runtime evidence mismatch: noncanonical bytes')
    record, stdout, version = compare(fixture, node)
    require(version == expected['node_version'],
            f"Node version mismatch: expected {expected['node_version']!r}, actual {version!r}")
    require(expected_raw == json_bytes(record), 'retained runtime evidence mismatch')
    require((fixture / 'donor-runtime-output.json').read_bytes() == stdout,
            'retained donor stdout mismatch')
    return record


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True, indent=2))
