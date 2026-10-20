"""Read-only public fixture verification; never execute archived code or weights.

The historical archive is externally committed by FIXTURE_SHA256. The explicit
expected digest parameter supports independently retained commitments, not resealing.
This reader is outside every experiment's historical source closure.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import statistics
import zipfile

from ncm_reference_v1.core import digest, inputs, json_bytes, read_bytes, source_bytes
from ncm_reference_v1.evidence import verify_result, verify_timings

FIXTURE_SHA256 = 'acb310148fdae63081ac603630fd6d58dd49a79a5766114c0b08c822506dc584'
DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/ncm-rtx-smoke-v1'
LIMIT = 4 * 1024 * 1024
REQUESTS = {
    'failed': '34f0df74d21797441b466de382053baf0b0eba2df0575360916499f5ebd2aabc',
    'retest': '764af970c917dee53df7adb07a07781c5fe4919f8fb692dfc5b5478c4e5753c9',
}
ARCHIVES = {
    'failed': '2d6fb9d71008431857a1e755690cb73f73221bd45e04dfda4edea52be9d163e9',
    'run.zip': '0843d0a153629217ef4b37ad8bb4b0070f6722e3af2aedd9f30a1dc2c50054c3',
    'replay.zip': 'e4c7fdb3b89799558a0857c88b57aa1c1fe2e4fa7cb02e22e87a156663645012',
}
OUTERS = {
    'failed': '6ade722cd398e3255c6d50537ce8c9d864c36ac03ecc38e4dbd02bdbe8c9c4c0',
    'retest': '0c03026722197c3a7d059b23e414e1bbd3dd2d4ceb434c9f391287a58180802f',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON literal: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def typed_equal(left, right):
    return json_bytes(left) == json_bytes(right)


def safe_name(name):
    require(isinstance(name, str) and bool(name), 'empty member name')
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and
            all(x not in ('.', '..') for x in path.parts) and
            not any(x in name for x in ('\\', ':', '\x00')) and
            not any(ord(x) < 32 for x in name), 'unsafe member name')


def read_zip(data):
    """Bounded member reads, with no filesystem extraction or archived imports."""
    require(len(data) <= LIMIT, 'compressed archive size limit')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        require(len(infos) <= 100 and sum(x.file_size for x in infos) <= LIMIT,
                'archive cardinality/expanded size limit')
        names = set()
        for item in infos:
            safe_name(item.filename)
            require(item.filename == item.orig_filename, 'truncated archive name')
            require(item.filename.casefold() not in names, 'duplicate/case-fold archive name')
            names.add(item.filename.casefold())
            require(not item.flag_bits & 1 and item.compress_type in (0, 8),
                    'encrypted/unsupported archive member')
            require(stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG),
                    'nonregular archive member')
            require(item.file_size <= 1024 * 1024 and
                    item.file_size <= max(65536, item.compress_size * 1000),
                    'member decompression bound')
        members = {}
        for item in infos:
            with archive.open(item) as stream:
                value = stream.read(item.file_size + 1)
            require(len(value) == item.file_size, 'member length mismatch')
            members[item.filename] = value
        return members


def verify_sources(members, label, identity):
    expected_identity = {
        'failed': ('7a9d2a446cd5bbafd74a6f43f8b5004bf5c2c2d3', 'ade155dc3683b90fbdec2789bcd54ce3fcafba9f'),
        'retest': ('2eb6b3d19a543880e135409ec1f5f0ebec51c35d', '32d7b164f0141132715cd02b37a5d7bae7e99afe'),
    }
    require((identity['commit'], identity['tree']) == expected_identity[label], 'producer Git identity mismatch')
    req_bytes = members['request.json']
    req = parse(req_bytes)
    require(digest(req_bytes) == REQUESTS[label] == identity['request_sha256'],
            'historical request identity mismatch')
    require(json_bytes(req) == req_bytes, 'noncanonical request')
    require(req['schema'] == 'ncm-reference-request-v1' and req['inputs'] == inputs(),
            'formal input or request schema mismatch')
    source = {name[7:]: data for name, data in members.items() if name.startswith('source/')}
    require(req['source'] == {name: digest(data) for name, data in source.items()},
            'source closure mismatch')
    blobs = {name: hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
             for name, data in source.items()}
    require(blobs == identity['git_blobs'], 'source Git blob identity mismatch')
    require(req['protocol'] == parse(source['scripts/ncm_reference_v1/protocol.json']),
            'protocol/source mismatch')
    if label == 'retest':
        require(source == source_bytes(), 'current NCM runtime changed from archived source')
    return req


def verify_completed(members, req):
    """Semantic checks on already bounded, source-bound successful attempt members."""
    source_names = {'source/' + name for name in req['source']}
    progress_names = {f'progress/progress-{i:03d}.json' for i in range(11)}
    require(set(members) == source_names | progress_names |
            {'request.json', 'status.json', 'worker.log', 'result.json', 'timings.json', 'inventory.json'},
            'incomplete or extra completed evidence')
    inventory = parse(members['inventory.json'])
    require(inventory == {name: digest(data) for name, data in members.items() if name != 'inventory.json'},
            'original inventory mismatch')
    require(typed_equal(parse(members['status.json']), {'returncode': 0, 'state': 'completed'}),
            'not a successful recorded attempt')
    result, timings = parse(members['result.json']), parse(members['timings.json'])
    # Authoritative current scorer and schema checker; these unchanged trusted
    # checkout modules do not import any archived source or load a checkpoint.
    verify_result(result, req)
    verify_timings(timings, req)
    outcomes, reconstructions = result['outcomes'], result['reconstructions']
    for row in outcomes + reconstructions:
        require(all(value is True for key, value in row['metrics'].items() if key != 'matching_cells')
                and row['metrics']['matching_cells'] == 81, 'archived exact-target checkpoint changed')
    for row in timings['measurements'][:6]:
        keys = {'phase', 'id', 'prepare_seconds', 'generator_seconds',
                'codec_and_transfer_seconds', 'total_seconds'}
        require(set(row) == keys, 'incomplete component timings')
        require(math.isclose(sum(row[k] for k in keys if k.endswith('_seconds') and k != 'total_seconds'),
                             row['total_seconds'], rel_tol=1e-12, abs_tol=1e-12),
                'timing components do not sum')
    progress = [parse(members[f'progress/progress-{i:03d}.json']) for i in range(11)]
    require(typed_equal(progress[0], dict(stage='loaded', environment=result['environment'],
                                load_seconds=timings['load_seconds'])), 'load progress mismatch')
    for index, row in enumerate(outcomes):
        require(typed_equal(progress[index + 1], dict(outcome=row, timing=timings['measurements'][index])),
                'generation progress mismatch')
    for index, row in enumerate(reconstructions):
        require(typed_equal(progress[index + 7], dict(reconstruction=row, timing=timings['measurements'][index + 6])),
                'reconstruction progress mismatch')
    diagnostics = [parse(line) for line in members['worker.log'].decode().splitlines()
                   if line.startswith('{"loader_contract"')]
    require(len(diagnostics) == 225 and all(row['loader_contract'] == 'ncm-exact-fixed-buffers-v2'
                                          for row in diagnostics), 'loader diagnostic inventory mismatch')
    require(all(type(row['fixed_buffer']) is bool and
                all(type(row[key]) is int and row[key] >= 0 for key in ('nan', 'positive_inf', 'negative_inf'))
                for row in diagnostics), 'diagnostic count/type mismatch')
    require(len({row['tensor'] for row in diagnostics}) == len(diagnostics),
            'duplicate tensor diagnostic')
    fixed = [row for row in diagnostics if row['fixed_buffer']]
    require(typed_equal(fixed, [dict(loader_contract='ncm-exact-fixed-buffers-v2', tensor='attn_mask_for_masking',
                          shape=[1, 1, 413, 413], dtype='torch.float32', nan=0, positive_inf=0,
                          negative_inf=115400, fixed_buffer=True)]), 'fixed mask diagnostics mismatch')
    require(all(row['nan'] == row['positive_inf'] == row['negative_inf'] == 0
                for row in diagnostics if not row['fixed_buffer']), 'unexpected nonfinite tensor diagnostic')
    warm = [row['total_seconds'] * 1000 for row in timings['measurements'] if row['phase'] == 'warm']
    return dict(state='completed', warm_boards=4, warm_valid=4, warm_clues_preserved=4,
                warm_solved=4, warm_exact_target=4, gold_codec_exact_target=4,
                warm_mean_ms=statistics.mean(warm), warm_range_ms=[min(warm), max(warm)],
                cold_load_ms=timings['load_seconds'] * 1000,
                first_inference_ms=timings['measurements'][0]['total_seconds'] * 1000,
                warmup_ms=timings['measurements'][1]['total_seconds'] * 1000,
                peak_allocated_bytes=timings['max_allocated_bytes'],
                peak_reserved_bytes=timings['max_reserved_bytes'],
                memory_scope=timings['memory_scope'])


def verify_failure(members, derivation, req):
    require(derivation['original_archive_sha256'] == ARCHIVES['failed'] and
            derivation['rejected_tensor_identified_in_original'] is False and
            derivation['redaction_count'] == 5, 'failure provenance mismatch')
    mapping = derivation['members']
    original = parse(members['original-inventory.json'])
    require(set(mapping) == set(original) | {'inventory.json'}, 'failure mapping inventory mismatch')
    require(set(members) == {row['derived_member'] for row in mapping.values()},
            'incomplete or extra failure evidence')
    require(len(members) == len(mapping), 'non-bijective failure mapping')
    for name, row in mapping.items():
        safe_name(name)
        dest = row['derived_member']
        require(digest(members[dest]) == row['derived_sha256'] and
                len(members[dest]) == row['derived_size'], 'derived member mismatch')
        require(row['original_sha256'] == (digest(members['original-inventory.json'])
                                          if name == 'inventory.json' else original[name]),
                'original inventory/mapping mismatch')
        if name != 'worker.log':
            require(row['transformation'] == 'byte-identical' and
                    row['original_sha256'] == row['derived_sha256'] and
                    row['original_size'] == row['derived_size'], 'unannounced failure transformation')
        else:
            require(dest == 'worker.redacted.log' and row['transformation'] ==
                    'redact-five-absolute-checkout-prefixes', 'wrong log transformation')
    expected = {'source/' + name for name in req['source']} | {
        'request.json', 'status.json', 'original-inventory.json', 'worker.redacted.log'}
    require(set(members) == expected and typed_equal(parse(members['status.json']),
            {'returncode': 1, 'state': 'failed'}), 'failure state/inventory mismatch')
    log = members['worker.redacted.log'].decode()
    require(log.count('[REDACTED_CHECKOUT]/') == 5 and
            'ValueError: nonfinite checkpoint tensor' in log and
            '/home/' not in log and '/Users/' not in log, 'redacted failure log mismatch')
    return dict(state='failed-before-inference', warm_solved=None, rejected_tensor_identified=False,
                original_error='ValueError: nonfinite checkpoint tensor',
                public_artifact='explicitly-derived-not-original-archive')


def verify_fixture(directory=DEFAULT_FIXTURE, expected_sha256=FIXTURE_SHA256):
    directory = Path(directory)
    require(not directory.is_symlink(), 'symlink fixture directory')
    raw = read_bytes(directory / 'manifest.json', LIMIT)
    require(digest(raw) == expected_sha256, 'external manifest SHA-256 mismatch')
    manifest = parse(raw)
    require(json_bytes(manifest) == raw, 'noncanonical public manifest')
    require(manifest['schema'] == 'ncm-rtx-public-archive-v1' and
            manifest['evidence_class'] == 'derived-public-engineering-fixture', 'wrong archive schema/class')
    expected_files = {'failed-run.public-derived.zip', 'run.zip', 'replay.zip',
                      'failed-verification.json', 'run-verification.json', 'replay-verification.json',
                      'replay-equality.json', 'run-exit.json', 'replay-exit.json'}
    require(set(manifest['files']) == expected_files, 'public file inventory mismatch')
    require({path.name for path in directory.iterdir()} == expected_files | {'manifest.json'},
            'unexpected or missing public file')
    require({name: item['sha256'] for name, item in manifest['outer_uploads'].items()} == OUTERS,
            'original outer provenance mismatch')
    data = {}
    for name, record in manifest['files'].items():
        safe_name(name)
        value = read_bytes(directory / name, LIMIT)
        require(len(value) == record['size'] and digest(value) == record['sha256'], 'public file digest/size mismatch')
        if record['origin']['transformation'] == 'byte-identical':
            require(record['origin']['original_sha256'] == digest(value), 'copied origin digest mismatch')
        data[name] = value
    identity = manifest['source_identities']
    failed = read_zip(data['failed-run.public-derived.zip'])
    failed_req = verify_sources(failed, 'failed', identity['failed'])
    summary = {'failed': verify_failure(failed, manifest['failure_derivation'], failed_req)}
    attempts = {}
    for label in ('run', 'replay'):
        name = label + '.zip'
        require(digest(data[name]) == ARCHIVES[name], 'original successful archive identity mismatch')
        attempts[label] = read_zip(data[name])
        req = verify_sources(attempts[label], 'retest', identity['retest'])
        summary[label] = verify_completed(attempts[label], req)
        receipt = parse(data[label + '-verification.json'])
        require(typed_equal(receipt, dict(inference_verified=False,
                limitation='Producer evidence, not hardware attestation or numerical replay',
                metrics_recomputed=True, samples=4, state='completed', structural_integrity=True,
                warm_solved=4)), 'offline verification receipt mismatch')
        elapsed = parse(data[label + '-exit.json'])
        timings = parse(attempts[label]['timings.json'])
        duration = elapsed['elapsed_seconds']
        require(type(elapsed['exit_code']) is int and elapsed['exit_code'] == 0 and type(duration) in (int, float) and
                math.isfinite(duration) and 0 <= duration < req['protocol']['worker_timeout_seconds'] and
                timings['load_seconds'] + sum(x['total_seconds'] for x in timings['measurements']) < duration,
                'outer CLI duration/worker bound mismatch')
    require(typed_equal(parse(data['failed-verification.json']),
            dict(inference_verified=False, state='failed', structural_integrity=True)), 'failed receipt mismatch')
    require(attempts['run']['result.json'] == attempts['replay']['result.json'], 'recorded replay result mismatch')
    require(typed_equal(parse(data['replay-equality.json']), dict(exact_result_match=True,
            expected_archive=ARCHIVES['run.zip'], replay_archive=ARCHIVES['replay.zip'],
            schema='ncm-reference-replay-v1', scope='same-environment inference, not training replay',
            timing_comparison=False)), 'replay equality receipt mismatch')
    summary.update(exact_recorded_result_match=True, result_sha256=digest(attempts['run']['result.json']),
                   hardware_attestation=False, new_inference_executed=False,
                   original_private_handoffs_reverified=False,
                   limitation=manifest['integrity_limit'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--expected-sha256', default=FIXTURE_SHA256)
    args = parser.parse_args()
    print(json.dumps(verify_fixture(args.fixture, args.expected_sha256), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
