# SPDX-License-Identifier: MPL-2.0
"""TT-003-FIT-001-EVIDENCE-REVIEW-001: offline, read-only independent auditor.

Only Python standard library. Never imports or executes archived source/weights,
loads native checkpoints, evaluates heldouts, or updates an optimizer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import zipfile

SEEDS = (17, 29, 43)
STEPS = (0, 64, 128, 192, 256)
EOS, BOS, VOCAB = 257, 256, 259
RESIDUES = (48, 49, 50)
VECTOR = struct.Struct('<259f')
MAX_ENTRIES = 2_000
MAX_UNCOMPRESSED = 256 * 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024
TOL = 1e-12
SHA_RE = re.compile(r'sha256:[0-9a-f]{64}\Z')


class EvidenceError(ValueError):
    """A claimed, complete evidence record failed its independent contract."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha(raw: bytes) -> str:
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def anchored_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return 'sha256:' + h.hexdigest()


def parse_anchor(raw: bytes) -> str:
    """Parse external SHA256SUMS style sidecar, not its untrusted filename."""
    try:
        text = raw.decode('ascii').strip()
    except UnicodeError as exc:
        raise EvidenceError('non-ASCII sidecar') from exc
    match = re.fullmatch(r'(?:sha256:)?([0-9a-f]{64})(?:[ \t]+\*?[^\r\n]+)?', text)
    need(match is not None, 'invalid or multi-entry external SHA-256 sidecar')
    return 'sha256:' + match.group(1)


def safe_archive(path: Path, expected_identity: str) -> tuple[zipfile.ZipFile, dict]:
    need(type(expected_identity) is str and SHA_RE.fullmatch(expected_identity) is not None,
         'exact externally supplied archive SHA-256 required')
    need(anchored_hash(path) == expected_identity, 'archive SHA-256 differs from external anchor')
    archive = zipfile.ZipFile(path)
    names, hashes, total = set(), {}, 0
    try:
        infos = archive.infolist()
        need(1 <= len(infos) <= MAX_ENTRIES, 'archive member count outside bound')
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name).parts
            need(name not in names and not name.startswith('/') and '\\' not in name
                 and ':' not in name and parts and all(s not in ('', '.', '..') for s in parts)
                 and PurePosixPath(name).as_posix() == name and not info.is_dir(),
                 'duplicate, unsafe or directory ZIP member')
            mode = (info.external_attr >> 16) & 0o170000
            need(mode in (0, stat.S_IFREG), 'symlink/non-file ZIP member rejected')
            need(info.file_size <= MAX_MEMBER and info.file_size >= 0,
                 'oversized ZIP member')
            total += info.file_size
            need(total <= MAX_UNCOMPRESSED, 'oversized decompressed archive')
            # Reading verifies ZIP CRC in addition to independently hashing bytes.
            raw = archive.read(info)
            need(len(raw) == info.file_size, 'ZIP member length mismatch')
            hashes[name] = {'identity': sha(raw), 'bytes': len(raw)}
            names.add(name)
    except BaseException:
        archive.close()
        raise
    return archive, hashes


def jdoc(archive: zipfile.ZipFile, name: str) -> dict | list:
    try:
        raw = archive.read(name)
        value = json.loads(raw)
    except (KeyError, ValueError, UnicodeError) as exc:
        raise EvidenceError('invalid/missing JSON: ' + name) from exc
    need(raw == canonical(value), 'noncanonical JSON: ' + name)
    return value


def jlines(archive: zipfile.ZipFile, name: str) -> list[dict]:
    try:
        raw = archive.read(name)
    except KeyError as exc:
        raise EvidenceError('missing JSONL: ' + name) from exc
    need(raw.endswith(b'\n'), 'incomplete JSONL: ' + name)
    out = []
    for line in raw.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise EvidenceError('malformed JSONL: ' + name) from exc
        need(type(value) is dict and line == canonical(value), 'noncanonical JSONL: ' + name)
        out.append(value)
    return out


def members_check(hashes: dict, prefix: str, recorded: list[dict],
                  extra: tuple[str, ...] = ()) -> None:
    need(type(recorded) is list, 'expected manifest member array')
    expected = {}
    for r in recorded:
        need(type(r) is dict and set(r) == {'name', 'identity', 'bytes'}
             and type(r['name']) is str and r['name'] not in expected,
             'invalid or duplicate member inventory')
        expected[r['name']] = {'identity': r['identity'], 'bytes': r['bytes']}
    actual = {p[len(prefix):]: v for p, v in hashes.items() if p.startswith(prefix)}
    for key in extra:
        actual.pop(key, None)
    need(actual == expected, 'inventory/hash mismatch at ' + prefix)


def panel_task(row: dict) -> str:
    task = row['observation']['task']
    return 'check_' + row['answer'] if task == 'check' else task


def classify(tokens: list[int], row: dict) -> dict:
    task, answer = row['observation']['task'], row['answer']
    need(type(tokens) is list and 1 <= len(tokens) <= 5
         and all(type(t) is int and 0 <= t < VOCAB for t in tokens)
         and task in ('check', 'apply', 'inverse'), 'invalid generated sequence')
    expected_len = 1 if task == 'check' else 4
    need(len(answer) == expected_len and set(answer) <= set('01' if task == 'check' else '012'),
         'invalid TRAIN answer')
    need(EOS not in tokens[:-1] and (tokens[-1] == EOS or len(tokens) == 5),
         'greedy sequence has invalid termination')
    eos = tokens[-1] == EOS
    body = tokens[:-1] if eos else tokens
    decoded = bytes(body).decode('ascii') if all(t < 128 for t in body) else None
    permitted = set('01' if task == 'check' else '012')
    valid = eos and decoded is not None and len(decoded) == expected_len and set(decoded) <= permitted
    exact = tokens == [*answer.encode('ascii'), EOS]
    if exact:
        category = 'exact'
    elif decoded is None or set(decoded) - permitted:
        category = 'invalid_symbol'
    elif not eos:
        category = 'missing_eos'
    elif len(body) < expected_len:
        category = 'premature_eos'
    elif len(body) > expected_len:
        category = 'excess_length'
    else:
        category = 'wrong_answer'
    return {'tokens': tokens, 'decoded': decoded, 'eos': eos,
            'answer_length': len(body), 'expected_length': expected_len,
            'format_valid': valid, 'exact': exact, 'classification': category}


def score(panel: list[dict], predictions: list[dict]) -> dict:
    need(len(panel) == len(predictions) == 72, 'incomplete panel/predictions')
    groups, classified = {}, []
    for row, pred in zip(panel, predictions):
        need(set(pred) == {'id', 'tokens'} and pred['id'] == row['id'],
             'prediction ID/order mismatch')
        tokens = pred['tokens']
        r = classify(tokens, row)
        classified.append(r)
        for key in ('all', panel_task(row), 'circuit:' + row['family']):
            group = groups.setdefault(key, {'rows': 0, 'exact': 0, 'valid': 0,
                'premature_eos': 0, 'missing_eos': 0, 'invalid': 0,
                'wrong_length': 0, 'eos_positions': {}, 'outputs': {}})
            group['rows'] += 1
            group['exact'] += int(r['exact'])
            group['valid'] += int(r['format_valid'])
            eos_pos = tokens.index(EOS) if EOS in tokens else None
            group['premature_eos'] += int(eos_pos is not None and eos_pos < len(row['answer']))
            group['missing_eos'] += int(eos_pos is None)
            group['invalid'] += int(r['classification'] == 'invalid_symbol')
            group['wrong_length'] += int(r['answer_length'] != len(row['answer']))
            k = str(eos_pos)
            group['eos_positions'][k] = group['eos_positions'].get(k, 0) + 1
            k = ','.join(map(str, tokens))
            group['outputs'][k] = group['outputs'].get(k, 0) + 1
    for value in groups.values():
        value['distinct_outputs'] = len(value['outputs'])
    positions = []
    vectors = [(r, p) for r, p in zip(panel, predictions) if r['observation']['task'] != 'check']
    need(len(vectors) == 36, 'incorrect vector task inventory')
    for pos in range(4):
        positions.append({'position': pos, 'rows': len(vectors),
            'targets': dict(sorted(Counter(r['answer'][pos] for r, _ in vectors).items())),
            'correct': sum(len(p['tokens']) > pos and p['tokens'][pos] == ord(r['answer'][pos])
                           for r, p in vectors),
            'nonzero_targets': sum(r['answer'][pos] != '0' for r, _ in vectors),
            'correct_nonzero': sum(r['answer'][pos] != '0' and len(p['tokens']) > pos
                                   and p['tokens'][pos] == ord(r['answer'][pos]) for r, p in vectors),
            'zero_outputs': sum(len(p['tokens']) > pos and p['tokens'][pos] == 48
                                for _, p in vectors)})
    return {'groups': groups, 'vector_positions': positions, 'classified': classified,
            'exact_fit': groups['all']['exact'] == 72}


def probs(logits: tuple[float, ...], target: int | None) -> dict:
    """Independent all-259-token logsumexp, binary EOS/rest odds and ranks."""
    need(len(logits) == VOCAB and all(math.isfinite(v) for v in logits),
         'invalid FP32 logit vector')
    maximum = max(logits)
    normalizer = math.log(math.fsum(math.exp(v - maximum) for v in logits))
    logp = [(v - maximum) - normalizer for v in logits]
    p = [math.exp(v) for v in logp]
    need(abs(math.fsum(p) - 1.0) <= TOL, 'invalid full-vocabulary normalization')
    rest = [v for i, v in enumerate(logits) if i != EOS]
    rest_max = max(rest)
    rest_lse = math.log(math.fsum(math.exp(v - rest_max) for v in rest))
    odds = (logits[EOS] - rest_max) - rest_lse
    argmax = max(range(VOCAB), key=logits.__getitem__)
    best = max(RESIDUES, key=logits.__getitem__)
    expected = {
        'argmax': argmax, 'best_residue_token': best,
        'target_rank': None if target is None else 1 + sum(
            v > logits[target] or (v == logits[target] and i < target)
            for i, v in enumerate(logits)),
        'target_residue_rank': None if target not in RESIDUES else 1 + sum(
            logits[i] > logits[target] or (logits[i] == logits[target] and i < target)
            for i in RESIDUES),
        'target_is_best_non_eos': None if target not in RESIDUES else target == max(
            (i for i in range(VOCAB) if i != EOS), key=logits.__getitem__),
        'eos_beats_target': None if target is None else logits[EOS] > logits[target],
        'eos_beats_target_loses_third': None if target is None else
            logits[EOS] > logits[target] and logits[argmax] > logits[EOS],
        'probability_underflow_count': sum(x == 0 for x in p),
    }
    expected['argmax_category'] = ('eos' if argmax == EOS else
        'correct_residue' if target in RESIDUES and argmax == target else
        ('other_residue' if target not in RESIDUES else 'incorrect_residue')
        if argmax in RESIDUES else 'other_vocabulary')
    four = [logits[x] for x in (*RESIDUES, EOS)]
    four_max = max(four)
    four_lse = math.log(math.fsum(math.exp(v - four_max) for v in four))
    rest_prob = math.fsum(v for i, v in enumerate(p) if i != EOS)
    check_odds = math.log(p[EOS]) - math.log(rest_prob) if (p[EOS] >= sys.float_info.min
                                    and rest_prob >= sys.float_info.min) else None
    scalars = {
        'eos_logit': logits[EOS], 'best_residue_logit': logits[best],
        'full_eos_probability': p[EOS], 'full_eos_log_probability': logp[EOS],
        'eos_rest_log_odds': odds, 'eos_rest_probability_check': check_odds,
        'eos_best_residue_margin': logits[EOS] - logits[best],
        'restricted_four_token_eos_probability': math.exp((logits[EOS] - four_max) - four_lse),
        'normalization_error': abs(math.fsum(p) - 1.0),
        'target_logit': None if target is None else logits[target],
        'target_probability': None if target is None else p[target],
        'target_log_probability': None if target is None else logp[target],
        'target_nll': None if target is None else -logp[target],
        'eos_target_margin': None if target is None else logits[EOS] - logits[target],
    }
    expected.update({key + '_hex': None if value is None else value.hex()
                     for key, value in scalars.items()})
    return expected


def metric_equal(actual: dict, expected: dict) -> bool:
    if set(actual) != set(expected):
        return False
    for key, value in expected.items():
        observed = actual[key]
        if key.endswith('_hex') and value is not None:
            if not isinstance(observed, str):
                return False
            try:
                observed_float = float.fromhex(observed)
                expected_float = float.fromhex(value)
            except ValueError:
                return False
            if not (math.isfinite(observed_float) and math.isclose(
                    observed_float, expected_float, abs_tol=TOL, rel_tol=TOL)):
                return False
        elif observed != value:
            return False
    return True


def summary_check(panel: list[dict], score_value: dict, index: list[dict],
                  metrics: list[dict], saved: dict) -> dict:
    for key in ('groups', 'classified', 'vector_positions', 'exact_fit'):
        need(score_value[key] == saved.get(key), 'summary disagreement: ' + key)
    need(saved.get('baseline_exact') == {'0': 18, '1': 18}, 'constant control disagreement')
    need(saved.get('improvement_over_baseline') == {
        '0': score_value['groups']['all']['exact'] - 18,
        '1': score_value['groups']['all']['exact'] - 18}, 'baseline improvement disagreement')
    lookup = {r['id']: r for r in panel}
    groups: dict[str, list[dict]] = defaultdict(list)
    vectors = [dict(position=p, total=0, correct_best_residue=0,
                    correct_non_eos=0, nonzero_targets=0,
                    correct_nonzero_best_residue=0, zero_best_residue=0) for p in range(4)]
    for rec, metric in zip(index, metrics):
        if rec['probe'] != 'gold':
            continue
        row = lookup[rec['row_id']]
        for key in ('all', panel_task(row), 'target:' + str(rec['target']),
                    'terminal_eos' if rec['target'] == EOS else 'answer_position'):
            groups[key].append(metric)
        if row['observation']['task'] != 'check' and rec['position'] < 4:
            r = vectors[rec['position']]
            target = rec['target']
            r['total'] += 1
            r['correct_best_residue'] += int(metric['best_residue_token'] == target)
            r['correct_non_eos'] += int(metric['target_is_best_non_eos'] is True)
            r['nonzero_targets'] += int(target != 48)
            r['correct_nonzero_best_residue'] += int(target != 48 and metric['best_residue_token'] == target)
            r['zero_best_residue'] += int(metric['best_residue_token'] == 48)
    need(vectors == saved.get('gold_vector_positions'), 'gold residue summary disagreement')
    recorded_groups = saved.get('gold_prefix')
    need(type(recorded_groups) is dict and set(recorded_groups) == set(groups),
         'gold prefix summary group coverage mismatch')
    for name, items in groups.items():
        observed = recorded_groups[name]
        need(observed['positions'] == len(items) and
             observed['target_argmax'] == sum(x['target_rank'] == 1 for x in items) and
             observed['zero_best_residue'] == sum(x['best_residue_token'] == 48 for x in items),
             'gold prefix group count mismatch')
        for summary_key, source_key in (('mean_target_nll_hex', 'target_nll_hex'),
              ('mean_eos_target_margin_hex', 'eos_target_margin_hex'),
              ('mean_eos_rest_log_odds_hex', 'eos_rest_log_odds_hex')):
            mean = math.fsum(float.fromhex(x[source_key]) for x in items) / len(items)
            need(math.isclose(float.fromhex(observed[summary_key]), mean,
                              rel_tol=TOL, abs_tol=TOL), 'gold prefix mean mismatch')
    return vectors


def checkpoint(archive: zipfile.ZipFile, seed: int, step: int,
               request: dict, hashes: dict, panel: list[dict]) -> dict:
    prefix = f'attempts/seed-{seed}/step-{step:03d}/'
    manifest = jdoc(archive, prefix + 'manifest.json')
    need(manifest['step'] == step and manifest['request_identity'] == request['_anchor'],
         'checkpoint source/request/step mismatch')
    members_check(hashes, prefix, manifest['members'], ('manifest.json',))
    metadata = jdoc(archive, prefix + 'metadata.json')
    need(metadata['step'] == step and metadata['seed'] == seed and
         metadata['cursor'] == step * 4 and metadata['request_identity'] == request['_anchor'],
         'checkpoint metadata mismatch')
    observation = jdoc(archive, prefix + 'observation.json')
    need(observation['before'] == observation['after'] and
         observation['optimizer_updates'] == 0 and
         observation['rng_unchanged'] is True and observation['state_unchanged'] is True,
         'checkpoint diagnostic mutated native state (recorded contract)')
    predictions = jdoc(archive, prefix + 'predictions.json')
    observed_score = score(panel, predictions)
    index = jlines(archive, prefix + 'index.jsonl')
    metrics = jlines(archive, prefix + 'metrics.jsonl')
    raw = archive.read(prefix + 'logits.f32le')
    expect_index = []
    for row, pred in zip(panel, predictions):
        prompt = [BOS, *row['prompt'].encode('ascii')]
        gold = [*row['answer'].encode('ascii'), EOS]
        for probe, emitted in (('gold', gold), ('greedy', pred['tokens'])):
            for pos in range(len(emitted)):
                expect_index.append({'row_id': row['id'], 'probe': probe,
                   'position': pos, 'prefix_tokens': prompt + emitted[:pos],
                   'target': gold[pos] if pos < len(gold) else None,
                   'offset_bytes': VECTOR.size * len(expect_index)})
    need(index == expect_index, 'prefix/target inventory mismatch')
    need(len(raw) == len(index) * VECTOR.size == len(metrics) * VECTOR.size,
         'FP32 full-vocabulary payload length mismatch')
    checked = []
    for i, (rec, observed) in enumerate(zip(index, metrics)):
        logits = VECTOR.unpack_from(raw, i * VECTOR.size)
        expected = probs(logits, rec['target'])
        need(metric_equal(observed, expected), 'derived token metric disagrees at record ' + str(i))
        if rec['probe'] == 'greedy':
            pred = next(p for p in predictions if p['id'] == rec['row_id'])
            need(expected['argmax'] == pred['tokens'][rec['position']],
                 'reported generated token is not full-vocabulary argmax')
        checked.append(expected)
    saved = jdoc(archive, prefix + 'summary.json')
    residues = summary_check(panel, observed_score, index, checked, saved)
    evaluation = jdoc(archive, prefix + 'evaluation.json')
    need(evaluation['heldout_scored'] is False and evaluation['geometry_loss'] == '0',
         'unsupported evaluation scope')
    exact_rows = [row for row, pred in zip(panel, predictions)
                  if classify(pred['tokens'], row)['exact']]
    exact_vec = {(row['observation']['word'], row['family'],
                  row['observation']['task'], row['answer']) for row in exact_rows
                 if row['observation']['task'] != 'check'}
    return {
        'seed': seed, 'update': step, 'exact': observed_score['groups']['all']['exact'],
        'tasks': {k: observed_score['groups'][k]['exact'] for k in
                  ('apply', 'inverse', 'check_0', 'check_1')},
        'premature_eos': observed_score['groups']['all']['premature_eos'],
        'distinct_outputs': observed_score['groups']['all']['distinct_outputs'],
        'gold_target_nll': float.fromhex(saved['gold_prefix']['all']['mean_target_nll_hex']),
        'native_train_loss': float.fromhex(evaluation['train_loss_hex']),
        'greedy_nonzero_residues': sum(x['correct_nonzero'] for x in observed_score['vector_positions']),
        'gold_nonzero_best_residues': sum(x['correct_nonzero_best_residue'] for x in residues),
        'correct_vector_rows': sorted([list(v) for v in exact_vec]),
        'complete_vector_fit': len(exact_vec),
        'vectors_exact_identity': sha(canonical(sorted(map(list, exact_vec)))),
        'positions': len(index),
        '_exact_ids': {r['id'] for r in exact_rows},
        '_vector_set': exact_vec,
        '_metadata': metadata,
    }


def review(archive_path: Path, archive_identity: str) -> dict:
    archive, hashes = safe_archive(archive_path, archive_identity)
    try:
        names = set(hashes)
        anchors = jdoc(archive, 'request-anchors.json')
        need(set(anchors) == {f'seed-{s}.json' for s in SEEDS},
             'missing or extra frozen seed anchor')
        dataset_manifest = jdoc(archive, 'dataset/manifest.json')
        members_check(hashes, 'dataset/', dataset_manifest['files'], ('manifest.json',))
        data_manifest_hash = hashes['dataset/manifest.json']['identity']
        need(dataset_manifest['split_counts'] == {'train': 972, 'validation': 648, 'test': 648}
             and dataset_manifest['heldout_scored'] is False
             and dataset_manifest['training_authorized'] is False,
             'dataset split/scoring declaration mismatch')
        train = jlines(archive, 'dataset/train.jsonl')
        need(len(train) == 972 and all(r['split'] == 'train' for r in train)
             and len({r['id'] for r in train}) == 972, 'invalid full TRAIN inventory')
        by_id = {r['id']: r for r in train}
        for member in dataset_manifest['source_closure']:
            path = 'source/' + member['path']
            need(path in hashes and hashes[path]['identity'] == member['identity'],
                 'TT-002 source closure differs from archived source bytes')
        requests, results, initial_ids = {}, [], set()
        for seed in SEEDS:
            base = f'attempts/seed-{seed}/'
            request = jdoc(archive, base + 'request.json')
            request_hash = hashes[base + 'request.json']['identity']
            need(request_hash == anchors[f'seed-{seed}.json'] and request['seed'] == seed
                 and request['schema'] == 'trinite.teacher-fit-001.v1.request'
                 and request['status'] == 'FROZEN_NOT_AUTHORIZED'
                 and request['dataset_manifest_identity'] == data_manifest_hash
                 and request['dataset_train_identity'] == hashes['dataset/train.jsonl']['identity'],
                 'frozen request or dataset identity mismatch')
            request['_anchor'] = request_hash
            requests[seed] = request
            panel = request['panel']
            need(len(panel) == 72 and len({r['id'] for r in panel}) == 72 and
                 all(r == by_id.get(r['id']) for r in panel),
                 'panel differs from original TT-002 TRAIN data')
            controls = {}
            for token in (48, 49):
                pred = [{'id': row['id'], 'tokens': [token] * len(row['answer']) + [EOS]}
                        for row in panel]
                controls[str(token - 48)] = score(panel, pred)
            need(controls == request['baselines'] and
                 all(value['groups']['all']['exact'] == 18 for value in controls.values()),
                 'original fixed control baseline disagreement')
            scheduled = request['selection']['visit_ids']
            need(scheduled == [panel[i % 72]['id'] for i in range(1024)] and
                 sum(len(by_id[r]['answer']) + 1 for r in scheduled) == 3590 and
                 request['schedule_identity'] == sha(canonical(scheduled)),
                 'frozen visits, schedule identity or target budget mismatch')
            need(request['limits']['updates'] == 256 and
                 request['limits']['batch_size'] == 4 and
                 request['limits']['retries'] == 0 and
                 request['limits']['target_tokens'] == 4096 and
                 request['native_config']['seed'] == seed,
                 'experimental treatment/limits differ')
            attempt_manifest = jdoc(archive, base + 'manifest.json')
            local_anchor = jdoc(archive, f'anchors/seed-{seed}.json')['manifest_identity']
            need(local_anchor == hashes[base + 'manifest.json']['identity'] and
                 attempt_manifest['status'] == 'COMPLETED' and
                 attempt_manifest['seed'] == seed and
                 attempt_manifest['request_identity'] == request_hash,
                 'attempt manifest / external anchor mismatch')
            members_check(hashes, base, attempt_manifest['members'], ('manifest.json',))
            grant = jdoc(archive, base + 'approval.json')
            need(grant['schema'] == 'trinite.teacher-fit-001.v1.owner-approval' and
                 grant['request_identity'] == request_hash and
                 grant['seed'] == seed and
                 grant['source_commit'] == request['source']['commit'] and
                 grant['training_authorized'] is True and
                 grant['limits'] == request['limits'] and
                 attempt_manifest['approval_identity'] == hashes[base + 'approval.json']['identity'],
                 'owner approval record binding differs (not proof of authenticity)')
            supervisor = jdoc(archive, base + 'supervisor.json')
            completed = jdoc(archive, base + 'completed.json')
            need(supervisor['status'] == 'COMPLETED' and
                 supervisor['child_returncode'] == 0 and
                 supervisor['process_resolved'] is True and
                 supervisor['timed_out'] is False and
                 supervisor['interrupted'] is False and
                 float.fromhex(supervisor['elapsed_seconds_hex']) <= 900 and
                 completed['optimizer_updates'] == 256 and
                 completed['target_tokens'] == 3590 and
                 completed['optimizer_replayed'] is False,
                 'unexpected completion or supervisor state')
            initial_ids.add(completed['initialization_identity'])
            need(len(completed['checkpoints']) == len(STEPS),
                 'attempt checkpoint count mismatch')
            replay = jdoc(archive, f'replay/seed-{seed}-forward-replay.json')
            need(replay['verified'] is True and replay['native_forward_replayed'] is True
                 and replay['optimizer_replayed'] is False and
                 replay['manifest_identity'] == local_anchor and
                 replay['request_identity'] == request_hash and
                 replay['steps_replayed'] == list(STEPS),
                 'native forward replay receipt claims differ')
            journal = jlines(archive, base + 'progress.jsonl')
            need(len(journal) == 256, 'incomplete optimizer journal')
            cumulative, elapsed, history = 0, 0., []
            for i, event in enumerate(journal):
                h = event['history']
                expected_batch = scheduled[i*4:(i+1)*4]
                need(h['step'] == i + 1 and h['examples'] == expected_batch,
                     'recorded updater schedule mismatch')
                tokens = sum(len(by_id[r]['answer']) + 1 for r in expected_batch)
                cumulative += tokens
                need(h['target_tokens'] == tokens,
                     'recorded supervised target mismatch')
                next_elapsed = float.fromhex(event['native_elapsed_hex'])
                need(elapsed <= next_elapsed <= 600,
                     'cumulative native time drift or overrun')
                elapsed = next_elapsed
                history.append(h)
            need(cumulative == 3590, 'journal target token total mismatch')
            series, previous = [], None
            for step in STEPS:
                row = checkpoint(archive, seed, step, request, hashes, panel)
                m = row['_metadata']
                need(m['history'] == history[:step] and m['target_tokens'] == sum(
                    h['target_tokens'] for h in history[:step]) and
                     m['initialization_identity'] == completed['initialization_identity'] and
                     (previous is None or m['native_elapsed_hex'] is not None),
                     'checkpoint trajectory/history mismatch')
                cp_hash = hashes[f'{base}step-{step:03d}/manifest.json']['identity']
                need(any(c['step'] == step and c['manifest_identity'] == cp_hash
                         for c in completed['checkpoints']),
                     'completed checkpoint digest differs')
                previous = row
                series.append(row)
            results.append(series)
        need(len(initial_ids) == 3, 'initial models are not independently initialized')
        first = requests[SEEDS[0]]
        baseline = {k: v for k, v in first.items() if k not in
                    ('seed', 'native_config', 'output_directory', '_anchor')}
        for seed in SEEDS[1:]:
            other = {k: v for k, v in requests[seed].items() if k not in
                     ('seed', 'native_config', 'output_directory', '_anchor')}
            need(other == baseline and requests[seed]['native_config'] ==
                 dict(first['native_config'], seed=seed),
                 'cross-seed experiment conditions differ')
        source = first['source']['members']
        members_check(hashes, 'source/', [{'name': x['path'], 'identity': x['identity'],
             'bytes': hashes.get('source/' + x['path'], {}).get('bytes')}
             for x in source])
        # Anchored source hashes above include all 1,279 files. No archived source executed.
        allowed = ({'synthesis.json', 'TT-003-FIT-001-TRIPLE-SYNTHESIS.md',
                    'request-anchors.json'}
                   | {f'anchors/seed-{s}.json' for s in SEEDS}
                   | {f'replay/seed-{s}-forward-replay.json' for s in SEEDS}
                   | {'dataset/manifest.json'}
                   | {'dataset/' + m['name'] for m in dataset_manifest['files']}
                   | {'source/' + m['path'] for m in source})
        for seed in SEEDS:
            attempt_prefix = f'attempts/seed-{seed}/'
            listed = jdoc(archive, attempt_prefix + 'manifest.json')['members']
            allowed.add(attempt_prefix + 'manifest.json')
            allowed.update(attempt_prefix + m['name'] for m in listed)
        need(names == allowed, 'unexpected/missing archive-level evidence member')
        synthesis = jdoc(archive, 'synthesis.json')
        need(synthesis['optimizer_replayed'] is False and synthesis['heldout_scored'] is False
             and {x['seed'] for x in synthesis['attempts']} == set(SEEDS),
             'synthesis scope/attempt inventory mismatch')
        for seed, series in zip(SEEDS, results):
            retained = next(x for x in synthesis['attempts'] if x['seed'] == seed)
            need(retained['status'] == 'COMPLETED' and retained['journal_updates'] == 256
                 and retained['journal_target_tokens'] == 3590 and
                 retained['manifest_identity'] == hashes[f'attempts/seed-{seed}/manifest.json']['identity']
                 and retained['optimizer_replayed'] is False and
                 len(retained['checkpoints']) == len(STEPS),
                 'synthesis record conflicts with retained attempt')
            for mine, theirs in zip(series, retained['checkpoints']):
                need(mine['update'] == theirs['step'] and
                     mine['exact'] == theirs['summary']['groups']['all']['exact'] and
                     mine['premature_eos'] == theirs['summary']['groups']['all']['premature_eos'],
                     'synthesis checkpoint conflicts with predictions')
        completed_rows = []
        for series in results:
            per_seed = []
            for row in series:
                sanitized = {k: v for k, v in row.items() if not k.startswith('_')}
                per_seed.append(sanitized)
            completed_rows.append(per_seed)
        final = [series[-1] for series in results]
        vector_sets = [r['_vector_set'] for r in final]
        overlap = vector_sets[0] & vector_sets[2]
        transitions = []
        for series in results:
            transition = []
            for before, after in zip(series, series[1:]):
                gained = after['_exact_ids'] - before['_exact_ids']
                lost = before['_exact_ids'] - after['_exact_ids']
                transition.append({'from': before['update'], 'to': after['update'],
                    'gained': len(gained), 'lost': len(lost)})
            transitions.append(transition)
        return {
            'schema': 'trinite.teacher-fit-001.evidence-review.v1',
            'archive_identity': archive_identity, 'archive_members_checked': len(hashes),
            'source_commit': first['source']['commit'], 'source_members_checked': len(source),
            'dataset_manifest_identity': data_manifest_hash,
            'exposed_train_only': True, 'optimizer_updates_per_seed': 256,
            'target_tokens_per_seed': 3590,
            'read_only': True, 'native_forward_replay_performed_by_this_tool': False,
            'optimizer_replay_performed': False,
            'recorded_native_forward_replay_receipts': 3,
            'baseline_exact': {'constant_zero': 18, 'constant_one': 18},
            'checkpoint_analysis': completed_rows,
            'exact_answer_transitions': transitions,
            'final_vector_overlap_17_43_count': len(overlap),
            'final_vector_overlap_17_43': sorted(map(list, overlap)),
            'conclusion': 'NONE_OF_THREE_SEEDS_FIT_72_OF_72',
        }
    finally:
        archive.close()
