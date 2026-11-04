# SPDX-License-Identifier: MPL-2.0
"""TT-003-FIT-001-EVIDENCE-REVIEW-001: offline, read-only independent auditor.

Only Python standard library. Never imports or executes archived source/weights,
loads native checkpoints, evaluates heldouts, or updates an optimizer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import zipfile
import tempfile

SEEDS = (17, 29, 43)
STEPS = (0, 64, 128, 192, 256)
EOS, BOS, VOCAB = 257, 256, 259
RESIDUES = (48, 49, 50)
VECTOR = struct.Struct('<259f')
MAX_ENTRIES = 2_000
MAX_UNCOMPRESSED = 256 * 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024
TOL = 1e-12
# Independently retained source/data commitments from the frozen owner handoff.
# These bind this *fixed historical experiment*, not the authenticity of its owner.
FROZEN_SOURCE_COMMIT = '308a0b73b210a691efbed8f71db5b92eb4d96262'
FROZEN_SOURCE_ID = 'sha256:2ed6c4bc5cc2aa1aaef47c789882ac88ab6bcdc2b2634e8da0255b8ecc28288d'
FROZEN_DATASET_ID = 'sha256:b90287c3549e2a877b4e23325b51f234f3e54eab3a48c0227f34c5f18d065687'
FROZEN_RIGHTS_ID = 'sha256:0d8f5b28ebbb2f308d2c17f5c16275e11c3479ea2d606393f5aff997a0ebfee5'
FROZEN_SOURCE_COUNT = 1279
# Independent deterministic reconstruction of the reviewed FIT-001 selector.
# Never load or execute the archived producer's Python source.
FAMILIES = ('S1', 'S2', 'W')
WORDS = tuple(''.join(t) for t in product('012', repeat=4))
WORD_DOMAIN = b'trinite.teacher-fit-001.word.v1\x00'
ROW_DOMAIN = b'trinite.teacher-fit-001.row.v1\x00'
FROZEN_LIMITS = {
    'native_seconds': 600, 'outer_seconds': 900, 'updates': 256,
    'target_tokens': 4096, 'batch_size': 4,
    'address_space_bytes': 8 * 1024**3, 'file_bytes': 16 * 1024**2,
    'run_bytes': 128 * 1024**2, 'retries': 0,
}
DATASET_FILES = frozenset({
    'dataset-rights.json', 'plan.json', 'privileged-targets.jsonl',
    'test-public.jsonl', 'train.jsonl', 'validation-public.jsonl',
    'weights-rights.json',
})
TT002_SOURCE_FILES = frozenset({
    'AGENTS.md', 'LICENSE', 'configs/teacher/gf3-dataset-v1.json',
    'configs/teacher/source-candidates-v1.json', 'docs/DATA.md',
    'docs/QUTRIT-CONFORMANCE-V1.md', 'docs/TEACHER-LAB-TT-002.md',
    'docs/TEACHER-LAB-V1.md', 'scripts/model_size_policy_v1.py',
    'scripts/queue_evaluation_v1.py', 'scripts/qutrit_conformance_v1.py',
    'scripts/teacher_dataset.py', 'scripts/teacher_dataset_v1/__init__.py',
    'scripts/teacher_dataset_v1/core.py', 'scripts/teacher_lab_v1/__init__.py',
    'scripts/teacher_lab_v1/core.py', 'src/trinite/__init__.py',
    'src/trinite/contracts.py', 'src/trinite/tokenizer.py',
})
REQUIRED_SOURCE_FILES = frozenset({
    'AGENTS.md', 'scripts/AGENTS.md', 'scripts/teacher_fit_v1/AGENTS.md',
    'docs/AGENT-IBAE-GOVERNANCE.md', 'docs/TEACHER-LAB-FIT-001-EXECUTION.md',
    'scripts/teacher_fit_v1/spec.py', 'scripts/teacher_fit_v1/evidence.py',
    'scripts/teacher_fit_v1/native.py', 'scripts/run_teacher_fit.py',
    'configs/teacher/tt003-fit-001-v1.json',
})
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
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


class AnchoredZip(zipfile.ZipFile):
    """Own the private snapshot so hash and ZIP inspection use identical bytes."""

    def __init__(self, snapshot):
        self._snapshot = snapshot
        super().__init__(snapshot, mode='r')

    def close(self):
        try:
            super().close()
        finally:
            self._snapshot.close()


def safe_archive(path: Path, expected_identity: str) -> tuple[zipfile.ZipFile, dict]:
    need(type(expected_identity) is str and SHA_RE.fullmatch(expected_identity) is not None,
         'exact externally supplied archive SHA-256 required')
    # A private, unlinked temporary file is the inspected object. Path replacement
    # or in-place archive modification after snapshot creation cannot redirect it.
    snapshot = tempfile.TemporaryFile(mode='w+b')
    archive = None
    try:
        h, length = hashlib.sha256(), 0
        with Path(path).open('rb') as original:
            while chunk := original.read(1024 * 1024):
                length += len(chunk)
                need(length <= MAX_ARCHIVE_BYTES, 'archive compressed byte budget exceeded')
                snapshot.write(chunk)
                h.update(chunk)
        need('sha256:' + h.hexdigest() == expected_identity,
             'archive SHA-256 differs from external anchor')
        snapshot.seek(0)
        archive = AnchoredZip(snapshot)
        names, hashes, total = set(), {}, 0
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
            raw = archive.read(info)
            need(len(raw) == info.file_size, 'ZIP member length mismatch')
            hashes[name] = {'identity': sha(raw), 'bytes': len(raw)}
            names.add(name)
    except BaseException:
        if archive is not None:
            archive.close()
        else:
            snapshot.close()
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



def fixed_inventories(source: dict, dataset_manifest: dict, hashes: dict) -> None:
    """Validate mandatory provenance coverage *before* hash consistency.

    Frozen identities are external handoff facts, not producer authentication.
    A re-anchored ZIP must not choose its own incomplete source/dataset scope.
    """
    need(type(source) is dict and set(source) == {'commit', 'members'} and
         source['commit'] == FROZEN_SOURCE_COMMIT and
         type(source['members']) is list and
         len(source['members']) == FROZEN_SOURCE_COUNT and
         sha(canonical(source)) == FROZEN_SOURCE_ID,
         'incomplete or changed frozen source closure')
    source_members = source['members']
    need(all(type(m) is dict and set(m) == {'path', 'identity'} and
             type(m['path']) is str and type(m['identity']) is str and
             SHA_RE.fullmatch(m['identity']) for m in source_members),
         'malformed frozen source inventory')
    source_names = {m['path'] for m in source_members}
    need(len(source_names) == FROZEN_SOURCE_COUNT and
         REQUIRED_SOURCE_FILES <= source_names and
         TT002_SOURCE_FILES <= source_names,
         'missing required source inventory members')
    need(type(dataset_manifest) is dict and
         dataset_manifest.get('schema') == 'trinite.teacher-gf3-dataset.v1' and
         type(dataset_manifest.get('files')) is list and
         len(dataset_manifest['files']) == len(DATASET_FILES),
         'incomplete TT-002 dataset member inventory')
    ds_files = dataset_manifest['files']
    need(all(type(m) is dict and set(m) == {'name', 'identity', 'bytes'} and
             type(m['name']) is str for m in ds_files) and
         {m['name'] for m in ds_files} == DATASET_FILES,
         'missing required dataset rights, train, heldout or plan member')
    need(hashes.get('dataset/manifest.json', {}).get('identity') == FROZEN_DATASET_ID and
         hashes.get('dataset/dataset-rights.json', {}).get('identity') == FROZEN_RIGHTS_ID,
         'dataset/rights differ from independent frozen identities')
    closure = dataset_manifest.get('source_closure')
    need(type(closure) is list and len(closure) == len(TT002_SOURCE_FILES) and
         all(type(m) is dict and set(m) == {'path', 'identity'} and
             type(m['path']) is str for m in closure) and
         {m['path'] for m in closure} == TT002_SOURCE_FILES,
         'TT-002 source closure inventory incomplete')
    by_source = {m['path']: m['identity'] for m in source_members}
    need(all(by_source.get(m['path']) == m['identity'] for m in closure),
         'TT-002 source closure not bound to frozen source inventory')


def approval_expected(request: dict, request_hash: str) -> dict:
    """Standalone exact mirror of original FIT spec.approval_expected."""
    return {'schema': 'trinite.teacher-fit-001.v1.owner-approval',
            'owner': 'QSOLKCB', 'scope': 'one-owner-local-native-cpu-fit-attempt',
            'request_identity': request_hash,
            'source_commit': request['source']['commit'],
            'seed': request['seed'],
            'dataset_manifest_identity': request['dataset_manifest_identity'],
            'schedule_identity': request['schedule_identity'],
            'environment_identity': sha(canonical(request['environment'])),
            'output_directory': request['output_directory'],
            'limits': request['limits'], 'training_authorized': True}


def validate_approval(grant: dict, request: dict, request_hash: str,
                      manifest_approval_identity: str, actual_identity: str) -> None:
    need(type(grant) is dict and grant == approval_expected(request, request_hash) and
         manifest_approval_identity == actual_identity,
         'complete owner approval contract differs (not proof of authenticity)')


def frozen_native_config(model: dict, seed: int) -> dict:
    """Independent copy of the frozen FIT-001 native treatment (no code imports)."""
    return {
        'schema': 'trinite.training-plan.v1', 'model': model, 'seed': seed,
        'steps': 256, 'batch_size': 4, 'learning_rate': '0.001',
        'beta1': '0.9', 'beta2': '0.999', 'optimizer_epsilon': '0.00000001',
        'weight_decay': '0.01', 'gradient_clip': '1', 'accumulation': 1,
        'schedule': 'constant', 'order': 'seeded-hash-rank-cycle.v1',
        'loss': 'answer-and-eos-target-mean.v1', 'validation_every': 64,
        'selection': 'final-step; no early stopping; no test evaluation',
        'max_seconds': 600, 'max_target_tokens': 4096,
    }


def frozen_panel_schedule(train: list[dict]) -> tuple[list[dict], dict]:
    """Independent reconstruction of the *entire* frozen FIT-001 selection.

    Source rows come from the archived TRAIN packet, not the request's panel.
    Selection ranks are domain-separated SHA-256 over ASCII text, with
    lexicographic tie-breaking. No archived Python is imported or executed.
    """
    need(type(train) is list and len(train) == 972,
         'FIT-001 selector requires exactly 972 TRAIN records')
    rows, ids = {}, set()
    for row in train:
        need(type(row) is dict and type(row.get('id')) is str and
             SHA_RE.fullmatch(row['id']) is not None and row['id'] not in ids and
             row.get('split') == 'train' and row.get('family') in FAMILIES and
             type(row.get('observation')) is dict,
             'FIT-001 TRAIN selector has invalid/duplicate source row')
        obs = row['observation']
        family, word, task = row['family'], obs.get('word'), obs.get('task')
        need(obs.get('circuit') == family and word in WORDS and
             task in ('apply', 'inverse', 'check'),
             'FIT-001 TRAIN selector has invalid structural observation')
        answer, tokens = row.get('answer'), row.get('supervised_tokens')
        if task == 'check':
            need(answer in ('0', '1') and type(tokens) is int and tokens == 2,
                 'FIT-001 TRAIN selector has invalid scalar target')
            view = answer
        else:
            need(type(answer) is str and len(answer) == 4 and
                 set(answer) <= set('012') and type(tokens) is int and tokens == 5,
                 'FIT-001 TRAIN selector has invalid vector target')
            view = ''
        key = (family, word, task, view)
        need(key not in rows, 'duplicate FIT-001 TRAIN structural task')
        ids.add(row['id'])
        rows[key] = row
    keys = {(family, word, task, view)
            for family in FAMILIES for word in WORDS
            for task, view in (('apply', ''), ('inverse', ''),
                               ('check', '0'), ('check', '1'))}
    need(set(rows) == keys, 'FIT-001 TRAIN structural inventory incomplete')

    def rank(domain: bytes, value: str) -> tuple[bytes, str]:
        return hashlib.sha256(domain + value.encode('ascii')).digest(), value

    words = sorted(WORDS, key=lambda word: rank(WORD_DOMAIN, word))[:6]
    chosen = [rows[family, word, task, view]
              for family in FAMILIES for word in words
              for task, view in (('apply', ''), ('inverse', ''),
                                 ('check', '0'), ('check', '1'))]
    ordered = sorted(chosen, key=lambda row: rank(ROW_DOMAIN, row['id']))
    scheduled = [ordered[i % 72] for i in range(1024)]
    targets = sum(row['supervised_tokens'] for row in scheduled)
    need(len(ordered) == 72 and targets == 3590,
         'independent FIT-001 target budget differs')
    selection = {
        'schema': 'trinite.teacher-fit-001-panel.v1',
        'selected_words': words,
        'panel_ids': [row['id'] for row in ordered],
        'visit_ids': [row['id'] for row in scheduled],
        'scheduled_target_tokens': targets,
        'seeds': list(SEEDS),
        'optimizer_updates_per_seed': 256,
        'batch_size': 4,
        'heldout_scored': False,
        'training_authorized': False,
    }
    return ordered, selection


def check_frozen_selection(request: dict, panel: list[dict],
                           selection: dict) -> list[str]:
    """Reject re-anchored changes in panel ordering, visits, or metadata."""
    need(canonical(request.get('panel')) == canonical(panel) and
         canonical(request.get('selection')) == canonical(selection) and
         request.get('panel_identity') == sha(canonical(panel)) and
         request.get('schedule_identity') == sha(canonical(selection['visit_ids'])),
         'frozen selector panel, ranking or visit schedule differs')
    return selection['visit_ids']


def canonical_native_time(raw: str) -> float:
    """A finite, nonnegative, canonical binary64 elapsed-second record."""
    need(type(raw) is str, 'invalid checkpoint native time')
    try:
        value = float.fromhex(raw)
    except (ValueError, OverflowError):
        raise EvidenceError('invalid checkpoint native time') from None
    need(math.isfinite(value) and value >= 0 and value.hex() == raw and
         math.copysign(1.0, value) > 0,
         'invalid checkpoint native time')
    return value


def check_native_time_trajectory(series: list[dict], journal: list[dict],
                                 completed: dict) -> None:
    """Cross-bind checkpoint times to the updater journal and completion."""
    need(len(series) == len(STEPS) and
         [r['update'] for r in series] == list(STEPS) and len(journal) == 256,
         'incomplete native elapsed-time trajectory')
    previous = 0.
    for row in series:
        step = row['update']
        actual_hex = row['_metadata']['native_elapsed_hex']
        elapsed = canonical_native_time(actual_hex)
        expected_hex = '0x0.0p+0' if step == 0 else journal[step - 1]['native_elapsed_hex']
        need(actual_hex == expected_hex and previous <= elapsed <= FROZEN_LIMITS['native_seconds'],
             'checkpoint native time differs from journal, decreases or exceeds budget')
        previous = elapsed
    final_hex = completed.get('native_elapsed_hex')
    final_time = canonical_native_time(final_hex)
    need(final_hex == series[-1]['_metadata']['native_elapsed_hex'] ==
         journal[-1]['native_elapsed_hex'] and
         final_time <= FROZEN_LIMITS['native_seconds'],
         'completion native time differs from final checkpoint/journal')


def check_frozen_treatment(request: dict, seed: int,
                           model: dict, design: dict) -> None:
    """Bind request treatment to independently frozen, source-hashed inputs."""
    need(type(request) is dict and request.get('model') == model and
         request.get('design') == design and
         request.get('model_parameters') == 90624 and
         request.get('lane') == 'dense' and
         request.get('geometry_loss') == '0' and
         request.get('limits') == FROZEN_LIMITS and
         request.get('native_config') == frozen_native_config(model, seed),
         'native configuration differs from frozen FIT-001 treatment')


def check_checkpoint_config(metadata: dict, request: dict) -> None:
    need(type(metadata) is dict and
         metadata.get('config') == request['native_config'],
         'checkpoint native configuration differs from frozen request')


def check_retained_environment(observed: dict, request: dict) -> None:
    """Recorded runtime must match the environment approved in this request.

    This tests self-consistency, not the actual hardware or producer identity.
    """
    need(type(observed) is dict and
         canonical(observed) == canonical(request['environment']),
         'retained execution environment differs from frozen CPU FP32 request')


def check_completed_receipt(completed: dict) -> None:
    """Reject any completion receipt that contradicts the TRAIN-only scope."""
    need(type(completed) is dict and
         completed.get('optimizer_updates') == 256 and
         completed.get('target_tokens') == 3590 and
         completed.get('optimizer_replayed') is False and
         completed.get('heldout_scored') is False,
         'completion receipt conflicts with frozen TRAIN-only treatment')


def check_observer_state(observation: dict, metadata: dict) -> None:
    """Observer's before/after must be those of this checkpoint, not a proxy."""
    need(type(metadata) is dict and type(observation) is dict and
         observation == {'before': metadata['identities'],
                         'after': metadata['identities'],
                         'optimizer_updates': 0,
                         'rng_unchanged': True, 'state_unchanged': True},
         'checkpoint observer state does not match retained model/optimizer')


def verified_loss(evaluation: dict, metadata: dict, step: int) -> float:
    """Finite nonnegative native loss, bound to retained validation history."""
    def number(raw):
        need(type(raw) is str, 'invalid nonnegative native loss encoding')
        try:
            val = float.fromhex(raw)
        except (ValueError, OverflowError):
            raise EvidenceError('invalid nonnegative native loss encoding') from None
        need(math.isfinite(val) and val >= 0 and val.hex() == raw,
             'invalid nonnegative native loss')
        return val

    value = number(evaluation.get('train_loss_hex'))
    initial = metadata.get('initial_train_loss_hex')
    number(initial)
    history = metadata.get('validation')
    need(type(history) is list and len(history) == step // 64 and
         all(type(v) is dict and set(v) == {'step', 'loss_hex'} and
             v['step'] == (i + 1) * 64 for i, v in enumerate(history)),
         'native validation loss inventory mismatch')
    for v in history:
        number(v['loss_hex'])
    expected = initial if step == 0 else history[-1]['loss_hex']
    need(evaluation['train_loss_hex'] == expected,
         'native checkpoint loss differs from retained validation/initial loss')
    return value


def check_loss_trajectory(series: list[dict]) -> None:
    """All checkpoints must share the same initial loss and validation prefix.

    Checking only each checkpoint's terminal loss allows prior validation
    values to be silently rewritten in later checkpoints of a re-anchored ZIP.
    """
    need(len(series) == len(STEPS) and
         [r['update'] for r in series] == list(STEPS),
         'incomplete checkpoint loss trajectory')
    initial = series[0]['_metadata']['initial_train_loss_hex']
    prior_validation = []
    for row in series:
        metadata = row['_metadata']
        expected = prior_validation + ([{'step': row['update'],
                                          'loss_hex': row['native_train_loss'].hex()}]
                                       if row['update'] else [])
        need(metadata['initial_train_loss_hex'] == initial and
             metadata['validation'] == expected,
             'checkpoint initial/validation loss trajectory contradicts prior checkpoints')
        prior_validation = expected


def fit_conclusion(final: list[dict]) -> tuple[list[int], str]:
    fitted = [r['seed'] for r in final if r['exact'] == 72]
    need(len(final) == len(SEEDS) and {r['seed'] for r in final} == set(SEEDS),
         'incomplete endpoint inventory')
    if not fitted:
        conclusion = 'NONE_OF_THREE_SEEDS_FIT_72_OF_72'
    elif len(fitted) == len(SEEDS):
        conclusion = 'ALL_THREE_SEEDS_FIT_72_OF_72'
    else:
        conclusion = 'SOME_OF_THREE_SEEDS_FIT_72_OF_72'
    return fitted, conclusion


def check_prefix_vectors(index: list[dict], raw: bytes) -> None:
    """Identical checkpoint+prefix must have exactly identical FP32 bytes.

    Checks cross-probe and cross-row cases, including matching continuations.
    """
    seen = {}
    for i, rec in enumerate(index):
        prefix = tuple(rec['prefix_tokens'])
        vector = raw[i*VECTOR.size:(i+1)*VECTOR.size]
        previous = seen.setdefault(prefix, vector)
        need(previous == vector,
             'inconsistent FP32 logits for identical checkpoint prefix at record ' + str(i))


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
    check_checkpoint_config(metadata, request)
    need(metadata['step'] == step and metadata['seed'] == seed and
         metadata['cursor'] == step * 4 and metadata['request_identity'] == request['_anchor'],
         'checkpoint metadata mismatch')
    observation = jdoc(archive, prefix + 'observation.json')
    check_observer_state(observation, metadata)
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
    check_prefix_vectors(index, raw)
    checked = []
    by_prediction = {p['id']: p for p in predictions}
    for i, (rec, observed) in enumerate(zip(index, metrics)):
        logits = VECTOR.unpack_from(raw, i * VECTOR.size)
        expected = probs(logits, rec['target'])
        need(metric_equal(observed, expected), 'derived token metric disagrees at record ' + str(i))
        if rec['probe'] == 'greedy':
            pred = by_prediction[rec['row_id']]
            need(expected['argmax'] == pred['tokens'][rec['position']],
                 'reported generated token is not full-vocabulary argmax')
        checked.append(expected)
    saved = jdoc(archive, prefix + 'summary.json')
    residues = summary_check(panel, observed_score, index, checked, saved)
    evaluation = jdoc(archive, prefix + 'evaluation.json')
    need(evaluation['heldout_scored'] is False and evaluation['geometry_loss'] == '0',
         'unsupported evaluation scope')
    native_loss = verified_loss(evaluation, metadata, step)
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
        'native_train_loss': native_loss,
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
        need(type(dataset_manifest.get('files')) is list and
             {m.get('name') for m in dataset_manifest['files'] if type(m) is dict} == DATASET_FILES and
             len(dataset_manifest['files']) == len(DATASET_FILES),
             'incomplete TT-002 dataset files before hashing')
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
        expected_panel, expected_selection = frozen_panel_schedule(train)
        # These source files are included in the separately pinned complete
        # source closure. Model JSON predates the canonical sorted encoding.
        fixed_design = jdoc(archive, 'source/configs/teacher/tt003-fit-001-v1.json')
        fixed_model = json.loads(archive.read('source/configs/teacher/gf3-teacher-90k-v1.json'))
        need(type(fixed_model) is dict and fixed_model.get('schema') == 'trinite.model-config.v1',
             'missing frozen teacher model configuration')
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
            if seed == SEEDS[0]:
                fixed_inventories(request['source'], dataset_manifest, hashes)
            need(request['dataset_rights_identity'] == FROZEN_RIGHTS_ID,
                 'frozen request rights identity differs')
            request['_anchor'] = request_hash
            requests[seed] = request
            scheduled = check_frozen_selection(request, expected_panel, expected_selection)
            panel = expected_panel
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
            need(sum(len(by_id[r]['answer']) + 1 for r in scheduled) == 3590,
                 'independent FIT-001 scheduled target budget mismatch')
            check_frozen_treatment(request, seed, fixed_model, fixed_design)
            attempt_manifest = jdoc(archive, base + 'manifest.json')
            local_anchor = jdoc(archive, f'anchors/seed-{seed}.json')['manifest_identity']
            need(local_anchor == hashes[base + 'manifest.json']['identity'] and
                 attempt_manifest['status'] == 'COMPLETED' and
                 attempt_manifest['seed'] == seed and
                 attempt_manifest['request_identity'] == request_hash,
                 'attempt manifest / external anchor mismatch')
            members_check(hashes, base, attempt_manifest['members'], ('manifest.json',))
            retained_environment = jdoc(archive, base + 'environment.json')
            check_retained_environment(retained_environment, request)
            grant = jdoc(archive, base + 'approval.json')
            validate_approval(grant, request, request_hash,
                              attempt_manifest['approval_identity'],
                              hashes[base + 'approval.json']['identity'])
            supervisor = jdoc(archive, base + 'supervisor.json')
            completed = jdoc(archive, base + 'completed.json')
            need(supervisor['status'] == 'COMPLETED' and
                 supervisor['child_returncode'] == 0 and
                 supervisor['process_resolved'] is True and
                 supervisor['timed_out'] is False and
                 supervisor['interrupted'] is False and
                 float.fromhex(supervisor['elapsed_seconds_hex']) <= 900,
                 'unexpected completion or supervisor state')
            check_completed_receipt(completed)
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
                next_elapsed = canonical_native_time(event['native_elapsed_hex'])
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
                     m['initialization_identity'] == completed['initialization_identity'],
                     'checkpoint trajectory/history mismatch')
                cp_hash = hashes[f'{base}step-{step:03d}/manifest.json']['identity']
                need(any(c['step'] == step and c['manifest_identity'] == cp_hash
                         for c in completed['checkpoints']),
                     'completed checkpoint digest differs')
                previous = row
                series.append(row)
            check_loss_trajectory(series)
            check_native_time_trajectory(series, journal, completed)
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
        fit_seeds, conclusion = fit_conclusion(final)
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
            'fit_seeds': fit_seeds, 'conclusion': conclusion,
        }
    finally:
        archive.close()
