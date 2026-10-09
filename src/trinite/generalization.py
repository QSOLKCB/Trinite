"""Freshly verified descriptive diagnostics; preparation cannot unlock gates."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from .learning_data import dataset as learning_dataset
from .curriculum_data import dataset as curriculum_dataset, source_receipt as curriculum_sources
from .tokenizer import EOS

PROTOCOL_IDENTITY = 'sha256:0deb6c563655e7b84ac41afe102ecbfaa580da80d3ac6608c604abb55e22d1ec'
PROTOCOL_COMMIT = '9924e82047b728e229094c37f5714bb268ebe6a3'
WORKLOADS = ('arithmetic', 'fold', 'lattice')


def read(path, limit=16*1024*1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ContractError('unsafe/missing diagnostic input')
    with path.open('rb') as stream: raw = stream.read(limit+1)
    if len(raw) > limit: raise ContractError('diagnostic input exceeds bound')
    return raw


def protocol():
    raw = read(Path(__file__).parent/'generalization-protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY: raise ContractError('diagnostic protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    # Imports remain lazy: admission and diagnostic arithmetic are stdlib-only;
    # the run verifier requires the same hash-locked CPU environment as its parent.
    from .learning_training import sources as learning_sources
    package = Path(__file__).resolve().parent
    return {**learning_sources(), **curriculum_sources(),
        'generalization.py': identity(read(package/'generalization.py')),
        'generalization-protocol.json': PROTOCOL_IDENTITY,
        'scripts/run_generalization.py': identity(read(package.parents[1]/'scripts/run_generalization.py'))}


def request_core():
    from .training import require_environment
    return {'schema': 'trinite.generalization-request.v1', 'protocol_identity': PROTOCOL_IDENTITY,
        'protocol_commit': PROTOCOL_COMMIT, 'protocol': protocol(), 'source': sources(),
        'environment': require_environment(),
        'curricula': {w: {'dataset_identity': identity(curriculum_dataset(w)[0]),
                         'manifest_identity': identity(curriculum_dataset(w)[1])} for w in WORKLOADS}}


def freeze_request(path):
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY}


def read_request(path, expected_identity):
    require_identity(expected_identity); raw = read(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('diagnostic request identity mismatch')
    record = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(record, set(expected)|{'created_at'}, 'diagnostic request')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('diagnostic request source/environment/data/protocol mismatch')
    try:
        if datetime.fromisoformat(record['created_at']).utcoffset() is None: raise ValueError('timezone required')
    except (ValueError, TypeError) as error:
        raise ContractError('invalid diagnostic request clock') from error
    return record, raw


def answer_body(tokens):
    """EOS must be final and unique; byte bodies must decode as strict ASCII."""
    if type(tokens) is not list or not tokens:
        raise ContractError('nonempty generated tokens required')
    for token in tokens: integer(token, 'diagnostic token', 0, 258)
    if any(t > 255 for t in tokens[:-1]) or tokens[-1] in (256, 258):
        return None, 'special-token'
    if tokens[-1] != EOS: return None, 'missing-eos'
    try: return bytes(tokens[:-1]).decode('ascii'), None
    except UnicodeDecodeError: return None, 'non-ascii'


def component_counts(workload, item, tokens):
    body, _ = answer_body(tokens)
    if workload == 'fold':
        labels = ('count', 'sum', 'squares'); expected = item['answer'].split(' ')
        match = re.fullmatch(r'(0|[1-9][0-9]*) (0|-?[1-9][0-9]*) (0|[1-9][0-9]*)', body or '', flags=re.ASCII)
    elif workload == 'lattice' and item['task'] == 'traversal':
        labels = ('x', 'y', 'z'); expected = re.fullmatch(r'L\[([0-2]),([0-2]),([0-2])\]', item['answer']).groups()
        match = re.fullmatch(r'L\[([0-2]),([0-2]),([0-2])\]', body or '', flags=re.ASCII)
    else: return {}
    return {label: int(match is not None and match.groups()[i] == expected[i]) for i, label in enumerate(labels)}


def corpus_coverage(examples):
    families = defaultdict(set); semantic = defaultdict(set)
    for item in examples:
        families[item['family_id']].add(item['split']); semantic[item['semantic_identity']].add(item['split'])
    if any(len(v) != 1 for v in [*families.values(), *semantic.values()]):
        raise ContractError('diagnostic corpus leaks a family/semantic item across splits')
    result = {}
    for task in sorted({r['task'] for r in examples}):
        train = [r for r in examples if r['task'] == task and r['split'] == 'train']
        if not train: raise ContractError('diagnostic task lacks training rows')
        answers = {r['answer'] for r in train}; prompt_bytes = set().union(*(set(r['prompt'].encode()) for r in train))
        result[task] = {'training_examples': len(train), 'training_answers': sorted(answers),
            'training_prompt_bytes': sorted(prompt_bytes), 'splits': {}}
        for split in ('validation', 'test'):
            selected = [r for r in examples if r['task'] == task and r['split'] == split]
            result[task]['splits'][split] = {'examples': len(selected),
                'answer_seen_in_train': sum(r['answer'] in answers for r in selected),
                'unseen_answers': sorted({r['answer'] for r in selected if r['answer'] not in answers}),
                'unseen_prompt_bytes': sorted(set().union(*(set(r['prompt'].encode()) for r in selected))-prompt_bytes)}
    return result


def diagnose(workload, examples, predictions, split):
    """Supplementary counts only, over upstream-verified full prediction rows."""
    if workload not in WORKLOADS or split not in ('train', 'test'): raise ContractError('invalid diagnostic selector')
    selected = {r['example_identity']: r for r in examples if r['split'] == split}
    if type(predictions) is not list or len(predictions) != len(selected): raise ContractError('incomplete diagnostic predictions')
    train_answers = defaultdict(set)
    for item in examples:
        if item['split'] == 'train': train_answers[item['task']].add(item['answer'])
    groups = {}; seen = set()
    for row in predictions:
        key = row['example_identity']
        if key not in selected or key in seen: raise ContractError('duplicate/unknown diagnostic prediction')
        seen.add(key); item = selected[key]; tokens = row['generated_tokens']
        body, error = answer_body(tokens)
        correct = tokens == [*item['answer'].encode(), EOS]
        if type(row['correct']) is not bool or row['correct'] != correct:
            raise ContractError('diagnostic prediction correctness mismatch')
        bucket = 'exact' if correct else error or 'wrong-answer'
        components = component_counts(workload, item, tokens)
        axes = {'task': item['task'], 'carrier': item['carrier'],
                'answer-coverage': 'seen' if item['answer'] in train_answers[item['task']] else 'unseen'}
        for axis, value in axes.items():
            name = axis+':'+value
            if name not in groups:
                groups[name] = {'examples': 0, 'correct': 0, 'answer_with_eos_correct': 0,
                    'errors': dict.fromkeys(('exact', 'wrong-answer', 'missing-eos', 'special-token', 'non-ascii'), 0),
                    'components': {}}
            group = groups[name]; group['examples'] += 1; group['correct'] += int(correct)
            group['answer_with_eos_correct'] += int(body == item['answer'])
            group['errors'][bucket] += 1
            for label, count in components.items():
                component = group['components'].setdefault(label, {'examples': 0, 'correct': 0})
                component['examples'] += 1; component['correct'] += count
    return dict(sorted(groups.items()))


def run_inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise ContractError('diagnostic run must be a stable directory')
    limits = protocol()['resources']; result = {}; total = 0
    for entry, path in enumerate(root.rglob('*'), 1):
        if entry > limits['max_run_files']: raise ContractError('diagnostic run inventory exceeds bound')
        if path.is_symlink() or not (path.is_file() or path.is_dir()): raise ContractError('unsafe diagnostic run member')
        if path.is_file():
            raw = read(path, 32*1024*1024); total += len(raw)
            result[path.relative_to(root).as_posix()] = identity(raw)
            if len(result) > limits['max_run_files'] or total > limits['max_run_bytes']:
                raise ContractError('diagnostic run inventory exceeds bound')
    return result


def analyze_run(root, request, request_identity):
    from .learning import summarize, check_predictions
    root = Path(root); _, request_raw = read_request(request, request_identity)
    before = run_inventory(root); settings = protocol(); parent = settings['parent']
    for name, key in (('request.json', 'request_identity'), ('summary.json', 'summary_identity'),
                      ('test-decision.json', 'decision_identity')):
        if before.get(name) != parent[key]: raise ContractError('diagnostic parent '+name+' identity mismatch')
    old = parse_json(read(root/'summary.json'), canonical=True)
    current = summarize(root, root/'request.json', parent['request_identity'], old['worker_errors'], publish=False)
    if json_bytes(current) != read(root/'summary.json') or current['outcome'] != 'completed':
        raise ContractError('diagnostic parent differs from fresh authoritative verification')
    reports = []; coverage = {}
    for workload in WORKLOADS:
        examples = parse_json(learning_dataset(workload)[0], canonical=True)['examples']
        coverage[workload] = corpus_coverage(examples)
        for seed in settings['matrix']['seeds']:
            for lane in settings['matrix']['lanes']:
                name = f'{workload}-{seed}-{lane}'; directory = root/name; splits = {}
                for split, file in (('train', 'training-predictions.json'), ('test', 'test-predictions.json')):
                    raw = read(directory/file); check_predictions(workload, split, raw)
                    splits[split] = diagnose(workload, examples, parse_json(raw, canonical=True)['predictions'], split)
                reports.append({'workload': workload, 'seed': seed, 'lane': lane, 'diagnostics': splits})
    if run_inventory(root) != before: raise ContractError('diagnostic parent files changed during verification/analysis')
    read_request(request, request_identity)
    return {'schema': 'trinite.generalization-report.v1', 'outcome': 'completed',
        'request_identity': request_identity, 'request_file_identity': identity(request_raw),
        'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
        'source': sources(), 'parent': parent, 'parent_files': len(before),
        'parent_inventory_identity': identity(json_bytes(before)), 'parent_inventory_unchanged': True,
        'fresh_parent_verification': 'closed evidence, checkpoints, regenerated final predictions/loss and summary',
        'corpus_coverage': coverage, 'cells': reports,
        'curriculum_preparation': request_core()['curricula'],
        'curriculum_training_completed': False, **settings['gates'], 'scope': settings['scope']}


def evidence_files(output, run):
    output, run = Path(output), Path(run)
    parent = protocol()['parent']
    inputs = {'diagnostic-request.json': read(output/'request.json'),
              'parent-request.json': read(run/'request.json'),
              'parent-summary.json': read(run/'summary.json'),
              'parent-decision.json': read(run/'test-decision.json')}
    for name, key in (('parent-request.json','request_identity'), ('parent-summary.json','summary_identity'),
                      ('parent-decision.json','decision_identity')):
        if identity(inputs[name]) != parent[key]: raise ContractError('diagnostic evidence parent changed')
    outputs = {'report.json': read(output/'report.json')}
    for workload in WORKLOADS:
        for name in ('dataset.json','manifest.json'):
            outputs['curriculum/'+workload+'/'+name] = read(output/'curriculum'/workload/name)
    return inputs, outputs


def retain_evidence(output, run):
    from .observation import BundleObserver
    inputs, outputs = evidence_files(output, run)
    observer = BundleObserver(Path(output)/'provenance')
    observer.record('generalization-diagnostic', inputs=inputs, outputs=outputs)
    verification = observer.finalize()
    with (Path(output)/'verification.json').open('xb') as stream: stream.write(json_bytes(verification))
    verify_evidence(output, run)


def verify_evidence(output, run):
    from .observation import verify_observation
    bundle = Path(output)/'provenance'
    manifest_raw = read(bundle/'manifest.json'); verification = verify_observation(bundle)
    if read(bundle/'manifest.json') != manifest_raw: raise ContractError('diagnostic manifest changed during verification')
    if read(Path(output)/'verification.json') != json_bytes(verification):
        raise ContractError('retained diagnostic verification receipt differs from fresh verification')
    manifest = parse_json(manifest_raw, canonical=True)
    if manifest['manifest_identity'] != verification['manifest_identity']:
        raise ContractError('diagnostic evidence logical identity mismatch')
    members = {a['content_identity'] for a in manifest['core']['artifacts']}; indexes = []
    for key in members:
        try: value = parse_json(read(bundle/'artifacts/sha256'/key[7:]))
        except ContractError: continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1': indexes.append(value)
    if len(indexes) != 1: raise ContractError('diagnostic evidence requires one artifact index')
    inputs, outputs = evidence_files(output, run)
    expected = {name: identity(raw) for name, raw in {**inputs, **outputs}.items()}
    if indexes[0]['artifacts'] != expected or not set(expected.values()) <= members:
        raise ContractError('diagnostic root files differ from closed evidence')
    return verification
