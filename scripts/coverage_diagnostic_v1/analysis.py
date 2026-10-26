"""Read-only, stdlib analysis of one immutable public negative-result subset.

This is not the original closed-custody verifier or a numerical prediction replay.
"""
from pathlib import Path
import sys

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from trinite.foundations_data import audit_bytes, dataset
from trinite.tokenizer import EOS
from unseen_evaluation_v1.packet import inventory, unchanged

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/unseen-negative-seed0-v1'
COMPLETION = 'sha256:2a4f1c03bdcf251262ee85277f731670c4aedfd925b062eac406c49db054c110'
REQUEST = 'sha256:317285b180ca6a101385533c8fbe7ff305d488a63541c19fb833958a6ca849e7'
MODEL = 'sha256:53bc540949ddd10f77e3bbc2b279ba48de8e16a515e33d3d11b02b9242198f47'
CORPUS = 'sha256:7b38b603ded9f0821f87c1eb3a45ebaeba83f1df3eaff3152801867720268133'
ROW_NAMES = tuple(f'row-{i:04d}.json' for i in range(128))
SOURCE_NAMES = (
    'scripts/run_coverage_diagnostic.py',
    'scripts/coverage_diagnostic_v1/__init__.py',
    'scripts/coverage_diagnostic_v1/analysis.py',
    'scripts/unseen_evaluation_v1/__init__.py',
    'scripts/unseen_evaluation_v1/packet.py',
    'src/trinite/__init__.py', 'src/trinite/contracts.py',
    'src/trinite/foundations_data.py', 'src/trinite/foundations_oracle.py',
    'src/trinite/foundations-protocol.json', 'src/trinite/tokenizer.py',
)


MODULE_SOURCES = {
    'coverage_diagnostic_v1': 'scripts/coverage_diagnostic_v1/__init__.py',
    'coverage_diagnostic_v1.analysis': 'scripts/coverage_diagnostic_v1/analysis.py',
    'unseen_evaluation_v1': 'scripts/unseen_evaluation_v1/__init__.py',
    'unseen_evaluation_v1.packet': 'scripts/unseen_evaluation_v1/packet.py',
    'trinite': 'src/trinite/__init__.py',
    'trinite.contracts': 'src/trinite/contracts.py',
    'trinite.foundations_data': 'src/trinite/foundations_data.py',
    'trinite.foundations_oracle': 'src/trinite/foundations_oracle.py',
    'trinite.tokenizer': 'src/trinite/tokenizer.py',
}


def source_receipt():
    for module, name in MODULE_SOURCES.items():
        actual = getattr(sys.modules.get(module), '__file__', None)
        if actual is None or Path(actual).resolve() != (ROOT/name).resolve():
            raise ContractError('analysis dependency imported from a different source root')
    return {name: identity(read_bytes(ROOT/name)) for name in SOURCE_NAMES}


def read_bytes(path, limit=1024*1024):
    """Bounded regular-file read; callers snapshot directories around analysis."""
    import stat
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ContractError('expected bounded regular file')
    with path.open('rb') as stream:
        data = stream.read(limit+1)
    after = path.lstat()
    if len(data) > limit or (before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino,
        after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ContractError('file changed or exceeded bound')
    return data


def corpus():
    raw, admission = dataset('arithmetic')
    audit_bytes(raw, admission)
    if identity(raw) != CORPUS:
        raise ContractError('historical corpus identity changed')
    examples = parse_json(raw, canonical=True)['examples']
    groups = {s: [e for e in examples if e['split'] == s]
              for s in ('train', 'validation', 'test')}
    if {s: len(v) for s, v in groups.items()} != {'train': 264, 'validation': 64, 'test': 64}:
        raise ContractError('historical corpus membership changed')
    return groups


def retained(fixture=FIXTURE):
    """Authenticate the exact public subset, not the omitted original packet."""
    fixture = Path(fixture)
    before = inventory(fixture)
    expected = {'.', 'README.md', 'completion.json', 'row-index.json', *ROW_NAMES}
    if set(before['members']) != expected:
        raise ContractError('public subset membership differs')
    completion_bytes = read_bytes(fixture/'completion.json')
    if identity(completion_bytes) != COMPLETION:
        raise ContractError('original externally retained completion identity differs')
    completion = parse_json(completion_bytes, canonical=True)
    exact_keys(completion, {'schema', 'request_identity', 'manifest_identity',
                           'verification_identity', 'artifacts'}, 'original completion')
    if completion['request_identity'] != REQUEST:
        raise ContractError('original request differs')
    index_bytes = read_bytes(fixture/'row-index.json')
    if identity(index_bytes) != completion['artifacts']['row-index.json']:
        raise ContractError('row index identity differs')
    index = parse_json(index_bytes, canonical=True)
    if set(index) != set(ROW_NAMES):
        raise ContractError('row index membership differs')
    values = []
    for name in ROW_NAMES:
        raw = read_bytes(fixture/name)
        if identity(raw) != index[name] or index[name] != completion['artifacts'][name]:
            raise ContractError('original row identity differs')
        value = parse_json(raw, canonical=True)
        exact_keys(value, {'row', 'targets'}, 'retained row')
        values.append(value)
    unchanged(fixture, before)
    return values


def support(item, train):
    """Describe input/output coverage within the SAME operation, not globally.

    Length and range are independent descriptive flags, not equivalent to novel
    answers. This function does not assign splits, choose training, or score a model.
    """
    peers = [e for e in train if e['task'] == item['task']]
    if not peers:
        raise ContractError('operation has no training support')
    a, b = item['formal']['a'], item['formal']['b']
    answers = {e['answer'] for e in peers}
    operands = {e['formal'][k] for e in peers for k in ('a', 'b')}
    families = {e['family_id'] for e in peers}
    lengths = {len(e['answer'].encode('ascii')) for e in peers}
    return {
        'family_seen_in_training': item['family_id'] in families,
        'gold_answer_seen_in_same_task_training': item['answer'] in answers,
        'operand_values_seen_in_same_task_training': a in operands and b in operands,
        'operand_range_extrapolation': min(a, b) < min(operands) or max(a, b) > max(operands),
        'operand_digit_length_extrapolation': max(len(str(abs(a))), len(str(abs(b)))) >
            max(len(str(abs(v))) for v in operands),
        'answer_byte_length_seen_in_same_task_training': len(item['answer'].encode('ascii')) in lengths,
    }


def _summary(values):
    rows = [v['row'] for v in values]
    targets = [t for v in values for t in v['targets']]
    answer_targets = [t for t in targets if not t['is_eos']]
    eos_targets = [t for t in targets if t['is_eos']]
    return {
        'rows': len(rows), 'semantic_items': len({r['semantic_identity'] for r in rows}),
        'families': len({r['family_id'] for r in rows}),
        'raw_answer_and_eos_correct': sum(r['generated_tokens'] == r['expected_tokens'] for r in rows),
        'teacher_forced_correct': sum(t['argmax_token'] == t['gold_token'] for t in targets),
        'teacher_forced_targets': len(targets),
        'teacher_forced_answer_correct': sum(t['argmax_token'] == t['gold_token'] for t in answer_targets),
        'teacher_forced_answer_targets': len(answer_targets),
        'teacher_forced_eos_correct': sum(t['argmax_token'] == t['gold_token'] for t in eos_targets),
        'teacher_forced_eos_targets': len(eos_targets),
        'distinct_generated_outputs': len({tuple(r['generated_tokens']) for r in rows}),
    }


def development_proposal():
    """Frozen review scope, deliberately not an executable experiment request."""
    return {
        'schema': 'trinite.coverage-development-proposal.v1',
        'status': 'review-only; no target execution or training authorization',
        'operation': 'add',
        'selection_reason': 'familiar-answer recombination failed; addition avoids novel same-task answer strings in historical heldouts',
        'historical_role': 'negative exploratory diagnosis only; never fresh confirmation',
        'primary_question': 'does a prespecified training-coverage intervention improve familiar-answer new-combination development performance?',
        'required_separate_strata': ['familiar-answer/new-family', 'novel-answer/new-family',
                                    'operand-range/length-extrapolation'],
        'held_fixed': ['dense architecture', 'tokenizer/carriers', 'optimizer',
                       'initialization seed set', 'update count', 'scheduled target/input exposure',
                       'raw greedy decoding and output cap'],
        'before_execution_requires': ['new separately versioned admitted development corpus',
            'family-disjoint splits and complete coverage table',
            'one coverage intervention and matched control frozen before target outcomes',
            'explicit budgets, seeds, update/target schedule and failure retention',
            'new source-bound implementation/request and green review',
            'separate owner authorization for local target execution'],
        'confirmation': 'reserve genuinely new multiple families with exposure review after development recipe freeze; do not generate or expose them here',
        'model_executed': False, 'training_enabled': False,
        'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
    }


def analyze(fixture=FIXTURE):
    source = source_receipt()
    groups = corpus()
    values = retained(fixture)
    selected = groups['validation'] + groups['test']
    if len(values) != len(selected):
        raise ContractError('all historical held-out rows required')
    observations = []
    for item, value in zip(selected, values):
        row, targets = value['row'], value['targets']
        for key in ('example_identity', 'semantic_identity', 'family_id', 'task', 'carrier', 'split'):
            if row[key] != item[key]:
                raise ContractError('retained row differs from admitted corpus')
        expected = [*item['answer'].encode('ascii'), EOS]
        if row['expected_tokens'] != expected or [t['gold_token'] for t in targets] != expected:
            raise ContractError('retained labels differ from admitted corpus')
        if any(t['example_identity'] != item['example_identity'] for t in targets):
            raise ContractError('target belongs to another row')
        flags = support(item, groups['train'])
        peers = [e for e in groups['train'] if e['task'] == item['task']]
        generated = row['generated_tokens']
        generated_answers = {(*e['answer'].encode('ascii'), EOS) for e in peers}
        observations.append({
            'example_identity': item['example_identity'], 'split': item['split'], 'task': item['task'],
            'family_id': item['family_id'], 'semantic_identity': item['semantic_identity'],
            'operand_magnitudes': sorted((abs(item['formal']['a']), abs(item['formal']['b']))),
            'support': flags,
            'generated_answer_seen_in_same_task_training': tuple(generated) in generated_answers,
        })
    splits = {}
    for split in ('validation', 'test'):
        group = [v for v in values if v['row']['split'] == split]
        observed = [o for o in observations if o['split'] == split]
        tasks = {}
        for task in ('add', 'subtract', 'multiply', 'divide'):
            task_values = [v for v in group if v['row']['task'] == task]
            task_obs = [o for o in observed if o['task'] == task]
            counts = {flag: sum(o['support'][flag] for o in task_obs) for flag in task_obs[0]['support']}
            tasks[task] = {'outcome': _summary(task_values), 'support_counts': counts,
                          'generated_answer_seen_in_training': sum(o['generated_answer_seen_in_same_task_training'] for o in task_obs)}
        splits[split] = {'outcome': _summary(group), 'by_task': tasks,
                         'operand_magnitude_families': sorted({tuple(o['operand_magnitudes']) for o in observed})}
        splits[split]['operand_magnitude_families'] = [list(v) for v in splits[split]['operand_magnitude_families']]
    if source != source_receipt():
        raise ContractError('analysis source changed during inspection')
    return {
        'schema': 'trinite.coverage-diagnostic.v1', 'sources': source,
        'historical_corpus_identity': CORPUS,
        'original_evaluator_commit': '4045f03f4a2d8ce2a68bc50625bf0c6c4b6208a9',
        'original_evaluation_request_identity': REQUEST,
        'original_evaluation_completion_identity': COMPLETION,
        'reported_original_model_identity': MODEL,
        'model_identity_scope': 'reported linkage from privately audited original packet; not independently derivable from this public subset',
        'verification_scope': 'original completion-to-index-to-row hashes, admitted labels/membership and recomputed descriptive counts; not full custody or prediction replay',
        'historical_exposure': 'one family per split, two carriers per item; exploratory, not independent trials or fresh confirmation',
        'by_split': splits, 'row_support': observations,
        'proposal': development_proposal(),
        'predictions_replayed': False, 'training_executed': False,
        'fresh_confirmation': False, 'learning_adequate': False,
        'phase5_ready': False, 'release_ready': False,
    }
