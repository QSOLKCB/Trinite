"""Separate, exact finite music admission; historical admissions are unchanged.

Only this commissioned formal generator and lossless adapter are admitted.
Independent audit regenerates the complete expected bytes; an identity or a
self-reported success field alone never admits arbitrary supplied content.
"""
from pathlib import Path
from importlib import import_module
import stat

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from symbolic_music_v1.corpus import generate, validate_rows
from symbolic_music_v1.oracle import annotate
from .codec import TASKS, encode_example, make_example, parse_prompt
from .protocol import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
POLICY = 'trinite.music-learning-data.v1'
MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_ADMISSION_BYTES = 1024 * 1024
SOURCES = (
    'scripts/music_learning_v1/__init__.py',
    'scripts/music_learning_v1/admission.py',
    'scripts/music_learning_v1/codec.py',
    'scripts/music_learning_v1/protocol.py',
    'scripts/symbolic_music_v1/__init__.py',
    'scripts/symbolic_music_v1/core.py',
    'scripts/symbolic_music_v1/corpus.py',
    'scripts/symbolic_music_v1/oracle.py',
    'scripts/run_symbolic_music.py',
    'src/trinite/__init__.py',
    'src/trinite/contracts.py',
    'src/trinite/tokenizer.py',
    'docs/SYMBOLIC-MUSIC.md',
    'docs/MUSIC-LEARNING.md',
)


def source_receipt():
    if ROOT != Path(__file__).resolve().parents[2]:
        raise ContractError('music admission requires the executing checkout')
    modules = {
        'music_learning_v1': 'scripts/music_learning_v1/__init__.py',
        'music_learning_v1.admission': 'scripts/music_learning_v1/admission.py',
        'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py',
        'music_learning_v1.protocol': 'scripts/music_learning_v1/protocol.py',
        'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py',
        'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py',
        'symbolic_music_v1.corpus': 'scripts/symbolic_music_v1/corpus.py',
        'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py',
        'trinite': 'src/trinite/__init__.py',
        'trinite.contracts': 'src/trinite/contracts.py',
        'trinite.tokenizer': 'src/trinite/tokenizer.py',
    }
    for module_name, relative in modules.items():
        location = getattr(import_module(module_name), '__file__', None)
        if type(location) is not str or Path(location).resolve() != ROOT / relative:
            raise ContractError('foreign music admission import: ' + module_name)
    sources = {}
    for name in SOURCES:
        path = ROOT / name
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != ROOT.parent):
            raise ContractError('symlink in music admission source')
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise ContractError('invalid bounded music admission source')
        sources[name] = identity(path.read_bytes())
    return sources


def dataset():
    """Regenerate all 160 parent rows and all 960 fixed task projections."""
    sources = source_receipt()
    revision = identity(json_bytes(sources))
    parents = generate()
    validate_rows(parents)
    if len(parents) != 160:
        raise ContractError('music parent inventory changed')
    examples = []
    for parent in parents:
        if annotate(parent['input']['score']) != parent['target']:
            raise ContractError('music independent parent annotation mismatch')
        for task in TASKS:
            adapted = make_example(parent['input'], task)
            encoded = encode_example(adapted)
            parsed = parse_prompt(adapted['prompt'])
            if parsed != {'task': task, 'score': parent['input']['score']}:
                raise ContractError('music prompt changed formal meaning')
            core = {**adapted, 'family_id': parent['family'], 'split': parent['split'],
                    'parent_id': parent['id'], 'source_ids': [POLICY],
                    'generator_revision': revision, 'seed': 0,
                    'formal': parent['input']['score'], 'verifier_outcome': 'verified',
                    'transformation_lineage': {'parent_identity': identity(json_bytes(parent)),
                                               'projection': task, 'carrier': 'music-ascii-v1'},
                    'meter': str(parsed['score']['meter']['numerator']) + '/' +
                             str(parsed['score']['meter']['denominator']),
                    'token_count': len(encoded.input_ids),
                    'answer_token_count': sum(encoded.loss_mask)}
            examples.append({**core, 'example_identity': identity(json_bytes(core))})
    if len(examples) != 960 or len({e['prompt'] for e in examples}) != 960:
        raise ContractError('music projection inventory or duplicate prompt')
    family_splits = {}
    for example in examples:
        family_splits.setdefault(example['family_id'], set()).add(example['split'])
    if len(family_splits) != 8 or any(len(s) != 1 for s in family_splits.values()):
        raise ContractError('music whole-family separation failed')
    split_counts = {s: sum(e['split'] == s for e in examples) for s in ('train', 'validation', 'test')}
    if split_counts != {'train': 480, 'validation': 240, 'test': 240}:
        raise ContractError('music fixed split counts changed')
    protocol_identity = identity(json_bytes(PROTOCOL))
    raw = json_bytes({'schema': POLICY, 'protocol_identity': protocol_identity,
                      'examples': examples})
    record = {
        'schema': 'trinite.music-learning-admission.v1', 'source_id': POLICY,
        'dataset_identity': identity(raw), 'sources': sources,
        'generator_revision': revision, 'protocol_identity': protocol_identity,
        'origin': 'bounded original formal monophonic notation generator; no imported compositions, recordings, sheets, teacher output or donor assets',
        'author': 'Codex deterministic generator commissioned by Trent Slade / QSOL-IMC',
        'generation': 'eight rest-mask parents, ten meter/group settings and two notation variants; six fixed task projections per score; split seed zero',
        'verification': 'full parent regeneration, independent integer-tick annotations, independent compact parser, exact formal roundtrip and native byte-token answer masks',
        'rights_basis': 'newly rendered formal timing/count facts and minimal symbolic notation; no third-party expressive music or prose embedded',
        'supporting_reference': {'formal_items': identity(json_bytes([p['input']['score'] for p in parents])),
                                 'exact_rendered_items': identity(json_bytes(examples)),
                                 'source_functions': ['symbolic_music_v1.corpus.generate',
                                                      'symbolic_music_v1.oracle.annotate',
                                                      'music_learning_v1.codec.parse_score',
                                                      'music_learning_v1.admission.dataset']},
        'scope': 'only these exact 160 public formal scores and 960 source-bound projections under this frozen music-only protocol; no blanket synthetic-data or external-source admission',
        'reviewer': 'automated source-specific rights, formal-fact, grammar, token-budget and whole-family audit under the owner-commissioned music training/evaluation implementation',
        'reviewed_on': '2026-10-27', 'outcome': 'admitted',
        'execution_authorization': 'separate explicit target-run authorization required; admission is not permission to execute training or inference',
        'limitations': 'not legal certification, a powered study or fresh confirmation; all families public/development-exposed and share construction grammar; signature/group annotations include easy readout',
        'counts': {'parents': 160, 'examples': 960, 'families': 8, **split_counts},
        'family_counts': {s: sum(next(iter(v)) == s for v in family_splits.values())
                          for s in split_counts},
        'exclusions': {'external_content': 0, 'dropped_for_model_success': 0,
                       'truncated': 0, 'invalid': 0},
    }
    if source_receipt() != sources:
        raise ContractError('music admission source changed during generation')
    return raw, json_bytes(record)


def validate_admission(admission):
    """Standalone exact policy validation, not a nonempty-field/hash check."""
    if type(admission) is bytes:
        if len(admission) > MAX_ADMISSION_BYTES:
            raise ContractError('music admission exceeds byte bound')
        value = parse_json(admission, canonical=True)
    elif type(admission) is dict:
        if len(json_bytes(admission)) > MAX_ADMISSION_BYTES:
            raise ContractError('music admission exceeds byte bound')
        value = admission
    else:
        raise ContractError('music admission must be exact bytes or a record')
    _, expected = dataset()
    if json_bytes(value) != expected:
        raise ContractError('music source-specific admission mismatch')
    return value


def audit_bytes(raw, admission):
    """Read-only deterministic source/input audit, not model or update replay."""
    if type(raw) is not bytes or len(raw) > MAX_DATASET_BYTES:
        raise ContractError('music dataset requires bounded exact bytes')
    if type(admission) is not bytes or len(admission) > MAX_ADMISSION_BYTES:
        raise ContractError('music admission requires bounded exact bytes')
    parse_json(raw, canonical=True)
    parse_json(admission, canonical=True)
    expected_raw, expected_admission = dataset()
    if raw != expected_raw or admission != expected_admission:
        raise ContractError('music exact source/input admission replay mismatch')
    return {'verified': True, 'dataset_identity': identity(raw),
            'admission_identity': identity(admission), 'examples': 960,
            'scope': 'exact formal source/input regeneration only; no training or inference replay'}
