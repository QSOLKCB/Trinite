"""Freeze model, historical exposure, admitted rows and evaluator before inference."""
from pathlib import Path

from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from trinite.training import require_environment
from native_diagnostic_v1.request import ROOT, read, sources as producer_sources
from .producer import inspect_packet
from .measurements import examples, train_examples

PROTOCOL = {
    'schema': 'trinite.unseen-evaluation-protocol.v1',
    'workload': 'arithmetic', 'lane': 'dense', 'checkpoint_step': 4096,
    'generation_cap': 5, 'context_length': 64,
    'partitions': ['validation', 'test'],
    'exposure': 'unseen to this checkpoint; historical research-exposed families; exploratory, not fresh confirmation',
    'selection': 'all historical validation/test rows; final4096 only; no outcome-based selection',
    'worker_seconds': 900, 'max_worker_log_bytes': 1024 * 1024,
    'max_artifact_bytes': 16 * 1024 * 1024, 'max_retained_payload_bytes': 32 * 1024 * 1024,
    'decoding': 'prompt-only greedy; one attempt; no repair, fallback, retries or gold prefix',
    'teacher_forcing': 'gold-prefix diagnostic, separately reported; not raw generation',
    'updates': 0, 'geometry_loss': '0', 'cache_mode': 'unsupported',
    'acceptance': 'descriptive counts only; no learning, Phase5, format or release gate',
    'independence': 'one family per split; correlated tasks, signs and carriers; no iid confidence interval',
}


def sources():
    paths = list((ROOT/'scripts/unseen_evaluation_v1').glob('*.py'))
    paths += [ROOT/'scripts/run_unseen_evaluation.py', ROOT/'docs/UNSEEN-EVALUATION.md']
    return {**producer_sources(), **{p.relative_to(ROOT).as_posix(): identity(read(p)) for p in sorted(paths)}}


def core(producer, request_identity, completion_identity, model_identity):
    binding = inspect_packet(producer, request_identity, completion_identity, model_identity)
    heldout, train = examples(), train_examples()
    return {'schema': 'trinite.unseen-evaluation-request.v1', 'protocol': PROTOCOL,
            'protocol_identity': identity(json_bytes(PROTOCOL)), 'producer': binding,
            'source': sources(), 'environment': require_environment(),
            'evaluation_examples_identity': identity(json_bytes(heldout)),
            'training_examples_identity': identity(json_bytes(train))}


def validate(raw, expected_identity, producer):
    require_identity(expected_identity)
    if identity(raw) != expected_identity:
        raise ContractError('unseen evaluation request identity mismatch')
    record = parse_json(raw, canonical=True)
    if type(record) is not dict or type(record.get('producer')) is not dict:
        raise ContractError('unseen evaluation request requires producer binding')
    binding = record['producer']
    try:
        expected = core(producer, binding['producer_request_identity'],
                        binding['producer_completion_identity'], binding['model_identity'])
    except KeyError as error:
        raise ContractError('incomplete producer binding') from error
    if raw != json_bytes(expected):
        raise ContractError('unseen evaluation source/environment/producer/protocol mismatch')
    return record


def separate(destination, producer):
    """No writes inside or over an original packet, including symlink aliases."""
    destination, producer = Path(destination).resolve(), Path(producer).resolve()
    if destination == producer or destination in producer.parents or producer in destination.parents:
        raise ContractError('output/request must be disjoint from original producer packet')
