"""Small explicit, pre-outcome model-only request and source binding."""
import os
from pathlib import Path
import stat
import sys

from trinite.contracts import (ContractError, ModelConfig, exact_keys, identity,
                              integer, json_bytes, parse_json, require_identity)

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {'workers': 1, 'threads': 1, 'interop_threads': 1, 'batch': 1,
          'max_prompts': 4, 'max_repetitions': 3, 'max_output_tokens': 16,
          'max_parameters': 5_000_000, 'max_context': 256,
          'checkpoint_bytes': 32 * 1024 * 1024, 'request_bytes': 1024 * 1024,
          'artifact_bytes': 4 * 1024 * 1024, 'worker_log_bytes': 64 * 1024,
          'retained_store_bytes': 32 * 1024 * 1024, 'provenance_payload_bytes': 64 * 1024 * 1024,
          'max_worker_seconds': 600, 'memory_ceiling_bytes': None}
PLAN_KEYS = {'config', 'lane', 'model_identity', 'checkpoint_identity', 'prompts',
             'repetitions', 'output_tokens', 'worker_seconds'}


def read(path, limit=LIMITS['artifact_bytes']):
    """Bounded regular read, rejecting symlink ancestors and replacement."""
    path = Path(path).absolute()
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ContractError('input ancestors must be non-link directories')
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ContractError('bounded regular input required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if signature(os.fstat(fd)) != signature(before):
            raise ContractError('input changed at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        if (len(raw) > limit or signature(os.fstat(fd)) != signature(before)
                or signature(path.lstat()) != signature(before)):
            raise ContractError('input changed during read')
        return raw
    finally:
        os.close(fd)


def plan(value):
    exact_keys(value, PLAN_KEYS, 'CPU ledger plan')
    config = ModelConfig.from_dict(value['config'])
    if config.parameter_count > LIMITS['max_parameters']:
        raise ContractError('predicted model parameters exceed admission budget')
    if value['lane'] not in ('dense', 'ternary', 'four-state'):
        raise ContractError('unsupported native lane')
    require_identity(value['model_identity']); require_identity(value['checkpoint_identity'])
    repetitions = integer(value['repetitions'], 'repetitions', 1, LIMITS['max_repetitions'])
    cap = integer(value['output_tokens'], 'output tokens', 1, LIMITS['max_output_tokens'])
    integer(value['worker_seconds'], 'worker seconds', 1, LIMITS['max_worker_seconds'])
    prompts = value['prompts']
    if type(prompts) is not list or not 1 <= len(prompts) <= LIMITS['max_prompts']:
        raise ContractError('bounded prompt list required')
    lengths = []
    for prompt in prompts:
        if type(prompt) is not str:
            raise ContractError('prompt must be exact UTF-8 text')
        length = 1 + len(prompt.encode('utf-8'))
        if length + cap - 1 > config.context_length:
            raise ContractError('prompt plus full output budget exceeds context; no truncation')
        if config.heads * (length + cap - 1) ** 2 > 4_194_304:
            raise ContractError('predicted attention cells exceed native admission budget')
        lengths.append(length)
    if len(set(prompts)) != len(prompts):
        raise ContractError('duplicate prompts; use fixed repetitions instead')
    return {'limits': LIMITS, 'maximum_forward_calls': 3 * repetitions * len(prompts) * cap,
            'maximum_processed_positions': 3 * repetitions * sum(cap*n + cap*(cap-1)//2 for n in lengths),
            'maximum_emitted_tokens_all_passes': 3 * repetitions * len(prompts) * cap,
            'predicted_unique_parameter_bytes': config.parameter_count * 4,
            'prompt_bindings': [{'utf8_identity': identity(p.encode('utf-8')),
                'initial_prefix_identity': identity(json_bytes([256, *p.encode('utf-8')])),
                'input_tokens_including_bos': n} for p, n in zip(prompts, lengths)]}


def sources():
    paths = [p for p in (ROOT/'src').rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    paths += list((ROOT/'scripts/cpu_cost_ledger_v1').glob('*.py'))
    paths += [ROOT/p for p in ('scripts/run_cpu_cost_ledger.py', 'requirements.lock', 'AGENTS.md',
        'docs/CPU-COST-LEDGER.md', 'scripts/native_diagnostic_v1/__init__.py',
        'scripts/native_diagnostic_v1/artifacts.py', 'scripts/unseen_evaluation_v1/__init__.py',
        'scripts/unseen_evaluation_v1/packet.py')]
    allowed = {p.absolute() for p in paths}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('trinite', 'cpu_cost_ledger_v1', 'native_diagnostic_v1', 'unseen_evaluation_v1') or name.startswith('provenance_'):
            file = getattr(module, '__file__', None)
            if file is None or Path(file).absolute() not in allowed or Path(file).is_symlink():
                raise ContractError('import outside source-bound checkout: ' + name)
    return {p.relative_to(ROOT).as_posix(): identity(read(p)) for p in sorted(paths)}


def environment():
    from trinite.training import require_environment
    result = require_environment()
    if result['interop_threads'] != 1:
        raise ContractError('ledger requires one interop thread')
    # Reuse the authoritative pinned backend gate, not its training label.
    return {**result, 'optimizer': 'none; inference only'}


def core(value):
    bounds = plan(value)
    return {'schema': 'trinite.cpu-cost-ledger-request.v1', 'plan': value, 'bounds': bounds,
            'source': sources(), 'environment': environment(),
            'decode_policy': 'unchanged trinite.comparison.greedy; BOS/UTF-8; lowest-ID argmax; EOS or output cap',
            'ordering': 'prompt order, then repetition; uninstrumented-before, instrumented, uninstrumented-after',
            'cache_mode': 'unsupported', 'training_updates': 0}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw) > LIMITS['request_bytes'] or identity(raw) != expected:
        raise ContractError('request identity/size differs')
    value = parse_json(raw, canonical=True)
    if type(value) is not dict or 'plan' not in value or raw != json_bytes(core(value['plan'])):
        raise ContractError('request source/environment/plan differs')
    return value
