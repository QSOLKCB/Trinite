"""Read-only, digest-bound diagnostics for Trinite's *unchanged* ternary STE.

No model update, scoring, RNG draw, synthetic teacher, or research-gate promotion.
This is a descriptive checkpoint reader, NOT a full optimizer/custody replay.
"""
from __future__ import annotations

from pathlib import Path
import re

import torch
from safetensors.torch import load as load_safetensors

from trinite.contracts import (ContractError, ModelConfig, exact_keys, identity,
                               integer, json_bytes, parse_json, require_identity)
from trinite.inspection import tensor_identity
from trinite.quantizer import quantize, require_float_tensor

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_IDENTITY = 'sha256:14261ecc9c6bf5f9487a71a1def5ce9115298de6cefd9fd9b702eb91a06bbebf'
PROTOCOL_PATH = Path(__file__).with_name('protocol.json')
SELECTION_SCHEMA = 'trinite.ternary-dynamics-selection.v1'
REQUEST_SCHEMA = 'trinite.ternary-dynamics-request.v2'
REPORT_SCHEMA = 'trinite.ternary-dynamics-report.v2'
SNAPSHOT_SCHEMAS = frozenset(('trinite.training-checkpoint.v1',
                              'trinite.comparison-checkpoint.v1'))
PARTS = ('q', 'k', 'v', 'o', 'up', 'down')
_SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z')
MAX_MODEL_TENSORS = 1024


def protocol() -> dict:
    data = PROTOCOL_PATH.read_bytes()
    if identity(data) != PROTOCOL_IDENTITY:
        raise ContractError('ternary-dynamics protocol bytes are not frozen v1')
    return parse_json(data, canonical=True)


def source_closure() -> dict[str, str]:
    """Byte identities for the *current observer*, never an older producer."""
    names = ('scripts/ternary_dynamics/protocol.json',
             'scripts/ternary_dynamics/dynamics.py',
             'scripts/ternary_dynamics/__init__.py',
             'scripts/analyze_ternary_dynamics.py',
             'src/trinite/contracts.py', 'src/trinite/model.py',
             'src/trinite/inspection.py', 'src/trinite/quantizer.py',
             'src/trinite/training.py', 'requirements.lock')
    return {name: identity((ROOT/name).read_bytes()) for name in names}


def observer_environment() -> dict:
    """Use the native CPU training guard as the observer's numeric trust gate.

    This observes the executing process, not the historical checkpoint's
    producer environment. Both identities are preserved independently.
    """
    from trinite.training import require_environment
    return require_environment()


def _bounded_file(path: Path, limit: int) -> bytes:
    """Stable-directory reader: no symlink traversal; hostile concurrent FS not certified."""
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ContractError('missing or symlinked ternary-dynamics artifact')
    if path.stat().st_size > limit:
        raise ContractError('ternary-dynamics artifact exceeds byte bound')
    with path.open('rb') as reader:
        result = reader.read(limit + 1)
    if len(result) > limit:
        raise ContractError('ternary-dynamics artifact exceeds byte bound')
    return result


def _member(root: Path, name: object) -> Path:
    if (type(name) is not str or not _SAFE_NAME.fullmatch(name) or
            name in ('.', '..') or name.startswith('..')):
        raise ContractError('snapshot member requires a safe single basename')
    return root/name


def selection(value: object) -> dict:
    r = exact_keys(value, {'schema', 'lane', 'seed', 'snapshots'}, 'ternary selection')
    if r['schema'] != SELECTION_SCHEMA or r['lane'] != 'ternary':
        raise ContractError('ternary selection requires the native ternary lane')
    integer(r['seed'], 'selection seed', 0, 2**31-1)
    snaps = r['snapshots']
    cfg = protocol()
    if (type(snaps) is not list or not cfg['minimum_snapshots_per_cell']
            <= len(snaps) <= cfg['maximum_snapshots_per_cell']):
        raise ContractError('ternary selection requires 2..16 snapshots')
    steps, names = [], []
    for item in snaps:
        exact_keys(item, {'step', 'payload', 'metadata'}, 'selected snapshot')
        integer(item['step'], 'selected step', 0, 2**31-1)
        for key in ('payload', 'metadata'):
            _member(Path('/'), item[key]); names.append(item[key])
        if not item['payload'].endswith('.safetensors') or not item['metadata'].endswith('.json'):
            raise ContractError('snapshot file extension does not match its declared format')
        steps.append(item['step'])
    if steps != sorted(set(steps)) or len(set(names)) != len(names):
        raise ContractError('snapshot steps and artifact names must be unique and ordered')
    return r


def freeze_request(selected: dict, input_root: Path) -> dict:
    """Hash existing bytes without examining model tensors or outcome metrics."""
    selection(selected)
    runtime = observer_environment()  # Refuse a foreign observer before sampling even file hashes.
    input_root = Path(input_root).absolute()
    snapshots = []
    for item in selected['snapshots']:
        payload = _bounded_file(_member(input_root, item['payload']), protocol()['maximum_tensor_file_bytes'])
        metadata = _bounded_file(_member(input_root, item['metadata']), protocol()['maximum_metadata_bytes'])
        snapshots.append({**item, 'payload_identity': identity(payload),
                          'metadata_identity': identity(metadata)})
    return {'schema': REQUEST_SCHEMA, 'protocol_identity': PROTOCOL_IDENTITY,
            'source': source_closure(), 'observer_environment': runtime,
            'selection': selected, 'snapshots': snapshots,
            'assurance': 'digest-bound-descriptive; not optimizer/custody replay'}


def _matrix_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    d, ff = config.width, config.feed_forward_width
    shapes = {'token_embedding': (config.vocabulary_size, d),
              'position_embedding': (config.context_length, d),
              'final_norm.weight': (d,)}
    for i in range(config.blocks):
        prefix = f'blocks.{i}.'
        shapes[prefix+'attention_norm.weight'] = (d,)
        shapes[prefix+'feed_forward_norm.weight'] = (d,)
        for part in PARTS:
            shapes[prefix+part+'.weight'] = (ff, d) if part == 'up' else (
                (d, ff) if part == 'down' else (d, d))
    return shapes


def _validate_checkpoint(payload: bytes, metadata: bytes, step: int, seed: int) -> tuple[dict, dict, dict]:
    m = parse_json(metadata, canonical=True)
    if type(m) is not dict or m.get('schema') not in SNAPSHOT_SCHEMAS:
        raise ContractError('unsupported native ternary snapshot schema')
    if type(m.get('step')) is not int or m['step'] != step or m.get('lane') != 'ternary':
        raise ContractError('snapshot step/lane disagrees with frozen selection')
    if type(m.get('plan')) is not dict or type(m['plan'].get('model')) is not dict:
        raise ContractError('snapshot lacks a valid model plan')
    plan = m['plan']
    if type(plan.get('seed')) is not int or plan['seed'] != seed:
        raise ContractError('snapshot seed differs from selected seed')
    try:
        cfg = ModelConfig.from_dict(plan['model'])
    except (ContractError, TypeError, ValueError) as err:
        raise ContractError('snapshot model configuration is invalid') from err
    if (type(m.get('source')) is not dict or type(m.get('environment')) is not dict or
            type(m.get('dataset_identity')) is not str or
            type(m.get('manifest_identity')) is not str or
            type(m.get('model_identity')) is not str):
        raise ContractError('snapshot lacks its source, environment or model bindings')
    # Both admitted ternary snapshot producers bind the original quantizer
    # as 'quantizer.py' in their native source receipt. Reject different or
    # missing implementations rather than reinterpret old weights through
    # today's forward/STE masks. Dense task-balance is not a ternary producer.
    current_quantizer = source_closure()['src/trinite/quantizer.py']
    if m['source'].get('quantizer.py') != current_quantizer:
        raise ContractError('snapshot producer quantizer source differs from this observer')
    for name in ('dataset_identity', 'manifest_identity', 'model_identity'):
        require_identity(m[name])
    try:
        tensors = load_safetensors(payload)
    except Exception as err:
        raise ContractError('snapshot is not a valid bounded safetensors payload') from err
    if type(tensors) is not dict or len(tensors) > MAX_MODEL_TENSORS:
        raise ContractError('snapshot tensor inventory exceeds bound')
    shapes = _matrix_shapes(cfg)
    model_names = {'model.'+name for name in shapes}
    actual = {key for key in tensors if key.startswith('model.')}
    if actual != model_names:
        raise ContractError('snapshot model parameter inventory differs from the declared architecture')
    model = {}
    for name, shape in shapes.items():
        val = tensors['model.'+name]
        require_float_tensor(val, 'snapshot '+name)
        if tuple(val.shape) != shape:
            raise ContractError('snapshot model tensor shape differs from the plan')
        model[name] = val
    computed = identity(json_bytes({name: tensor_identity(v) for name, v in model.items()}))
    if computed != m['model_identity']:
        raise ContractError('snapshot model tensors disagree with retained model identity')
    for name, tensor in tensors.items():
        if name.startswith('model.'):
            continue
        if name.startswith('rng.'):
            if m['schema'] != 'trinite.training-checkpoint.v1' or name not in (
                    'rng.torch_cpu', 'rng.numpy_keys'):
                raise ContractError('snapshot has an unexpected RNG tensor')
            continue  # Its original reader, not this diagnostic, owns RNG-state validation.
        if not name.startswith('optimizer.'):
            raise ContractError('snapshot has unexpected non-model tensor keys')
        suffix = name[len('optimizer.'):]
        match = next(((param, slot) for param in shapes for slot in (
            'step', 'exp_avg', 'exp_avg_sq') if suffix == param+'.'+slot), None)
        if match is None:
            raise ContractError('snapshot has an unexpected optimizer tensor')
        param, slot = match
        require_float_tensor(tensor, 'snapshot optimizer moment')
        if tensor.shape != (() if slot == 'step' else shapes[param]):
            raise ContractError('snapshot optimizer moment shape is invalid')
        if slot == 'exp_avg_sq' and bool(torch.any(tensor < 0)):
            raise ContractError('snapshot optimizer variance is negative')
    context = {'seed': seed, 'lane': m['lane'],
               'plan_identity': identity(json_bytes(plan)),
               'source_identity': identity(json_bytes(m['source'])),
               'environment_identity': identity(json_bytes(m['environment'])),
               'dataset_identity': m['dataset_identity'],
               'manifest_identity': m['manifest_identity']}
    return m, tensors, context


def _nonzero(value: torch.Tensor) -> int:
    return int(torch.count_nonzero(value).item())


def inspect_weight(weight: torch.Tensor, *, moment: torch.Tensor | None = None) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Read-only measurement of the *actual* v0 quantizer and its strict mask."""
    require_float_tensor(weight, 'inspected weight')
    if weight.ndim != 2 or not 1 <= weight.numel() <= 2_000_000:
        raise ContractError('inspected weight requires a bounded matrix')
    if moment is not None:
        require_float_tensor(moment, 'retained optimizer moment')
        if moment.shape != weight.shape:
            raise ContractError('optimizer moment shape disagrees with latent weight')
    with torch.no_grad():
        observed = quantize(weight)
        codes = observed.codes.detach().clone()
        scale = observed.scale.detach()
        units = weight.detach() / scale
        inside = units.abs() < 1  # exact mask used by quantize(); ties excluded
        mag = units.abs()
        zero_boundary = (mag - 0.5).abs() <= .015625
        ste_boundary = (mag - 1.).abs() <= .015625
        counts = {str(n): int(torch.count_nonzero(codes == n)) for n in (-1, 0, 1)}
        n = int(weight.numel())
        report = {'tensor_identity': tensor_identity(weight),
                  'scale_hex': float(scale.item()).hex(),
                  'code_identity': tensor_identity(codes),
                  'elements': n, 'codes': counts,
                  'ste_mask_enabled': _nonzero(inside),
                  'ste_mask_disabled': n-_nonzero(inside),
                  'near_zero_code_boundary': _nonzero(zero_boundary),
                  'near_ste_support_boundary': _nonzero(ste_boundary),
                  'normalized_max_abs_hex': float(mag.max().item()).hex(),
                  'moment_nonzero_inside': None,
                  'moment_nonzero_outside': None}
        if moment is not None:
            report['moment_nonzero_inside'] = _nonzero((moment != 0) & inside)
            report['moment_nonzero_outside'] = _nonzero((moment != 0) & ~inside)
    return report, codes, weight.detach().clone()


def transition(prior: tuple[torch.Tensor, torch.Tensor], later: tuple[torch.Tensor, torch.Tensor]) -> dict:
    previous_codes, previous_weights = prior
    new_codes, new_weights = later
    if previous_codes.shape != new_codes.shape or previous_weights.shape != new_weights.shape:
        raise ContractError('snapshot matrix shapes changed across steps')
    changed = previous_codes != new_codes
    latent_changed = previous_weights != new_weights
    return {'code_changed': _nonzero(changed),
            'code_stable': int(changed.numel())-_nonzero(changed),
            'latent_changed': _nonzero(latent_changed),
            'latent_changed_code_stable': _nonzero(latent_changed & ~changed),
            'code_changed_latent_stable': _nonzero(changed & ~latent_changed),
            'zero_to_nonzero': _nonzero((previous_codes == 0) & (new_codes != 0)),
            'nonzero_to_zero': _nonzero((previous_codes != 0) & (new_codes == 0)),
            'negative_to_positive': _nonzero((previous_codes == -1) & (new_codes == 1)),
            'positive_to_negative': _nonzero((previous_codes == 1) & (new_codes == -1))}


def report_for(request: dict, input_root: Path) -> dict:
    """Revalidate the complete frozen request and all snapshot bytes per read."""
    runtime = observer_environment()  # Fail closed in a foreign CPU/kernel/dependency lane.
    exact_keys(request, {'schema', 'protocol_identity', 'source', 'observer_environment',
                         'selection', 'snapshots', 'assurance'}, 'ternary dynamics request')
    if (request['schema'] != REQUEST_SCHEMA or request['protocol_identity'] != PROTOCOL_IDENTITY
            or request['source'] != source_closure() or
            request['assurance'] != 'digest-bound-descriptive; not optimizer/custody replay'):
        raise ContractError('ternary-dynamics request source/protocol differs from checked observer')
    if request['observer_environment'] != runtime:
        raise ContractError('observer runtime differs from the frozen selection request')
    selection(request['selection'])
    if type(request['snapshots']) is not list or len(request['snapshots']) != len(request['selection']['snapshots']):
        raise ContractError('ternary-dynamics selected snapshot inventory differs')
    root = Path(input_root).absolute()
    steps = []
    expected_context = None
    prior = None
    transitions = []
    for plan_item, item in zip(request['selection']['snapshots'], request['snapshots']):
        exact_keys(item, {'step', 'payload', 'metadata', 'payload_identity', 'metadata_identity'}, 'snapshot commitment')
        if any(item[k] != plan_item[k] for k in ('step', 'payload', 'metadata')):
            raise ContractError('snapshot does not match predeclared selection')
        for name in ('payload_identity', 'metadata_identity'):
            require_identity(item[name])
        raw = _bounded_file(_member(root, item['payload']), protocol()['maximum_tensor_file_bytes'])
        meta = _bounded_file(_member(root, item['metadata']), protocol()['maximum_metadata_bytes'])
        if (identity(raw), identity(meta)) != (item['payload_identity'], item['metadata_identity']):
            raise ContractError('snapshot member bytes differ from externally bound request')
        metadata, tensors, context = _validate_checkpoint(raw, meta, item['step'], request['selection']['seed'])
        if (metadata['schema'] == 'trinite.training-checkpoint.v1' and
                metadata.get('tensor_file_identity') != item['payload_identity']):
            raise ContractError('native checkpoint tensor identity differs from metadata')
        if expected_context is None:
            expected_context = context
        elif context != expected_context:
            raise ContractError('snapshot model/source/dataset/plan/environment context drift')
        per_matrix = {}
        current = {}
        for name in sorted(k for k in _matrix_shapes(ModelConfig.from_dict(metadata['plan']['model']))
                           if k.endswith('.weight') and any(k.endswith('.'+p+'.weight') for p in PARTS)):
            key = 'model.'+name
            moment = tensors.get('optimizer.'+name+'.exp_avg')
            metric, code, weight = inspect_weight(tensors[key], moment=moment)
            per_matrix[name] = metric
            if prior is not None:
                transitions[-1]['matrices'][name] = transition(prior[name], (code, weight))
            current[name] = (code, weight)
        if prior is not None:
            transitions[-1]['to_step'] = item['step']
        prior = current
        steps.append({'step': item['step'], 'payload_identity': item['payload_identity'],
                      'metadata_identity': item['metadata_identity'],
                      'model_identity': metadata['model_identity'],
                      'matrices': per_matrix})
        if len(steps) < len(request['snapshots']):
            transitions.append({'from_step': item['step'], 'to_step': None, 'matrices': {}})
    if observer_environment() != runtime:
        raise ContractError('observer runtime changed during numeric inspection')
    return {'schema': REPORT_SCHEMA, 'protocol_identity': PROTOCOL_IDENTITY,
            'request_identity': identity(json_bytes(request)), 'observer_source': source_closure(),
            'observer_environment': runtime, 'context': expected_context, 'samples': steps, 'transitions': transitions,
            'evidence_class': 'DESCRIPTIVE', 'full_optimizer_replay': False,
            'incoming_training_gradients_available': False,
            'phase5_ready': False, 'release_ready': False,
            'interpretation': 'STE mask availability and code transitions are observations; utility/causality and task learning not established'}
