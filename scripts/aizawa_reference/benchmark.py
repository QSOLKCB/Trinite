"""Pure-CPU Aizawa synthetic reference; no model, training or result selection.

Procedure is frozen in protocol.json. The existing pinned Trinite geometry
implementation alone owns distance/difference/curvature/alignment semantics.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import struct
import sys

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from trinite import geometry

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name('protocol.json')
PROTOCOL_IDENTITY = 'sha256:cb904c512d45d8abba55a258c6b649e320c2ca37a7c04c9464d4c5623ca4857f'
PROTOCOL_SCHEMA = 'trinite.aizawa-reference-protocol.v1'
REQUEST_SCHEMA = 'trinite.aizawa-reference-request.v1'
REPORT_SCHEMA = 'trinite.aizawa-reference-report.v1'
SOURCE_FILES = (
    'scripts/aizawa_reference/__init__.py',
    'scripts/aizawa_reference/benchmark.py',
    'scripts/aizawa_reference/protocol.json',
    'scripts/run_aizawa_reference.py',
    'src/trinite/__init__.py',
    'src/trinite/contracts.py',
    'src/trinite/geometry.py',
    'src/trinite/_geo_reference.py',
    'src/trinite/geometry-pin.json',
)


def protocol() -> dict:
    """Reject changed frozen procedure before any outcome calculation."""
    raw = PROTOCOL_PATH.read_bytes()
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('Aizawa protocol differs from the separately frozen revision')
    value = parse_json(raw, canonical=True)
    if value['schema'] != PROTOCOL_SCHEMA:
        raise ContractError('unsupported Aizawa protocol')
    return value


def source_closure() -> dict[str, str]:
    sources = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        if (path.is_symlink() or not path.is_file() or
                any(p.is_symlink() for p in path.parents)):
            raise ContractError('unsafe or missing Aizawa reference source: ' + name)
        sources[name] = identity(path.read_bytes())
    return sources


def runtime() -> dict:
    """Exact numerical replay scope is the actual interpreter and libm host."""
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'libc': list(platform.libc_ver()),
        'byteorder': sys.byteorder,
        'float_mantissa_bits': sys.float_info.mant_dig,
        'float_radix': sys.float_info.radix,
        'float64_pack_size': struct.calcsize('<d'),
    }


def freeze_request() -> dict:
    """Bind reviewed sources and numerical environment without examining outcomes."""
    value = protocol()
    pin = geometry.verify_geometry_pin()
    return {
        'schema': REQUEST_SCHEMA, 'protocol_identity': PROTOCOL_IDENTITY,
        'sources': source_closure(), 'runtime': runtime(),
        'geometry_pin': pin,
        'conditions': ['native', 'shuffle-seed-11', 'shuffle-seed-37',
                       'rademacher-seed-17', 'smooth-periodic', 'affine-line',
                       'reverse-diagnostic'],
        'sampling_points': value['numerics']['samples'],
        'evidence_class': 'SIMULATION',
    }


def _finite_vector(value: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(value) != 3 or any(type(x) is not float or not math.isfinite(x) for x in value):
        raise ContractError('nonfinite Aizawa state or derivative')
    return value


def field(state: tuple[float, float, float], params: tuple[float, ...]) -> tuple[float, float, float]:
    """Aizawa autonomous ODE, evaluated in frozen expression order."""
    x, y, z = _finite_vector(state)
    a, b, c, d, e, f = params
    return _finite_vector(((z - b) * x - d * y,
                           d * x + (z - b) * y,
                           c + a * z - z ** 3 / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * (x ** 3)))


def rk4_step(state: tuple[float, float, float], h: float,
             params: tuple[float, ...]) -> tuple[float, float, float]:
    if type(h) is not float or not math.isfinite(h) or h <= 0:
        raise ContractError('RK4 step size must be positive binary64')
    k1 = field(state, params)
    k2 = field(tuple(state[i] + (h / 2.0) * k1[i] for i in range(3)), params)
    k3 = field(tuple(state[i] + (h / 2.0) * k2[i] for i in range(3)), params)
    k4 = field(tuple(state[i] + h * k3[i] for i in range(3)), params)
    return _finite_vector(tuple(state[i] + (h / 6.0) * (
        k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(3)))


def _parameters(p: dict) -> tuple[float, ...]:
    # The exact decimal *strings* are frozen. Float parsing occurs only here.
    return tuple(float(p['parameters'][k]) for k in ('a', 'b', 'c', 'd', 'e', 'f'))


def integrate(p: dict, steps: int) -> dict:
    """Return 201 exact sampled coordinates and SHA-256 of all steps, incl. t=0.

    Fixed-step integration counts are not the same as t_eval output samples.
    Hashing the complete packed stream makes the discarded intermediate states
    independently recomputable without claiming that the samples contain them.
    """
    steps_coarse = p['numerics']['coarse_steps']
    steps_fine = p['numerics']['fine_steps']
    count = p['numerics']['samples']
    if (type(steps) is not int or steps not in (steps_coarse, steps_fine)
            or steps % (count - 1) != 0):
        raise ContractError('nonfrozen integration horizon or sample stride')
    stride = steps // (count - 1)
    h = float(p['numerics']['duration']) / steps
    params = _parameters(p)
    state = tuple(float(x) for x in p['initial_state'])
    hasher = hashlib.sha256()
    sampled = []
    for i in range(steps + 1):
        hasher.update(struct.pack('<ddd', *_finite_vector(state)))
        if i % stride == 0:
            sampled.append([float(x) for x in state])
        if i < steps:
            state = rk4_step(state, h, params)
    if len(sampled) != count:
        raise ContractError('Aizawa sampler failed its frozen cardinality')
    return {'steps': steps, 'h_hex': h.hex(), 'stream_states': steps + 1,
            'full_binary64_stream_identity': 'sha256:' + hasher.hexdigest(),
            'sampled': sampled}


def _coordinate_errors(coarse: list[list[float]], fine: list[list[float]], upto: int) -> dict:
    if len(coarse) != len(fine) or type(upto) is not int or not 1 <= upto <= len(coarse):
        raise ContractError('mismatched convergence arrays')
    values = [abs(coarse[i][j] - fine[i][j]) for i in range(upto) for j in range(3)]
    return {'sample_count': upto, 'max_abs_coordinate_error_hex': max(values).hex(),
            'mean_abs_coordinate_error_hex': (math.fsum(values) / len(values)).hex()}


def _shuffle(sampled: list[list[float]], seed: int) -> tuple[list[list[float]], list[int]]:
    order = sorted(range(len(sampled)), key=lambda i: hashlib.sha256(json_bytes({
        'index': i, 'seed': seed, 'domain': 'trinite.aizawa.temporal-shuffle.v1'
    })).digest())
    if order == list(range(len(sampled))):
        raise ContractError('temporal shuffle did not change sample order')
    return [sampled[i][:] for i in order], order


def _controls(sampled: list[list[float]], p: dict) -> list[dict]:
    t_step = float(p['numerics']['duration']) / (len(sampled) - 1)
    controls = [{'name': 'native', 'category': 'primary-simulated-reference',
                 'sampled': [row[:] for row in sampled]}]
    for seed in p['controls']['temporal_shuffle_seeds']:
        rows, order = _shuffle(sampled, seed)
        controls.append({'name': f'shuffle-seed-{seed}', 'category': 'temporal-permutation-null',
                         'permutation': order, 'sampled': rows})
    controls.append({'name': 'rademacher-seed-17', 'category': 'independent-sign-null',
                     'sampled': geometry.rademacher(sampled, 'aizawa-reference-v1',
                                                   p['controls']['rademacher_seed'])})
    periodic, line = [], []
    for i in range(len(sampled)):
        t = i * t_step
        periodic.append([math.cos(t), math.sin(t), 0.5 * math.sin(2.0 * t)])
        line.append([t / float(p['numerics']['duration']), 0.0, 0.0])
    controls += [
        {'name': 'smooth-periodic', 'category': 'nonchaotic-analytic-null', 'sampled': periodic},
        {'name': 'affine-line', 'category': 'nonchaotic-affine-null', 'sampled': line},
        {'name': 'reverse-diagnostic', 'category': 'orientation-diagnostic-not-null',
         'sampled': list(reversed([row[:] for row in sampled]))},
    ]
    return controls


def _encode_points(rows: list[list[float]]) -> list[list[str]]:
    return [[float(x).hex() for x in row] for row in geometry.points(rows)]


def measure(request: dict) -> dict:
    """Run the frozen synthetic experiment; never import Trinite's neural path."""
    exact_keys(request, set(freeze_request()), 'Aizawa reference request')
    if json_bytes(request) != json_bytes(freeze_request()):
        raise ContractError('Aizawa request source/protocol/environment drift')
    p = protocol()
    initial_runtime = runtime()
    initial_sources = source_closure()
    coarse = integrate(p, p['numerics']['coarse_steps'])
    fine = integrate(p, p['numerics']['fine_steps'])
    early = _coordinate_errors(coarse['sampled'], fine['sampled'],
                               p['numerics']['prefix_samples_for_convergence'])
    full = _coordinate_errors(coarse['sampled'], fine['sampled'], len(coarse['sampled']))
    threshold = float(p['numerics']['early_max_pointwise_coordinate_error_upper_bound'])
    if float.fromhex(early['max_abs_coordinate_error_hex']) > threshold:
        raise ContractError('Aizawa early-horizon RK4 refinement convergence failed')
    comparison = []
    original_metrics = None
    for condition in _controls(coarse['sampled'], p):
        rows = condition['sampled']
        metrics = geometry.metrics(rows)
        label = condition['name']
        record = {
            'name': label, 'category': condition['category'],
            'points_hex': _encode_points(rows),
            'points_identity': identity(json_bytes(_encode_points(rows))),
            'metrics': metrics,
            'alignment_to_native_order1_hex': geometry.alignment(
                coarse['sampled'], rows, order=p['analysis']['alignment_order'],
                mode=p['analysis']['alignment_mode'].split(' ')[0]).hex(),
        }
        if 'permutation' in condition:
            record['permutation'] = condition['permutation']
            record['permutation_identity'] = identity(json_bytes(condition['permutation']))
        if label == 'native':
            original_metrics = metrics
        if label == 'reverse-diagnostic' and (
                metrics['path_length_hex'] != original_metrics['path_length_hex'] or
                metrics['menger_hex'] != list(reversed(original_metrics['menger_hex']))):
            raise ContractError('orientation reversal unexpectedly changed invariant metrics')
        comparison.append(record)
    if len(comparison) != len(request['conditions']):
        raise ContractError('missing Aizawa controls')
    if [x['name'] for x in comparison] != request['conditions']:
        raise ContractError('reordered Aizawa controls')
    if runtime() != initial_runtime or source_closure() != initial_sources:
        raise ContractError('Aizawa environment/source changed during numerical analysis')
    return {
        'schema': REPORT_SCHEMA, 'protocol_identity': PROTOCOL_IDENTITY,
        'request_identity': identity(json_bytes(request)),
        'sources': initial_sources, 'runtime': initial_runtime,
        'evidence_class': 'SIMULATION', 'learning_claim': 'not evaluated',
        'neural_alignment': 'not evaluated; no neural state supplied',
        'full_optimizer_replay': False, 'geometry_loss': '0',
        'phase5_ready': False, 'release_ready': False,
        'coarse_integrator': {k:v for k,v in coarse.items() if k!='sampled'},
        'fine_integrator': {k:v for k,v in fine.items() if k!='sampled'},
        'refined_points_hex': _encode_points(fine['sampled']),
        'refined_points_identity': identity(json_bytes(_encode_points(fine['sampled']))),
        'early_refinement': early, 'full_horizon_refinement_descriptive': full,
        'controls': comparison,
    }
