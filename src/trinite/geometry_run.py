"""Frozen request binding and explicit observational composition; no intervention."""
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import resource
import time

from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .data import audit_directory
from .geometry import alignment, metrics, rademacher, verify_geometry_pin
from .observation import BundleObserver

PROTOCOL_REVISION = '7ebe475f1e62f4675e2525dbed7ddba441db1998'
PROTOCOL_IDENTITY = 'sha256:d6f35530f9b0dc04a1836e4b8e2026ca16e2a690eb0825802405cef9e4795931'
REQUEST_SCHEMA = 'trinite.geometry-request.v1'


def protocol():
    raw = (Path(__file__).resolve().parent/'geometry-protocol.json').read_bytes()
    if identity(raw) != PROTOCOL_IDENTITY: raise ContractError('unsupported geometry protocol bytes')
    return parse_json(raw, canonical=True)


def source():
    from .training import source_receipt
    root = Path(__file__).resolve().parent
    return {**source_receipt(), **{n: identity((root/n).read_bytes()) for n in (
        'capture.py', 'geometry.py', 'geometry_run.py', '_geo_reference.py',
        'geometry-pin.json', 'geometry-protocol.json', 'GEO-LICENSE.txt')}}


def inputs(dataset, training_config, dense_checkpoint, ternary_checkpoint):
    from .checkpoint import read_checkpoint
    from .training import RunConfig, require_environment
    p = protocol(); require_environment(); audit_directory(dataset)
    plan = RunConfig.load(training_config)
    if plan.content_identity != p['training_plan_identity']:
        raise ContractError('observation requires the frozen seed-0 tiny training plan')
    raw = {'dataset.json': (Path(dataset)/'dataset.json').read_bytes(),
           'dataset-manifest.json': (Path(dataset)/'manifest.json').read_bytes(),
           'training-plan.json': json_bytes(plan.to_dict())}
    if identity(raw['dataset.json']) != p['dataset_identity'] or identity(raw['dataset-manifest.json']) != p['dataset_manifest_identity']:
        raise ContractError('observation requires the admitted frozen formal-v1 bytes')
    for lane, directory in [('dense', dense_checkpoint), ('ternary', ternary_checkpoint)]:
        payload, metadata = read_checkpoint(directory)
        if len(payload) > 512*1024: raise ContractError('tiny checkpoint exceeds observation input budget')
        m = parse_json(metadata, canonical=True)
        if (m.get('schema') != 'trinite.training-checkpoint.v1' or m.get('lane') != lane
                or type(m.get('step')) is not int or m['step'] != 60
                or m.get('plan_identity') != plan.content_identity
                or m.get('tensor_file_identity') != identity(payload)):
            raise ContractError('observation requires an elected final-step checkpoint for each lane')
        raw[lane+'-tensors.safetensors'], raw[lane+'-metadata.json'] = payload, metadata
    return plan, raw


def freeze_observation(output, *, dataset, training_config, dense_checkpoint, ternary_checkpoint):
    from .training import environment
    _, raw = inputs(dataset, training_config, dense_checkpoint, ternary_checkpoint)
    request = {'schema': REQUEST_SCHEMA, 'protocol': protocol(), 'protocol_identity': PROTOCOL_IDENTITY,
               'protocol_revision': PROTOCOL_REVISION, 'inputs': {n: identity(b) for n, b in raw.items()},
               'source': source(), 'environment': environment(), 'geometry_pin': verify_geometry_pin(),
               'created_at': datetime.now(timezone.utc).isoformat(), 'clock_assurance': 'local; not independent registration'}
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    encoded = json_bytes(request); (root/'request.json').write_bytes(encoded)
    return {'request_identity': identity(encoded), 'protocol_revision': PROTOCOL_REVISION,
            'protocol_identity': PROTOCOL_IDENTITY, 'request_file': str(root/'request.json')}


def read_request(path, expected_identity, raw):
    from .training import environment
    require_identity(expected_identity)
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ContractError('request must be a regular file')
    with path.open('rb') as stream: data = stream.read(65537)
    if len(data) > 65536 or identity(data) != expected_identity: raise ContractError('frozen request bytes differ')
    r = exact_keys(parse_json(data, canonical=True), {'schema', 'protocol', 'protocol_identity', 'protocol_revision',
        'inputs', 'source', 'environment', 'geometry_pin', 'created_at', 'clock_assurance'}, 'geometry request')
    expected = {'schema': REQUEST_SCHEMA, 'protocol': protocol(), 'protocol_identity': PROTOCOL_IDENTITY,
                'protocol_revision': PROTOCOL_REVISION, 'inputs': {n: identity(b) for n, b in raw.items()},
                'source': source(), 'environment': environment(), 'geometry_pin': verify_geometry_pin()}
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('request protocol/input/source/environment/pin mismatch')
    if type(r['created_at']) is not str or r['clock_assurance'] != 'local; not independent registration':
        raise ContractError('invalid request clock assurance')
    try:
        if datetime.fromisoformat(r['created_at']).utcoffset() is None: raise ValueError('timezone required')
    except ValueError as error: raise ContractError('invalid request timestamp') from error
    return data


def summarize(records, check_budget):
    # Every record is already span/tensor audited. Six families, with 3x2 items.
    families = {}
    for r in records: families.setdefault(r['family_id'], []).append(r)
    if len(families) != 6 or any(len(v) != 6 for v in families.values()):
        raise ContractError('observation requires exactly six complete training families')
    result = {}
    for layer in protocol()['layers']:
        conditions = {}
        for condition in ('native', 'reverse-token-order', 'deterministic-rademacher-vectors'):
            per_family = []
            for family, group in sorted(families.items()):
                same, different, pairs = [], [], []
                # GEO-NULL-REUSE-001: fixed seed/key/layer/shape, immutable
                # audited prompt points. Every pair consumes this exact null.
                nulls = {}
                if condition == 'deterministic-rademacher-vectors':
                    for item in group:
                        check_budget()
                        nulls[item['example_identity']] = rademacher(
                            item['points'][layer], item['example_identity']+':'+layer)
                for a, b in itertools.combinations(sorted(group, key=lambda r: r['example_identity']), 2):
                    same_item = a['semantic_identity'] == b['semantic_identity'] and a['carrier'] != b['carrier']
                    different_item = a['semantic_identity'] != b['semantic_identity'] and a['carrier'] == b['carrier']
                    if not (same_item or different_item): continue
                    check_budget()
                    left, right = a['points'][layer], b['points'][layer]
                    if condition == 'reverse-token-order': left, right = left[::-1], right[::-1]
                    elif condition == 'deterministic-rademacher-vectors':
                        left = nulls[a['example_identity']]
                        right = nulls[b['example_identity']]
                    score = alignment(left, right)
                    (same if same_item else different).append(score)
                    pairs.append({'left': a['example_identity'], 'right': b['example_identity'],
                                  'type': 'same-item/different-carrier' if same_item else 'same-carrier/different-item',
                                  'cosine_hex': score.hex()})
                if len(same) != 3 or len(different) != 6: raise ContractError('incomplete control pairing')
                contrast = math.fsum(same)/3-math.fsum(different)/6
                per_family.append({'family_id': family, 'contrast_hex': contrast.hex(), 'pairs': pairs})
            values = [float.fromhex(r['contrast_hex']) for r in per_family]
            conditions[condition] = {'family_mean_hex': (math.fsum(values)/6).hex(),
                                     'family_min_hex': min(values).hex(), 'family_max_hex': max(values).hex(),
                                     'families': per_family,
                                     'evidence_class': 'SIMULATION' if condition.startswith('deterministic') else 'OBSERVATION'}
        result[layer] = conditions
    return result


def measure_observation(output, *, request, request_identity, dataset, training_config,
                        dense_checkpoint, ternary_checkpoint, observer=True, observer_factory=BundleObserver):
    import random
    import numpy as np
    import torch
    from .capture import audit_capture, capture_prompt
    from .checkpoint import checkpoint_bytes, load_checkpoint_bytes
    from .inspection import inventory
    from .training import model_identity
    if type(observer) is not bool: raise ContractError('observer mode requires bool')
    plan, raw = inputs(dataset, training_config, dense_checkpoint, ternary_checkpoint)
    request_bytes = read_request(request, request_identity, raw)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    (root/'request.json').write_bytes(request_bytes)
    raw['request.json'] = request_bytes
    clock, observation_seconds = time.monotonic(), 0.
    errors, observation_errors, collector, verification, results = [], [], None, None, {}
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def check_budget():
        if time.monotonic()-clock > protocol()['max_seconds']: raise ContractError('observation exceeded stage time budget')
    def observed(call):
        nonlocal observation_seconds
        start = time.monotonic()
        try: return call()
        except Exception as error:
            observation_errors.append(type(error).__name__+': '+str(error)); return None
        finally: observation_seconds += time.monotonic()-start
    try:
        if observer: collector = observed(lambda: observer_factory(root/'provenance'))
        examples = [e for e in parse_json(raw['dataset.json'])['examples'] if e['split'] == 'train']
        capture_bytes, retained_capture_bytes = 0, 0
        for lane in ('dense', 'ternary'):
            check_budget()
            state = load_checkpoint_bytes(raw[lane+'-tensors.safetensors'], raw[lane+'-metadata.json'],
                                          raw['dataset.json'], raw['dataset-manifest.json'], plan, lane)
            before = checkpoint_bytes(state)
            inv = inventory(state.model)
            (root/(lane+'-inventory.json')).write_bytes(json_bytes(inv))
            records, trajectories, details = [], [], []
            for example in sorted(examples, key=lambda e: e['example_identity']):
                check_budget()
                encoded = capture_prompt(state.model, example['prompt'])
                capture_bytes += len(encoded)
                if capture_bytes > protocol()['max_capture_bytes']: raise ContractError('captures exceed aggregate 4 MiB budget')
                record, pts = audit_capture(encoded)
                if record['model_inventory_identity'] != inv['inventory_identity']:
                    raise ContractError('capture model identity changed')
                metadata = {k: example[k] for k in ('example_identity', 'semantic_identity', 'family_id', 'carrier')}
                records.append({**metadata, 'capture': record})
                trajectories.append({**metadata, 'points': pts})
                details.append({'example_identity': example['example_identity'],
                                'metrics': {layer: metrics(p) for layer, p in pts.items()}})
            if before != checkpoint_bytes(state): raise ContractError('capture changed model/optimizer/RNG/progress state')
            artifact = {'schema': 'trinite.capture-set.v1', 'lane': lane, 'model_identity': model_identity(state.model),
                        'checkpoint': {n: identity(raw[lane+'-'+n]) for n in ('tensors.safetensors', 'metadata.json')},
                        'request_identity': request_identity, 'source': source(), 'records': records}
            capture_artifact = json_bytes(artifact)
            retained_capture_bytes += len(capture_artifact)
            if retained_capture_bytes > protocol()['max_capture_bytes']:
                raise ContractError('retained capture sets exceed aggregate 4 MiB budget')
            (root/(lane+'-captures.json')).write_bytes(capture_artifact)
            comparisons = summarize(trajectories, check_budget)
            lane_result = {'schema': 'trinite.geometry-lane-result.v1', 'lane': lane,
                           'capture_identity': identity(capture_artifact), 'request_identity': request_identity,
                           'comparisons': comparisons, 'trajectory_metrics': details}
            lane_bytes = json_bytes(lane_result)
            (root/(lane+'-geometry.json')).write_bytes(lane_bytes)
            results[lane] = {'capture_identity': identity(capture_artifact), 'geometry_identity': identity(lane_bytes),
                             'primary_contrast_hex': comparisons['final']['native']['family_mean_hex']}
            if collector:
                observed(lambda: collector.record('capture-and-measure-'+lane, inputs=raw, outputs={
                    lane+'-captures.json': capture_artifact, lane+'-geometry.json': lane_bytes,
                    lane+'-inventory.json': json_bytes(inv)}))
        check_budget()
        result = {'schema': 'trinite.geometry-result.v1', 'outcome': 'completed', 'evidence_class': 'OBSERVATION',
                  'protocol_id': protocol()['protocol_id'], 'protocol_identity': PROTOCOL_IDENTITY,
                  'protocol_revision': PROTOCOL_REVISION, 'request_identity': request_identity,
                  'replication_status': 'not_attempted', 'result_status': 'inconclusive', 'lanes': results,
                  'uncertainty': protocol()['uncertainty'], 'dataset_exposure': 'verified/exposed: training split',
                  'limitations': [protocol()['claim_scope'], 'Operands vary; logical operation type is fixed',
                                  'No independent replication, fresh confirmatory evaluation or mechanism claim',
                                  'Both-path reversal tests orientation invariance, not independent order scrambling']}
    except Exception as error:
        errors.append(type(error).__name__+': '+str(error))
        result = {'schema': 'trinite.geometry-result.v1', 'outcome': 'failed', 'lanes': results, 'errors': errors,
                  'request_identity': request_identity, 'result_status': 'inconclusive'}
    finally:
        # Loading a training checkpoint restores its RNG. Observation keeps caller RNGs unchanged.
        random.setstate(py_rng); np.random.set_state(np_rng); torch.set_rng_state(torch_rng)
    result_bytes = json_bytes(result); (root/'result.json').write_bytes(result_bytes)
    if collector:
        observed(lambda: collector.record('geometry-result', inputs=raw, outputs={'result.json': result_bytes}))
        verification = observed(collector.finalize)
    if verification is not None: (root/'verification.json').write_bytes(json_bytes(verification))
    eligible = bool(observer and verification and verification['integrity_verified'] and verification['manifest_scope']=='closed'
                    and not verification['known_missing_artifacts'] and not verification['errors'] and not errors and not observation_errors)
    report = {'schema': 'trinite.geometry-run-report.v1', 'compute_outcome': result['outcome'],
              'result_identity': identity(result_bytes), 'observer_enabled': observer,
              'observer_evidence_eligible': eligible, 'release_ready': False,
              'errors': errors, 'observation_errors': observation_errors, 'verification': verification,
              'wall_seconds_hex': (time.monotonic()-clock).hex(), 'observer_seconds_hex': observation_seconds.hex(),
              'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'rss_scope': 'Linux process lifetime; includes checkpoint load/imports/observer'}
    (root/'report.json').write_bytes(json_bytes(report))
    return report
