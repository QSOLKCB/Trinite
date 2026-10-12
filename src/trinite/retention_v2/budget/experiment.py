"""V2 request/checkpoint/report composition and explicit inference/replay assurance."""
from pathlib import Path
import math
import resource
import time

from ...contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from ...comparison_checkpoint import tensor_payload, progress, restore_tensors
from ...foundations import score_examples
from ...model import Decoder
from ...training import evaluate, model_identity, require_environment, run_steps
from . import spec
from .. import evidence
from ..criteria import training_failures
from ..experiment import exposure, solve_classes


class VerificationFailure(ContractError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def request_core(run_profile, packs):
    spec.profile_settings(run_profile)
    return {'schema': 'trinite.maths-retention-budget-request.v2', 'run_profile': run_profile,
            'protocol': spec.protocol(), 'protocol_identity': spec.PROTOCOL_IDENTITY,
            'source': spec.sources(), 'environment': require_environment(),
            'corpora': {k: {'dataset_identity': identity(p.raw), 'manifest_identity': identity(p.manifest)}
                        for k, p in packs.items()}}


def freeze_request(path, run_profile='research'):
    packs = spec.corpora()
    return {'request_identity': evidence.freeze(path, request_core(run_profile, packs)),
            'protocol_identity': spec.PROTOCOL_IDENTITY, 'planned_cells': len(spec.cells()), 'run_profile': run_profile}


def read_request(path, expected_identity):
    raw = evidence.read_bytes(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('v2 experiment request identity mismatch')
    run_profile = parse_json(raw, canonical=True)['run_profile']
    packs = spec.corpora()
    r, raw = evidence.read_frozen(path, expected_identity, request_core(run_profile, packs))
    return r, raw, packs


def recheck_request(path, expected_identity, req, packs):
    # Reuse only this invocation's already audited immutable bytes. Source,
    # environment, protocol and request bytes are read freshly at both boundaries.
    evidence.read_frozen(path, expected_identity, request_core(req['run_profile'], packs))


def context(state, seed, profile, request_identity, run_profile):
    require_identity(request_identity); spec.cell_name(seed, profile)
    if state.config != spec.plan_for(seed, run_profile, profile) or state.lane != 'dense':
        raise ContractError('v2 checkpoint plan/profile mismatch')
    return {'schema': 'trinite.maths-retention-budget-checkpoint.v2', 'seed': seed, 'profile': profile,
            'run_profile': run_profile, 'request_identity': request_identity, 'protocol_identity': spec.PROTOCOL_IDENTITY,
            'source': spec.sources(), 'environment': require_environment(), 'plan': state.config.to_dict(),
            'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity,
            'initialization_identity': state.initialization_identity, 'initial_train_loss_hex': state.initial_train_loss,
            'cycle_identity': identity(json_bytes([r['identity'] for r in state.data['train']])),
            'diagnostic_view_identity': identity(json_bytes([r['identity'] for r in spec.diagnostic_rows(state)]))}


def snapshot(state, seed, profile, request_identity, run_profile):
    return tensor_payload(state), json_bytes({**context(state, seed, profile, request_identity, run_profile),
            'model_identity': model_identity(state.model), 'step': state.step, 'cursor': state.cursor,
            'target_tokens': state.target_tokens, 'history': state.history, 'train_diagnostics': state.validation})


def restore(payload, metadata, seed, profile, request_identity, run_profile):
    r = parse_json(metadata, canonical=True); state = spec.state_for(seed, profile, run_profile)
    expected = context(state, seed, profile, request_identity, run_profile)
    exact_keys(r, set(expected)|{'model_identity', 'step', 'cursor', 'target_tokens', 'history', 'train_diagnostics'}, 'v2 checkpoint')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('v2 checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = r['step'], r['cursor'], r['target_tokens']
    state.history, state.validation = r['history'], r['train_diagnostics']
    progress(state)
    return restore_tensors(state, payload, r['model_identity'])


def report_for(state, seed, profile, request_identity, run_profile, packs, seconds_hex, rss):
    initial = Decoder(state.config.model, lane='dense', seed=seed)
    if model_identity(initial) != state.initialization_identity: raise ContractError('v2 initial model mismatch')
    scores, predictions, failures = {}, {}, {}
    settings = spec.profile_settings(run_profile)
    mapping = {e['example_identity']: e for p in packs.values() for e in p.examples}
    for kind, pack in packs.items():
        examples = spec.scoring_examples(pack, run_profile)
        for stage, model, splits in (('initial', initial, ('train',)), ('final', state.model, ('train',))):
            for split in splits:
                key = kind+'-'+stage+'-'+split
                values, rows = score_examples(model, examples, split, settings['generation_cap'])
                # Exact task coverage/counts come from admission, never the caller's map.
                training_failures(values, spec.expected_counts(examples, split))
                scores[key] = {**values, 'drift_diagnostics': spec.drift(rows, mapping)}
                if kind == 'maths': scores[key]['solve_classes'] = solve_classes(rows, mapping)
                predictions[key] = {'predictions': rows}
        failures[kind] = training_failures(scores[kind+'-final-train'], spec.expected_counts(examples, 'train'))
    return {**context(state, seed, profile, request_identity, run_profile), 'schema': 'trinite.maths-retention-budget-report.v2',
        'outcome': 'completed', 'model_identity': model_identity(state.model), 'steps': state.step,
        'target_tokens': state.target_tokens, 'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
        'final_train_loss_hex': evaluate(state.model, spec.diagnostic_rows(state), state.config.batch_size),
        'training_wall_seconds_hex': seconds_hex, 'peak_process_rss_kib': rss,
        'scores': scores, 'exposure': exposure(state, packs), 'training_gate_failures': failures,
        'scoring_view': settings['scoring_view'], 'held_out_scored': False, 'learning_adequacy': False,
        'scope': spec.protocol()['scope']}, predictions


def train_cell(root, seed, profile, request, request_identity):
    req, raw, packs = read_request(request, request_identity)
    run_profile = req['run_profile']; root = evidence.safe_path(root); root.mkdir(parents=True, exist_ok=False)
    try:
        state = spec.state_for(seed, profile, run_profile); started = time.perf_counter(); run_steps(state)
        seconds = time.perf_counter()-started
        payload, metadata = snapshot(state, seed, profile, request_identity, run_profile)
        (root/'tensors.safetensors').write_bytes(payload); (root/'metadata.json').write_bytes(metadata)
        evidence.write(root/'steps.json', {'steps': state.history, 'train_diagnostics': state.validation})
        report, predictions = report_for(state, seed, profile, request_identity, run_profile, packs,
                                         seconds.hex(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        evidence.write(root/'report.json', report); evidence.write(root/'predictions.json', predictions)
        recheck_request(request, request_identity, req, packs)
        evidence.seal(root, spec.inputs(raw, packs))
        return report
    except Exception as error:
        evidence.write(root/'failure.json', {'schema': 'trinite.retention-budget-failure.v2', 'error': type(error).__name__+': '+str(error),
                                           'learning_adequacy': False, 'phase5_ready': False})
        raise


def freeze_verification_request(path, root, request, request_identity):
    req, raw, _ = read_request(request, request_identity); root = evidence.safe_path(root)
    if evidence.read_bytes(root/'request.json') != raw: raise ContractError('v2 root experiment request differs')
    core = {'schema': 'trinite.retention-budget-verification-request.v2', 'request_identity': request_identity,
            'protocol_identity': spec.PROTOCOL_IDENTITY, 'environment_identity': identity(json_bytes(req['environment'])),
            'root_artifacts': evidence.commitments(root, evidence.ROOT_ARTIFACTS),
            'cells': {spec.cell_name(*c): evidence.commitments(root/spec.cell_name(*c), evidence.ARTIFACTS) for c in spec.cells()}}
    return {'verification_request_identity': evidence.freeze(path, core), 'request_identity': request_identity}


def read_verification_request(path, expected_identity, request, request_identity):
    req, _, _ = read_request(request, request_identity)
    return _read_verification_request(path, expected_identity, req, request_identity)


def _read_verification_request(path, expected_identity, req, request_identity):
    raw = evidence.read_bytes(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('v2 verification request identity mismatch')
    value = parse_json(raw, canonical=True)
    exact_keys(value, {'schema', 'request_identity', 'protocol_identity', 'environment_identity', 'root_artifacts', 'cells', 'created_at'}, 'v2 verification request')
    names = {spec.cell_name(*c) for c in spec.cells()}
    if type(value['cells']) is not dict or set(value['cells']) != names:
        raise ContractError('v2 verification request cell coverage mismatch')
    for artifacts, allowed in [(value['root_artifacts'], evidence.ROOT_ARTIFACTS), *[(v, evidence.ARTIFACTS) for v in value['cells'].values()]]:
        if type(artifacts) is not dict or not set(artifacts) <= set(allowed): raise ContractError('v2 unexpected artifact commitment names')
        for v in artifacts.values(): require_identity(v)
    core = {'schema': 'trinite.retention-budget-verification-request.v2', 'request_identity': request_identity,
            'protocol_identity': spec.PROTOCOL_IDENTITY, 'environment_identity': identity(json_bytes(req['environment'])),
            'root_artifacts': value['root_artifacts'], 'cells': value['cells']}
    return evidence.read_frozen(path, expected_identity, core)[0]


def receipt_for(request_identity, verification_identity, artifacts, tier):
    if tier not in ('inference', 'replay'): raise ContractError('unknown v2 verification tier')
    return {'schema': 'trinite.retention-budget-verification.v2', 'request_identity': request_identity,
            'verification_request_identity': verification_identity, 'artifacts': artifacts, 'tier': tier,
            'checks': {k: False for k in spec.protocol()['verification_checks']},
            'outcome': 'failed', 'error': None, 'replay_environment_identity': None, 'learning_adequacy': False}


def verify_cell(root, seed, profile, request, request_identity, verification, verification_identity, *, tier='inference'):
    name = spec.cell_name(seed, profile)
    req, raw, packs = read_request(request, request_identity)
    v = _read_verification_request(verification, verification_identity, req, request_identity)
    receipt = receipt_for(request_identity, verification_identity, v['cells'][name], tier)
    try:
        evidence.check_commitments(root, v['cells'][name], evidence.ARTIFACTS)
        bound = evidence.read_bound(root, spec.inputs(raw, packs)); receipt['checks']['closed_custody'] = True
        run_profile = req['run_profile']
        state = restore(bound['tensors.safetensors'], bound['metadata.json'], seed, profile, request_identity, run_profile)
        if state.step != state.config.steps: raise ContractError('v2 requires final checkpoint')
        for k in ('request_and_source_binding', 'safe_tensor_structure', 'scheduled_visits_and_targets'): receipt['checks'][k] = True
        report = parse_json(bound['report.json'], canonical=True)
        wall = float.fromhex(report['training_wall_seconds_hex'])
        if not math.isfinite(wall) or not 0 < wall <= state.config.max_seconds or wall.hex() != report['training_wall_seconds_hex']:
            raise ContractError('v2 invalid wall budget')
        integer(report['peak_process_rss_kib'], 'v2 RSS', 1, 2**63-1)
        expected, predictions = report_for(state, seed, profile, request_identity, run_profile, packs, wall.hex(), report['peak_process_rss_kib'])
        if (json_bytes(expected) != bound['report.json'] or json_bytes(predictions) != bound['predictions.json']
                or json_bytes({'steps': state.history, 'train_diagnostics': state.validation}) != bound['steps.json']
                or state.validation[-1]['loss_hex'] != report['final_train_loss_hex']):
            raise ContractError('v2 fresh predictions/loss/report mismatch')
        receipt['checks']['final_predictions_and_loss'] = True
        if tier == 'replay':
            replay = spec.state_for(seed, profile, run_profile); run_steps(replay)
            payload, metadata = snapshot(replay, seed, profile, request_identity, run_profile)
            if payload != bound['tensors.safetensors'] or metadata != bound['metadata.json']:
                raise ContractError('v2 complete optimizer/history replay mismatch')
            if json_bytes({'steps': replay.history, 'train_diagnostics': replay.validation}) != bound['steps.json']:
                raise ContractError('v2 full replay step copy mismatch')
            receipt['checks']['complete_optimizer_replay'] = True
            receipt['replay_environment_identity'] = identity(json_bytes(require_environment()))
        recheck_request(request, request_identity, req, packs)
        evidence.check_commitments(root, v['cells'][name], evidence.ARTIFACTS)
        receipt['outcome'] = 'passed'
        return {'seed': seed, 'profile': profile, 'outcome': 'completed', 'report': report, 'verification': receipt}
    except (ContractError, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        receipt['error'] = type(error).__name__+': '+str(error)
        raise VerificationFailure(receipt['error'], receipt) from error


def aggregate(records, errors, request_identity, verification_identity, tier):
    if [(r['seed'], r['profile']) for r in records] != list(spec.cells()):
        raise ContractError('v2 summary must retain every fixed cell in order')
    required = set(spec.protocol()['verification_checks'])-{'complete_optimizer_replay'}
    for record in records:
        spec.cell_name(record['seed'], record['profile'])
        receipt = record['verification']
        if (receipt['request_identity'] != request_identity or receipt['verification_request_identity'] != verification_identity
                or receipt['tier'] != tier or set(receipt['checks']) != set(spec.protocol()['verification_checks'])
                or any(type(v) is not bool for v in receipt['checks'].values())):
            raise ContractError('v2 aggregate verification context mismatch')
        if record['outcome'] == 'completed' and (receipt['outcome'] != 'passed'
                or not all(receipt['checks'][k] for k in required)
                or receipt['checks']['complete_optimizer_replay'] != (tier == 'replay')):
            raise ContractError('v2 complete cell lacks the requested assurance checks')
    complete = all(r['outcome']=='completed' for r in records) and not errors
    pairing, contrasts = [], {}
    if complete:
        for seed in (0, 1):
            a, b = [next(r['report'] for r in records if r['seed']==seed and r['profile']==p) for p in spec.PROFILES]
            for key in ('initialization_identity', 'initial_train_loss_hex', 'dataset_identity', 'manifest_identity', 'run_profile', 'diagnostic_view_identity'):
                if a[key] != b[key]: pairing.append('seed '+str(seed)+' paired '+key+' differs')
            run_profile = a['run_profile']
            if (a['steps'] != spec.plan_for(seed, run_profile, 'control').steps
                    or b['steps'] != 2*a['steps']
                    or any(a['scores'][k] != b['scores'][k] for k in ('maths-initial-train', 'legacy-initial-train'))):
                pairing.append('seed '+str(seed)+' budget/initial scores differ')
            for report in (a,b):
                for kind in ('legacy','maths'):
                    if report['exposure'][kind]['example_visits'] != report['steps']*4:
                        pairing.append('seed '+str(seed)+' balanced visits differ')
        if not pairing:
            for seed in (0, 1):
                a, b = [next(r['report'] for r in records if r['seed']==seed and r['profile']==p) for p in spec.PROFILES]
                contrasts[str(seed)] = {kind: {'control_correct': a['scores'][kind+'-final-train']['correct'],
                    'candidate_correct': b['scores'][kind+'-final-train']['correct'],
                    'examples': a['scores'][kind+'-final-train']['examples']} for kind in ('legacy', 'maths')}
    if complete and not pairing:
        packs = spec.corpora()
        for record in records:
            report = record['report']
            expected = {kind: training_failures(report['scores'][kind+'-final-train'],
                        spec.expected_counts(spec.scoring_examples(pack,report['run_profile']),'train'))
                        for kind,pack in packs.items()}
            if report['training_gate_failures'] != expected:
                raise ContractError('budget aggregate training gates disagree with admitted task counts')
    joint_pass = (complete and not pairing and tier=='replay'
                  and all(r['report']['run_profile']=='research' for r in records)
                  and all(not failures for r in records if r['profile']=='candidate'
                          for failures in r['report']['training_gate_failures'].values()))
    return {'schema': 'trinite.maths-retention-budget-summary.v2', 'request_identity': request_identity,
            'verification_request_identity': verification_identity, 'tier': tier, 'cells': records,
            'outcome': 'completed' if complete and not pairing else 'failed', 'worker_errors': errors,
            'pairing_errors': pairing, 'contrasts': contrasts, 'selected_candidate': None,
            'joint_training_accepted': joint_pass, 'held_out_scored': False,
            'learning_adequacy': False, 'phase5_ready': False, 'format_selection_eligible': False,
            'release_ready': False, 'scope': spec.protocol()['scope']}


def summarize(root, request, request_identity, verification, verification_identity, *, tier='inference'):
    req, request_raw, packs = read_request(request, request_identity)
    v = _read_verification_request(verification, verification_identity, req, request_identity)
    if evidence.read_bytes(Path(root)/'request.json') != request_raw:
        raise ContractError('v2 retained experiment request copy differs')
    evidence.check_commitments(root, v['root_artifacts'], evidence.ROOT_ARTIFACTS)
    errors = parse_json(evidence.read_bytes(Path(root)/'worker-errors.json'), canonical=True)
    if (type(errors) is not dict or not set(errors) <= {spec.cell_name(*c) for c in spec.cells()}
            or any(type(x) is not str or not x.strip() for x in errors.values())):
        raise ContractError('v2 invalid worker error inventory')
    records = []
    for seed, profile in spec.cells():
        name = spec.cell_name(seed, profile)
        try:
            if name in errors: raise ContractError(errors[name])
            records.append(verify_cell(Path(root)/name, seed, profile, request, request_identity, verification, verification_identity, tier=tier))
        except (ContractError, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            receipt = getattr(error, 'receipt', receipt_for(request_identity, verification_identity, v['cells'][name], tier))
            if receipt['error'] is None: receipt['error'] = type(error).__name__+': '+str(error)
            records.append({'seed': seed, 'profile': profile, 'outcome': 'failed', 'error': receipt['error'], 'verification': receipt})
    # A later cell's replay must not leave earlier successes trusted against
    # changed retained bytes. Recheck commitments AND actual custody/root copies
    # at the summary boundary without repeating numerical replay or caching IO.
    for record in records:
        if record['outcome'] != 'completed': continue
        name = spec.cell_name(record['seed'], record['profile'])
        try:
            evidence.check_commitments(Path(root)/name, v['cells'][name], evidence.ARTIFACTS)
            evidence.read_bound(Path(root)/name, spec.inputs(request_raw, packs))
        except (ContractError, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            receipt = record['verification']
            receipt['outcome'] = 'failed'; receipt['checks']['closed_custody'] = False
            receipt['error'] = 'summary boundary: '+type(error).__name__+': '+str(error)
            record['outcome'] = 'failed'; record['error'] = receipt['error']
    evidence.check_commitments(root, v['root_artifacts'], evidence.ROOT_ARTIFACTS)
    recheck_request(request, request_identity, req, packs)
    return aggregate(records, errors, request_identity, verification_identity, tier)
