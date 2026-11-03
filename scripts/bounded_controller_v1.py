"""External engineering controller. Policy capability: public token prefix -> token."""
from pathlib import Path
import time

import omi_module_v1 as omi
import qutrit_conformance_v1 as qutrit
import queue_evaluation_v1 as gate
from trinite.tokenizer import ByteTokenizer, EOS

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    gate.require(type(name) is str and name in ('omi', 'qutrit'), 'unknown module')
    return omi if name == 'omi' else qutrit


def validate(request):
    gate.keys(request, ('schema', 'module', 'observation', 'mode', 'budget'))
    gate.require(request['schema'] == 'trinite.bounded-episode.v1', 'unknown schema')
    mod = module(request['module'])
    mod.validate(request['observation'])
    gate.require(type(request['mode']) is str and request['mode'] in ('direct', 'stepwise'), 'unknown mode')
    if request['mode'] == 'stepwise':
        gate.require(mod is omi and request['observation']['task'] in ('forward', 'backward'), 'stepwise only OMI forward/backward')
    budget = request['budget']
    gate.keys(budget, ('calls', 'tokens', 'prefix_positions'))
    for key, cap in (('calls', 8), ('tokens', 40), ('prefix_positions', 4920)):
        gate.integer(budget[key], 1, cap)
    return request


def sources():
    result = {**omi.sources(), **qutrit.sources()}
    for relative in ('scripts/bounded_controller_v1.py', 'scripts/run_bounded_controller_v1.py',
                     'docs/BOUNDED-CONTROLLER-V1.md'):
        result[relative] = gate.digest(gate.read(ROOT / relative))
    return result


def freeze(request):
    # Engineering request binding only: not a scientific run approval/freeze.
    validate(request)
    return dict(schema='trinite.bounded-engineering-freeze.v1', request=gate.decode(gate.encode(request)),
                sources=sources())


def run(frozen, next_token):
    """Trusted synchronous callback; never given targets, receipts or evaluator state.

    Each invocation represents ONE uncached full-prefix generation operation.
    The callback is not a security sandbox or a checkpoint loader.
    """
    gate.keys(frozen, ('schema', 'request', 'sources'))
    gate.require(frozen['schema'] == 'trinite.bounded-engineering-freeze.v1', 'unknown freeze')
    gate.require(frozen['sources'] == sources(), 'stale producer')
    request = gate.decode(gate.encode(validate(frozen['request'])))
    mod, observation = module(request['module']), request['observation']
    original = gate.decode(gate.encode(observation))
    budget = request['budget']
    steps = observation['horizon'] if request['mode'] == 'stepwise' else 1
    seen = {observation['word']}
    calls = tokens = operations = positions = 0
    policy_seconds = runner_seconds = 0.0
    trace, answer, status = [], None, 'step_cap'
    started = time.perf_counter()
    for step in range(steps):
        if request['mode'] == 'stepwise':
            observation = omi.public(observation['word'], original['task'], 1)
        ledger = mod.ledger(observation)
        reserve = ledger['emitted_tokens_including_eos']
        cost = ledger['full_prefix_forward_positions']
        if calls + 1 > budget['calls'] or operations + reserve > budget['tokens'] or positions + cost > budget['prefix_positions']:
            status = 'budget'; break
        before = time.perf_counter()
        # Observe only. Privileged evaluation is never dispatched in this function.
        admitted = mod.native('observe', observation)
        runner_seconds += time.perf_counter() - before
        gate.require(admitted == observation, 'public admission mismatch')
        prefix = tuple(mod.public_only(admitted))
        calls += 1
        emitted = []
        rejected_type = None
        for _ in range(reserve):
            before = time.perf_counter()
            operations += 1
            positions += len(prefix) + len(emitted)
            try:
                token = next_token(prefix + tuple(emitted))
            except Exception as exc:
                rejected_type = type(exc).__name__[:120]
                status = 'policy_error'
                policy_seconds += time.perf_counter() - before
                break
            policy_seconds += time.perf_counter() - before
            if type(token) is not int or not (0 <= token < 256 or token == EOS):
                rejected_type = type(token).__name__[:120]
                status = 'malformed_token'; break
            emitted.append(token)
            tokens += 1
            if token == EOS:
                break
        trace.append(dict(step=step, public_prompt=mod.prompt(admitted), emitted=emitted,
                          reserved_tokens=reserve, prefix_tokens=len(prefix), rejected_type=rejected_type))
        if status in ('malformed_token', 'policy_error'):
            break
        if not emitted or emitted[-1] != EOS:
            status = 'missing_eos'; break
        raw = emitted[:-1]
        alphabet = '012' if mod is qutrit else '0123456789ABCDEF'
        if ledger['answer_bytes'] == 1:
            alphabet = '01' if mod is qutrit else '1248'
        if len(raw) != ledger['answer_bytes'] or any(t > 127 or chr(t) not in alphabet for t in raw):
            status = 'malformed_answer'; break
        answer = bytes(raw).decode('ascii')
        if request['mode'] == 'stepwise':
            word = int(answer, 16)
            if word in seen and step + 1 < steps:
                status = 'repeated_state'; break
            seen.add(word)
            observation = omi.public(word, original['task'], 1)
        if step + 1 == steps:
            status = 'complete'
    gate.require(frozen['sources'] == sources(), 'producer changed during episode')
    return dict(schema='trinite.bounded-episode-result.v1', request=request, status=status,
                answer=answer, trace=trace, source_identity=frozen['sources'],
                accounting=dict(calls=calls, emitted_tokens=tokens, full_prefix_operations=operations,
                                processed_prefix_positions=positions, policy_seconds=policy_seconds,
                                runner_seconds=runner_seconds, elapsed_seconds=time.perf_counter()-started,
                                measured_peak_memory_bytes=None, actual_neural_forwards=None,
                                optimizer_updates=0),
                policy_binding=dict(kind='trusted injected callback', source_bound=False, configuration_bound=False),
                evidence_class='engineering policy execution; not trained-model performance')


def evaluate(result):
    """Separate post-episode oracle capability. Never returns a correction to policy."""
    request = validate(result['request'])
    expected = module(request['module']).expected(request['observation'])
    return dict(expected=expected, correct=result['status'] == 'complete' and result['answer'] == expected,
                evidence_class='public engineering conformance; not held-out learning')


class ConstantPolicy:
    """Deliberately input-insensitive engineering control, not an oracle/model."""
    def __init__(self, answer):
        gate.require(type(answer) is str and answer.isascii() and 1 <= len(answer) <= 4, 'invalid constant')
        self.answer = tuple(answer.encode('ascii')) + (EOS,)

    def __call__(self, prefix):
        # Separator '=' has byte 61; answer alphabets cannot contain it.
        offset = len(prefix) - 1 - max(i for i, token in enumerate(prefix) if token == 61)
        return self.answer[offset] if offset < len(self.answer) else EOS
