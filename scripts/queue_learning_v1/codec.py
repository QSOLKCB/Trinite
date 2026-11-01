"""Lossless public queue carrier. Labels are accepted only by supervised()."""
import re

import queue_evaluation_v1 as gate
from trinite.contracts import ContractError
from trinite.tokenizer import ByteTokenizer, EOS

ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'
TASK = 'forecast.dropped'
CONTEXT = 256
GENERATION = 5  # Four decimal digits (32*255 <= 8160), then EOS; '?' is abstention.


def compact(observation_bytes):
    obs = gate.observation(observation_bytes)
    header = f"Q1;{obs['capacity']};{obs['observe_ticks']};{obs['horizon']};"
    rows = []
    for row in obs['rows']:
        mask = sum(int(v) << i for i, v in enumerate(row['observed']))
        rows.append(str(int(row['outage'])) + ALPHABET[mask] + ''.join(
            ALPHABET[v // 64] + ALPHABET[v % 64] for v in row['values']))
    return header + ''.join(rows)


def expand(text):
    """Independent reader: fixed-width rows, implicit contiguous tick numbers."""
    if type(text) is not str or not text.isascii() or len(text) > 4096:
        raise ContractError('bounded ASCII queue carrier required')
    parts = text.split(';')
    if len(parts) != 5 or parts[0] != 'Q1':
        raise ContractError('queue carrier version/header mismatch')
    numbers = parts[1:4]
    if any(re.fullmatch(r'(0|[1-9][0-9]{0,2})', v) is None for v in numbers):
        raise ContractError('noncanonical queue header')
    capacity, n, horizon = map(int, numbers)
    if len(parts[4]) != 14 * n:
        raise ContractError('queue carrier row length mismatch')
    rows = []
    for tick in range(n):
        raw = parts[4][tick * 14:(tick + 1) * 14]
        if raw[0] not in '01' or any(c not in ALPHABET for c in raw[1:]):
            raise ContractError('queue carrier alphabet mismatch')
        mask = ALPHABET.index(raw[1])
        rows.append({'tick': tick, 'outage': raw[0] == '1',
                     'values': [64 * ALPHABET.index(raw[i]) + ALPHABET.index(raw[i+1])
                                for i in range(2, 14, 2)],
                     'observed': [bool(mask & (1 << i)) for i in range(6)]})
    obs = {'schema': 'trinite.queue-observation.v1', 'capacity': capacity,
           'observe_ticks': n, 'horizon': horizon, 'rows': rows}
    return gate.observation(gate.encode(obs))


def prompt(observation_bytes, task=TASK):
    if task != TASK or type(task) is not str:
        raise ContractError('unknown queue task')
    # Strict whitelist is enforced before encoding; no request or row metadata.
    obs = gate.observation(observation_bytes)
    carrier = compact(observation_bytes)
    if expand(carrier) != obs:
        raise ContractError('queue roundtrip changed public observation')
    value = carrier + '?' + task + '='
    if 1 + len(value.encode('ascii')) + GENERATION > CONTEXT:
        raise ContractError('queue input and fixed generation allowance exceed context 256')
    return value


def supervised(observation_bytes, label):
    obs = gate.observation(observation_bytes)
    gate.integer(label, 0, 32 * obs['horizon'])
    prefix = prompt(observation_bytes)
    answer = str(label)
    encoding = ByteTokenizer(CONTEXT).encode(prefix + answer, answer_start=len(prefix))
    if sum(encoding.loss_mask) != len(answer) + 1:
        raise ContractError('queue answer/EOS mask mismatch')
    return encoding


def output(tokens, observation_bytes):
    """No repair, stripping, truncation, answer-conditioned decoding or fallback."""
    obs = gate.observation(observation_bytes)
    if (type(tokens) is not list or not 1 <= len(tokens) <= GENERATION
            or any(type(t) is not int or not 0 <= t <= 258 for t in tokens)
            or tokens[-1] != EOS or any(t > 255 for t in tokens[:-1])):
        return {'status': 'malformed'}
    raw = bytes(tokens[:-1])
    if raw == b'?':
        return {'status': 'abstain'}
    if re.fullmatch(rb'(0|[1-9][0-9]{0,3})', raw) is None:
        return {'status': 'malformed'}
    value = int(raw)
    if value > 32 * obs['horizon']:
        return {'status': 'malformed'}
    return {'status': 'prediction', 'value': value}
