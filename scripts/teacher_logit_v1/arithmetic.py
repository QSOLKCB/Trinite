# SPDX-License-Identifier: MPL-2.0
"""Standard-library binary64 analysis of unmodified binary32 observations."""
import math
import struct
import sys
from trinite.contracts import ContractError

VOCAB = 259
EOS = 257
RESIDUES = (48, 49, 50)
VECTOR = struct.Struct('<259f')
TOL = 1e-12


def checked(values):
    z = list(values)
    if len(z) != VOCAB or any(type(v) is not float or not math.isfinite(v) for v in z):
        raise ContractError('exactly 259 finite float logits required')
    return z


def pack(values):
    z = checked(values)
    try:
        raw = VECTOR.pack(*z)
    except (OverflowError, struct.error) as exc:
        raise ContractError('FP32 representation failed') from exc
    if list(VECTOR.unpack(raw)) != z:
        raise ContractError('logits must already be exactly representable FP32')
    return raw


def unpack(raw):
    if type(raw) is not bytes or len(raw) != VECTOR.size:
        raise ContractError('fixed-width complete FP32 vector required')
    return checked(VECTOR.unpack(raw))


def shifted_lse(values):
    maximum = max(values)
    return maximum, math.log(math.fsum(math.exp(v - maximum) for v in values))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    a = math.exp(x)
    return a / (1.0 + a)


def metrics(values, target):
    z = checked(values)
    if target is not None and (type(target) is not int or target not in (*RESIDUES, EOS)):
        raise ContractError('unsupported TRAIN reference target')
    m, lse = shifted_lse(z)
    logp = [(v-m)-lse for v in z]
    p = [math.exp(v) for v in logp]
    error = abs(math.fsum(p)-1.0)
    if error > TOL:
        raise ContractError('full-vocabulary normalization failed')
    rest = [v for i, v in enumerate(z) if i != EOS]
    rm, rlse = shifted_lse(rest)
    odds = (z[EOS]-rm)-rlse
    if not math.isclose(logistic(odds), p[EOS], abs_tol=TOL, rel_tol=TOL):
        raise ContractError('EOS/rest consistency failed')
    # Independent probability-ratio check only when both represented masses
    # are normal binary64 values. Positive subnormals have lost relative
    # precision during exp(); their logarithms are not a reliable cross-check
    # of the stable log-space odds. Never alter the original probabilities,
    # tolerances, or log-space metrics to accommodate that representation loss.
    rest_p = math.fsum(v for i, v in enumerate(p) if i != EOS)
    odds_check = None
    if p[EOS] >= sys.float_info.min and rest_p >= sys.float_info.min:
        odds_check = math.log(p[EOS])-math.log(rest_p)
        if not math.isclose(odds, odds_check, abs_tol=TOL, rel_tol=TOL):
            raise ContractError('EOS/rest probability-ratio inconsistency')
    top = max(range(VOCAB), key=z.__getitem__)  # first index on equal logits
    best = max(RESIDUES, key=z.__getitem__)
    rank = None if target is None else 1 + sum(
        v > z[target] or (v == z[target] and i < target) for i, v in enumerate(z))
    rrank = None if target not in RESIDUES else 1 + sum(
        z[i] > z[target] or (z[i] == z[target] and i < target) for i in RESIDUES)
    if top == EOS:
        category = 'eos'
    elif target in RESIDUES and top == target:
        category = 'correct_residue'
    elif top in RESIDUES:
        category = 'other_residue' if target not in RESIDUES else 'incorrect_residue'
    else:
        category = 'other_vocabulary'
    cm, clse = shifted_lse([z[i] for i in (*RESIDUES, EOS)])
    result = {
        'argmax': top, 'argmax_category': category,
        'target_rank': rank, 'target_residue_rank': rrank,
        'best_residue_token': best,
        'target_is_best_non_eos': None if target not in RESIDUES else target == max(
            (i for i in range(VOCAB) if i != EOS), key=z.__getitem__),
        'eos_beats_target': None if target is None else z[EOS] > z[target],
        'eos_beats_target_loses_third': None if target is None else z[EOS] > z[target] and z[top] > z[EOS],
        'probability_underflow_count': sum(v == 0 for v in p),
    }
    scalars = {
        'eos_logit': z[EOS], 'best_residue_logit': z[best],
        'full_eos_probability': p[EOS], 'full_eos_log_probability': logp[EOS],
        'eos_rest_log_odds': odds, 'eos_rest_probability_check': odds_check,
        'eos_best_residue_margin': z[EOS]-z[best],
        'restricted_four_token_eos_probability': math.exp((z[EOS]-cm)-clse),
        'normalization_error': error,
        'target_logit': None if target is None else z[target],
        'target_probability': None if target is None else p[target],
        'target_log_probability': None if target is None else logp[target],
        'target_nll': None if target is None else -logp[target],
        'eos_target_margin': None if target is None else z[EOS]-z[target],
    }
    for name, value in scalars.items():
        if value is not None and not math.isfinite(value):
            raise ContractError('nonfinite derived metric')
        result[name+'_hex'] = None if value is None else value.hex()
    return result
