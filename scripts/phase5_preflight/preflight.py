"""Read-only, signed Phase 5 *preflight*; never an intervention authority.

This boundary intentionally cannot start geometry-on training or issue an
executable authorization. Evidence integrity, empirical learning and permission
are different predicates. The existing frozen research verifiers remain the
only source of numerical/gate facts.
"""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity

PACKET_SCHEMA = 'trinite.phase5-preflight-packet.v1'
PROTOCOL_SCHEMA = 'trinite.phase5-geometry-proposal.v1'
POLICY_SCHEMA = 'trinite.phase5-trust-policy.v1'
RECEIPT_SCHEMA = 'trinite.phase5-preflight-receipt.v1'
ROLES = ('protocol-steward', 'independent-verifier')
FORMATS = ('dense', 'ternary')
CONDITIONS = ('geometry-off', 'geometry-on', 'auxiliary-null')
_SHA1 = re.compile(r'[0-9a-f]{40}\Z')

_ED25519_SPKI_PREFIX = bytes.fromhex('302a300506032b6570032100')


def public_key_der(pem: str) -> bytes:
    """Accept one Ed25519 SPKI PEM, rejecting EC/RSA/Ed448 and trailing material.

    The exact PEM bytes are still bound by the v1 key_identity for compatibility;
    quorum independence instead compares the canonical public key itself.
    """
    if type(pem) is not str or len(pem) > 2048 or not pem.isascii():
        raise ContractError('phase5 trusted key requires bounded ASCII PEM')
    header = '-----BEGIN PUBLIC KEY-----\n'
    footer = '-----END PUBLIC KEY-----\n'
    if not pem.startswith(header) or not pem.endswith(footer):
        raise ContractError('phase5 trusted key requires one PUBLIC KEY object')
    lines = pem[len(header):-len(footer)].splitlines()
    if (not lines or any(not line or len(line) > 64 or
                         re.fullmatch('[A-Za-z0-9+/]*={0,2}', line) is None
                         for line in lines)):
        raise ContractError('phase5 public key PEM body is malformed')
    joined = ''.join(lines)
    try:
        der = base64.b64decode(joined, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError('phase5 public key PEM is not canonical base64') from exc
    if base64.b64encode(der).decode('ascii') != joined:
        raise ContractError('phase5 public key base64 is not canonical')
    if len(der) != len(_ED25519_SPKI_PREFIX) + 32 or not der.startswith(_ED25519_SPKI_PREFIX):
        raise ContractError('phase5 public key must be Ed25519 SubjectPublicKeyInfo')
    return der


def validate_matrix(rows: object, seeds: list[int], name: str) -> None:
    expected = [{'seed': seed, 'format': fmt, 'condition': condition}
                for seed in seeds for fmt in FORMATS for condition in CONDITIONS]
    if type(rows) is not list or len(rows) != len(expected):
        raise ContractError(name + ' missing/reordered geometry and null controls')
    for row in rows:
        exact_keys(row, {'seed', 'format', 'condition'}, name + ' cell')
        integer(row['seed'], name + ' seed', 0, 2**31 - 1)
        if type(row['format']) is not str or row['format'] not in FORMATS:
            raise ContractError(name + ' format invalid')
        if type(row['condition']) is not str or row['condition'] not in CONDITIONS:
            raise ContractError(name + ' condition invalid')
    if json_bytes(rows) != json_bytes(expected):
        raise ContractError(name + ' missing/reordered geometry and null controls')


def safe_read(path: Path, *, limit: int = 1024 * 1024) -> bytes:
    """Never accept path links or unbounded/unstructured authorization input."""
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ContractError('missing or symlinked phase5 input')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ContractError('phase5 input exceeds byte limit')
    return data


def read_json(path: Path, *, limit: int = 1024 * 1024) -> dict:
    data = parse_json(safe_read(path, limit=limit), canonical=True)
    if type(data) is not dict:
        raise ContractError('phase5 requires canonical JSON object')
    return data


def aware_utc(raw: object, name: str) -> datetime:
    if type(raw) is not str or not raw.endswith('Z'):
        raise ContractError(name + ' must use UTC RFC3339 Z')
    try:
        result = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except ValueError as err:
        raise ContractError(name + ' is invalid') from err
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0 or result.isoformat().replace('+00:00', 'Z') != raw:
        raise ContractError(name + ' must be canonical UTC')
    return result


def exact_hash(value: object, label: str) -> str:
    try:
        return require_identity(value)
    except ContractError as err:
        raise ContractError(label + ': ' + str(err)) from err


def validate_protocol(protocol: object) -> None:
    exact_keys(protocol, {'schema', 'objective', 'seeds', 'matrix', 'budgets', 'margins',
                          'source_identity', 'data_admission_identity', 'evaluation_identity',
                          'preregistration_witness_identity'}, 'phase5 proposal')
    if protocol['schema'] != PROTOCOL_SCHEMA:
        raise ContractError('unsupported phase5 proposal schema')
    for k in ('source_identity', 'data_admission_identity', 'evaluation_identity', 'preregistration_witness_identity'):
        exact_hash(protocol[k], k)
    objective = exact_keys(protocol['objective'], {'formula', 'implementation_identity', 'control_formula',
                                                   'weight', 'differentiability_receipt_identity'}, 'geometry objective')
    for k in ('implementation_identity', 'differentiability_receipt_identity'):
        exact_hash(objective[k], k)
    for k in ('formula', 'control_formula'):
        if type(objective[k]) is not str or not 1 <= len(objective[k]) <= 512 or 'TODO' in objective[k].upper():
            raise ContractError('phase5 objective must be fully specified: '+k)
    # This proposal is not a numerical proof, but intervention strength must
    # at least be an unambiguous canonical scalar, not "NaN", "disabled", or
    # a post-hoc arbitrary string. Freeze nonzero 0 < weight <= 1, with at most
    # six decimal places, no exponent, leading/trailing zero ambiguity or sign.
    if (type(objective['weight']) is not str or
            re.fullmatch(r'(?:0\.[0-9]{0,5}[1-9]|1)', objective['weight']) is None):
        raise ContractError('phase5 objective weight must be canonical in (0,1]')
    seeds = protocol['seeds']
    if type(seeds) is not list or not 2 <= len(seeds) <= 16 or len(set(map(str, seeds))) != len(seeds):
        raise ContractError('phase5 needs >=2 distinct paired seeds')
    for s in seeds:
        integer(s, 'phase5 seed', 0, 2**31-1)
    if seeds != sorted(seeds):
        raise ContractError('phase5 seeds must be in canonical order')
    validate_matrix(protocol['matrix'], seeds, 'phase5 matrix')
    budget = exact_keys(protocol['budgets'], {'updates_per_cell', 'target_tokens_per_cell',
                                             'max_wall_seconds_per_cell'}, 'phase5 budgets')
    for k in budget:
        integer(budget[k], k, 1, 2**31-1)
    margins = exact_keys(protocol['margins'], {'min_quality_delta', 'max_cost_ratio',
                                              'uncertainty_method'}, 'phase5 margins')
    if (not isinstance(margins['min_quality_delta'], str) or
            not re.fullmatch(r'-?0\.[0-9]{1,6}|0|1', margins['min_quality_delta']) or
            type(margins['max_cost_ratio']) is not str or
            not re.fullmatch(r'[1-9][0-9]{0,2}(?:\.[0-9]{1,6})?', margins['max_cost_ratio']) or
            margins['uncertainty_method'] not in ('paired-seed-bootstrap-v1', 'paired-seed-exact-v1')):
        raise ContractError('phase5 quality/cost/uncertainty margins must be frozen')


def validate_packet(packet: object) -> None:
    exact_keys(packet, {'schema', 'proposal', 'request', 'prerequisites'}, 'phase5 packet')
    if packet['schema'] != PACKET_SCHEMA:
        raise ContractError('unsupported phase5 packet schema')
    validate_protocol(packet['proposal'])
    request = exact_keys(packet['request'], {'proposal_identity', 'source_tree', 'source_identity',
        'data_admission_identity', 'environment_identity', 'run_nonce', 'issued_at', 'expires_at', 'matrix'}, 'phase5 request')
    if request['proposal_identity'] != identity(json_bytes(packet['proposal'])):
        raise ContractError('phase5 proposal identity changed')
    if type(request['source_tree']) is not str or _SHA1.fullmatch(request['source_tree']) is None:
        raise ContractError('phase5 Git tree identity is invalid')
    for k in ('source_identity', 'data_admission_identity', 'environment_identity'):
        exact_hash(request[k], k)
    validate_matrix(request['matrix'], packet['proposal']['seeds'], 'phase5 request matrix')
    if (request['source_identity'] != packet['proposal']['source_identity'] or
            request['data_admission_identity'] != packet['proposal']['data_admission_identity'] or
            json_bytes(request['matrix']) != json_bytes(packet['proposal']['matrix'])):
        raise ContractError('phase5 request/proposal material or matrix mismatch')
    if type(request['run_nonce']) is not str or not re.fullmatch('[0-9a-f]{32,64}', request['run_nonce']):
        raise ContractError('phase5 nonce requires >=128-bit externally provided hex')
    start = aware_utc(request['issued_at'], 'issued_at')
    end = aware_utc(request['expires_at'], 'expires_at')
    if not 0 < (end - start).total_seconds() <= 86400:
        raise ContractError('phase5 authorization window must be <=24 hours')
    prereq = exact_keys(packet['prerequisites'], {'foundations_request_identity', 'foundations_summary_identity',
           'full_optimizer_replay_identity', 'fresh_family_confirmation_identity',
           'fold_composed_adequacy_identity'}, 'phase5 prerequisites')
    for k, v in prereq.items():
        exact_hash(v, k)


def verify_quorum(packet: dict, signatures: object, policy: dict) -> None:
    """Verify independent Ed25519 signatures over *exact* canonical packet bytes.

    Trust policy identity is checked separately by the caller against an
    externally held digest; the packet may not self-declare trusted keys.
    """
    exact_keys(policy, {'schema', 'keys', 'required_roles'}, 'phase5 trust policy')
    if policy['schema'] != POLICY_SCHEMA or policy['required_roles'] != list(ROLES):
        raise ContractError('phase5 trust policy roles/schema invalid')
    keys = policy['keys']
    if type(keys) is not list or len(keys) != len(ROLES):
        raise ContractError('phase5 requires two distinct trusted key records')
    allowed = {}
    mathematical_keys = set()
    for key in keys:
        exact_keys(key, {'role', 'key_identity', 'public_key_pem'}, 'phase5 trusted key')
        role, pem = key['role'], key['public_key_pem']
        if (type(role) is not str or role not in ROLES or role in allowed or
                type(key['key_identity']) is not str):
            raise ContractError('invalid/duplicate phase5 trusted key')
        der = public_key_der(pem)
        if identity(pem.encode('ascii')) != key['key_identity']:
            raise ContractError('phase5 trusted key PEM identity mismatch')
        if der in mathematical_keys:
            raise ContractError('phase5 independent key identities required')
        mathematical_keys.add(der)
        allowed[role] = key
    if set(allowed) != set(ROLES):
        raise ContractError('phase5 independent signer roles required')
    if type(signatures) is not list or len(signatures) != len(ROLES):
        raise ContractError('phase5 requires independent signature quorum')
    openssl = shutil.which('openssl')
    if not openssl:
        raise ContractError('OpenSSL required to verify phase5 signatures')
    with tempfile.TemporaryDirectory(prefix='trinite-p5-verify-') as directory:
        root = Path(directory)
        payload_path = root/'packet.json'
        payload_path.write_bytes(json_bytes(packet))
        verified = set()
        for row in signatures:
            exact_keys(row, {'role', 'key_identity', 'algorithm', 'signature_b64'}, 'phase5 signature')
            role = row['role']
            if role not in allowed or role in verified or row['algorithm'] != 'ed25519':
                raise ContractError('phase5 unsupported/duplicate signer role')
            if row['key_identity'] != allowed[role]['key_identity']:
                raise ContractError('phase5 signature key is not in external trust policy')
            try:
                signature = base64.b64decode(row['signature_b64'], validate=True)
            except (binascii.Error, TypeError, ValueError) as err:
                raise ContractError('phase5 signature is not canonical base64') from err
            if len(signature) != 64 or base64.b64encode(signature).decode('ascii') != row['signature_b64']:
                raise ContractError('phase5 Ed25519 signature bytes invalid')
            pub = root/'public.pem'
            sig = root/'signature.bin'
            pub.write_bytes(allowed[role]['public_key_pem'].encode('ascii'))
            sig.write_bytes(signature)
            try:
                run = subprocess.run([openssl, 'pkeyutl', '-verify', '-pubin', '-inkey', str(pub),
                    '-rawin', '-in', str(payload_path), '-sigfile', str(sig)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False, timeout=20)
            except (OSError, subprocess.TimeoutExpired) as err:
                raise ContractError('phase5 signature verifier failed') from err
            if run.returncode != 0:
                raise ContractError('phase5 cryptographic signature verification failed')
            verified.add(role)
    if verified != set(ROLES):
        raise ContractError('phase5 signer threshold unmet')


def assess(packet: dict, signatures: object, policy: dict, policy_identity: str,
           *, now: datetime | None = None, foundations_result: dict | None = None) -> dict:
    """Generate a deterministic rejection receipt, never a release authorization.

    foundations_result may only come from the authoritative read-only verifier,
    not a caller-supplied flag in the candidate packet.
    """
    checks = {key: False for key in ('protocol_and_matrix', 'external_policy_identity', 'signature_quorum',
        'request_window', 'foundations_learning', 'optimizer_full_replay', 'fresh_confirmation',
        'fold_composed_adequacy', 'preregistration_witness', 'geometry_execution_authorized')}
    errors = []
    try:
        validate_packet(packet)
        checks['protocol_and_matrix'] = True
    except (ContractError, TypeError, ValueError) as err:
        errors.append(str(err))
    try:
        exact_hash(policy_identity, 'trust policy identity')
        if policy_identity != identity(json_bytes(policy)):
            raise ContractError('external trust policy digest mismatch')
        checks['external_policy_identity'] = True
        if checks['protocol_and_matrix']:
            verify_quorum(packet, signatures, policy)
            checks['signature_quorum'] = True
    except (ContractError, TypeError, ValueError, UnicodeError, OverflowError) as err:
        errors.append(str(err))
    if checks['protocol_and_matrix']:
        current = now if now is not None else datetime.now(timezone.utc)
        try:
            if current.tzinfo is None or current.utcoffset().total_seconds() != 0:
                raise ContractError('UTC evaluation clock required')
            start = aware_utc(packet['request']['issued_at'], 'issued_at')
            end = aware_utc(packet['request']['expires_at'], 'expires_at')
            if not start <= current <= end:
                raise ContractError('phase5 request expired or not yet issued')
            checks['request_window'] = True
        except (ContractError, TypeError, ValueError) as err:
            errors.append(str(err))
    if foundations_result is not None:
        # An operator cannot supply this property in packet JSON; only the CLI
        # calls the repository's read-only foundations verifier and checks its
        # canonical stdout / retained external identities before passing it in.
        prereq = packet.get('prerequisites')
        if (type(foundations_result) is dict and type(prereq) is dict
                and foundations_result.get('schema') == 'trinite.phase5-foundations-verification.v1'
                and foundations_result.get('request_identity') == prereq.get('foundations_request_identity')
                and foundations_result.get('summary_identity') == prereq.get('foundations_summary_identity')
                and foundations_result.get('outcome') == 'completed'
                and foundations_result.get('dense_learning_adequate') is True
                and foundations_result.get('verified_training_cells') == 27):
            checks['foundations_learning'] = True
        else:
            errors.append('original foundations adequacy is not independently verified')
    else:
        errors.append('original foundations learning evidence not supplied')
    # Conservatively unavailable in this first increment: the original
    # foundations verifier checks retained score/checkpoint derivations but does
    # *not* replay every optimizer update. Nor does an external hash establish
    # fresh family confirmation, composed-answer adequacy or witnessed freeze.
    for key, reason in (
        ('optimizer_full_replay', 'no independent full optimizer replay binding'),
        ('fresh_confirmation', 'no verified fresh-family confirmation'),
        ('fold_composed_adequacy', 'no verified original fold/composed adequacy'),
        ('preregistration_witness', 'no independently verified pre-outcome witness'),
        ('geometry_execution_authorized', 'no implemented/reviewed geometry-on runner')):
        errors.append(reason)
    return {'schema': RECEIPT_SCHEMA, 'outcome': 'blocked', 'phase5_authorized': False,
            'phase5_outcome_accepted': False, 'request_identity': identity(json_bytes(packet)),
            'trust_policy_identity': policy_identity, 'checks': checks, 'errors': errors,
            'scope': 'read-only prospective preflight; not a geometry-on execution token'}
