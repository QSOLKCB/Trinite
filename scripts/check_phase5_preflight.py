"""Offline signed Phase 5 preflight; only creates an immutable rejection receipt."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from phase5_preflight.preflight import assess, read_json, safe_read

MAX_LOG_BYTES = 16384
FOUNDATIONS_STDOUT_LIMIT = 1024 * 1024
FOUNDATIONS_TIMEOUT = 2400
FOUNDATIONS_SCOPE = 'integrity/score/checkpoint/gate replay; no new generation or independent experiment'


class FoundationsVerifierFailure(ContractError):
    """Preserve bounded, SHA-256-identified child-process evidence on failure."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def utc_now():
    return datetime.now(timezone.utc)


def safe_absolute(path):
    """Normalize `..` aliases, but fail on symlinked spelling or ancestry."""
    raw = Path(path).absolute()
    absolute = Path(os.path.abspath(path))
    for p in (raw, *raw.parents, absolute, *absolute.parents):
        if p.is_symlink():
            raise ContractError('phase5 path traversal through symlink denied')
    return absolute


def _log_record(path):
    digest = hashlib.sha256()
    preview = bytearray()
    count = 0
    with path.open('rb') as stream:
        while True:
            block = stream.read(65536)
            if not block:
                break
            digest.update(block)
            count += len(block)
            if len(preview) < MAX_LOG_BYTES:
                preview.extend(block[:MAX_LOG_BYTES-len(preview)])
    return {'sha256': 'sha256:' + digest.hexdigest(), 'total_bytes': count,
            'truncated': count > MAX_LOG_BYTES,
            'prefix_b64': base64.b64encode(preview).decode('ascii')}


def verify_foundations(root, request_identity, summary_identity):
    """Invoke the *existing* verifier, retain child diagnostics without modifying run."""
    root = safe_absolute(root)
    if not root.is_dir():
        raise ContractError('unsafe or missing foundations run root')
    runner = ROOT / 'scripts/verify_foundations_run.py'
    argv = [sys.executable, str(runner), str(root), '--request-identity', request_identity,
            '--summary-identity', summary_identity]
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='trinite-p5-foundations-') as directory:
        base = Path(directory)
        stdout_file, stderr_file = base / 'stdout.bin', base / 'stderr.bin'
        with stdout_file.open('xb') as stdout, stderr_file.open('xb') as stderr:
            try:
                result = subprocess.run(argv, cwd=ROOT, shell=False, env=env,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    check=False, timeout=FOUNDATIONS_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                info = {'state': 'timed-out', 'timeout_seconds': FOUNDATIONS_TIMEOUT,
                        'stdout': _log_record(stdout_file), 'stderr': _log_record(stderr_file)}
                raise FoundationsVerifierFailure('foundations verification timed out', info) from exc
            except OSError as exc:
                info = {'state': 'spawn-error', 'error_type': type(exc).__name__,
                        'stdout': _log_record(stdout_file), 'stderr': _log_record(stderr_file)}
                raise FoundationsVerifierFailure('foundations verifier could not start', info) from exc
        info = {'state': 'completed' if result.returncode == 0 else 'rejected',
                'returncode': result.returncode, 'stdout': _log_record(stdout_file),
                'stderr': _log_record(stderr_file)}
        if result.returncode:
            raise FoundationsVerifierFailure('authoritative foundations verification rejected the run', info)
        if stdout_file.stat().st_size > FOUNDATIONS_STDOUT_LIMIT:
            raise FoundationsVerifierFailure('foundations verifier stdout exceeds bound', info)
        stdout = stdout_file.read_bytes()
        # Existing verify_foundations_run.py calls print(json_bytes(...).decode()).
        # Canonical json_bytes already ends with LF; print adds exactly ONE more.
        if not stdout.endswith(b'\n\n') or stdout.endswith(b'\n\n\n'):
            raise FoundationsVerifierFailure('unexpected foundations stdout transport framing', info)
        try:
            record = parse_json(stdout[:-1], canonical=True)
            exact_keys(record, {'summary_identity', 'request_identity', 'verified_training_cells',
                                 'outcome', 'dense_learning_adequate', 'scope'}, 'foundations verifier receipt')
            if (record['request_identity'] != request_identity or
                    record['summary_identity'] != summary_identity or
                    type(record['verified_training_cells']) is not int or
                    not 0 <= record['verified_training_cells'] <= 27 or
                    type(record['dense_learning_adequate']) is not bool or
                    record['outcome'] not in ('blocked', 'completed') or
                    record['scope'] != FOUNDATIONS_SCOPE):
                raise ContractError('foundations verifier returned an unbound or invalid receipt')
        except (ContractError, ValueError, TypeError) as exc:
            raise FoundationsVerifierFailure('foundations verifier receipt failed strict validation', info) from exc
        return {'schema': 'trinite.phase5-foundations-verification.v1', **record}


def tool_source_manifest():
    """Identify local tool entries and existing foundations source closure.

    This is a *reporting* identity, not an externally approved verifier identity.
    """
    paths = [ROOT/'scripts/check_phase5_preflight.py', ROOT/'scripts/verify_foundations_run.py',
             ROOT/'scripts/run_foundations.py', ROOT/'requirements.lock',
             *sorted((ROOT/'scripts/phase5_preflight').glob('*.py')),
             *sorted((ROOT/'src/trinite').rglob('*.py')),
             *sorted((ROOT/'src/trinite').rglob('*.json')),
             ROOT/'src/trinite/cpu-dependencies.lock']
    return {str(path.relative_to(ROOT)): identity(safe_read(path)) for path in paths}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--signatures', type=Path, required=True)
    p.add_argument('--trust-policy', type=Path, required=True)
    p.add_argument('--trust-policy-identity', required=True,
                   help='Out-of-band SHA256 of canonical JSON trust policy; never read this from the packet')
    p.add_argument('--foundations-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        sources = [safe_absolute(a) for a in (args.packet, args.signatures, args.trust_policy)]
        output = safe_absolute(args.output)
        foundations_root = safe_absolute(args.foundations_run) if args.foundations_run else None
        if output in sources or any(output in s.parents for s in sources):
            raise ContractError('phase5 output overlaps immutable input')
        if foundations_root and (output == foundations_root or foundations_root in output.parents):
            raise ContractError('phase5 output overlaps retained foundations evidence root')
        packet = read_json(sources[0])
        signature_bytes = safe_read(sources[1])
        signatures = parse_json(signature_bytes, canonical=True)
        policy = read_json(sources[2])
        started = utc_now()
        result = assess(packet, signatures, policy, args.trust_policy_identity, now=started)
        prior, prior_error, diagnostic = None, None, None
        if foundations_root and all(result['checks'][key] for key in (
                'protocol_and_matrix', 'external_policy_identity', 'signature_quorum', 'request_window')):
            try:
                refs = packet['prerequisites']
                prior = verify_foundations(foundations_root, refs['foundations_request_identity'],
                                           refs['foundations_summary_identity'])
            except FoundationsVerifierFailure as exc:
                prior_error, diagnostic = str(exc), exc.diagnostics
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                prior_error = str(exc)
            # A slow verifier cannot preserve an approval window that has expired.
            result = assess(packet, signatures, policy, args.trust_policy_identity,
                            now=utc_now(), foundations_result=prior)
        # Always recompute on a fresh UTC clock, including after a slow verifier.
        assessed = utc_now()
        result = assess(packet, signatures, policy, args.trust_policy_identity,
                        now=assessed, foundations_result=prior)
        if prior_error:
            result['errors'].append('foundations verification: ' + prior_error)
        if diagnostic is not None:
            result['foundations_diagnostics'] = diagnostic
        result['assessment_started_at'] = started.isoformat().replace('+00:00', 'Z')
        result['assessment_completed_at'] = assessed.isoformat().replace('+00:00', 'Z')
        result['signature_bundle_identity'] = identity(signature_bytes)
        manifest = tool_source_manifest()
        result['preflight_source_manifest'] = manifest
        result['preflight_source_manifest_identity'] = identity(json_bytes(manifest))
        # Exclusive fail-closed retention; no old packet or evidence-root mutation.
        output.parent.mkdir(parents=True, exist_ok=True)
        safe_absolute(output)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(output, flags, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(json_bytes(result))
            stream.flush()
            os.fsync(stream.fileno())
        print(json_bytes({'receipt_identity': identity(json_bytes(result)),
                          'outcome': result['outcome'], 'phase5_authorized': False}).decode(), end='')
        return 1
    except (ContractError, OSError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        print('phase5-preflight: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
