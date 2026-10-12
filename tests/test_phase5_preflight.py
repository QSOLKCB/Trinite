"""Adversarial and positive *signature* conformance; no empirical gate pass."""
import base64
import copy
import fcntl
import errno
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from phase5_preflight.preflight import assess, validate_packet, verify_quorum, public_key_der
import importlib.util
from unittest import mock


def preflight_cli():
    spec = importlib.util.spec_from_file_location('phase5_cli_test', ROOT/'scripts/check_phase5_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(label):
    return identity(label.encode())


def utc(t):
    return t.isoformat(timespec='seconds').replace('+00:00', 'Z')


class Phase5AuthorizationPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.openssl = shutil.which('openssl')
        if cls.openssl is None:
            raise AssertionError('OpenSSL Ed25519 is required for preflight signature tests')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.keys = {}
        self.trust = []
        for role in ('protocol-steward', 'independent-verifier'):
            private = root / (role+'.key')
            public = root / (role+'.pem')
            subprocess.run([self.openssl, 'genpkey', '-algorithm', 'ed25519', '-out', str(private)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            subprocess.run([self.openssl, 'pkey', '-in', str(private), '-pubout', '-out', str(public)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            pem = public.read_text('ascii')
            self.trust.append({'role': role, 'key_identity': identity(pem.encode('ascii')),
                               'public_key_pem': pem})
            self.keys[role] = private
        self.policy = {'schema': 'trinite.phase5-trust-policy.v1',
                       'keys': self.trust, 'required_roles': ['protocol-steward', 'independent-verifier']}
        self.policy_identity = identity(json_bytes(self.policy))
        now = datetime.now(timezone.utc).replace(microsecond=0)
        self.now = now
        seeds = [0, 1]
        matrix = [{'seed': seed, 'format': fmt, 'condition': c}
                  for seed in seeds for fmt in ('dense', 'ternary')
                  for c in ('geometry-off', 'geometry-on', 'auxiliary-null')]
        protocol = {'schema': 'trinite.phase5-geometry-proposal.v1',
                    'objective': {'formula': 'matched per-pair angle loss', 'implementation_identity': digest('impl'),
                                  'control_formula': 'matched null auxiliary loss', 'weight': '0.02',
                                  'differentiability_receipt_identity': digest('gradient-conformance')},
                    'seeds': seeds, 'matrix': matrix,
                    'budgets': {'updates_per_cell': 8192, 'target_tokens_per_cell': 1048576,
                                'max_wall_seconds_per_cell': 960},
                    'margins': {'min_quality_delta': '-0.05', 'max_cost_ratio': '1.25',
                                'uncertainty_method': 'paired-seed-exact-v1'},
                    'source_identity': digest('source'), 'data_admission_identity': digest('admission'),
                    'evaluation_identity': digest('eval'), 'preregistration_witness_identity': digest('witness')}
        self.packet = {'schema': 'trinite.phase5-preflight-packet.v1', 'proposal': protocol,
            'request': {'proposal_identity': identity(json_bytes(protocol)),
                        'source_tree': 'a'*40, 'source_identity': digest('source'),
                        'data_admission_identity': digest('admission'),
                        'environment_identity': digest('environment'), 'run_nonce': 'b'*32,
                        'issued_at': utc(now-timedelta(minutes=3)),
                        'expires_at': utc(now+timedelta(hours=1)), 'matrix': matrix},
            'prerequisites': {'foundations_request_identity': digest('foundations-request'),
                'foundations_summary_identity': digest('foundations-summary'),
                'full_optimizer_replay_identity': digest('full-replay'),
                'fresh_family_confirmation_identity': digest('fresh'),
                'fold_composed_adequacy_identity': digest('fold')}}

    def signatures(self, packet=None):
        packet = self.packet if packet is None else packet
        base = Path(self.tmp.name)
        payload = base/'packet.json'
        payload.write_bytes(json_bytes(packet))
        records = []
        for k in self.trust:
            role = k['role']
            out = base/(role+'.sig')
            subprocess.run([self.openssl, 'pkeyutl', '-sign', '-inkey', str(self.keys[role]),
                            '-rawin', '-in', str(payload), '-out', str(out)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            records.append({'role': role, 'key_identity': k['key_identity'],
                            'algorithm': 'ed25519',
                            'signature_b64': base64.b64encode(out.read_bytes()).decode('ascii')})
        return records

    def test_valid_quorum_cannot_issue_geometry_authorization(self):
        signatures = self.signatures()
        result = assess(self.packet, signatures, self.policy, self.policy_identity, now=self.now)
        self.assertTrue(all(result['checks'][k] for k in ('protocol_and_matrix',
                        'external_policy_identity', 'signature_quorum', 'request_window')))
        self.assertFalse(result['checks']['foundations_learning'])
        self.assertFalse(result['checks']['optimizer_full_replay'])
        self.assertFalse(result['phase5_authorized'])
        self.assertFalse(result['phase5_outcome_accepted'])
        self.assertEqual(result['outcome'], 'blocked')
        self.assertEqual(result, parse_json(json_bytes(result), canonical=True))

    def test_one_byte_mutation_breaks_signatures(self):
        signatures = self.signatures()
        self.packet['request']['run_nonce'] = 'c'*32
        result = assess(self.packet, signatures, self.policy, self.policy_identity, now=self.now)
        self.assertFalse(result['checks']['signature_quorum'])
        self.assertIn('cryptographic signature', '\n'.join(result['errors']))

    def test_missing_control_rejected(self):
        self.packet['proposal']['matrix'].pop()
        with self.assertRaisesRegex(ContractError, 'matrix missing'):
            validate_packet(self.packet)
        result = assess(self.packet, [], self.policy, self.policy_identity, now=self.now)
        self.assertFalse(result['checks']['protocol_and_matrix'])
        self.assertFalse(result['phase5_authorized'])

    def test_duplicate_signer_and_role_cannot_make_quorum(self):
        signatures = self.signatures()
        signatures[1] = dict(signatures[0], role='independent-verifier')
        with self.assertRaisesRegex(ContractError, 'not in external trust policy'):
            verify_quorum(self.packet, signatures, self.policy)
        self.policy['keys'][1]['key_identity'] = self.policy['keys'][0]['key_identity']
        self.policy['keys'][1]['public_key_pem'] = self.policy['keys'][0]['public_key_pem']
        with self.assertRaisesRegex(ContractError, 'independent key'):
            verify_quorum(self.packet, self.signatures(), self.policy)

    def test_policy_must_be_pinned_externally(self):
        result = assess(self.packet, self.signatures(), self.policy, digest('different-policy'), now=self.now)
        self.assertFalse(result['checks']['external_policy_identity'])
        self.assertFalse(result['checks']['signature_quorum'])
        self.assertFalse(result['phase5_authorized'])

    def test_expired_and_future_windows_are_rejected(self):
        result = assess(self.packet, self.signatures(), self.policy, self.policy_identity,
                        now=self.now+timedelta(hours=2))
        self.assertFalse(result['checks']['request_window'])
        self.assertFalse(result['phase5_authorized'])

    def test_missing_evidence_and_unsigned_confirmation_cannot_promote(self):
        signatures = self.signatures()
        forged = {'schema': 'trinite.phase5-foundations-verification.v1',
                  'request_identity': digest('foundations-request'),
                  'summary_identity': digest('foundations-summary'), 'outcome': 'completed',
                  'dense_learning_adequate': True, 'verified_training_cells': 27}
        result = assess(self.packet, signatures, self.policy, self.policy_identity,
                        now=self.now, foundations_result=forged)
        # An in-process result is NOT an optimizer replay or an authorization.
        self.assertFalse(result['phase5_authorized'])
        self.assertFalse(result['checks']['optimizer_full_replay'])
        self.assertFalse(result['checks']['geometry_execution_authorized'])

    def test_noncanonical_protocol_identity_rejected(self):
        self.packet['request']['proposal_identity'] = digest('altered')
        with self.assertRaisesRegex(ContractError, 'proposal identity changed'):
            validate_packet(self.packet)

    def test_request_admission_and_tree_binding_cannot_drift(self):
        self.packet['request']['data_admission_identity'] = digest('different-admission')
        with self.assertRaisesRegex(ContractError, 'request/proposal material'):
            validate_packet(self.packet)
        self.packet['request']['data_admission_identity'] = digest('admission')
        self.packet['request']['source_tree'] = 'main'
        with self.assertRaisesRegex(ContractError, 'Git tree identity'):
            validate_packet(self.packet)

    def test_unknown_field_and_unsigned_objective_rejected(self):
        self.packet['proposal']['objective']['formula'] = 'TODO'
        with self.assertRaisesRegex(ContractError, 'fully specified'):
            validate_packet(self.packet)
        self.packet['proposal']['objective']['formula'] = 'matched per-pair angle loss'
        self.packet['request']['result_was_great'] = True
        with self.assertRaisesRegex(ContractError, 'requires exactly'):
            validate_packet(self.packet)

    def test_cli_retains_rejection_exclusively(self):
        root = Path(self.tmp.name)
        for name, obj in [('packet.json', self.packet), ('signatures.json', self.signatures()),
                          ('policy.json', self.policy)]:
            (root/name).write_bytes(json_bytes(obj))
        cmd = [sys.executable, str(ROOT/'scripts/check_phase5_preflight.py'), '--packet', str(root/'packet.json'),
               '--signatures', str(root/'signatures.json'), '--trust-policy', str(root/'policy.json'),
               '--trust-policy-identity', self.policy_identity, '--output', str(root/'receipt.json')]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=os.environ.copy() | {'PYTHONPATH': str(ROOT/'src')}, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        stored = parse_json((root/'receipt.json').read_bytes(), canonical=True)
        self.assertFalse(stored['phase5_authorized'])
        old = (root/'receipt.json').read_bytes()
        again = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=os.environ.copy() | {'PYTHONPATH': str(ROOT/'src')}, check=False)
        self.assertEqual(again.returncode, 2)
        self.assertEqual(old, (root/'receipt.json').read_bytes())


    def test_same_ed25519_public_key_with_different_pem_wrapping_is_not_a_quorum(self):
        original = self.trust[0]['public_key_pem']
        content = ''.join(original.splitlines()[1:-1])
        alias = ('-----BEGIN PUBLIC KEY-----\n' +
                 '\n'.join(content[i:i+32] for i in range(0, len(content), 32)) +
                 '\n-----END PUBLIC KEY-----\n')
        self.assertNotEqual(original, alias)
        self.assertEqual(public_key_der(original), public_key_der(alias))
        signatures = self.signatures()
        self.policy['keys'][1]['public_key_pem'] = alias
        self.policy['keys'][1]['key_identity'] = identity(alias.encode('ascii'))
        signatures[1] = dict(signatures[0], role='independent-verifier',
                             key_identity=self.policy['keys'][1]['key_identity'])
        self.policy_identity = identity(json_bytes(self.policy))
        with self.assertRaisesRegex(ContractError, 'independent key identities'):
            verify_quorum(self.packet, signatures, self.policy)
        receipt = assess(self.packet, signatures, self.policy, self.policy_identity, now=self.now)
        self.assertFalse(receipt['checks']['signature_quorum'])
        self.assertFalse(receipt['phase5_authorized'])

    def test_non_ed25519_ec_rsa_ed448_and_concatenated_pem_cannot_sign(self):
        root = Path(self.tmp.name)
        for label, algorithm, options in (
            ('ecdsa', 'EC', ['-pkeyopt', 'ec_paramgen_curve:secp224r1']),
            ('rsa', 'RSA', ['-pkeyopt', 'rsa_keygen_bits:2048']),
            ('ed448', 'ED448', [])):
            with self.subTest(algorithm=label):
                private, public = root/(label+'.key'), root/(label+'.pub')
                subprocess.run([self.openssl, 'genpkey', '-algorithm', algorithm, *options,
                                '-out', str(private)], check=True, capture_output=True)
                subprocess.run([self.openssl, 'pkey', '-in', str(private), '-pubout',
                                '-out', str(public)], check=True, capture_output=True)
                with self.assertRaisesRegex(ContractError, 'Ed25519'):
                    public_key_der(public.read_text('ascii'))
                signatures = self.signatures()
                bad = public.read_text('ascii')
                self.policy['keys'][1]['public_key_pem'] = bad
                self.policy['keys'][1]['key_identity'] = identity(bad.encode())
                receipt = assess(self.packet, signatures, self.policy,
                                 identity(json_bytes(self.policy)), now=self.now)
                self.assertFalse(receipt['checks']['signature_quorum'])
                self.assertFalse(receipt['phase5_authorized'])
        pem = self.trust[0]['public_key_pem']
        for other in (pem + pem, pem + 'ignored\n', pem.replace('\n', '\r\n')):
            with self.assertRaises(ContractError):
                public_key_der(other)

    def test_boolean_seeds_rejected_in_both_matrices_even_if_resigned(self):
        for target in ('proposal', 'request'):
            with self.subTest(target=target):
                from copy import deepcopy
                packet = deepcopy(self.packet)
                packet[target]['matrix'][0]['seed'] = False
                if target == 'proposal':
                    packet['request']['proposal_identity'] = identity(json_bytes(packet['proposal']))
                with self.assertRaisesRegex(ContractError, 'seed must be an integer'):
                    validate_packet(packet)
        validate_packet(self.packet)

    def test_foundations_stdout_exact_transport_and_strict_identity(self):
        cli = preflight_cli()
        root = Path(self.tmp.name)
        refs = self.packet['prerequisites']
        record = {'summary_identity': refs['foundations_summary_identity'],
                  'request_identity': refs['foundations_request_identity'],
                  'verified_training_cells': 27, 'outcome': 'blocked',
                  'dense_learning_adequate': False,
                  'scope': cli.FOUNDATIONS_SCOPE}
        framing = json_bytes(record) + b'\n'
        def fake_process(*args, stdout, stderr, **kwargs):
            stdout.write(framing)
            stderr.write(b'')
            stdout.flush()
            return subprocess.CompletedProcess(args[0], 0)
        with mock.patch.object(cli.subprocess, 'run', side_effect=fake_process):
            result = cli.verify_foundations(root, refs['foundations_request_identity'],
                                            refs['foundations_summary_identity'])
        self.assertEqual(result['outcome'], 'blocked')
        self.assertFalse(result['dense_learning_adequate'])
        for bad in (json_bytes(record), framing+b'\n', framing+b'{{}}',
                    json_bytes(dict(record, summary_identity=digest('wrong')))+b'\n'):
            def fake_bad(*args, stdout, stderr, **kwargs):
                stdout.write(bad)
                stdout.flush()
                return subprocess.CompletedProcess(args[0], 0)
            with mock.patch.object(cli.subprocess, 'run', side_effect=fake_bad):
                with self.assertRaises(cli.FoundationsVerifierFailure):
                    cli.verify_foundations(root, refs['foundations_request_identity'],
                                           refs['foundations_summary_identity'])

    def test_verifier_failures_keep_bounded_stdout_stderr_and_identity(self):
        cli = preflight_cli()
        root = Path(self.tmp.name)
        def fake_process(*args, stdout, stderr, **kwargs):
            stdout.write(b'X'*20000)
            stderr.write(b'foreign environment: wrong CPU lock\n')
            stdout.flush();stderr.flush()
            return subprocess.CompletedProcess(args[0], 2)
        with mock.patch.object(cli.subprocess, 'run', side_effect=fake_process):
            with self.assertRaises(cli.FoundationsVerifierFailure) as got:
                cli.verify_foundations(root, digest('request'), digest('summary'))
        info = got.exception.diagnostics
        self.assertEqual(info['returncode'], 2)
        self.assertEqual(info['stdout']['total_bytes'], 20000)
        self.assertTrue(info['stdout']['truncated'])
        self.assertEqual(base64.b64decode(info['stderr']['prefix_b64']),
                         b'foreign environment: wrong CPU lock\n')
        self.assertEqual(info['stderr']['sha256'], digest('foreign environment: wrong CPU lock\n'))

    def test_cli_rejects_receipt_inside_evidence_root_before_writing(self):
        root = Path(self.tmp.name)
        for name, obj in [('packet.json', self.packet), ('signatures.json', self.signatures()),
                          ('policy.json', self.policy)]:
            (root/name).write_bytes(json_bytes(obj))
        retained = root/'retained';retained.mkdir()
        (retained/'sentinel.txt').write_bytes(b'keep evidence immutable')
        options = ['--packet', str(root/'packet.json'), '--signatures', str(root/'signatures.json'),
                   '--trust-policy', str(root/'policy.json'),
                   '--trust-policy-identity', self.policy_identity,
                   '--foundations-run', str(retained)]
        before = {str(p.relative_to(retained)): p.read_bytes()
                  for p in retained.rglob('*') if p.is_file()}
        for output in (retained/'receipt.json', retained/'nested'/'receipt.json',
                       retained/'../retained'/'alias.json'):
            with self.subTest(path=str(output)):
                cmd = [sys.executable, str(ROOT/'scripts/check_phase5_preflight.py'),
                       *options, '--output', str(output)]
                result = subprocess.run(cmd, capture_output=True, check=False,
                                        env=os.environ.copy() | {'PYTHONPATH':str(ROOT/'src')})
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(b'overlaps retained', result.stderr)
                self.assertEqual(before,{str(p.relative_to(retained)):p.read_bytes()
                                        for p in retained.rglob('*') if p.is_file()})
        alias = root/'retained-alias'
        alias.symlink_to(retained, target_is_directory=True)
        run = subprocess.run([sys.executable,str(ROOT/'scripts/check_phase5_preflight.py'),
                              *options,'--output',str(alias/'other.json')],
                             capture_output=True,check=False,env=os.environ.copy() | {'PYTHONPATH':str(ROOT/'src')})
        self.assertEqual(run.returncode, 2)
        self.assertEqual(before,{str(p.relative_to(retained)):p.read_bytes()
                                for p in retained.rglob('*') if p.is_file()})

    def test_cli_rechecks_expiry_after_foundations_verifier(self):
        cli = preflight_cli()
        root = Path(self.tmp.name)
        for name, obj in [('packet.json', self.packet), ('signatures.json', self.signatures()),
                          ('policy.json', self.policy)]:
            (root/name).write_bytes(json_bytes(obj))
        retained = root/'retained';retained.mkdir()
        with mock.patch.object(cli,'verify_foundations',return_value=None) as verifier:
            with mock.patch.object(cli,'utc_now', side_effect=(
                    self.now, self.now+timedelta(hours=2), self.now+timedelta(hours=2))):
                with mock.patch.object(cli,'tool_source_manifest',return_value={'test':digest('test')}):
                    exit_code = cli.main(['--packet',str(root/'packet.json'),
                        '--signatures',str(root/'signatures.json'),
                        '--trust-policy',str(root/'policy.json'),
                        '--trust-policy-identity',self.policy_identity,
                        '--foundations-run',str(retained), '--output',str(root/'receipt.json')])
        self.assertEqual(exit_code,1)
        self.assertEqual(verifier.call_count,1)
        result = parse_json((root/'receipt.json').read_bytes(),canonical=True)
        self.assertFalse(result['checks']['request_window'])
        self.assertFalse(result['phase5_authorized'])



    def test_ecdsa_64_byte_signatures_labelled_ed25519_never_make_quorum(self):
        root = Path(self.tmp.name)
        (root/'packet.json').write_bytes(json_bytes(self.packet))
        signatures = []
        for index, role in enumerate(('protocol-steward','independent-verifier')):
            private = root/(role+'.ec.key')
            public = root/(role+'.ec.pub')
            subprocess.run([self.openssl,'genpkey','-algorithm','EC','-pkeyopt',
                            'ec_paramgen_curve:secp224r1','-out',str(private)],
                            check=True,capture_output=True)
            subprocess.run([self.openssl,'pkey','-in',str(private),'-pubout','-out',str(public)],
                            check=True,capture_output=True)
            signature = root/(role+'.ec.sig')
            for _ in range(100):
                subprocess.run([self.openssl,'pkeyutl','-sign','-inkey',str(private),
                                '-rawin','-in',str(root/'packet.json'),'-out',str(signature)],
                                check=True,capture_output=True)
                if signature.stat().st_size == 64:
                    break
            self.assertEqual(signature.stat().st_size,64)
            pem = public.read_text('ascii')
            self.policy['keys'][index] = {'role': role,'key_identity': identity(pem.encode()),
                                           'public_key_pem':pem}
            signatures.append({'role': role,'key_identity':identity(pem.encode()),
                               'algorithm':'ed25519',
                               'signature_b64':base64.b64encode(signature.read_bytes()).decode()})
        result = assess(self.packet,signatures,self.policy,identity(json_bytes(self.policy)),now=self.now)
        self.assertFalse(result['checks']['signature_quorum'])
        self.assertIn('Ed25519', '\n'.join(result['errors']))
        self.assertFalse(result['phase5_authorized'])

    def test_cli_retains_child_failure_diagnostics_in_blocked_receipt(self):
        cli=preflight_cli()
        root=Path(self.tmp.name)
        for name,obj in [('packet.json',self.packet),('signatures.json',self.signatures()),
                         ('policy.json',self.policy)]:
            (root/name).write_bytes(json_bytes(obj))
        retained=root/'retained';retained.mkdir()
        (retained/'sentinel').write_bytes(b'immutable')
        diagnostics={'state':'rejected','returncode':3,
                     'stdout':{'prefix_b64':'','sha256':digest(''),'total_bytes':0,'truncated':False},
                     'stderr':{'prefix_b64':base64.b64encode(b'wrong environment').decode(),
                               'sha256':digest('wrong environment'),'total_bytes':17,'truncated':False}}
        with mock.patch.object(cli,'verify_foundations',side_effect=cli.FoundationsVerifierFailure(
                                 'authoritative foundations verification rejected the run',diagnostics)):
            with mock.patch.object(cli,'tool_source_manifest',return_value={'fixture':digest('fixture')}):
                result = cli.main(['--packet',str(root/'packet.json'),
                    '--signatures',str(root/'signatures.json'),
                    '--trust-policy',str(root/'policy.json'),
                    '--trust-policy-identity',self.policy_identity,
                    '--foundations-run',str(retained),'--output',str(root/'receipt.json')])
        self.assertEqual(result,1)
        record=parse_json((root/'receipt.json').read_bytes(),canonical=True)
        self.assertEqual(record['foundations_diagnostics'],diagnostics)
        self.assertIn('authoritative foundations verification rejected', '\n'.join(record['errors']))
        self.assertFalse(record['phase5_authorized'])
        self.assertEqual((retained/'sentinel').read_bytes(),b'immutable')
        self.assertEqual(sorted(p.name for p in retained.iterdir()),['sentinel'])

    def test_objective_strength_must_be_canonical_bounded_nonzero_decimal(self):
        for value in ('0.02', '0.000001', '1'):
            with self.subTest(valid=value):
                packet = copy.deepcopy(self.packet)
                packet['proposal']['objective']['weight'] = value
                packet['request']['proposal_identity'] = identity(json_bytes(packet['proposal']))
                validate_packet(packet)
        for value in ('disabled', 'NaN', '-100', '0', '0.000000', '-0.01',
                      '1.1', '2', '00.1', '.2', '0.02e0', '0.020', '0.0000001',
                      0.02, True, None):
            with self.subTest(invalid=value):
                packet = copy.deepcopy(self.packet)
                packet['proposal']['objective']['weight'] = value
                with self.assertRaisesRegex(ContractError, 'objective weight'):
                    validate_packet(packet)

    def test_malformed_canonical_proposals_return_exit_two_without_receipt(self):
        cli = preflight_cli()
        base = Path(self.tmp.name)
        signatures = base/'signatures.json'
        policy = base/'policy.json'
        signatures.write_bytes(json_bytes(self.signatures()))
        policy.write_bytes(json_bytes(self.policy))
        invalids = []
        unexpected = copy.deepcopy(self.packet)
        unexpected['proposal']['unexpected'] = True
        invalids.append(unexpected)
        weight = copy.deepcopy(self.packet)
        weight['proposal']['objective']['weight'] = 'disabled'
        invalids.append(weight)
        missing = copy.deepcopy(self.packet)
        missing['proposal']['matrix'].pop()
        invalids.append(missing)
        wrong_types = copy.deepcopy(self.packet)
        wrong_types['proposal']['matrix'][0]['seed'] = False
        invalids.append(wrong_types)
        for idx, packet in enumerate(invalids):
            with self.subTest(case=idx):
                path = base/f'malformed-{idx}.json'
                output = base/f'bad-receipt-{idx}.json'
                path.write_bytes(json_bytes(packet))
                with mock.patch.object(cli, 'tool_source_manifest', side_effect=AssertionError('must not retain malformed proposal')):
                    returned = cli.main(['--packet', str(path), '--signatures', str(signatures),
                        '--trust-policy', str(policy), '--trust-policy-identity', self.policy_identity,
                        '--output', str(output)])
                self.assertEqual(returned, 2)
                self.assertFalse(output.exists())

    def test_poisoned_tmpdir_and_child_scratch_cannot_touch_foundations_run(self):
        cli = preflight_cli()
        base = Path(self.tmp.name)
        retained = base/'retained-foundations';retained.mkdir()
        (retained/'original.json').write_bytes(b'important preserved evidence')
        before = {p.name:p.read_bytes() for p in retained.iterdir()}
        selected = []
        def malicious_environment(*args, stdout, stderr, env, **kwargs):
            for key in ('TMPDIR', 'TMP', 'TEMP'):
                temp_root = Path(env[key])
                self.assertNotEqual(temp_root, retained)
                self.assertNotIn(retained, temp_root.parents)
                self.assertTrue(temp_root.is_dir())
            self.assertEqual(env['TMPDIR'], env['TMP'])
            self.assertEqual(env['TMPDIR'], env['TEMP'])
            selected.append(env['TMPDIR'])
            with tempfile.TemporaryDirectory(dir=env['TMPDIR']) as child:
                (Path(child)/'would-be-evidence').write_bytes(b'isolated')
            stderr.write(b'valid intentional rejection')
            stderr.flush()
            return subprocess.CompletedProcess(args[0], 1)
        with mock.patch.dict(os.environ, {'TMPDIR':str(retained),'TEMP':str(retained),'TMP':str(retained)}):
            with mock.patch.object(cli.subprocess, 'run', side_effect=malicious_environment):
                with self.assertRaises(cli.FoundationsVerifierFailure):
                    cli.verify_foundations(retained, digest('request'), digest('summary'))
        self.assertEqual(len(selected), 1)
        self.assertFalse(Path(selected[0]).exists())
        self.assertEqual(before, {p.name:p.read_bytes() for p in retained.iterdir()})

    def test_receipt_parent_descriptor_blocks_swapped_symlink_race(self):
        cli = preflight_cli()
        base = Path(self.tmp.name)
        retained = base/'immutable';retained.mkdir()
        (retained/'original').write_bytes(b'original retained bytes')
        directory = base/'untrusted';directory.mkdir()
        moved = base/'untrusted-before-swap'
        payload = json_bytes({'outcome':'blocked','phase5_authorized':False})
        original_open = os.open
        swapped = []
        def intercept_open(path, flags, *args, **kwargs):
            if path == 'receipt.json' and flags & os.O_CREAT:
                self.assertIn('dir_fd', kwargs)
                os.rename(directory, moved)
                directory.symlink_to(retained, target_is_directory=True)
                swapped.append(True)
            return original_open(path, flags, *args, **kwargs)
        with mock.patch.object(cli.os, 'open', side_effect=intercept_open):
            cli.write_receipt(directory/'receipt.json', payload)
        self.assertEqual(swapped, [True])
        self.assertEqual((moved/'receipt.json').read_bytes(), payload)
        self.assertFalse((retained/'receipt.json').exists())
        self.assertEqual((retained/'original').read_bytes(), b'original retained bytes')
        with self.assertRaises(OSError):
            cli.write_receipt(directory/'another.json', payload)

    def test_receipt_fsyncs_file_directory_and_new_ancestor_entries(self):
        cli = preflight_cli()
        base = Path(self.tmp.name)
        output = base/'nested'/'nested-again'/'receipt.json'
        payload = json_bytes({'outcome':'blocked'})
        kinds = []
        original_fsync = os.fsync
        def track_fsync(fd):
            kinds.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
            return original_fsync(fd)
        with mock.patch.object(cli.os, 'fsync', side_effect=track_fsync):
            cli.write_receipt(output, payload)
        self.assertEqual(kinds[0], 'file')
        self.assertGreaterEqual(kinds.count('directory'), 3)
        self.assertEqual(output.read_bytes(), payload)
        with self.assertRaises(FileExistsError):
            cli.write_receipt(output, payload)

    def test_preflight_manifest_discloses_separate_tool_and_retained_foundations_sources(self):
        cli=preflight_cli()
        manifest=cli.tool_source_manifest()
        for name in ('scripts/check_phase5_preflight.py',
                     'scripts/phase5_preflight/preflight.py',
                     'scripts/phase5_preflight/__init__.py',
                     'src/trinite/contracts.py',
                     'scripts/verify_foundations_run.py'):
            self.assertIn(name,manifest)
            self.assertEqual(manifest[name],identity((ROOT/name).read_bytes()))
        self.assertNotIn('src/trinite/phase5_preflight.py',manifest)



    def test_signature_temporary_workspace_ignores_poisoned_tmpdir(self):
        cli = preflight_cli()
        from phase5_preflight import preflight as verifier
        base = Path(self.tmp.name)
        retained = base/'immutable-run'; retained.mkdir()
        (retained/'original.json').write_bytes(b'preexisting evidence')
        before = {p.name:p.read_bytes() for p in retained.iterdir()}
        for name, obj in [('packet.json', self.packet),
                          ('signatures.json', self.signatures()), ('policy.json', self.policy)]:
            (base/name).write_bytes(json_bytes(obj))
        recorded = []
        original_memfd = verifier._memfd

        def observe_memfd(name, content=b'', *, seal=True):
            recorded.append((name, seal))
            return original_memfd(name, content, seal=seal)

        with mock.patch.dict(os.environ, {'TMPDIR':str(retained), 'TMP':str(retained),
                                           'TEMP':str(retained)}):
            with mock.patch.object(verifier, '_memfd', side_effect=observe_memfd):
                with mock.patch.object(cli, 'verify_foundations', return_value=None):
                    with mock.patch.object(cli,'tool_source_manifest',return_value={'test':digest('test')}):
                        result = cli.main(['--packet',str(base/'packet.json'),
                            '--signatures',str(base/'signatures.json'),
                            '--trust-policy',str(base/'policy.json'),
                            '--trust-policy-identity', self.policy_identity,
                            '--foundations-run',str(retained),
                            '--output',str(base/'blocked.json')])
        self.assertEqual(result,1)
        # Two assessments, each with a packet and 2x public key/signature
        # descriptors. No named signature scratch directory is created.
        self.assertEqual([label for label, _ in recorded].count('signed-packet'), 2)
        self.assertTrue(all(sealed for _, sealed in recorded))
        self.assertEqual(before,{p.name:p.read_bytes() for p in retained.iterdir()})
        receipt=parse_json((base/'blocked.json').read_bytes(),canonical=True)
        self.assertTrue(receipt['checks']['signature_quorum'])
        self.assertFalse(receipt['phase5_authorized'])

    def test_malformed_signer_schema_returns_two_but_bad_crypto_returns_one(self):
        cli = preflight_cli()
        base = Path(self.tmp.name)
        signatures = self.signatures()
        scenarios=[]
        wrong_policy = copy.deepcopy(self.policy);wrong_policy['extra']=True
        scenarios.append((signatures, wrong_policy))
        wrong_signature = copy.deepcopy(signatures);wrong_signature[0]['extra']='unrecognized'
        scenarios.append((wrong_signature, self.policy))
        duplicate = copy.deepcopy(signatures);duplicate[1]['role']=duplicate[0]['role']
        scenarios.append((duplicate, self.policy))
        invalid_b64 = copy.deepcopy(signatures);invalid_b64[0]['signature_b64']='broken?'
        scenarios.append((invalid_b64, self.policy))
        invalid_policy_key = copy.deepcopy(self.policy)
        invalid_policy_key['keys'][0]['public_key_pem']='not a PEM key'
        scenarios.append((signatures, invalid_policy_key))
        (base/'packet.json').write_bytes(json_bytes(self.packet))
        for index,(rows,policy) in enumerate(scenarios):
            (base/'signatures.json').write_bytes(json_bytes(rows))
            (base/'policy.json').write_bytes(json_bytes(policy))
            receipt=base/f'malformed-signer-{index}.json'
            with self.subTest(index=index):
                with mock.patch.object(cli,'tool_source_manifest',side_effect=AssertionError('no receipt for malformed signer structures')):
                    outcome=cli.main(['--packet',str(base/'packet.json'),
                        '--signatures',str(base/'signatures.json'),
                        '--trust-policy',str(base/'policy.json'),
                        '--trust-policy-identity',identity(json_bytes(policy)),
                        '--output',str(receipt)])
                self.assertEqual(outcome,2)
                self.assertFalse(receipt.exists())
        valid_shape=copy.deepcopy(signatures)
        valid_shape[0]['signature_b64']=base64.b64encode(b'!'*64).decode('ascii')
        (base/'signatures.json').write_bytes(json_bytes(valid_shape))
        (base/'policy.json').write_bytes(json_bytes(self.policy))
        with mock.patch.object(cli, 'tool_source_manifest',return_value={'test':digest('test')}):
            outcome=cli.main(['--packet',str(base/'packet.json'),
                '--signatures',str(base/'signatures.json'),
                '--trust-policy',str(base/'policy.json'),
                '--trust-policy-identity',self.policy_identity,
                '--output',str(base/'crypto-denial.json')])
        self.assertEqual(outcome,1)
        denied=parse_json((base/'crypto-denial.json').read_bytes(),canonical=True)
        self.assertFalse(denied['checks']['signature_quorum'])
        self.assertFalse(denied['phase5_authorized'])

    def test_fake_openssl_on_path_cannot_forge_signature_quorum(self):
        base = Path(self.tmp.name)
        fakebin = base/'spoof';fakebin.mkdir()
        fake = fakebin/'openssl'
        fake.write_text('#!/bin/sh\nexit 0\n')
        fake.chmod(0o755)
        forged = copy.deepcopy(self.signatures())
        forged[0]['signature_b64'] = base64.b64encode(b'Z'*64).decode('ascii')
        with mock.patch.dict(os.environ, {'PATH':str(fakebin)}):
            result=assess(self.packet, forged, self.policy, self.policy_identity, now=self.now)
        self.assertFalse(result['checks']['signature_quorum'])
        self.assertIn('cryptographic signature', '\n'.join(result['errors']))
        self.assertIsNone(result['signature_verifier'])
        with mock.patch.dict(os.environ, {'PATH':str(fakebin)}):
            actual=assess(self.packet, self.signatures(), self.policy,
                          self.policy_identity, now=self.now)
        self.assertTrue(actual['checks']['signature_quorum'])
        verifier=actual['signature_verifier']
        self.assertEqual(verifier['path'],'/usr/bin/openssl')
        self.assertEqual(verifier['identity'],identity(Path('/usr/bin/openssl').read_bytes()))
        self.assertEqual(verifier['execution'],'pinned-linux-file-descriptor.v1')

    def test_expiry_rechecked_after_last_crypto_and_before_receipt(self):
        cli=preflight_cli()
        base=Path(self.tmp.name)
        for name, value in [('packet.json',self.packet),
                            ('signatures.json',self.signatures()),('policy.json',self.policy)]:
            (base/name).write_bytes(json_bytes(value))
        completed=self.now+timedelta(hours=2)
        with mock.patch.object(cli,'utc_now',side_effect=[self.now,completed]):
            with mock.patch.object(cli,'tool_source_manifest',return_value={'test':digest('test')}):
                returned=cli.main(['--packet',str(base/'packet.json'),
                    '--signatures',str(base/'signatures.json'),
                    '--trust-policy',str(base/'policy.json'),
                    '--trust-policy-identity',self.policy_identity,
                    '--output',str(base/'receipt-post-crypto.json')])
        self.assertEqual(returned,1)
        result=parse_json((base/'receipt-post-crypto.json').read_bytes(),canonical=True)
        self.assertTrue(result['checks']['signature_quorum'])
        self.assertFalse(result['checks']['request_window'])
        self.assertEqual(result['assessment_completed_at'],utc(completed))
        self.assertIn('at final assessment completion','\n'.join(result['errors']))
        self.assertFalse(result['phase5_authorized'])


    def test_signature_verifier_uses_sealed_pinned_inputs_instead_of_paths(self):
        from phase5_preflight import preflight as verifier
        actual_run = subprocess.run
        checked = []
        expected_signatures = self.signatures()
        def inspect_crypto(*args, **kwargs):
            argv = args[0]
            if ('pkeyutl' in argv and '-verify' in argv):
                # All inputs are sealed before the invocation, so writing via
                # the original fd (or through /proc/self/fd) is impossible.
                referenced = [argv[argv.index(option) + 1] for option in
                              ('-inkey', '-in', '-sigfile')]
                self.assertEqual(len(referenced), 3)
                for pathname in referenced:
                    self.assertTrue(pathname.startswith('/proc/self/fd/'))
                    fd = int(pathname.rsplit('/', 1)[1])
                    self.assertIn(fd, kwargs['pass_fds'])
                    seals = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
                    self.assertEqual(seals & fcntl.F_SEAL_WRITE, fcntl.F_SEAL_WRITE)
                    with self.assertRaises(OSError):
                        os.write(fd, b'forged-input')
                    checked.append(pathname)
            return actual_run(*args, **kwargs)
        with mock.patch.object(verifier.subprocess, 'run', side_effect=inspect_crypto):
            receipt = assess(self.packet, expected_signatures, self.policy,
                             self.policy_identity, now=self.now)
        self.assertTrue(receipt['checks']['signature_quorum'])
        self.assertEqual(len(checked), 6)
        self.assertFalse(receipt['phase5_authorized'])

    def test_foundations_output_path_replacement_cannot_forge_positive_learning(self):
        cli = preflight_cli()
        root = Path(self.tmp.name)
        refs = self.packet['prerequisites']
        real = {'summary_identity': refs['foundations_summary_identity'],
                'request_identity': refs['foundations_request_identity'],
                'verified_training_cells': 27, 'outcome': 'blocked',
                'dense_learning_adequate': False, 'scope': cli.FOUNDATIONS_SCOPE}
        forged = dict(real, outcome='completed', dense_learning_adequate=True)
        observed = []
        def replaced_file(*args, stdout, stderr, env, **kwargs):
            stdout.write(json_bytes(real) + b'\n')
            stdout.flush()
            scratch = Path(env['TMPDIR'])
            # Replace a file with the name the previous implementation
            # reopened. The pinned anonymous descriptor must be unaffected.
            (scratch/'stdout.bin').write_bytes(json_bytes(forged) + b'\n')
            self.assertIsInstance(stdout.name, int)
            observed.append(stdout.name)
            return subprocess.CompletedProcess(args[0], 0)
        with mock.patch.object(cli.subprocess, 'run', side_effect=replaced_file):
            result = cli.verify_foundations(root, refs['foundations_request_identity'],
                                            refs['foundations_summary_identity'])
        self.assertEqual(result['outcome'], 'blocked')
        self.assertFalse(result['dense_learning_adequate'])
        self.assertEqual(len(observed), 1)

    def test_bad_external_policy_identity_is_malformed_not_scientific_block(self):
        cli = preflight_cli()
        root = Path(self.tmp.name)
        for name, content in (('packet.json', self.packet),
                              ('signatures.json', self.signatures()), ('policy.json', self.policy)):
            (root/name).write_bytes(json_bytes(content))
        for index, invalid in enumerate(('garbage', 'sha256:abcdef',
                                         'sha256:'+'A'*64, True)):
            with self.subTest(invalid=invalid):
                output = root/f'malformed-digest-{index}.json'
                rc = cli.main(['--packet', str(root/'packet.json'),
                    '--signatures',str(root/'signatures.json'),
                    '--trust-policy', str(root/'policy.json'),
                    '--trust-policy-identity', str(invalid), '--output',str(output)])
                self.assertEqual(rc, 2)
                self.assertFalse(output.exists())
        # Syntactically valid but incorrect digest is still a normal blocked
        # research assessment with a retained result, not malformed input.
        output = root/'mismatched-external-policy.json'
        rc = cli.main(['--packet', str(root/'packet.json'),
            '--signatures',str(root/'signatures.json'),
            '--trust-policy', str(root/'policy.json'),
            '--trust-policy-identity',digest('wrong-policy'), '--output',str(output)])
        self.assertEqual(rc,1)
        result = parse_json(output.read_bytes(), canonical=True)
        self.assertFalse(result['checks']['external_policy_identity'])

    def test_receipt_partial_write_failure_is_unlinked_and_retryable(self):
        cli = preflight_cli()
        root = Path(self.tmp.name)
        receipt = root/'partial'/'receipt.json'
        payload = json_bytes({'outcome':'blocked', 'scope':'write-fault'})
        actual_fdopen = os.fdopen
        triggered = []

        class HalfWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def write(self, content):
                self.stream.write(content[:max(1,len(content)//2)])
                triggered.append(True)
                raise OSError(errno.ENOSPC, 'simulated partial write')
            def flush(self):
                return self.stream.flush()
            def fileno(self):
                return self.stream.fileno()

        def faulting_fdopen(fd, mode, *args, **kwargs):
            stream = actual_fdopen(fd, mode, *args, **kwargs)
            return HalfWriter(stream) if mode == 'wb' else stream

        with mock.patch.object(cli.os,'fdopen', side_effect=faulting_fdopen):
            with self.assertRaises(OSError):
                cli.write_receipt(receipt, payload)
        self.assertEqual(triggered, [True])
        self.assertFalse(receipt.exists())
        cli.write_receipt(receipt, payload)
        self.assertEqual(receipt.read_bytes(), payload)

    def test_receipt_io_or_sync_failure_cleans_exclusive_file_for_retry(self):
        cli = preflight_cli()
        root = Path(self.tmp.name)
        receipt = root/'nested'/'receipt.json'
        content = json_bytes({'schema':'test', 'outcome':'blocked'})
        true_fsync = os.fsync
        for failed_kind in ('file', 'directory'):
            with self.subTest(failed_kind=failed_kind):
                fault = [False]
                def failing_fsync(fd):
                    is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
                    kind = 'directory' if is_dir else 'file'
                    if kind == failed_kind and not fault[0]:
                        fault[0] = True
                        raise OSError(errno.ENOSPC, 'simulated full filesystem')
                    return true_fsync(fd)
                with mock.patch.object(cli.os, 'fsync', side_effect=failing_fsync):
                    with self.assertRaises(OSError):
                        cli.write_receipt(receipt, content)
                self.assertTrue(fault[0])
                self.assertFalse(receipt.exists(), 'failed file must be unlinked')
                cli.write_receipt(receipt, content)
                self.assertEqual(receipt.read_bytes(), content)
                receipt.unlink()


if __name__ == '__main__':
    unittest.main()
