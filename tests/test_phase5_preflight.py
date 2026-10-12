"""Adversarial and positive *signature* conformance; no empirical gate pass."""
import base64
import os
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



if __name__ == '__main__':
    unittest.main()
