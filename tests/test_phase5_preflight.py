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

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.phase5_preflight import assess, validate_packet, verify_quorum

ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == '__main__':
    unittest.main()
