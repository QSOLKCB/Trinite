"""Pinned upstream mapping and closed retained-evidence rejection; stdlib only."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from provenance_core import EventCore, EventEnvelope, EvidenceClass, canonical_json_bytes
from provenance_verify import verify_bundle
from trinite.contracts import ContractError, identity
from trinite.observation import BundleObserver, verify_observation, verify_pin

ROOT = Path(__file__).resolve().parents[1]


class ObservationTests(unittest.TestCase):
    def bundle(self, root):
        collector = BundleObserver(root)
        event = collector.record('fixture-transform', inputs={'source': b'1+1'}, outputs={'result': b'2'})
        report = collector.finalize()
        self.assertTrue(report['integrity_verified'])
        self.assertEqual(report['manifest_scope'], 'closed')
        self.assertEqual(report['known_missing_artifacts'], 0)
        return collector, event

    def test_acquisition_receipt_binds_all_exact_upstream_blobs(self):
        pin = verify_pin(ROOT)
        self.assertEqual(len(pin['checked_files']), 24)
        self.assertEqual(len(verify_pin()['checked_files']), 21)
        with patch('trinite.observation.PIN_IDENTITY', identity(b'unsupported receipt')):
            with self.assertRaises(ContractError): verify_pin(ROOT)

    def test_actual_upstream_canonical_events_and_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'bundle'; collector, event = self.bundle(root)
            expected = EventEnvelope.seal(EventCore(EvidenceClass.DERIVED, 'trinite.cpu-runner',
                                       'fixture-transform', inputs=(identity(b'1+1'),), outputs=(identity(b'2'),)))
            self.assertEqual(event, expected.event_identity)
            self.assertEqual((root/'events/sha256'/f'{event[7:]}.json').read_bytes(),
                             canonical_json_bytes(expected.to_dict()))
            self.assertEqual(verify_observation(root), verify_bundle(root).to_dict())
            with self.assertRaises(ContractError): collector.retain('later', b'3')
            with self.assertRaises(ContractError): collector.finalize()

    def test_tampering_missing_retained_content_and_extra_members_rejected(self):
        for corruption in ('tamper', 'missing', 'extra', 'schema'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/'bundle'; self.bundle(root)
                member = root/'artifacts/sha256'/identity(b'2')[7:]
                if corruption == 'tamper': member.write_bytes(b'3')
                elif corruption == 'missing': member.unlink()
                elif corruption == 'extra': (root/'unlisted.json').write_bytes(b'{}\n')
                else:
                    manifest = root/'manifest.json'
                    manifest.write_bytes(manifest.read_bytes().replace(b'provenance.manifest.v1', b'provenance.manifest.v9'))
                with self.assertRaises(ContractError): verify_observation(root)

    def test_unresolved_derivation_is_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'bundle'; collector = BundleObserver(root)
            envelope = EventEnvelope.seal(EventCore(EvidenceClass.DERIVED, 'trinite.cpu-runner', 'bad-reference',
                                                    inputs=(identity(b'unretained input'),)))
            collector.events[envelope.event_identity] = envelope
            collector._json(root/'events/sha256'/f'{envelope.event_identity[7:]}.json', envelope.to_dict())
            report = collector.finalize()
            self.assertFalse(report['integrity_verified'])
            self.assertTrue(report['errors'])
            with self.assertRaises(ContractError): verify_observation(root)

    def test_derivation_requires_source_and_names_cannot_rebind(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = BundleObserver(Path(directory)/'bundle')
            with self.assertRaises(ContractError):
                collector.record('invalid', inputs={}, outputs={'out': b'2'})
            with self.assertRaises(ContractError): collector.retain('out', b'3')
            with self.assertRaises(ContractError):
                collector.record('bad-enum', inputs={'source': b'1'}, outputs={}, evidence_class='LOCAL')


if __name__ == '__main__': unittest.main()
