"""Named exact reuse invariants; unchanged upstream serializer/verifier oracle."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError
from trinite.observation import BundleObserver, MAX_BUNDLE_BYTES

ROOT = Path(__file__).resolve().parents[1]


class CollectorReuseTests(unittest.TestCase):
    def test_obs_reuse_001_exact_immutable_bytes_skip_factory_only_for_same_name(self):
        with tempfile.TemporaryDirectory() as directory:
            c=BundleObserver(Path(directory)/'bundle')
            original=b'x'*8192; equal=bytes(bytearray(original))
            self.assertIsNot(original,equal)
            with patch.object(c.ArtifactRecord,'from_bytes',wraps=c.ArtifactRecord.from_bytes) as factory:
                key=c.retain('source',original)
                self.assertEqual(c.retain('source',original),key)
                self.assertEqual(c.retain('source',equal),key)
                self.assertEqual(factory.call_count,1)
                self.assertEqual(c.retain('alias',equal),key)
                self.assertEqual(factory.call_count,2)
                with self.assertRaises(ContractError): c.retain('source',original[:-1]+b'y')
                with self.assertRaises(ContractError): c.retain('source',bytearray(original))
            self.assertEqual(len(c._content),1)
            self.assertIs(c._content[key],original)
            self.assertEqual(c.total_bytes,8192)
            self.assertLessEqual(sum(map(len,c._content.values())),MAX_BUNDLE_BYTES)
            self.assertTrue(c.finalize()['integrity_verified'])
            self.assertEqual(c._content,{})
            with self.assertRaises(ContractError): c.retain('source',original)

    def test_reuse_never_masks_retained_file_tampering_before_final_verification(self):
        for corruption in ('tamper','delete'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                c=BundleObserver(Path(directory)/'bundle');content=b'checked bytes';key=c.retain('source',content)
                path=c.path/'artifacts/sha256'/key[7:]
                if corruption=='tamper':path.write_bytes(b'changed bytes')
                else:path.unlink()
                c.record('derive',inputs={'source':content},outputs={'out':b'answer'})
                r=c.finalize()
                self.assertFalse(r['integrity_verified'])
                self.assertTrue(r['errors'])

    def test_reuse_state_is_collector_local_and_failed_publication_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            a=BundleObserver(Path(directory)/'a');b=BundleObserver(Path(directory)/'b')
            with patch.object(a,'_json',side_effect=OSError('injected publication failure')):
                with self.assertRaises(OSError): a.retain('x',b'x')
            self.assertEqual(a._content,{})
            self.assertEqual(a.names,{})
            key=a.retain('x',b'x')
            self.assertEqual(b._content,{})
            self.assertEqual(b.retain('x',b'y'),next(iter(b.artifacts)))
            self.assertNotIn(key,b.artifacts)


if __name__=='__main__':unittest.main()
