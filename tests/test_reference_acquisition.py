"""Acquisition lifecycle simulations; no network or pretrained downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('acquisition',ROOT/'scripts/acquire_reference_model.py')
acquisition=importlib.util.module_from_spec(spec);spec.loader.exec_module(acquisition)


class AcquisitionTests(unittest.TestCase):
    def test_stale_partial_restarts_and_only_verified_bytes_are_promoted(self):
        raw=b'complete model asset';entry={'path':'asset.bin','bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest(),'git_blob':None}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pin=root/'src/trinite/reference-model-pin.json';pin.parent.mkdir(parents=True)
            pin.write_text(json.dumps({'model':'simulation','revision':'fixed','files':[entry]}))
            cache=root/'cache';cache.mkdir();partial=cache/'asset.bin.partial';partial.write_bytes(b'interrupted')
            with patch.object(acquisition,'ROOT',root),patch.object(acquisition.urllib.request,'urlopen',return_value=io.BytesIO(raw)) as fetch:
                acquisition.acquire(cache);self.assertEqual(fetch.call_count,1)
            self.assertEqual((cache/'asset.bin').read_bytes(),raw);self.assertFalse(partial.exists())
            (cache/'asset.bin').unlink();partial.write_bytes(b'interrupted again')
            with patch.object(acquisition,'ROOT',root),patch.object(acquisition.urllib.request,'urlopen',return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):acquisition.acquire(cache)
            self.assertFalse((cache/'asset.bin').exists())

    def test_partial_symlink_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pin=root/'src/trinite/reference-model-pin.json';pin.parent.mkdir(parents=True)
            pin.write_text(json.dumps({'model':'simulation','revision':'fixed','files':[{'path':'asset.bin'}]}))
            cache=root/'cache';cache.mkdir();other=root/'unrelated';other.write_bytes(b'keep')
            (cache/'asset.bin.partial').symlink_to(other)
            with patch.object(acquisition,'ROOT',root),patch.object(acquisition.urllib.request,'urlopen') as fetch:
                with self.assertRaises(ValueError):acquisition.acquire(cache)
                fetch.assert_not_called()
            self.assertEqual(other.read_bytes(),b'keep')


if __name__=='__main__':unittest.main()
