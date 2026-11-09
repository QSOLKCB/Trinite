"""OLLAMA-LOG-001 synthetic offline evidence tests; never launches Ollama."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ollama_receipt", ROOT / "scripts" / "ollama_log_receipt_001.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def private_file(path, raw):
    path.write_bytes(raw)
    path.chmod(0o600)
    return path


LOG_TEXT = (
    'time=2026-11-10T02:23:49.273+10:30 level=INFO source=routes.go:2474 '
    'msg="server config" env="map[OLLAMA_NO_CLOUD:true]"\n'
    'time=2026-11-10T02:23:49.276+10:30 level=INFO source=routes.go:2476 '
    'msg="Ollama cloud disabled: true"\n'
    'time=2026-11-10T02:23:49.281+10:30 level=INFO source=routes.go:2536 '
    'msg="Listening on 127.0.0.1:11434 (version 0.40.15)"\n'
    'time=2026-11-10T02:23:53.049+10:30 level=INFO source=types.go:32 '
    'msg="inference compute" id=0 library=CUDA '
    'description="NVIDIA GeForce RTX 5060 Ti" driver=13.2 '
    'total="15.5 GiB" available="14.6 GiB"\n'
    '[GIN] 2026/11/10 - 02:23:59 | 200 |  83.55µs | '
    '127.0.0.1 | GET "/api/version"\n'
    '[GIN] 2026/11/10 - 02:24:50 | 200 |  5.00ms | '
    '127.0.0.1 | POST "/api/chat"\n'
).encode("utf-8")


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = private_file(self.root / "ollama-test.log", LOG_TEXT)
        self.experiment = private_file(
            self.root / "manifest.json",
            mod.canonical({
                "schema": "trinite.mtc-principal-validation.manifest.v1",
                "members": {"plan.json": "sha256:" + "a" * 64}
            })
        )
        self.anchor = mod.sha(self.experiment.read_bytes())
        self.receipt = self.root / "receipt.json"

    def seal(self, confirmed=True, label="session-001"):
        mod.seal_command(SimpleNamespace(
            log=self.log, experiment_manifest=self.experiment,
            experiment_identity=self.anchor, session_label=label,
            out=self.receipt, confirm_finalized=confirmed,
        ))
        return mod.sha(self.receipt.read_bytes())

    def verify(self, anchor):
        mod.verify_command(SimpleNamespace(
            receipt=self.receipt, receipt_identity=anchor,
            log=self.log, experiment_manifest=self.experiment,
        ))

    def test_realistic_startup_log_metadata_and_reproducible_verification(self):
        anchor = self.seal()
        record = json.loads(self.receipt.read_bytes())
        self.assertEqual(record["schema"], mod.SCHEMA)
        log = record["log"]
        self.assertEqual(log["size_bytes"], len(LOG_TEXT))
        self.assertEqual(log["sha256"], mod.sha(LOG_TEXT))
        self.assertEqual(log["line_count"], 6)
        self.assertEqual(log["invalid_utf8_line_count"], 0)
        self.assertEqual(log["listen_events"], [{
            "host": "127.0.0.1:11434", "version": "0.40.15"
        }])
        self.assertEqual(log["cloud_disabled_events"], {
            "observed_true": 1, "observed_false": 0
        })
        self.assertEqual(log["gpu_discovery_events"], [{
            "id": "0", "library": "CUDA",
            "description": "NVIDIA GeForce RTX 5060 Ti",
            "driver": "13.2", "total": "15.5 GiB",
            "available": "14.6 GiB",
        }])
        interval = log["offset_timestamp_interval"]
        self.assertEqual(interval["count"], 4)
        self.assertEqual(interval["first_utc"], "2026-11-09T15:53:49.273000+00:00")
        self.assertEqual(interval["last_utc"], "2026-11-09T15:53:53.049000+00:00")
        gin = log["unzoneds_access_interval"]
        self.assertEqual(gin["count"], 2)
        self.assertEqual(gin["last_literal"], "2026/11/10 - 02:24:50")
        self.assertEqual(log["http_observations"], {
            "gin_requests": 2, "gin_chat_requests": 1
        })
        self.assertEqual(record["experiment_manifest"]["sha256"], self.anchor)
        self.assertEqual(record["experiment_manifest"]["schema_observed"],
                         "trinite.mtc-principal-validation.manifest.v1")
        self.assertFalse(record["experiment_manifest"]["experiment_verified_by_this_tool"])
        self.assertFalse(record["operator_association_is_execution_evidence"])
        self.assertFalse(record["training_admission"])
        self.assertFalse(record["finalization"]["machine_attested"])
        self.assertNotIn("server config", self.receipt.read_text())
        self.verify(anchor)
        self.assertEqual(self.receipt.stat().st_mode & 0o077, 0)
        self.assertEqual(self.experiment.read_bytes(), mod.canonical({
            "schema": "trinite.mtc-principal-validation.manifest.v1",
            "members": {"plan.json": "sha256:" + "a" * 64}
        }))

    def test_declines_to_seal_without_finalization_declaration(self):
        with self.assertRaisesRegex(ValueError, "confirmation"):
            self.seal(confirmed=False)
        self.assertFalse(self.receipt.exists())

    def test_fails_on_wrong_external_experiment_manifest_digest(self):
        with self.assertRaisesRegex(ValueError, "external identity mismatch"):
            mod.seal_command(SimpleNamespace(
                log=self.log, experiment_manifest=self.experiment,
                experiment_identity="sha256:" + "0"*64,
                session_label="session-001", out=self.receipt,
                confirm_finalized=True,
            ))
        self.assertFalse(self.receipt.exists())

    def test_log_bytes_mutation_breaks_receipt_verification(self):
        digest = self.seal()
        with self.log.open("ab") as stream:
            stream.write(b"appended event\n")
        with self.assertRaisesRegex(ValueError, "differs from receipt"):
            self.verify(digest)

    def test_experiment_manifest_mutation_breaks_verification(self):
        digest = self.seal()
        private_file(self.experiment, mod.canonical({
            "schema": "trinite.modified.manifest.v1",
            "members": {"plan.json": "sha256:" + "b"*64}
        }))
        with self.assertRaisesRegex(ValueError, "external identity mismatch"):
            self.verify(digest)

    def test_receipt_tampering_rejected_with_external_anchor(self):
        digest = self.seal()
        private_file(self.receipt, self.receipt.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "external receipt identity mismatch"):
            self.verify(digest)

    def test_closed_log_must_have_complete_last_line(self):
        private_file(self.log, LOG_TEXT[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete or concurrently changing"):
            self.seal()
        self.assertFalse(self.receipt.exists())

    def test_oversize_line_refused_before_receipt_created(self):
        private_file(self.log, b"x" * (mod.MAX_LINE + 1) + b"\n")
        with self.assertRaisesRegex(ValueError, "oversized"):
            self.seal()
        self.assertFalse(self.receipt.exists())

    def test_symlink_and_hardlink_inputs_rejected(self):
        link = self.root / "alias.log"
        link.symlink_to(self.log)
        original = self.log
        self.log = link
        with self.assertRaises((ValueError, OSError)):
            self.seal()
        self.log = original
        other = self.root / "hardlink.log"
        os.link(self.log, other)
        with self.assertRaisesRegex(ValueError, "single-link"):
            self.seal()
        self.assertFalse(self.receipt.exists())

    def test_fails_closed_on_group_readable_log(self):
        self.log.chmod(0o640)
        with self.assertRaisesRegex(ValueError, "private single-link"):
            self.seal()
        self.assertFalse(self.receipt.exists())

    def test_startup_failure_log_does_not_invent_gpu_version_or_session(self):
        private_file(self.log, b"Error: listen tcp 127.0.0.1:11434: bind: address already in use\n")
        digest = self.seal()
        record = json.loads(self.receipt.read_bytes())
        self.assertEqual(record["log"]["listen_events"], [])
        self.assertEqual(record["log"]["gpu_discovery_events"], [])
        self.assertEqual(record["log"]["offset_timestamp_interval"]["count"], 0)
        self.assertEqual(record["log"]["cloud_disabled_events"]["observed_true"], 0)
        self.verify(digest)

    def test_invalid_utf8_is_accounted_for_without_discards(self):
        private_file(self.log, LOG_TEXT + b"\xff\xfe unknown binary\n")
        digest = self.seal()
        record = json.loads(self.receipt.read_bytes())
        self.assertEqual(record["log"]["invalid_utf8_line_count"], 1)
        self.assertEqual(record["log"]["sha256"], mod.sha(self.log.read_bytes()))
        self.verify(digest)

    def test_source_closure_drift_detected_even_if_receipt_untampered(self):
        digest = self.seal()
        with patch.object(mod, "source_closure", return_value={"drift": "true"}):
            with self.assertRaisesRegex(ValueError, "differs from receipt"):
                self.verify(digest)


    def test_large_repeated_startup_events_seal_and_verify_consistently(self):
        # Reproduces the P2 review: 25,000 event lines generated a >1 MiB
        # receipt, which seal once accepted but verify rejected.
        repeated = (
            'time=2026-11-09T00:00:00Z '
            'msg="Listening on 127.0.0.1:11434 (version 0.40.15)"\n'
        ).encode("ascii") * 25_000
        self.assertLess(len(repeated), mod.MAX_LOG)
        private_file(self.log, repeated)
        digest = self.seal(label="repeated-listening-events")
        receipt_bytes = self.receipt.read_bytes()
        self.assertGreater(len(receipt_bytes), mod.MAX_MANIFEST)
        self.assertLessEqual(len(receipt_bytes), mod.MAX_RECEIPT)
        record = json.loads(receipt_bytes)
        self.assertEqual(record["log"]["line_count"], 25_000)
        self.assertEqual(len(record["log"]["listen_events"]), 25_000)
        self.verify(digest)

    def test_seal_rejects_receipt_above_shared_limit_before_creating_file(self):
        # Lower the shared bound inside this test to make the case quick.
        # The same 25,000-line repro is bigger than 1 MiB as a receipt.
        repeated = (
            'time=2026-11-09T00:00:00Z '
            'msg="Listening on 127.0.0.1:11434 (version 0.40.15)"\n'
        ).encode("ascii") * 25_000
        private_file(self.log, repeated)
        with patch.object(mod, "MAX_RECEIPT", mod.MAX_MANIFEST):
            with self.assertRaisesRegex(ValueError, "receipt exceeds limit"):
                self.seal()
        self.assertFalse(self.receipt.exists())

    def test_verify_uses_same_receipt_limit_as_seal(self):
        digest = self.seal()
        actual_size = self.receipt.stat().st_size
        with patch.object(mod, "MAX_RECEIPT", actual_size - 1):
            with self.assertRaisesRegex(ValueError, "receipt exceeds limit"):
                self.verify(digest)
        self.verify(digest)

    def test_preexisting_receipt_not_overwritten(self):
        digest = self.seal()
        raw = self.receipt.read_bytes()
        with self.assertRaises(FileExistsError):
            self.seal()
        self.assertEqual(self.receipt.read_bytes(), raw)
        self.verify(digest)

    def test_bad_manifest_envelope_is_rejected(self):
        private_file(self.experiment, b'{"schema":"not-trinite","members":{}}\n')
        self.anchor = mod.sha(self.experiment.read_bytes())
        with self.assertRaisesRegex(ValueError, "unsupported experiment manifest"):
            self.seal()

    def test_wrong_receipt_anchor_fails_closed(self):
        self.seal()
        with self.assertRaisesRegex(ValueError, "external receipt identity mismatch"):
            self.verify("sha256:" + "a" * 64)


if __name__ == "__main__":
    unittest.main()
