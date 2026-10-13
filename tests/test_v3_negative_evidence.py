"""Immutable public v3 record checks, not numerical replay or new admission."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from trinite.contracts import identity, json_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/procedural-qec-v3-negative-20261013"
# Independent commitments retained before public reporting. Do not regenerate
# these from a modified catalogue or use tests to reseal historical evidence.
ORIGINALS = {'FINAL-SHA256SUMS.txt': (322, '80dbac149fae06dea04dd857db2922c92ddefc17624b842e03af4583135af31a'),
 'archive-receipt.json': (806, 'dbdd539b15026f69b42bd44bd32ee2369fc848b0e1806373676a19c950d678ae'),
 'final-byte-parity.json': (314,
                            '7bf06248bebd4039b2c5e41ffcf0b4886c72e62e91cfa6fbb092bf273ee1b531'),
 'historical-control-audit.json': (3896,
                                   '81149b4351d2d8142a088f34fc0b5027b75be591a0ff1758bf591237a5b2bf5c'),
 'independent-historical-control-comparison.json': (2580,
                                                    '7e0160ac9ca5c6241dd8001bbc8775c6dd1c51266755533705debd819d9a087b'),
 'read-only-byte-parity.json': (138,
                                '582921dacd0de86a84ade8d4aa415c9f7d010b0495c072c8e841c8560bfce244'),
 'read-only-replay.json': (300, 'c69b61ebfc9d17abf2d2e1be6022cc02c05fe8856e851c9d9169526d6df6c51e'),
 'request.json': (23432, '7ee0b351cf737807cf888a66494fd1438714698f2c1b9834ea2073c0ee5c66ff'),
 'source-reconstruction-verification.json': (570,
                                             'a91d725445c8919c175422e5061b55afad2ee30714d97fd7331fcfc611fb49ef'),
 'summary.json': (152424, 'c3ca81a4ef2a8aff90ce01ddd4984f4ce555d99eb559f5a2650ca9a6aaa3352b'),
 'transport-verification.json': (667,
                                 '34cfad1d4eefa169134a0ddcb1de864791376f3a082896737a67d73a0b585db3'),
 'verification-request.json': (3090,
                               '6a52d8bcc33dc1c6a85624126e0001525a768f9deb24c8715bf46e9dfe8155b9')}


def require_originals(payloads):
    if set(payloads) != set(ORIGINALS):
        raise ValueError("original record inventory differs")
    for name, raw in payloads.items():
        size, digest = ORIGINALS[name]
        if (type(raw) is not bytes or len(raw) != size
                or hashlib.sha256(raw).hexdigest() != digest):
            raise ValueError("original record bytes differ: " + name)


class V3NegativeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads((FIXTURE / "evidence-index.json").read_bytes())
        cls.payloads = {name: (FIXTURE / name).read_bytes() for name in ORIGINALS}
        require_originals(cls.payloads)
        cls.request = json.loads(cls.payloads["request.json"])
        cls.summary = json.loads(cls.payloads["summary.json"])

    def test_exact_original_bytes_and_catalogue_inventory(self):
        self.assertEqual({p.name for p in FIXTURE.iterdir()}, set(ORIGINALS) | {"evidence-index.json"})
        self.assertEqual(set(self.index["retained_originals"]), set(ORIGINALS))
        for name, (size, digest) in ORIGINALS.items():
            self.assertEqual(self.index["retained_originals"][name],
                             {"bytes": size, "sha256": digest, "unchanged_copy": True})
        for name, field in (("request.json", "request_identity"),
                            ("verification-request.json", "verification_request_identity"),
                            ("summary.json", "summary_identity")):
            self.assertEqual(self.index[field], identity(self.payloads[name]))

    def test_all_four_original_reports_and_negative_gates_remain_bound(self):
        summary, request = self.summary, self.request
        self.assertEqual({(c["seed"], c["arm"]) for c in summary["cells"]},
                         {(s, a) for s in (0, 1) for a in ("final-only", "procedural")})
        self.assertEqual(len(summary["cells"]), 4)
        self.assertEqual(summary["outcome"], "completed")
        self.assertEqual(summary["pairing_errors"], [])
        self.assertEqual(summary["worker_errors"], {})
        for field in ("training_gate_pass", "learning_adequacy", "phase5_ready",
                      "format_selection_eligible", "held_out_scored", "release_ready"):
            self.assertIs(summary[field], False)
            self.assertIs(self.index["gates"][field], False)
        self.assertIsNone(summary["selected_arm"])
        for cell in summary["cells"]:
            report = cell["report"]
            self.assertEqual(identity(json_bytes(report)), cell["report_identity"])
            self.assertEqual(cell["outcome"], "completed")
            self.assertEqual(len(cell["checks"]), 6)
            self.assertTrue(all(value is True for value in cell["checks"].values()))
            self.assertEqual(report["steps"], 4096)
            self.assertEqual(report["request_identity"], identity(self.payloads["request.json"]))
            for field in ("dataset_identity", "manifest_identity", "protocol_identity", "source", "environment"):
                self.assertEqual(report[field], request[field])
            self.assertIn("fold-sum", report["training_task_failures"])
        self.assertEqual(len(request["source"]), 156)
        self.assertEqual(request["environment"]["threads"], 1)
        self.assertEqual(request["environment"]["interop_threads"], 9)

    def test_independent_ordering_failure_receipt_is_the_same_original(self):
        original = self.payloads["independent-historical-control-comparison.json"]
        self.assertEqual(original, (ROOT / "docs/evidence/procedural-qec-control-order/comparison.json").read_bytes())
        comparison = json.loads(original)
        self.assertEqual(comparison["dataset_rows"], 3876)
        self.assertEqual(comparison["changed_example_identity_count"], 3876)
        self.assertIs(comparison["only_example_and_generator_id_differ"], True)
        final = [c for c in comparison["cells"] if c["arm"] == "final-only"]
        self.assertEqual([c["numeric_visit_differences"] for c in final], [32576, 32521])
        self.assertTrue(all(c["current_v2_function_matches_current_v3"] is True for c in final))
        self.assertIs(self.index["historical_v2"]["historical_control_preserved"], False)
        self.assertIs(self.index["historical_v2"]["causal_cursor_only_comparison_eligible"], False)

    def test_altered_record_cannot_be_blessed_by_resealed_catalogue(self):
        for name in ORIGINALS:
            with self.subTest(name=name):
                changed = dict(self.payloads)
                changed[name] = changed[name] + b" "
                forged = copy.deepcopy(self.index)
                forged["retained_originals"][name].update(
                    bytes=len(changed[name]), sha256=hashlib.sha256(changed[name]).hexdigest())
                self.assertNotEqual(forged["retained_originals"][name], self.index["retained_originals"][name])
                with self.assertRaises(ValueError):
                    require_originals(changed)

    def test_missing_extra_and_mutable_originals_rejected(self):
        name = next(iter(ORIGINALS))
        missing = dict(self.payloads); missing.pop(name)
        extra = {**self.payloads, "unexpected.json": b"{}"}
        mutable = {**self.payloads, name: bytearray(self.payloads[name])}
        for altered in (missing, extra, mutable):
            with self.assertRaises(ValueError):
                require_originals(altered)
