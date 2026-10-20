"""Post-publication regression retained outside the frozen v1 source closure."""
import copy
import unittest

from trinite.contracts import ContractError, json_bytes, parse_json
from batch_composition_v1 import spec, validation
from test_validation_review import synthetic_context


class CanonicalRoundTripRegression(unittest.TestCase):
    def test_frozen_v1_rejects_its_serialized_report_without_order_restoration(self):
        with synthetic_context() as (report, binding):
            raw = json_bytes(report)
            loaded = parse_json(raw, canonical=True)
            with self.assertRaisesRegex(ContractError, "recomputed comparisons mismatch"):
                validation.validate_report(loaded, binding)
            restored = copy.deepcopy(loaded)
            for cell in restored["cells"]:
                cell["layouts"] = {name: cell["layouts"][name] for name in spec.LAYOUTS}
            self.assertEqual(json_bytes(restored), raw)
            self.assertTrue(validation.validate_report(restored, binding)["conforms"])
            self.assertEqual(json_bytes(loaded), raw)


if __name__ == "__main__":
    unittest.main()
