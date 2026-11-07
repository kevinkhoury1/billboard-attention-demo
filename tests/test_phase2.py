import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billboard.metrics.cost_metrics import build_cost_metrics
from billboard.metrics.funnel import build_funnel


class MarketingEvidence(unittest.TestCase):
    def test_viewability_requires_crossing_during_a_visit(self):
        run = {"site_id": "unit_test", "zone_kind": "billboard_visibility", "vehicle_passages": 2,
               "median_complete_zone_time_s": 1.0, "median_complete_visit_speed_kmh": None,
               "measurement": {"observation_s": 10.0}}
        events = [{"kind": "line_crossing", "vehicle_class": "car", "track_id": 1, "start_s": 1.5},
                  {"kind": "line_crossing", "vehicle_class": "car", "track_id": 2, "start_s": 4.0},
                  {"kind": "zone_visit", "track_id": 1, "start_s": 1.0, "end_s": 2.0,
                   "entry_observed": True, "exit_observed": True}]
        evidence = {"occupancy": {"car": {"mean": 1.5, "low": 1.25, "high": 1.75, "source": "Arithmetic test"}},
                    "pedestrians": {"site_id": "unit_test", "end_s": 10.0, "passing": 3,
                                    "viewable": 2, "source": "Arithmetic test"}}
        report = build_funnel(run, events, evidence)
        self.assertEqual(report["vehicle_passages"], 2)
        self.assertEqual(report["viewable_vehicle_passages"], 1)
        self.assertEqual(report["estimated_ots_all_counted_modes"], 3.5)
        self.assertEqual(report["occupancy_sensitivity"]["high"], 3.75)
        traffic_only = build_funnel({**run, "zone_kind": "traffic_only"}, events, {"occupancy": evidence["occupancy"]})
        self.assertIsNone(traffic_only["estimated_ots_vehicle_occupants"])

    def test_cost_and_audience_must_match_booking_scope(self):
        base = {"display_id": "unit_test", "start_date": "2026-11-01", "end_date": "2026-12-01",
                "scope": "one static face", "source": "Arithmetic test"}
        cost = {**base, "kind": "quote", "amount": 500, "currency": "USD"}
        claim = {**base, "kind": "owner_claim", "metric": "ots", "value": 10000}
        independent = {**base, "kind": "independent", "metric": "ots", "value": 8000}
        attentive = {**base, "kind": "independent", "metric": "attentive_passages", "value": 2000}
        report = build_cost_metrics("unit_test", {"cost": cost, "audiences": [claim, independent, attentive]})
        self.assertEqual(report["owner_cpm_by_claim_type"]["ots"], 50)
        self.assertEqual(report["independent_ots_cpm"], 62.5)
        self.assertEqual(report["independent_to_claimed_ots_ratio"], .8)
        self.assertEqual(report["attentive_cpm"], 250)
        for change in ({"end_date": "2026-11-15"}, {"display_id": "other"}, {"scope": "both faces"}, {"value": 0}):
            with self.subTest(change=change):
                mismatch = build_cost_metrics("unit_test", {"cost": cost, "audiences": [{**attentive, **change}]})
                self.assertIsNone(mismatch["attentive_cpm"])


if __name__ == "__main__":
    unittest.main()
