import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billboard.analytics.zone_events import TrackPoint, analyze_tracks
from billboard.calibration.homography import RoadCalibration
from billboard.config import Site


def point(track_id, t, x, y, inside, world=None):
    return TrackPoint(track_id, round(t * 10), t, "car", .9, x, y, inside, *(world or (None, None)))


class RoadMeasurements(unittest.TestCase):
    def setUp(self):
        self.site = Site.from_mapping({
            "site_id": "unit_test", "scenario": "test", "camera": "fixed", "zone_kind": "traffic_only",
            "zone_basis": "Analytic test geometry", "frame_size": [1200, 1000],
            "zone": [[300, 450], [900, 450], [900, 800], [300, 800]],
            "counting_line": [[300, 600], [900, 600]], "crossing_direction": "negative_to_positive",
        })

    def test_line_count_uses_direction_and_actual_segment(self):
        samples = [point(1, 0, 500, 550, True), point(1, .1, 500, 650, True),
                   point(1, .2, 500, 550, True), point(1, .3, 500, 650, True),
                   point(2, 0, 500, 650, True), point(2, .1, 500, 550, True),
                   point(3, 0, 1050, 550, False), point(3, .1, 1050, 650, False)]
        events, metrics = analyze_tracks(samples, self.site, .5, .4)
        self.assertEqual(metrics["vehicle_passages"], 1)
        crossing, = [event for event in events if event["kind"] == "line_crossing"]
        self.assertEqual(crossing["track_id"], 1)
        self.assertTrue(0 < crossing["start_s"] < .1)

    def test_interrupted_visits_do_not_extend_dwell(self):
        samples = [point(1, 0, 500, 400, False), point(1, .1, 500, 500, True),
                   point(1, .2, 500, 650, True), point(1, .3, 500, 850, False),
                   point(2, 0, 500, 500, True), point(2, 1.5, 500, 650, True)]
        _, metrics = analyze_tracks(samples, self.site, .5, .4)
        self.assertEqual(metrics["complete_zone_visits"], 1)
        self.assertEqual(metrics["partial_zone_visits"], 2)
        self.assertAlmostEqual(metrics["median_complete_zone_time_s"], .2)
        self.assertEqual(metrics["vehicle_passages"], 1)
        self.assertIsNone(metrics["median_complete_visit_speed_kmh"])

    def test_calibrated_distance_is_used_only_inside_calibration_region(self):
        calibration = RoadCalibration({"source": "Analytic test geometry", "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                                       "world_points_m": [[0, 0], [10, 0], [10, 10], [0, 10]]})
        self.assertIsNone(calibration.pixel_to_world((110, 50)))
        left, right = calibration.pixel_to_world((10, 10)), calibration.pixel_to_world((30, 10))
        samples = [point(1, 0, 500, 400, False), point(1, .25, 500, 500, True, left),
                   point(1, .75, 500, 650, True, right), point(1, 1.0, 500, 850, False)]
        _, metrics = analyze_tracks(samples, self.site, .5, .5)
        self.assertAlmostEqual(metrics["median_complete_visit_speed_kmh"], 14.4)


if __name__ == "__main__":
    unittest.main()
