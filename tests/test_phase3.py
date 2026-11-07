import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from billboard.attention.attention_rule import Observation, SceneGeometry, summarize_attention


class AttentionCoverage(unittest.TestCase):
    def test_uncalibrated_site_reports_coverage_without_an_attention_rate(self):
        config = {"max_gap_s": .3, "min_assessable_s": .4, "min_facing_s": .4, "cone_degrees": 20}
        samples = {1: [Observation(t, True, None) for t in (0, .2, .4)]}
        result = summarize_attention(samples, config, calibrated=False)
        self.assertEqual(result["pedestrian_tracks_with_sustained_usable_pose"], 1)
        self.assertIsNone(result["billboard_facing_rate_among_assessable"])
        self.assertEqual(result["unassessable_pedestrian_tracks"], 1)

    def test_brief_or_interrupted_glances_do_not_count_as_sustained(self):
        config = {"max_gap_s": .3, "min_assessable_s": .4, "min_facing_s": .4, "cone_degrees": 20}
        samples = {1: [Observation(t, True, 10) for t in (0, .2, .4)],
                   2: [Observation(t, True, angle) for t, angle in ((0, 10), (.2, 40), (.4, 10))],
                   3: [Observation(t, True, 10) for t in (0, .2, .7, .9)]}
        result = summarize_attention(samples, config, calibrated=True)
        self.assertEqual(result["assessable_pedestrian_tracks"], 2)
        self.assertEqual(result["billboard_facing_rate_among_assessable"], .5)
        self.assertAlmostEqual(result["observed_sustained_facing_seconds"], .4)

    def test_screen_direction_is_measured_from_head_not_camera(self):
        setup = {"source": "Analytic geometry test", "target_camera_m": [1, 0, 0],
                 "fixed_head_camera_m": [0, 0, 2], "pose_axis_signs": [1, 1]}
        result = SceneGeometry(setup).angle((0, 0), np.eye(3))
        self.assertAlmostEqual(result, 26.565051, places=5)


if __name__ == "__main__":
    unittest.main()
