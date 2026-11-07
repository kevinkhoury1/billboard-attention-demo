from dataclasses import dataclass

import numpy as np


class SceneGeometry:
    """Locate the head at a measured mark or intersect its ray with a reference plane."""

    def __init__(self, data: dict):
        self.source = data["source"].strip()
        def vector(field, length):
            value = np.asarray(data[field], dtype=float)
            if value.shape != (length,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {field}")
            return value

        self.target = vector("target_camera_m", 3)
        self.signs = vector("pose_axis_signs", 2)
        if not self.source or not np.isin(self.signs, [-1, 1]).all():
            raise ValueError("Scene calibration needs measured geometry and verified pose axis signs")
        self.fixed_head = vector("fixed_head_camera_m", 3) if "fixed_head_camera_m" in data else None
        if self.fixed_head is not None:
            if self.fixed_head[2] <= 0 or np.linalg.norm(self.target - self.fixed_head) < 1e-6:
                raise ValueError("The measured head mark must be in front of the camera and separate from the target")
        else:
            self.intrinsics = vector("intrinsics_px", 4)
            self.normal = vector("head_plane_normal", 3)
            self.depth = vector("depth_range_m", 2)
            self.offset = float(data["head_plane_offset_m"])
            if (not np.isfinite(self.offset) or np.linalg.norm(self.normal) < 1e-6 or abs(self.offset) < 1e-6
                    or min(self.intrinsics[:2]) <= 0 or not 0 < self.depth[0] < self.depth[1]):
                raise ValueError("Invalid camera intrinsics, head-height plane or depth range")

    def angle(self, center: tuple[float, float], rotation: np.ndarray) -> float | None:
        head = self.fixed_head
        if head is None:
            fx, fy, cx, cy = self.intrinsics
            ray = np.array([(center[0] - cx) / fx, (center[1] - cy) / fy, 1.0])
            projection = self.normal @ ray
            if abs(projection) < 1e-6:
                return None
            depth = self.offset / projection
            if not self.depth[0] <= depth <= self.depth[1]:
                return None
            head = ray * depth
        to_screen = self.target - head
        forward = -rotation[:, 2] * np.array([*self.signs, 1.0])
        product = np.linalg.norm(forward) * np.linalg.norm(to_screen)
        if product < 1e-9 or not np.isfinite(product):
            return None
        return float(np.degrees(np.arccos(np.clip(forward @ to_screen / product, -1, 1))))


@dataclass(frozen=True)
class Observation:
    time_s: float
    pose_available: bool
    angle_degrees: float | None


def longest_runs(samples: list[Observation], predicate, max_gap: float) -> list[float]:
    runs = []
    first = last = None
    for sample in samples:
        if not predicate(sample):
            if first is not None:
                runs.append(last - first)
            first = last = None
            continue
        if first is None:
            first = sample.time_s
        elif sample.time_s - last > max_gap + 1e-8:
            runs.append(last - first)
            first = sample.time_s
        last = sample.time_s
    if first is not None:
        runs.append(last - first)
    return runs


def summarize_attention(tracks: dict[int, list[Observation]], config: dict, calibrated: bool) -> dict:
    coverage = assessable = facing = 0
    facing_seconds = 0.0
    gap = config["max_gap_s"]
    for samples in tracks.values():
        quality = longest_runs(samples, lambda sample: sample.pose_available, gap)
        coverage += max(quality, default=-1) + 1e-8 >= config["min_assessable_s"]
        if not calibrated:
            continue
        valid = longest_runs(samples, lambda sample: sample.angle_degrees is not None, gap)
        if max(valid, default=-1) + 1e-8 < config["min_assessable_s"]:
            continue
        assessable += 1
        hits = longest_runs(samples, lambda sample: sample.angle_degrees is not None
                            and sample.angle_degrees <= config["cone_degrees"], gap)
        sustained = [duration for duration in hits if duration + 1e-8 >= config["min_facing_s"]]
        if sustained:
            facing += 1
            facing_seconds += sum(sustained)
    return {
        "pedestrian_tracks": len(tracks),
        "pedestrian_tracks_with_sustained_usable_pose": coverage,
        "sustained_pose_coverage": coverage / len(tracks) if tracks else None,
        "assessable_pedestrian_tracks": assessable if calibrated else None,
        "billboard_facing_pedestrian_tracks": facing if assessable else None,
        "pedestrian_tracks_without_sustained_facing": assessable - facing if assessable else None,
        "unassessable_pedestrian_tracks": len(tracks) - assessable,
        "billboard_facing_rate_among_assessable": facing / assessable if assessable else None,
        "observed_sustained_facing_seconds": facing_seconds if assessable else None,
    }
