from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from math import dist
from statistics import median

import cv2
import numpy as np

from billboard.config import Point, Site


@dataclass(frozen=True)
class TrackPoint:
    track_id: int
    frame: int
    time_s: float
    vehicle_class: str
    confidence: float
    x: float
    y: float
    in_zone: bool
    world_x_m: float | None
    world_y_m: float | None

    @property
    def world(self) -> Point | None:
        if self.world_x_m is None or self.world_y_m is None:
            return None
        return self.world_x_m, self.world_y_m


def contour(vertices: tuple[Point, ...]) -> np.ndarray:
    polygon = np.asarray(vertices, dtype=np.float32)
    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1:
        raise ValueError("The road zone must be a convex polygon with points in perimeter order")
    return polygon


def line_side(line: tuple[Point, ...], x: float, y: float) -> float:
    (ax, ay), (bx, by) = line
    return (bx - ax) * (y - ay) - (by - ay) * (x - ax)


def line_event(samples: list[TrackPoint], site: Site) -> tuple[float, str] | None:
    previous = None
    for point in samples:
        side = line_side(site.counting_line, point.x, point.y)
        if abs(side) < 1e-6:
            continue
        if previous is not None:
            prior, prior_side = previous
            if side * prior_side < 0:
                direction = "negative_to_positive" if side > 0 else "positive_to_negative"
                fraction = prior_side / (prior_side - side)
                x = prior.x + fraction * (point.x - prior.x)
                y = prior.y + fraction * (point.y - prior.y)
                (ax, ay), (bx, by) = site.counting_line
                along = (x - ax) * (bx - ax) + (y - ay) * (by - ay)
                if (site.crossing_direction in ("both", direction)
                        and 0 <= along <= (bx - ax) ** 2 + (by - ay) ** 2):
                    return prior.time_s + fraction * (point.time_s - prior.time_s), direction
        previous = point, side
    return None


def segments(samples: list[TrackPoint], max_gap_s: float):
    start = 0
    for index, (previous, point) in enumerate(pairwise(samples), start=1):
        if point.time_s - previous.time_s > max_gap_s:
            yield samples[start:index]
            start = index
    yield samples[start:]


def representative_speed(samples: list[TrackPoint], window_s: float) -> float | None:
    """Median displacement speed over windows entirely inside the calibrated area."""
    speeds = []
    start = 0
    for end, point in enumerate(samples):
        world = point.world
        if world is None:
            start = end + 1
            continue
        while start + 1 < end and point.time_s - samples[start + 1].time_s >= window_s:
            start += 1
        elapsed = point.time_s - samples[start].time_s
        origin = samples[start].world
        if origin is not None and elapsed >= window_s:
            speeds.append(3.6 * dist(origin, world) / elapsed)
    return median(speeds) if speeds else None


def zone_events(samples: list[TrackPoint], vehicle_class: str, window_s: float):
    start = None
    for index in range(len(samples) + 1):
        inside = index < len(samples) and samples[index].in_zone
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            entry_seen = start > 0
            exit_seen = index < len(samples)
            first = samples[start]
            last = samples[index - 1]
            entry = (samples[start - 1].time_s + first.time_s) / 2 if entry_seen else first.time_s
            exit_ = (last.time_s + samples[index].time_s) / 2 if exit_seen else last.time_s
            yield {
                "kind": "zone_visit", "track_id": first.track_id, "vehicle_class": vehicle_class,
                "start_s": entry, "end_s": exit_, "duration_s": exit_ - entry,
                "entry_observed": entry_seen, "exit_observed": exit_seen,
                "speed_kmh": representative_speed(samples[start:index], window_s), "direction": None,
            }
            start = None


def analyze_tracks(points: list[TrackPoint], site: Site, max_gap_s: float, speed_window_s: float) -> tuple[list[dict], dict]:
    by_id = defaultdict(list)
    for point in points:
        by_id[point.track_id].append(point)
    events = []
    crossing_counts = Counter()
    for track_id, samples in by_id.items():
        class_weights = Counter()
        for point in samples:
            class_weights[point.vehicle_class] += point.confidence
        vehicle_class = class_weights.most_common(1)[0][0]
        counted = False
        for segment in segments(samples, max_gap_s):
            events.extend(zone_events(segment, vehicle_class, speed_window_s))
            crossing = line_event(segment, site) if not counted else None
            if crossing:
                stamp, direction = crossing
                crossing_counts[vehicle_class] += 1
                counted = True
                events.append({
                    "kind": "line_crossing", "track_id": track_id, "vehicle_class": vehicle_class,
                    "start_s": stamp, "end_s": None, "duration_s": None, "entry_observed": None,
                    "exit_observed": None, "speed_kmh": None, "direction": direction,
                })
    events.sort(key=lambda event: event["start_s"])
    visits = [event for event in events if event["kind"] == "zone_visit"]
    complete = [event for event in visits if event["entry_observed"] and event["exit_observed"]]
    speeds = [event["speed_kmh"] for event in complete if event["speed_kmh"] is not None]
    summary = {
        "vehicle_passages": sum(crossing_counts.values()),
        "passages_by_class": dict(sorted(crossing_counts.items())),
        "zone_tracks_observed": len({event["track_id"] for event in visits}),
        "complete_zone_visits": len(complete),
        "partial_zone_visits": len(visits) - len(complete),
        "median_complete_zone_time_s": median(event["duration_s"] for event in complete) if complete else None,
        "complete_visits_with_speed": len(speeds),
        "median_complete_visit_speed_kmh": median(speeds) if speeds else None,
    }
    return events, summary
