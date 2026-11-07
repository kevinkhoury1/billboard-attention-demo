from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import yaml

Point = tuple[float, float]
CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def points(values: list, minimum: int, maximum: int, field: str) -> tuple[Point, ...]:
    if not isinstance(values, list) or not minimum <= len(values) <= maximum:
        raise ValueError(f"{field} needs {minimum} to {maximum} points")
    result = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Each {field} point needs an x and y coordinate")
        x, y = map(float, value)
        if not isfinite(x) or not isfinite(y):
            raise ValueError(f"{field} coordinates must be finite")
        result.append((x, y))
    if len(set(result)) != len(result):
        raise ValueError(f"{field} points must be distinct")
    return tuple(result)


@dataclass(frozen=True)
class Site:
    site_id: str
    scenario: str
    zone_kind: str
    zone_basis: str
    frame_size: tuple[int, int]
    zone: tuple[Point, ...]
    counting_line: tuple[Point, ...]
    crossing_direction: str
    billboard: tuple[Point, ...] | None
    calibration: dict | None

    @classmethod
    def from_yaml(cls, path: Path) -> "Site":
        return cls.from_mapping(read_yaml(path))

    @classmethod
    def from_mapping(cls, data: dict) -> "Site":
        if data.get("camera") != "fixed":
            raise ValueError("Phase 1 zone geometry requires fixed-camera footage")
        kind = data["zone_kind"]
        if kind not in {"traffic_only", "billboard_visibility"}:
            raise ValueError("zone_kind must be traffic_only or billboard_visibility")
        basis = str(data.get("zone_basis") or "").strip()
        if not basis:
            raise ValueError("Describe the evidence for the zone in zone_basis")
        size = tuple(data["frame_size"])
        if len(size) != 2 or any(type(v) is not int or v < 2 for v in size):
            raise ValueError("frame_size must contain positive integer width and height")
        width, height = size
        direction = data.get("crossing_direction", "both")
        if direction not in {"both", "negative_to_positive", "positive_to_negative"}:
            raise ValueError("Invalid crossing_direction")
        zone = points(data["zone"], 3, 100, "zone")
        line = points(data["counting_line"], 2, 2, "counting_line")
        billboard = points(data["billboard"], 4, 4, "billboard") if data.get("billboard") else None
        if kind == "billboard_visibility" and billboard is None:
            raise ValueError("A billboard visibility zone requires an annotated billboard")
        calibration = data.get("calibration")
        image_points = points(calibration["image_points"], 4, 4, "calibration image points") if calibration else ()
        for x, y in (*zone, *line, *(billboard or ()), *image_points):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Point {(x, y)} is outside the annotated frame {size}")
        site_id, scenario = str(data["site_id"]).strip(), str(data["scenario"]).strip()
        if not site_id or not scenario:
            raise ValueError("site_id and scenario are required")
        return cls(site_id, scenario, kind, basis, size, zone, line, direction, billboard, calibration)
