from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from xml.etree import ElementTree

from billboard.config import CLASSES

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class VehicleBox:
    class_name: str
    box: Box
    track_id: int | None = None
    confidence: float | None = None


class CvatVideo:
    def __init__(self, path: Path):
        root = ElementTree.parse(path).getroot()
        if root.tag != "annotations" or root.findall("image"):
            raise ValueError("Export CVAT for video annotations with vehicle tracks")
        metadata = root.find("./meta/task")
        self.size = None
        self.last_frame = None
        if metadata is not None:
            size = metadata.find("original_size")
            if size is not None:
                self.size = int(size.findtext("width")), int(size.findtext("height"))
            stop = metadata.findtext("stop_frame")
            self.last_frame = int(stop) if stop else None
            if metadata.findtext("start_frame", "0") != "0" or metadata.findtext("frame_filter", "").strip() not in {"", "step=1"}:
                raise ValueError("Export annotations for the original untrimmed video with no CVAT frame filter")
        self.tracks = []
        seen = set()
        for track in root.findall("track"):
            name = track.attrib["label"].lower().strip()
            if name not in CLASSES.values():
                continue
            track_id = int(track.attrib["id"])
            if track_id in seen:
                raise ValueError(f"Duplicate CVAT track ID {track_id}")
            seen.add(track_id)
            boxes = {}
            for shape in track.findall("box"):
                frame = int(shape.attrib["frame"])
                if frame < 0 or frame in boxes:
                    raise ValueError(f"Invalid or duplicate frame {frame} on CVAT track {track_id}")
                box = tuple(float(shape.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr"))
                if any(not isfinite(coord) for coord in box) or box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError(f"Invalid box at frame {frame} on CVAT track {track_id}")
                boxes[frame] = shape.attrib.get("outside") == "1", box
            if boxes:
                self.tracks.append((name, track_id, sorted(boxes), boxes))
        if not self.tracks:
            raise ValueError("The XML has no car, motorcycle, bus or truck box tracks")

    def frame(self, index: int) -> list[VehicleBox]:
        output = []
        for name, track_id, frames, boxes in self.tracks:
            position = bisect_right(frames, index) - 1
            if position < 0:
                continue
            left = frames[position]
            outside, box = boxes[left]
            if outside:
                continue
            if position + 1 < len(frames) and left != index:
                right = frames[position + 1]
                fraction = (index - left) / (right - left)
                box = tuple(start + fraction * (end - start) for start, end in zip(box, boxes[right][1]))
            output.append(VehicleBox(name, box, track_id))
        return output
