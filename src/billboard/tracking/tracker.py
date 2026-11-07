from dataclasses import dataclass

import numpy as np

from billboard.config import CLASSES


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    x: float
    y: float
    box: tuple[float, float, float, float]

    @property
    def vehicle_class(self) -> str:
        return self.class_name


class ObjectTracker:
    def __init__(self, config: dict, weights: str, device: str | None, class_names: dict[int, str]):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        if any(self.model.names.get(key) != name for key, name in class_names.items()):
            raise ValueError("Use YOLO weights with the expected COCO classes")
        self.class_names = class_names
        self.thresholds = config["class_confidence"]
        self.min_area = config["min_box_area_px"]
        self.args = {
            "tracker": "bytetrack.yaml", "persist": True, "verbose": False,
            "classes": list(class_names), "imgsz": config["image_size"],
            "conf": config["tracker_input_confidence"], "iou": config["nms_iou"],
        }
        if device is not None:
            self.args["device"] = device

    def update(self, frame: np.ndarray) -> list[TrackedObject]:
        boxes = self.model.track(frame, **self.args)[0].boxes
        if boxes is None or boxes.id is None:
            return []
        tracked = []
        for track_id, category, confidence, (left, top, right, bottom) in zip(
            boxes.id.int().cpu().tolist(), boxes.cls.int().cpu().tolist(),
            boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist(), strict=True,
        ):
            name = self.class_names[category]
            if confidence >= self.thresholds[name] and (right - left) * (bottom - top) >= self.min_area:
                tracked.append(TrackedObject(track_id, name, confidence, (left + right) / 2, bottom,
                                             (left, top, right, bottom)))
        return tracked


class VehicleTracker(ObjectTracker):
    def __init__(self, config: dict, weights: str, device: str | None):
        super().__init__(config, weights, device, CLASSES)
