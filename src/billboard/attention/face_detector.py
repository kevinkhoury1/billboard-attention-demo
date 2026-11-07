from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]
    score: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


class Scrfd:
    """Run the three-stride SCRFD detector. No landmark or recognition output is retained."""

    def __init__(self, weights: Path, size: int, confidence: float):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        # The official graph declares 640px output sizes even with a dynamic input.
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(weights), sess_options=options, providers=["CPUExecutionProvider"])
        input_ = self.session.get_inputs()[0]
        self.name = input_.name
        self.height, self.width = (dimension if isinstance(dimension, int) else size for dimension in input_.shape[2:])
        if min(self.height, self.width) < 32 or self.height % 32 or self.width % 32:
            raise ValueError("SCRFD input dimensions must be positive multiples of 32")
        if len(self.session.get_outputs()) not in (6, 9):
            raise ValueError("Expected a three-stride SCRFD ONNX model")
        self.output_names = [output.name for output in self.session.get_outputs()[:6]]
        self.confidence = confidence
        self.centers = []
        for stride in (8, 16, 32):
            y, x = np.mgrid[:self.height // stride, :self.width // stride]
            self.centers.append(np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.float32) * stride)

    def detect(self, image: np.ndarray) -> list[Face]:
        height, width = image.shape[:2]
        if min(height, width) < 8:
            return []
        scale = min(self.height / height, self.width / width)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:h, :w] = cv2.resize(image, (w, h))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (self.width, self.height), (127.5,) * 3, swapRB=True)
        outputs = self.session.run(self.output_names, {self.name: blob})
        faces, rectangles, confidences = [], [], []
        for index, stride in enumerate((8, 16, 32)):
            scores = outputs[index].reshape(-1)
            distances = outputs[index + 3].reshape(-1, 4)
            centers = self.centers[index]
            anchors, remainder = divmod(len(scores), len(centers))
            if remainder or anchors < 1 or len(scores) != len(distances):
                raise ValueError("Unexpected SCRFD output layout")
            centers = np.repeat(centers, anchors, axis=0)
            for k in np.flatnonzero(scores >= self.confidence):
                x, y = centers[k]
                if x >= w or y >= h:
                    continue
                l, t, r, b = distances[k] * stride
                x1, x2 = float((x - l) * width / w), float((x + r) * width / w)
                y1, y2 = float((y - t) * height / h), float((y + b) * height / h)
                if x2 <= x1 or y2 <= y1:
                    continue
                faces.append(Face((x1, y1, x2, y2), float(scores[k])))
                rectangles.append((x1, y1, x2 - x1, y2 - y1))
                confidences.append(float(scores[k]))
        if not faces:
            return []
        selected = cv2.dnn.NMSBoxes(rectangles, confidences, self.confidence - 1e-6, .4)
        return [faces[int(i)] for i in np.asarray(selected).reshape(-1)]
