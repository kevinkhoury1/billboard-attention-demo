import cv2
import numpy as np

from billboard.config import Point, points


class RoadCalibration:
    def __init__(self, data: dict):
        self.source = str(data.get("source") or "").strip()
        if not self.source:
            raise ValueError("Calibration needs the source of the measured road distances")
        image = points(data["image_points"], 4, 4, "calibration image points")
        world = points(data["world_points_m"], 4, 4, "calibration world points")
        self.region = np.asarray(image, dtype=np.float32)
        world_region = np.asarray(world, dtype=np.float32)
        for region in (self.region, world_region):
            if not cv2.isContourConvex(region) or abs(cv2.contourArea(region)) < 1e-6:
                raise ValueError("Calibration points must form a convex quadrilateral in perimeter order")
        self.matrix = cv2.getPerspectiveTransform(self.region, world_region, solveMethod=cv2.DECOMP_SVD)
        if not np.isfinite(self.matrix).all() or np.linalg.matrix_rank(self.matrix) != 3:
            raise ValueError("Calibration does not define a usable perspective transform")

    def pixel_to_world(self, point: Point) -> Point | None:
        if cv2.pointPolygonTest(self.region, point, False) < 0:
            return None
        mapped = self.matrix @ np.array([*point, 1.0])
        if abs(mapped[2]) < 1e-12:
            return None
        x, y = mapped[:2] / mapped[2]
        return float(x), float(y)
