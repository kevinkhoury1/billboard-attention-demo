from pathlib import Path

import cv2
import numpy as np


def euler_degrees(rotation: np.ndarray) -> np.ndarray:
    """6DRepNet's x, y, z order, expressed as pitch, yaw, roll."""
    sy = np.hypot(rotation[0, 0], rotation[1, 0])
    yaw = np.arctan2(-rotation[2, 0], sy)
    if sy < 1e-6:
        return np.degrees([np.arctan2(-rotation[1, 2], rotation[1, 1]), yaw, 0.0])
    return np.degrees([np.arctan2(rotation[2, 1], rotation[2, 2]), yaw,
                       np.arctan2(rotation[1, 0], rotation[0, 0])])


class HeadPose:
    def __init__(self, weights: Path, device: str = "cpu"):
        import torch
        from sixdrepnet.model import SixDRepNet

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = SixDRepNet(backbone_name="RepVGG-B1g2", backbone_file="", deploy=True, pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.mean = np.array([.485, .456, .406], dtype=np.float32)
        self.std = np.array([.229, .224, .225], dtype=np.float32)

    def predict(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        scale = 224 / min(h, w)
        rgb = cv2.cvtColor(cv2.resize(crop, (round(w * scale), round(h * scale))), cv2.COLOR_BGR2RGB)
        cy, cx = (rgb.shape[0] - 224) // 2, (rgb.shape[1] - 224) // 2
        image = (rgb[cy:cy + 224, cx:cx + 224].astype(np.float32) / 255 - self.mean) / self.std
        tensor = self.torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)[None])).to(self.device)
        with self.torch.inference_mode():
            return self.model(tensor)[0].cpu().numpy()
