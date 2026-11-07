import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billboard.attention.face_detector import Scrfd
from billboard.attention.head_pose import HeadPose, euler_degrees
from billboard.attention.quality_gate import pose_crop, quality_reason
from billboard.config import read_yaml

BIWI_FILE = re.compile(r"(?:^|/)(\d\d)/frame_(\d+)_(rgb\.(?:jpg|png)|pose\.txt)$")
BIWI_SOURCE = "https://docs.fast.ai/tutorial.vision.html#points"
BIWI_TERMS = "https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose"


def biwi_pairs(archive: Path, stride: int):
    pending = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            match = BIWI_FILE.search(member.name)
            if not member.isfile() or match is None:
                continue
            sequence, frame, kind = match.groups()
            if int(frame) % stride:
                continue
            if not 0 < member.size < (2_000_000 if kind.startswith("rgb") else 4096):
                raise ValueError("Unexpected BIWI member size")
            key = sequence, int(frame)
            record = pending.setdefault(key, {})
            with source.extractfile(member) as file:
                record["rgb" if kind.startswith("rgb") else "pose"] = file.read()
            if len(record) == 2:
                del pending[key]
                yield key, record["rgb"], record["pose"]
    if pending:
        raise ValueError("The BIWI archive has unmatched RGB or pose records in the selected sample")


def biwi_euler(raw: bytes) -> np.ndarray:
    # Convert the RGB-D benchmark orientation to the Euler convention used for RGB head pose.
    # This follows the published HopeNet BIWI loader's axis convention.
    values = np.fromstring(raw.decode("ascii"), sep=" ")
    if values.size != 12 or not np.isfinite(values).all():
        raise ValueError("Unexpected BIWI pose annotation")
    angles = euler_degrees(values[:9].reshape(3, 3).T)
    return angles * np.array([1, -1, -1])


def metric(errors: list[np.ndarray]) -> dict:
    if not errors:
        return {"frames": 0, "mean_absolute_error_deg": None, "mae_by_axis_deg": None}
    values = np.stack(errors)
    return {"frames": len(errors), "mean_absolute_error_deg": float(values.mean()),
            "mae_by_axis_deg": dict(zip(("pitch", "yaw", "roll"), map(float, values.mean(axis=0)), strict=True))}


def run(args) -> dict:
    config = read_yaml(args.config)
    detector = Scrfd(args.face_weights, config["face_input_size"], config["face_confidence"])
    estimator = HeadPose(args.pose_weights, args.device)
    sizes = sorted(set(args.degrade_widths), reverse=True)
    native = []
    paired_native = []
    reduced = {width: [] for width in sizes}
    statuses = Counter()
    by_sequence = defaultdict(Counter)
    for (sequence, _), rgb, pose in biwi_pairs(args.archive, args.stride):
        by_sequence[sequence]["sampled"] += 1
        image = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode a sampled BIWI frame")
        faces = detector.detect(image)
        if not faces:
            statuses["no_face"] += 1
            continue
        face = max(faces, key=lambda item: item.score)
        reason = quality_reason(image, face, config)
        if reason is not None:
            statuses[reason] += 1
            continue
        target = biwi_euler(pose)
        crop = pose_crop(image, face)
        errors = np.abs((euler_degrees(estimator.predict(crop)) - target + 180) % 360 - 180)
        if not np.isfinite(errors).all():
            statuses["pose_failed"] += 1
            continue
        statuses["scored"] += 1
        by_sequence[sequence]["scored"] += 1
        native.append(errors)
        native_width = face.box[2] - face.box[0]
        if sizes and native_width >= max(sizes):
            paired_native.append(errors)
            for width in sizes:
                scale = width / native_width
                small = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                          max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                degraded = euler_degrees(estimator.predict(small))
                reduced[width].append(np.abs((degraded - target + 180) % 360 - 180))
        if len(native) % 40 == 0:
            print(f"Scored {len(native)} BIWI frames", file=sys.stderr, flush=True)
    sampled = sum(statuses.values())
    if not sampled:
        raise ValueError("No matching BIWI RGB and pose records were found")
    with args.archive.open("rb") as file:
        archive_hash = hashlib.file_digest(file, "sha256").hexdigest()
    return {
        "dataset": "BIWI Kinect Head Pose, fast.ai RGB archive", "archive": args.archive.name,
        "archive_sha256": archive_hash, "source": BIWI_SOURCE, "dataset_terms": BIWI_TERMS,
        "sample_rule": f"Frame number divisible by {args.stride} in each recording",
        "frames_sampled": sampled, "recordings_sampled": len(by_sequence),
        "recordings_with_scored_frames": sum(counts["scored"] > 0 for counts in by_sequence.values()),
        "sample_outcomes": dict(sorted(statuses.items())), "scored_fraction": len(native) / sampled,
        "models": {"face": args.face_weights.name, "pose": args.pose_weights.name},
        "native_resolution": metric(native),
        "resolution_stress_test": {"native_paired": metric(paired_native),
                                   "reduced_face_width_px": {str(width): metric(reduced[width]) for width in sizes}},
        "scope": "Head-pose error on sampled BIWI frames that pass the quality gate. Resize stress test uses paired real faces.",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate head pose on real BIWI RGB/pose pairs without extracting images.")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--degrade-widths", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/attention.yaml")
    parser.add_argument("--face-weights", type=Path, default=ROOT / "weights/scrfd_500m.onnx")
    parser.add_argument("--pose-weights", type=Path, default=ROOT / "weights/6DRepNet_300W_LP_AFLW2000.pth")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stride < 1 or any(width < 8 for width in args.degrade_widths):
        parser.error("Use a positive stride and resize widths of at least eight pixels")
    try:
        if args.output and args.output.exists():
            raise FileExistsError(args.output)
        result = json.dumps(run(args), indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                file.write(result + "\n")
        print(result)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(2, f"Head-pose evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
