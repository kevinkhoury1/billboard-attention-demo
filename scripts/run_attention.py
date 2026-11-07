import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from math import ceil, floor, isfinite
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from billboard.analytics.zone_events import contour
from billboard.attention.attention_rule import Observation, SceneGeometry, summarize_attention
from billboard.attention.face_detector import Face, Scrfd
from billboard.attention.head_pose import HeadPose
from billboard.attention.quality_gate import pose_crop, quality_reason
from billboard.config import Site, points, read_yaml
from billboard.io.results import video_provenance
from billboard.io.video_reader import read_frames
from billboard.tracking.tracker import ObjectTracker


def person_face(frame, person, detector):
    left, top, right, bottom = person.box
    width, height = right - left, bottom - top
    x0, x1 = max(0, floor(left - .12 * width)), min(frame.shape[1], ceil(right + .12 * width))
    y0, y1 = max(0, floor(top - .08 * height)), min(frame.shape[0], ceil(top + .50 * height))
    candidates = []
    for face in detector.detect(frame[y0:y1, x0:x1]):
        l, t, r, b = face.box
        mapped = Face((l + x0, t + y0, r + x0, b + y0), face.score)
        x, y = mapped.center
        if left <= x <= right and top - .04 * height <= y <= top + .42 * height:
            centered = abs(x - person.x) / width
            candidates.append((face.score - .5 * centered, mapped))
    return max(candidates, key=lambda pair: pair[0], default=None)


def assign_faces(frame, people, detector) -> dict[int, Face]:
    candidates = [(person.track_id, choice) for person in people if (choice := person_face(frame, person, detector))]
    if not candidates:
        return {}
    boxes = []
    scores = []
    for _, (rank, face) in candidates:
        x1, y1, x2, y2 = face.box
        boxes.append((x1, y1, x2 - x1, y2 - y1))
        scores.append(rank)
    selected = cv2.dnn.NMSBoxes(boxes, scores, 0.0, .45)
    return {candidates[int(i)][0]: candidates[int(i)][1][1] for i in np.asarray(selected).reshape(-1)}


def run(args) -> dict:
    config = read_yaml(args.config)
    site = Site.from_yaml(args.site)
    spec = read_yaml(args.scene)
    if spec["site_id"] != site.site_id or spec["video"] != args.video.name:
        raise ValueError("Pedestrian geometry does not match this site and clip")
    for key in ("sample_fps", "min_face_width_px", "min_assessable_s", "min_facing_s", "max_gap_s"):
        if not isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["max_gap_s"] < 1 / config["sample_fps"]:
        raise ValueError("max_gap_s must allow adjacent sampled frames")
    if not 0 < config["cone_degrees"] < 90:
        raise ValueError("cone_degrees must be between 0 and 90 degrees")
    vertices = points(spec["pedestrian_zone"], 3, 30, "pedestrian zone")
    if any(not (0 <= x < site.frame_size[0] and 0 <= y < site.frame_size[1]) for x, y in vertices):
        raise ValueError("Pedestrian zone extends outside the source frame")
    zone = contour(vertices)
    geometry = SceneGeometry(spec["geometry"]) if spec.get("geometry") is not None else None
    end = float(spec["end_s"])
    if not isfinite(end) or end <= 0:
        raise ValueError("Scene end_s must be positive")
    provenance = video_provenance(args.licence_log, args.video)
    if "YOLO_CONFIG_DIR" not in os.environ:
        settings_dir = ROOT / ".ultralytics"
        settings_dir.mkdir(exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    traffic = read_yaml(ROOT / "config/config.yaml")
    person_config = {**traffic, "image_size": config["person_image_size"],
                     "class_confidence": {"person": config["person_confidence"]}}
    detector = Scrfd(args.face_weights, config["face_input_size"], config["face_confidence"])
    pose = HeadPose(args.pose_weights, f"cuda:{args.device}" if args.device.isdigit() else args.device)
    tracker = ObjectTracker(person_config, str(args.person_weights), args.device, {0: "person"})
    tracks = defaultdict(list)
    reasons = Counter()
    seen_face = set()
    count_frames = 0
    elapsed = 0.0
    for _, elapsed, frame in read_frames(args.video, config["sample_fps"], end):
        if (frame.shape[1], frame.shape[0]) != site.frame_size:
            raise ValueError("Video dimensions do not match the site")
        people = [person for person in tracker.update(frame) if cv2.pointPolygonTest(zone, (person.x, person.y), False) >= 0]
        faces = assign_faces(frame, people, detector)
        for person in people:
            face = faces.get(person.track_id)
            angle = None
            usable = False
            if face is None:
                reasons["no_associated_face"] += 1
            else:
                seen_face.add(person.track_id)
                reason = quality_reason(frame, face, config)
                if reason is not None:
                    reasons[reason] += 1
                else:
                    rotation = pose.predict(pose_crop(frame, face))
                    usable = bool(np.isfinite(rotation).all())
                    reasons["pose_estimated" if usable else "pose_failed"] += 1
                    if usable and geometry is not None:
                        angle = geometry.angle(face.center, rotation)
            tracks[person.track_id].append(Observation(elapsed, usable, angle))
        count_frames += 1
        if count_frames % 20 == 0:
            print(f"Processed {count_frames} samples through {elapsed:.1f}s", file=sys.stderr, flush=True)
    if count_frames < 2:
        raise ValueError("At least two sampled video frames are needed")
    return {
        "phase": 3, "site_id": site.site_id, "video": args.video.name, "provenance": provenance,
        "scope": "Detected pedestrian tracks in the marked area, with temporary within-clip track IDs",
        "zone_basis": spec["zone_basis"], "frames_processed": count_frames, "observation_s": elapsed,
        "target_geometry_source": geometry.source if geometry else None,
        "attention_status": "geometry_available" if geometry else "unavailable_no_scene_calibration",
        "pedestrian_tracks_with_a_detected_face": len(seen_face),
        "person_frame_quality_counts": dict(sorted(reasons.items())),
        "settings": config,
        "models": {"person": args.person_weights.name, "tracker": "ByteTrack", "face": args.face_weights.name,
                   "head_pose": args.pose_weights.name},
        **summarize_attention(tracks, config, geometry is not None),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure usable pedestrian head-pose coverage and calibrated facing estimates.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/attention.yaml")
    parser.add_argument("--licence-log", type=Path, default=ROOT / "data/licence_log.csv")
    parser.add_argument("--face-weights", type=Path, default=ROOT / "weights/scrfd_500m.onnx")
    parser.add_argument("--pose-weights", type=Path, default=ROOT / "weights/6DRepNet_300W_LP_AFLW2000.pth")
    parser.add_argument("--person-weights", type=Path, default=ROOT / "weights/yolo11m.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="New aggregate JSON file")
    args = parser.parse_args()
    output = args.output or ROOT / "data/outputs" / f"attention_{args.site.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    try:
        if output.exists():
            raise FileExistsError(output)
        result = json.dumps(run(args), indent=2, allow_nan=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as file:
            file.write(result + "\n")
        print(result)
        print(f"Saved {output.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Attention run failed: {error}\n")


if __name__ == "__main__":
    main()
