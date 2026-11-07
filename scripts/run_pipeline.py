import argparse
import json
import os
import sys
from datetime import datetime
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2

from billboard.analytics.zone_events import TrackPoint, analyze_tracks, contour
from billboard.calibration.homography import RoadCalibration
from billboard.config import CLASSES, Site, read_yaml
from billboard.io.results import save_results, video_provenance
from billboard.io.video_reader import read_frames
from billboard.tracking.tracker import VehicleTracker


def run(args) -> dict:
    config = read_yaml(args.config)
    if set(config["class_confidence"]) != set(CLASSES.values()):
        raise ValueError("class_confidence must specify the four supported vehicle classes")
    for name in ("tracker_input_confidence", "nms_iou", *config["class_confidence"]):
        value = config[name] if name in config else config["class_confidence"][name]
        if not 0 < value <= 1:
            raise ValueError(f"Invalid confidence or IoU for {name}")
    if config["tracker_input_confidence"] > min(config["class_confidence"].values()):
        raise ValueError("Tracking input confidence must be no higher than any reporting threshold")
    for key in ("min_box_area_px", "max_track_gap_s", "speed_window_s", "sample_fps"):
        if not isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if type(config["image_size"]) is not int or config["image_size"] < 32 or config["image_size"] % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    site = Site.from_yaml(args.site)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    provenance = video_provenance(args.licence_log, args.video)
    output = args.output or ROOT / "data" / "outputs" / f"{args.site.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    calibration = RoadCalibration(site.calibration) if site.calibration else None
    zone = contour(site.zone)
    model = args.model or config["model"]
    project_weights = ROOT / "weights" / model
    if project_weights.is_file():
        model = str(project_weights)
    track_points = []
    tracker = None
    processed = 0
    elapsed = 0.0
    for index, elapsed, frame in read_frames(args.video, config["sample_fps"], args.limit_seconds):
        if (frame.shape[1], frame.shape[0]) != site.frame_size:
            raise ValueError(f"Video frame size differs from annotation {site.frame_size}")
        if tracker is None:
            if "YOLO_CONFIG_DIR" not in os.environ:
                settings_dir = ROOT / ".ultralytics"
                settings_dir.mkdir(exist_ok=True)
                os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
            tracker = VehicleTracker(config, model, args.device)
        for vehicle in tracker.update(frame):
            anchor = vehicle.x, vehicle.y
            in_zone = cv2.pointPolygonTest(zone, anchor, False) >= 0
            world = calibration.pixel_to_world(anchor) if calibration else None
            track_points.append(TrackPoint(vehicle.track_id, index, elapsed, vehicle.vehicle_class,
                                           vehicle.confidence, *anchor, in_zone, *(world or (None, None))))
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} samples through {elapsed:.1f}s", file=sys.stderr, flush=True)
    if processed < 2:
        raise ValueError("At least two sampled video frames are needed")
    events, summary = analyze_tracks(track_points, site, config["max_track_gap_s"], config["speed_window_s"])
    metrics = {
        "phase": 1, "site_id": site.site_id, "scenario": site.scenario,
        "zone_kind": site.zone_kind, "zone_basis": site.zone_basis, "provenance": provenance,
        "measurement": {"model": Path(model).name, "tracker": "ByteTrack", "frames_processed": processed,
                        "observation_s": elapsed, "crossing_direction": site.crossing_direction,
                        "calibration_source": calibration.source if calibration else None, "settings": config},
        **summary,
    }
    save_results(output, track_points, events, metrics)
    print(f"Saved {output.resolve()}", file=sys.stderr)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Count vehicles and measure road-zone visits in real footage.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument("--licence-log", type=Path, default=ROOT / "data" / "licence_log.csv")
    parser.add_argument("--output", type=Path, help="New folder for the results")
    parser.add_argument("--model", help="Override the model, for example yolo11s.pt for a quick test")
    parser.add_argument("--image-size", type=int, help="Override detector input size, for example 1280")
    parser.add_argument("--device", help="For example cpu or 0 for the first CUDA device")
    parser.add_argument("--limit-seconds", type=float, help="Stop early for a smoke test")
    args = parser.parse_args()
    if args.limit_seconds is not None and (not isfinite(args.limit_seconds) or args.limit_seconds <= 0):
        parser.error("--limit-seconds must be positive and finite")
    try:
        print(json.dumps(run(args), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Pipeline failed: {error}\n")


if __name__ == "__main__":
    main()
