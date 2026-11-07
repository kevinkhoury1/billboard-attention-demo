import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from billboard.analytics.zone_events import contour
from billboard.config import CLASSES, Site, read_yaml
from billboard.evaluation.cvat_video import CvatVideo, VehicleBox
from billboard.evaluation.threshold_tuning import tune_thresholds
from billboard.evaluation.vision_metrics import detection_metrics, tracking_metrics
from billboard.io.results import video_provenance
from billboard.io.video_reader import read_frames
from billboard.tracking.tracker import VehicleTracker


def predictor(args, config):
    if "YOLO_CONFIG_DIR" not in os.environ:
        settings = ROOT / ".ultralytics"
        settings.mkdir(exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(settings)
    if args.detector == "rfdetr":
        try:
            from rfdetr import RFDETRMedium
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ImportError as error:
            raise ValueError("Install the optional rfdetr package to compare RF-DETR Medium") from error
        options = {"device": f"cuda:{args.device}" if args.device.isdigit() else args.device}
        if args.weights:
            options["pretrain_weights"] = args.weights
        elif (ROOT / "weights" / "rf-detr-medium.pth").is_file():
            options["pretrain_weights"] = str(ROOT / "weights" / "rf-detr-medium.pth")
        else:
            os.environ.setdefault("RF_HOME", str(ROOT / "weights"))
        model = RFDETRMedium(**options)

        def infer(frame):
            result = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=.01)
            names = result.data.get("class_name")
            if names is None:
                names = [COCO_CLASSES[int(category)] for category in result.class_id]
            return [VehicleBox(name, tuple(map(float, box)), confidence=float(confidence))
                    for name, box, confidence in zip(names, result.xyxy, result.confidence, strict=True)
                    if name in CLASSES.values()]

        return "RF-DETR Medium", infer

    weights = args.weights or config["model"]
    local = ROOT / "weights" / weights
    if local.is_file():
        weights = str(local)
    if args.task == "tracking":
        tracker = VehicleTracker(config, weights, args.device)

        def infer(frame):
            return [VehicleBox(row.vehicle_class, row.box, row.track_id, row.confidence) for row in tracker.update(frame)]

        return Path(weights).name, infer

    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    if any(model.names.get(key) != name for key, name in CLASSES.items()):
        raise ValueError("Use YOLO weights with the COCO vehicle class mapping")

    def infer(frame):
        result = model.predict(frame, classes=list(CLASSES), conf=.01, iou=config["nms_iou"],
                               imgsz=config["image_size"], device=args.device, verbose=False)[0].boxes
        return [VehicleBox(CLASSES[category], tuple(box), confidence=confidence)
                for category, box, confidence in zip(result.cls.int().cpu().tolist(), result.xyxy.cpu().tolist(),
                                                      result.conf.cpu().tolist(), strict=True)]

    return Path(weights).name, infer


def evaluate(args) -> dict:
    config = read_yaml(args.config)
    labels = CvatVideo(args.annotations)
    provenance = video_provenance(args.licence_log, args.video)
    site = Site.from_yaml(args.site) if args.site else None
    region = contour(site.zone) if site else None
    if labels.last_frame is not None and args.last_frame > labels.last_frame:
        raise ValueError("Requested frame range extends beyond the CVAT annotation task")

    def within_region(items):
        if region is None:
            return items
        return [item for item in items if cv2.pointPolygonTest(region, ((item.box[0] + item.box[2]) / 2, item.box[3]), False) >= 0]

    frames, times = [], []
    warmup_frames = 0
    model = infer = None
    for index, _, frame in read_frames(args.video, config["sample_fps"]):
        if index > args.last_frame:
            break
        if index < args.first_frame and args.task == "detection":
            continue
        size = frame.shape[1], frame.shape[0]
        if (labels.size and size != labels.size) or (site and size != site.frame_size):
            raise ValueError("The video dimensions differ from the annotation or site geometry")
        if infer is None:
            model, infer = predictor(args, config)
        start = perf_counter()
        predictions = infer(frame)
        if index < args.first_frame:
            warmup_frames += 1
            continue
        times.append(perf_counter() - start)
        frames.append((index, within_region(labels.frame(index)), within_region(predictions)))
    if not frames:
        raise ValueError("No sampled video frames fall in the labelled frame range")
    if args.task == "tracking" and len(frames) < 2:
        raise ValueError("Tracking evaluation needs at least two labelled sampled frames")
    metrics = detection_metrics(frames) if args.task == "detection" else tracking_metrics(frames)
    if args.tune_thresholds:
        metrics["threshold_candidates_development_only"] = tune_thresholds(frames)
    return {
        "task": args.task, "video": args.video.name, "source": provenance["source_url"],
        "annotations": str(args.annotations.resolve()), "model": model, "device": args.device,
        "region": site.site_id if site else "whole_frame", "sample_fps": config["sample_fps"],
        "yolo_input_size": config["image_size"] if args.detector == "yolo" else None,
        "first_evaluated_frame": frames[0][0], "last_evaluated_frame": frames[-1][0], "evaluated_frames": len(frames),
        "tracker_warmup_frames": warmup_frames,
        "inference_fps_after_first_frame": (len(times) - 1) / sum(times[1:]) if len(times) > 1 else None,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate vehicle detection or tracking on manually annotated CVAT video frames.")
    parser.add_argument("--task", choices=("detection", "tracking"), required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path, help="Extracted CVAT for video annotations.xml")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--last-frame", type=int, required=True, help="Last fully labelled original frame, inclusive")
    parser.add_argument("--site", type=Path, help="Evaluate only the marked road zone, if that was the labelled region")
    parser.add_argument("--detector", choices=("yolo", "rfdetr"), default="yolo")
    parser.add_argument("--tune-thresholds", action="store_true", help="Find confidence candidates on development footage only")
    parser.add_argument("--weights", help="Local weights override")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument("--licence-log", type=Path, default=ROOT / "data" / "licence_log.csv")
    parser.add_argument("--output", type=Path, help="Optional new JSON report")
    args = parser.parse_args()
    if args.first_frame < 0 or args.last_frame < args.first_frame:
        parser.error("Provide an inclusive, nonnegative range of fully labelled frames")
    if args.task == "tracking" and args.detector != "yolo":
        parser.error("This tracking evaluation uses the operational YOLO plus ByteTrack pipeline")
    if args.task == "tracking" and args.tune_thresholds:
        parser.error("Threshold tuning is a detection evaluation option")
    try:
        report = json.dumps(evaluate(args), indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(report + "\n")
        print(report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Vision evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
