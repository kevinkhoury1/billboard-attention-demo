"""Live demo: YOLO11m + ByteTrack on every vehicle and person, drawn on the video.

Writes an annotated video and a small summary JSON per clip. This is a visual demo of
detection and tracking; it does not estimate who looked at the billboard (see README, Phase 3).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
COCO = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLOURS = {"person": (0, 165, 255), "car": (0, 220, 0), "motorcycle": (255, 0, 255), "bus": (255, 200, 0), "truck": (0, 220, 220)}


def side(line, point):
    (ax, ay), (bx, by) = line
    return (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)


def run(video: Path, site: dict, args) -> dict:
    from ultralytics import YOLO

    weights = ROOT / "weights" / "yolo11m.pt"
    weights.parent.mkdir(exist_ok=True)
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(3)), int(capture.get(4))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    scale = min(1.0, 1280 / max(width, height))
    out_size = (int(width * scale), int(height * scale))
    out_path = args.out / f"{video.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    zone = np.array(site["zone"], np.int32)
    line = site["counting_line"]
    direction = site["crossing_direction"]
    previous_side, seen, crossed = {}, {n: set() for n in COCO.values()}, set()
    limit = int(args.limit_seconds * fps) if args.limit_seconds else None
    results = model.track(source=str(video), stream=True, persist=True, tracker="bytetrack.yaml",
                          classes=list(COCO), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
    for index, result in enumerate(results):
        if limit and index >= limit:
            break
        frame = result.orig_img.copy()
        thick = max(2, int(max(width, height) / 600))
        cv2.polylines(frame, [zone], True, (0, 200, 0), thick)
        if site.get("billboard"):
            cv2.polylines(frame, [np.array(site["billboard"], np.int32)], True, (255, 120, 0), thick)
        cv2.line(frame, tuple(line[0]), tuple(line[1]), (0, 255, 255), thick * 2)
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().tolist(), boxes.cls.int().tolist()):
                name = COCO[cls]
                x1, y1, x2, y2 = (int(v) for v in box)
                foot = ((x1 + x2) / 2, y2)
                seen[name].add(track_id)
                if name != "person":
                    now = side(line, foot)
                    before = previous_side.get(track_id)
                    inside = cv2.pointPolygonTest(zone, foot, False) >= 0
                    if before is not None and inside:
                        wanted = (before < 0 <= now) if direction == "negative_to_positive" else (before > 0 >= now)
                        if wanted:
                            crossed.add(track_id)
                    previous_side[track_id] = now
                colour = (0, 0, 255) if track_id in crossed else COLOURS[name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
                cv2.putText(frame, f"{name} {track_id}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.6, thick * 0.45), colour, max(1, thick - 1))
        banner = (f"{video.stem}  t={index / fps:.1f}s | vehicles seen {sum(len(seen[n]) for n in seen if n != 'person')}"
                  f" | people seen {len(seen['person'])} | line crossings (red) {len(crossed)}")
        cv2.rectangle(frame, (0, 0), (width, int(70 * max(width, height) / 1000)), (0, 0, 0), -1)
        cv2.putText(frame, banner, (10, int(48 * max(width, height) / 1000)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.7, max(width, height) / 1800), (255, 255, 255), thick)
        small = cv2.resize(frame, out_size)
        writer.write(small)
        if args.show:
            cv2.imshow("YOLO11m + ByteTrack (q to skip)", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    writer.release()
    cv2.destroyAllWindows()
    summary = {
        "video": video.name,
        "seconds_processed": round((index + 1) / fps, 2),
        "vehicles_tracked": {n: len(seen[n]) for n in seen if n != "person"},
        "people_tracked": len(seen["person"]),
        "line_crossings_in_marked_direction": len(crossed),
        "device": str(args.device),
        "annotated_video": out_path.name,
    }
    (args.out / f"{video.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", default=["toronto_01", "london_01", "highway_01"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit-seconds", type=float, default=None)
    parser.add_argument("--show", action="store_true", help="open a live preview window")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "outputs" / "demo_videos")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.videos:
        site = yaml.safe_load((ROOT / "config" / "sites" / f"{name}.yaml").read_text(encoding="utf-8"))
        print(json.dumps(run(ROOT / "data" / "raw_videos" / f"{name}.mp4", site, args)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
