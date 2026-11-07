"""Package the detection demo (annotated videos + strict pipeline runs) for the dashboard.

Reads data/outputs/demo_videos/ and data/outputs/demo_<clip>/metrics.json, writes
data/outputs/phase7_demo/ with a short GIF, key frames and detection_demo.json per clip.
"""
import json
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"
DEST = OUT / "phase7_demo"
CLIPS = {
    "toronto_01": {"city": "Toronto", "run": "demo_toronto", "window_s": 12.0},
    "london_01": {"city": "London", "run": "demo_london_01", "window_s": 12.0},
}


def grab(video, seconds, max_side):
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    scale = max_side / max(frame.shape[:2])
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))


def main():
    (DEST / "frames").mkdir(parents=True, exist_ok=True)
    report = {}
    for clip, info in CLIPS.items():
        video = OUT / "demo_videos" / f"{clip}_annotated.mp4"
        demo = json.loads((OUT / "demo_videos" / f"{clip}_summary.json").read_text(encoding="utf-8"))
        strict = json.loads((OUT / info["run"] / "metrics.json").read_text(encoding="utf-8"))
        seconds = demo["seconds_processed"]
        for index, fraction in enumerate((0.1, 0.5, 0.9)):
            frame = grab(video, seconds * fraction, 1100)
            if frame is not None:
                cv2.imwrite(str(DEST / "frames" / f"{clip}_{index}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        gif_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE, colors=128)
                      for t in range(0, int(min(seconds, 12) * 4)) if (f := grab(video, t / 4, 560)) is not None]
        gif_frames[0].save(DEST / f"{clip}.gif", save_all=True, append_images=gif_frames[1:], duration=250, loop=0)
        report[clip] = {
            "city": info["city"],
            "demo": {
                "seconds": seconds,
                "vehicles_tracked_ids": demo["vehicles_tracked"],
                "people_tracked_ids": demo["people_tracked"],
                "line_crossings": demo["line_crossings_in_marked_direction"],
            },
            "strict": {
                "window_s": info["window_s"] or seconds,
                "vehicle_passages": strict["vehicle_passages"],
                "zone_tracks_observed": strict["zone_tracks_observed"],
                "complete_zone_visits": strict["complete_zone_visits"],
                "partial_zone_visits": strict["partial_zone_visits"],
            },
        }
    (DEST / "detection_demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
