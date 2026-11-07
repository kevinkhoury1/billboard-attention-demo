import argparse
import sys
from math import isfinite
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from billboard.config import Site
from billboard.io.video_reader import read_frames


def main():
    parser = argparse.ArgumentParser(description="Manually tally crossings on the marked line without showing AI predictions.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--start-seconds", type=float, default=0, help="Start time in the original video")
    parser.add_argument("--limit-seconds", type=float, help="End time in the original video")
    parser.add_argument("--paused", action="store_true", help="Pause on the first frame of the review window")
    args = parser.parse_args()
    if not isfinite(args.start_seconds) or args.start_seconds < 0 or (args.limit_seconds is not None
            and (not isfinite(args.limit_seconds) or args.limit_seconds <= args.start_seconds)):
        parser.error("Enter a nonnegative start time and an end time after it")
    site = Site.from_yaml(args.site)
    title = "Manual review - billboard traffic"
    count, paused = 0, args.paused
    direction = {"both": "both ways", "negative_to_positive": "arrow direction",
                 "positive_to_negative": "arrow direction"}[site.crossing_direction]
    try:
        for _, stamp, frame in read_frames(args.video, 20, args.limit_seconds):
            if stamp < args.start_seconds or (args.limit_seconds is not None and stamp >= args.limit_seconds):
                continue
            height, width = frame.shape[:2]
            if (width, height) != site.frame_size:
                parser.exit(2, "Video size differs from the site annotation.\n")
            scale = min(1, 1180 / width, 650 / height)
            image = cv2.resize(frame, (round(width * scale), round(height * scale)))
            polygon = np.rint(np.asarray(site.zone) * scale).astype(np.int32)
            (ax, ay), (bx, by) = site.counting_line
            a = tuple(round(value * scale) for value in (ax, ay))
            b = tuple(round(value * scale) for value in (bx, by))
            if site.billboard:
                billboard = np.rint(np.asarray(site.billboard) * scale).astype(np.int32)
                cv2.polylines(image, [billboard], True, (255, 205, 60), 2)
            cv2.polylines(image, [polygon], True, (80, 220, 90), 2)
            cv2.line(image, a, b, (40, 230, 250), 2)
            if site.crossing_direction != "both":
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = float(np.hypot(dx, dy))
                if length > 0:
                    center = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
                    sign = 1 if site.crossing_direction == "negative_to_positive" else -1
                    offset = round(-sign * 20 * dy / length), round(sign * 20 * dx / length)
                    cv2.arrowedLine(image, (center[0] - offset[0], center[1] - offset[1]),
                                    (center[0] + offset[0], center[1] + offset[1]), (40, 230, 250), 2)
            while True:
                display = cv2.copyMakeBorder(image, 0, 51, 0, 0, cv2.BORDER_CONSTANT)
                h = image.shape[0]
                cv2.putText(display, f"{stamp:5.2f}s  |  Manual tally: {count}  |  Crossings: {direction}",
                            (8, h + 19), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cv2.putText(display, "C: count  |  U: undo  |  Space: pause  |  N: next frame while paused  |  Esc: finish",
                            (8, h + 40), cv2.FONT_HERSHEY_SIMPLEX, .50, (255, 255, 255), 1)
                cv2.imshow(title, display)
                key = cv2.waitKey(100) & 0xFF
                if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key == ord("c"):
                    count += 1
                elif key == ord("u"):
                    count = max(0, count - 1)
                elif key == ord(" "):
                    paused = not paused
                if not paused or key == ord("n"):
                    break
    finally:
        cv2.destroyAllWindows()
        print(f"Manual tally: {count}")


if __name__ == "__main__":
    main()
