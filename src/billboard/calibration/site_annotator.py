from pathlib import Path

import cv2
import yaml

from billboard.analytics.zone_events import contour
from billboard.calibration.homography import RoadCalibration
from billboard.config import Site
from billboard.io.video_reader import read_frames


def annotate(video: Path, destination: Path, site_id: str, scenario: str, basis: str,
             billboard_visible: bool, direction: str, world_points: list[float] | None,
             calibration_source: str | None) -> bool:
    frames = read_frames(video, 1)
    try:
        _, _, frame = next(frames)
    finally:
        frames.close()
    height, width = frame.shape[:2]
    preview_size = (min(1200, width), min(780, height))
    scale = min(preview_size[0] / width, preview_size[1] / height)
    preview = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    preview_height, preview_width = preview.shape[:2]
    steps = [("zone", 3, 30, "Click the road zone corners in perimeter order (convex)."),
             ("counting_line", 2, 2, "Click the two endpoints across the lanes you want to count.")]
    if billboard_visible:
        steps.append(("billboard", 4, 4, "Click the four corners of the billboard in perimeter order."))
    if world_points is not None:
        steps.append(("image_points", 4, 4, "Click the road landmarks in the order of --world-points."))
    colors = [(40, 230, 80), (70, 205, 250), (240, 115, 110), (230, 130, 240)]
    selection: dict[str, list] = {key: [] for key, *_ in steps}
    title = "Billboard site annotation"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    try:
        for name, minimum, maximum, message in steps:
            picked = selection[name]

            def on_click(event, x, y, _flags, _param):
                if (event == cv2.EVENT_LBUTTONDOWN and 0 <= x < preview_width
                        and 0 <= y < preview_height and len(picked) < maximum):
                    picked.append([min(width - 1, round(x / scale)), min(height - 1, round(y / scale))])

            cv2.setMouseCallback(title, on_click)
            while True:
                display = cv2.copyMakeBorder(preview, 0, 56, 0, 0, cv2.BORDER_CONSTANT)
                for (key, *_), color in zip(steps, colors):
                    coords = [tuple(round(value * scale) for value in point) for point in selection[key]]
                    for x, y in coords:
                        cv2.circle(display, (x, y), 4, color, -1)
                    for left, right in zip(coords, coords[1:]):
                        cv2.line(display, left, right, color, 2)
                    if key != "counting_line" and len(coords) > 2:
                        cv2.line(display, coords[-1], coords[0], color, 2)
                cv2.putText(display, message, (8, preview_height + 21), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
                cv2.putText(display, "Enter: confirm  |  U: undo  |  R: reset  |  Esc: cancel", (8, preview_height + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
                cv2.imshow(title, display)
                key = cv2.waitKey(30) & 0xFF
                if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    return False
                if key == ord("u") and picked:
                    picked.pop()
                if key == ord("r"):
                    picked.clear()
                if key in (10, 13) and minimum <= len(picked) <= maximum:
                    break
    finally:
        cv2.destroyAllWindows()

    calibration = None
    if world_points is not None:
        calibration = {"source": calibration_source, "image_points": selection["image_points"],
                       "world_points_m": [world_points[i:i + 2] for i in range(0, 8, 2)]}
        RoadCalibration(calibration)
    site_data = {
        "site_id": site_id, "scenario": scenario, "camera": "fixed",
        "zone_kind": "billboard_visibility" if billboard_visible else "traffic_only", "zone_basis": basis,
        "frame_size": [width, height], "zone": selection["zone"], "counting_line": selection["counting_line"],
        "crossing_direction": direction, "billboard": selection.get("billboard"), "calibration": calibration,
    }
    contour(Site.from_mapping(site_data).zone)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        yaml.safe_dump(site_data, output, sort_keys=False)
    return True
