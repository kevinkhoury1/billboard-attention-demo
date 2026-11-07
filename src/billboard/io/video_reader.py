from collections.abc import Iterator
from pathlib import Path

import numpy as np

try:
    import av
except ImportError:  # e.g. PyAV's DLL blocked by an Application Control policy
    av = None


def _read_frames_cv2(path: Path, sample_fps: float, limit_seconds: float | None) -> Iterator[tuple[int, float, np.ndarray]]:
    """OpenCV fallback; timestamps come from CAP_PROP_POS_MSEC instead of PyAV presentation times."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"No video stream in {path}")
    start = None
    next_sample = 0.0
    try:
        index = 0
        while True:
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, frame = capture.read()
            if not ok:
                break
            if start is None:
                start = stamp
            elapsed = stamp - start
            if limit_seconds is not None and elapsed > limit_seconds:
                break
            if elapsed + 1e-8 >= next_sample:
                next_sample = (int((elapsed + 1e-8) * sample_fps) + 1) / sample_fps
                yield index, elapsed, frame
            index += 1
    finally:
        capture.release()


def read_frames(path: Path, sample_fps: float, limit_seconds: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Decode against presentation times and sample without duplicating frames."""
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if av is None:
        yield from _read_frames_cv2(path, sample_fps, limit_seconds)
        return
    with av.open(str(path)) as video:
        if not video.streams.video:
            raise ValueError(f"No video stream in {path}")
        start = previous = None
        next_sample = 0.0
        for index, frame in enumerate(video.decode(video=0)):
            if frame.time is None:
                raise ValueError(f"Frame {index} has no presentation timestamp")
            stamp = float(frame.time)
            if previous is not None and stamp < previous:
                raise ValueError(f"Nonmonotonic video timestamps at frame {index}")
            previous = stamp
            if start is None:
                start = stamp
            elapsed = stamp - start
            if limit_seconds is not None and elapsed > limit_seconds:
                break
            if elapsed + 1e-8 < next_sample:
                continue
            next_sample = (int((elapsed + 1e-8) * sample_fps) + 1) / sample_fps
            yield index, elapsed, frame.to_ndarray(format="bgr24")
