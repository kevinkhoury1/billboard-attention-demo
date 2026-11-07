from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np


def read_frames(path: Path, sample_fps: float, limit_seconds: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Decode against presentation times and sample without duplicating frames."""
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
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
