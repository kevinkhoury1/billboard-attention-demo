import csv
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from billboard.analytics.zone_events import TrackPoint

TRACK_SCHEMA = pa.schema([
    ("track_id", pa.int64()), ("frame", pa.int64()), ("time_s", pa.float64()),
    ("vehicle_class", pa.string()), ("confidence", pa.float64()), ("x", pa.float64()), ("y", pa.float64()),
    ("in_zone", pa.bool_()), ("world_x_m", pa.float64()), ("world_y_m", pa.float64()),
])
EVENT_SCHEMA = pa.schema([
    ("kind", pa.string()), ("track_id", pa.int64()), ("vehicle_class", pa.string()),
    ("start_s", pa.float64()), ("end_s", pa.float64()), ("duration_s", pa.float64()),
    ("entry_observed", pa.bool_()), ("exit_observed", pa.bool_()),
    ("speed_kmh", pa.float64()), ("direction", pa.string()),
])


def video_provenance(manifest: Path, video: Path) -> dict:
    fields = ("source_url", "creator", "licence", "licence_url", "date_checked", "location", "conditions")
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        matches = [row for row in csv.DictReader(file) if row.get("filename") == video.name]
    if len(matches) != 1 or any(not (matches[0].get(key) or "").strip() for key in fields):
        raise ValueError(f"Add one complete licence_log.csv row for {video.name} before running the clip")
    return matches[0]


def save_results(output: Path, track_points: list[TrackPoint], events: list[dict], metrics: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        stage = Path(folder) / "results"
        stage.mkdir()
        track_table = pa.Table.from_pylist([asdict(point) for point in track_points], schema=TRACK_SCHEMA)
        event_table = pa.Table.from_pylist(events, schema=EVENT_SCHEMA)
        pq.write_table(track_table, stage / "tracks.parquet")
        pq.write_table(event_table, stage / "events.parquet")
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output.resolve())
