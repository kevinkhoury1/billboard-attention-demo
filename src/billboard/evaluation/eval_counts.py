import csv
from math import isfinite
from pathlib import Path


def evaluate_counts(run: dict, events: list[dict], reference: Path) -> dict:
    site, video = run["site_id"], run["provenance"]["filename"]
    observed_to = run["measurement"]["observation_s"]
    crossing_times = [event["start_s"] for event in events if event["kind"] == "line_crossing"]
    with reference.open(encoding="utf-8-sig", newline="") as file:
        reviews = [row for row in csv.DictReader(file) if row["site_id"] == site and row["video"] == video]
    if not reviews:
        raise ValueError(f"No review windows for {site} / {video}")
    windows = []
    for row in reviews:
        start, end = float(row["start_s"]), float(row["end_s"])
        if not isfinite(start) or not isfinite(end) or not 0 <= start < end <= observed_to:
            raise ValueError(f"Manual window [{start}, {end}) must be inside the AI observation (0–{observed_to}s)")
        lower = int(row["confirmed"])
        additional = int(row["possible_additional"]) if row["possible_additional"].strip() else None
        if lower < 0 or (additional is not None and additional < 0) or not row["source"].strip():
            raise ValueError("Each review needs nonnegative counts and a source")
        upper = lower + additional if additional is not None else None
        predicted = sum(start <= time < end for time in crossing_times)
        exact = lower == upper
        windows.append({
            "start_s": start, "end_s": end, "manual_confirmed": lower, "manual_maximum": upper,
            "ai_crossings": predicted, "manual_exact": exact,
            "consistent_with_manual_bounds": predicted >= lower and (upper is None or predicted <= upper),
            "signed_error": predicted - lower if exact else None,
            "absolute_percent_error": 100 * abs(predicted - lower) / lower if exact and lower else None,
            "source": row["source"], "notes": row.get("notes", ""),
        })
    windows.sort(key=lambda row: row["start_s"])
    if any(left["end_s"] > right["start_s"] for left, right in zip(windows, windows[1:])):
        raise ValueError("Manual review windows overlap. Use nonoverlapping windows for the summary.")
    exact_windows = [window for window in windows if window["manual_exact"]]
    percentage_errors = [window["absolute_percent_error"] for window in exact_windows if window["absolute_percent_error"] is not None]
    return {
        "site_id": site, "clip": video, "windows": windows, "exact_windows": len(exact_windows),
        "mean_absolute_error": sum(abs(w["signed_error"]) for w in exact_windows) / len(exact_windows) if exact_windows else None,
        "mape_percent": sum(percentage_errors) / len(percentage_errors) if percentage_errors else None,
    }
