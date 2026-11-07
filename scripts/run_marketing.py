import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billboard.config import read_yaml
from billboard.metrics.cost_metrics import build_cost_metrics
from billboard.metrics.funnel import build_funnel


def make_report(runs: list[Path], evidence: dict) -> list[dict]:
    results = []
    for run in runs:
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        events = pq.read_table(run / "events.parquet").to_pylist()
        clip = metrics["provenance"]["filename"]
        site_id = metrics["site_id"]
        results.append({
            "site_id": site_id, "clip": clip, "source": metrics["provenance"]["source_url"],
            "phase1_run": str(run.resolve()), "zone_kind": metrics["zone_kind"],
            "observation_s": metrics["measurement"]["observation_s"],
            "funnel": build_funnel(metrics, events, evidence.get("clips", {}).get(clip, {})),
            "cost_metrics": build_cost_metrics(site_id, evidence.get("placements", {}).get(site_id, {})),
        })
    return results


def display(value: float | None, digits: int = 1) -> str:
    return "Unavailable" if value is None else f"{value:,.{digits}f}"


def markdown(rows: list[dict]) -> str:
    lines = ["# Phase 2 billboard comparison", "",
             "| Site and clip | Observed seconds | AI vehicle crossings | Crossings in potential viewing zone | Complete zone visits | Median complete dwell (s) | Estimated OTS (all counted modes) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        f = row["funnel"]
        site = f"{row['site_id']} / {row['clip']}".replace("|", "\\|")
        lines.append(f"| {site} | {row['observation_s']:.2f} | {f['vehicle_passages']} | "
                     f"{display(f['viewable_vehicle_passages'], 0)} | "
                     f"{f['complete_zone_visits']}/{f['zone_visits']} | "
                     f"{display(f['median_complete_zone_time_s'], 2)} | {display(f['estimated_ots_all_counted_modes'])} |")
    for row in rows:
        f, c = row["funnel"], row["cost_metrics"]
        currency = f" {c['currency']}" if c["currency"] else ""
        owner_cpm = ", ".join(f"{metric} {display(value, 2)}" for metric, value in c["owner_cpm_by_claim_type"].items())
        lines.extend(["", f"## {row['site_id']} / {row['clip']}", "",
                      f"- Vehicle-occupant OTS — {display(f['estimated_ots_vehicle_occupants'])}",
                      f"- Owner CPM by claim definition{currency} — {owner_cpm or 'Unavailable'}",
                      f"- Independent OTS CPM{currency} — {display(c['independent_ots_cpm'], 2)}",
                      f"- Attentive CPM{currency} — {display(c['attentive_cpm'], 2)}",
                      f"- [Original footage]({row['source']})"])
        lines.extend(f"- {note}" for note in f["notes"] + c["notes"])
    lines.extend(["", "Treat clip results as observations for their recorded intervals. Compare prices only in a "
                  "common currency and at comparable booking and audience scope.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Create a marketing scorecard from real Phase 1 runs and sourced placement evidence.")
    parser.add_argument("--run", type=Path, action="append", required=True, help="Phase 1 results folder. Repeat for comparison.")
    parser.add_argument("--evidence", type=Path, default=ROOT / "config" / "marketing_evidence.yaml")
    parser.add_argument("--output", type=Path, help="New scorecard folder")
    args = parser.parse_args()
    output = args.output or ROOT / "data" / "outputs" / f"phase2_{datetime.now():%Y%m%d_%H%M%S_%f}"
    try:
        rows = make_report(args.run, read_yaml(args.evidence))
        report_json = json.dumps({"phase": 2, "placements": rows}, indent=2, default=str, allow_nan=False)
        report_md = markdown(rows)
        output.mkdir(parents=True, exist_ok=False)
        (output / "marketing.json").write_text(report_json + "\n", encoding="utf-8")
        (output / "marketing.md").write_text(report_md, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Scorecard failed: {error}\n")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
