import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billboard.evaluation.eval_counts import evaluate_counts


def main():
    parser = argparse.ArgumentParser(description="Compare AI counts with manually reviewed crossing windows.")
    parser.add_argument("--run", required=True, type=Path, help="Phase 1 results folder covering the review windows")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "labels" / "manual_crossings.csv")
    parser.add_argument("--output", type=Path, help="Optional new JSON report")
    args = parser.parse_args()
    try:
        metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
        events = pq.read_table(args.run / "events.parquet").to_pylist()
        result = json.dumps(evaluate_counts(metrics, events, args.labels), indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(result + "\n")
        print(result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Count evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
