import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from billboard.calibration.site_annotator import annotate


def main():
    parser = argparse.ArgumentParser(description="Mark a fixed-camera road scene using the first video frame.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--site", required=True, type=Path, help="New site YAML to save")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--zone-basis", required=True, help="Explain how the zone was chosen")
    parser.add_argument("--billboard", action="store_true", help="Mark a real billboard and its potential visibility zone")
    parser.add_argument("--direction", choices=("both", "negative_to_positive", "positive_to_negative"), default="both")
    parser.add_argument("--world-points", type=float, nargs=8, metavar="M", help="Four measured XY pairs in metres")
    parser.add_argument("--calibration-source", help="Source of the measured road distances")
    args = parser.parse_args()
    if bool(args.world_points) != bool(args.calibration_source):
        parser.error("Provide --world-points and --calibration-source together")
    if args.site.exists():
        parser.error(f"Site YAML already exists: {args.site}")
    try:
        saved = annotate(args.video, args.site, args.site_id, args.scenario, args.zone_basis,
                         args.billboard, args.direction, args.world_points, args.calibration_source)
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(2, f"Annotation failed: {error}\n")
    print(f"Saved {args.site}" if saved else "Annotation cancelled.")


if __name__ == "__main__":
    main()
