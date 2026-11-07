from collections import Counter, defaultdict
from math import isfinite


def number(value, field: str, minimum: float = 0) -> float:
    if type(value) not in (int, float) or not isfinite(value) or value < minimum:
        raise ValueError(f"{field} must be a finite number of at least {minimum}")
    return value


def source(record: dict, field: str) -> str:
    value = record.get("source")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} requires a source")
    return value.strip()


def people_from_vehicles(counts: Counter, occupancy: dict, variant: str = "mean") -> float | None:
    if any(vehicle not in occupancy for vehicle in counts):
        return None
    return sum(count * occupancy[vehicle].get(variant, occupancy[vehicle]["mean"])
               for vehicle, count in counts.items())


def build_funnel(run: dict, events: list[dict], evidence: dict) -> dict:
    crossings = [event for event in events if event["kind"] == "line_crossing"]
    visits = [event for event in events if event["kind"] == "zone_visit"]
    if len(crossings) != run["vehicle_passages"] or len({event["track_id"] for event in crossings}) != len(crossings):
        raise ValueError("Phase 1 metrics and crossing records disagree")
    counts = Counter(event["vehicle_class"] for event in crossings)
    complete = sum(event["entry_observed"] and event["exit_observed"] for event in visits)
    has_billboard = run["zone_kind"] == "billboard_visibility"
    eligible = None
    if has_billboard:
        by_track = defaultdict(list)
        for visit in visits:
            by_track[visit["track_id"]].append(visit)
        eligible = Counter(event["vehicle_class"] for event in crossings if any(
            visit["start_s"] <= event["start_s"] <= visit["end_s"] for visit in by_track[event["track_id"]]
        ))

    occupancy = evidence.get("occupancy", {})
    for vehicle, entry in occupancy.items():
        source(entry, f"occupancy for {vehicle}")
        mean = number(entry["mean"], f"{vehicle} mean occupancy", 1)
        low = number(entry.get("low", mean), f"{vehicle} low occupancy", 1)
        high = number(entry.get("high", mean), f"{vehicle} high occupancy", 1)
        if not low <= mean <= high:
            raise ValueError(f"{vehicle} occupancy must satisfy low <= mean <= high")

    pedestrians = evidence.get("pedestrians")
    if pedestrians is not None:
        source(pedestrians, "pedestrian count")
        if pedestrians.get("site_id") != run["site_id"] or pedestrians.get("end_s") != run["measurement"]["observation_s"]:
            raise ValueError("Pedestrian counts must match the site and cover time 0 to the clip run's observation_s")
        for field in ("passing", "viewable"):
            value = pedestrians.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"pedestrians.{field} must be a nonnegative integer")
        if pedestrians.get("viewable") is not None and (
            not has_billboard or pedestrians.get("passing") is None or pedestrians["viewable"] > pedestrians["passing"]
        ):
            raise ValueError("Viewable pedestrians need a billboard zone and cannot exceed passing pedestrians")

    passing_people = people_from_vehicles(counts, occupancy)
    eligible_people = people_from_vehicles(eligible, occupancy) if eligible is not None else None
    pedestrians_passing = pedestrians.get("passing") if pedestrians is not None else None
    pedestrians_viewable = pedestrians.get("viewable") if pedestrians is not None else None
    ots = eligible_people + pedestrians_viewable if eligible_people is not None and pedestrians_viewable is not None else None
    sensitivity = None
    if eligible_people is not None and any("low" in occupancy[v] or "high" in occupancy[v] for v in eligible):
        pedestrian_offset = pedestrians_viewable if pedestrians_viewable is not None else 0
        sensitivity = {
            "scope": "vehicle_occupants_and_pedestrians" if pedestrians_viewable is not None else "vehicle_occupants",
            "low": people_from_vehicles(eligible, occupancy, "low") + pedestrian_offset,
            "high": people_from_vehicles(eligible, occupancy, "high") + pedestrian_offset,
        }

    notes = []
    if not has_billboard:
        notes.append("This clip has a traffic-only road zone. Billboard viewability and OTS are unavailable.")
    if visits and complete < len(visits):
        notes.append(f"{len(visits) - complete} of {len(visits)} zone visits were partial. Dwell uses complete visits only.")
    if complete == 1:
        notes.append("The dwell figure comes from one complete visit. Collect more complete visits before comparing locations.")
    if counts and passing_people is None:
        notes.append("Vehicle occupancy evidence is incomplete. Person counts are unavailable for unmeasured vehicle classes.")
    if has_billboard and pedestrians_viewable is None:
        notes.append("No matching pedestrian visibility count. Any vehicle-occupant OTS covers vehicle occupants only.")
    return {
        "vehicle_passages": len(crossings),
        "passages_by_class": dict(sorted(counts.items())),
        "estimated_passing_vehicle_occupants": passing_people,
        "pedestrian_passages": pedestrians_passing,
        "viewable_vehicle_passages": sum(eligible.values()) if eligible is not None else None,
        "viewable_vehicles_by_class": dict(sorted(eligible.items())) if eligible is not None else None,
        "estimated_ots_vehicle_occupants": eligible_people,
        "viewable_pedestrians": pedestrians_viewable,
        "estimated_ots_all_counted_modes": ots,
        "occupancy_sensitivity": sensitivity,
        "assessable_people": None,
        "attentive_people": None,
        "attention_rate": None,
        "zone_visits": len(visits),
        "complete_zone_visits": complete,
        "complete_visit_share": complete / len(visits) if visits else None,
        "median_complete_zone_time_s": run["median_complete_zone_time_s"],
        "median_complete_visit_speed_kmh": run["median_complete_visit_speed_kmh"],
        "evidence": evidence,
        "notes": notes,
    }
