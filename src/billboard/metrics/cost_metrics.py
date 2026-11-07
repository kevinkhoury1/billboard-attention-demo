from datetime import date

from billboard.metrics.funnel import number, source

AUDIENCE_METRICS = {"circulation", "ots", "likelihood_to_see", "attentive_passages", "attentive_seconds"}


def period_and_scope(record: dict) -> tuple[str, date, date, str]:
    start, end = date.fromisoformat(str(record["start_date"])), date.fromisoformat(str(record["end_date"]))
    display, scope = record["display_id"], record["scope"]
    if not isinstance(display, str) or not display.strip() or not isinstance(scope, str) or not scope.strip() or start >= end:
        raise ValueError("Evidence needs a display_id, scope and valid booking dates")
    return display, start, end, scope


def build_cost_metrics(site_id: str, evidence: dict) -> dict:
    cost = evidence.get("cost")
    audiences = {}
    for row in evidence.get("audiences", []):
        kind, metric = row["kind"], row["metric"]
        if kind not in {"owner_claim", "independent"} or metric not in AUDIENCE_METRICS:
            raise ValueError(f"Invalid audience kind or metric for {site_id}")
        key = kind, metric
        if key in audiences:
            raise ValueError(f"Provide one {kind} / {metric} record per placement")
        period_and_scope(row)
        number(row["value"], f"{kind} {metric}")
        source(row, f"{kind} {metric}")
        audiences[key] = row

    if cost is not None:
        period_and_scope(cost)
        number(cost["amount"], "booking cost")
        source(cost, "booking cost")
        currency = cost["currency"]
        if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isalpha() or currency != currency.upper():
            raise ValueError("Use a three-letter uppercase currency code")
        if cost["kind"] not in {"quote", "paid"}:
            raise ValueError("Cost kind must be quote or paid")

    def matching(row: dict | None) -> bool:
        return (cost is not None and row is not None and row["display_id"] == site_id
                and period_and_scope(row) == period_and_scope(cost))

    def cost_per_thousand(row: dict | None) -> float | None:
        if not matching(row) or row["value"] == 0:
            return None
        return 1000 * cost["amount"] / row["value"]

    ots = audiences.get(("independent", "ots"))
    attentive = audiences.get(("independent", "attentive_passages"))
    attentive_seconds = audiences.get(("independent", "attentive_seconds"))
    quoted_cpm = {metric: cost_per_thousand(row) for (kind, metric), row in audiences.items()
                  if kind == "owner_claim" and metric != "attentive_seconds"}
    claimed_ots = audiences.get(("owner_claim", "ots"))
    ots_ratio = None
    if (ots is not None and claimed_ots is not None and ots["display_id"] == site_id
            and period_and_scope(ots) == period_and_scope(claimed_ots) and claimed_ots["value"] > 0):
        ots_ratio = ots["value"] / claimed_ots["value"]
    notes = []
    if cost is None:
        notes.append("Provide a sourced booking quote or invoice for this display to calculate cost metrics.")
    elif not any(matching(row) for row in audiences.values()):
        notes.append("No audience record matches the booking's display, dates and scope.")
    elif any(not matching(row) for row in audiences.values()):
        notes.append("Some audience records differ from the booking's display, dates or scope. Their CPM is unavailable.")
    if not matching(attentive):
        notes.append("Attentive CPM requires independent attentive passages covering the booking scope and dates.")
    return {
        "currency": cost["currency"] if cost else None,
        "owner_cpm_by_claim_type": quoted_cpm,
        "independent_ots_cpm": cost_per_thousand(ots),
        "attentive_cpm": cost_per_thousand(attentive),
        "attentive_passages_per_1000_currency": (1000 * attentive["value"] / cost["amount"]
                                                 if matching(attentive) and cost["amount"] > 0 else None),
        "cost_per_1000_attentive_seconds": cost_per_thousand(attentive_seconds),
        "independent_to_claimed_ots_ratio": ots_ratio,
        "evidence": evidence,
        "notes": notes,
    }
