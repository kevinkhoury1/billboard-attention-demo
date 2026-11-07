import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "outputs" / "phase5"


def main():
    data = json.loads((OUTPUT / "muttart_observations.json").read_text(encoding="utf-8"))
    before, during = (data["periods"][period] for period in ("before", "during"))
    treated_change = round(during["treated"] - before["treated"], 1)
    comparison_change = round(during["comparison"] - before["comparison"], 1)
    weekly = data["weekly"]
    first_campaign_week = data["first_campaign_week"]
    result = {
        "source": data["source"],
        "outcome": data["outcome"],
        "treated_change_pp": treated_change,
        "comparison_change_pp": comparison_change,
        "descriptive_difference_in_differences_pp": round(treated_change - comparison_change, 1),
        "weekly_pre_campaign_gap_pp": {
            row["week"]: round(row["treated"] - row["comparison"], 1)
            for row in weekly if row["week"] < first_campaign_week
        },
        "causal_lift": None,
        "scope": "Published venue-level aggregates. The pre-campaign gap was already widening."
    }
    (OUTPUT / "muttart_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    weeks = [row["week"] for row in weekly]
    figure, axes = plt.subplots(figsize=(9, 4.9), layout="constrained")
    axes.axvspan(first_campaign_week - 0.5, weeks[-1] + 0.5, color="#e8f4f2", zorder=0)
    axes.axhline(0, color="#526579", linewidth=0.9)
    axes.grid(axis="y", color="#dce3ea", linewidth=0.7)
    axes.set_axisbelow(True)
    axes.plot(weeks, [row["treated"] for row in weekly], color="#00685f", marker="o", linewidth=2.3,
              label="Muttart Conservatory (advertised)")
    axes.plot(weeks, [row["comparison"] for row in weekly], color="#695797", marker="s", linestyle="--",
              linewidth=2.0, label="S & S Center (comparison)")
    axes.set_title("Paid attendance was already diverging before the billboard campaign",
                   loc="left", fontsize=13, fontweight="bold", pad=12)
    axes.text(2.5, 87, "BEFORE CAMPAIGN", ha="center", fontsize=9, color="#344b62")
    axes.text(6.5, 87, "BILLBOARDS RUNNING", ha="center", fontsize=9, color="#00685f")
    axes.set(xlim=(0.65, weeks[-1] + 0.35), ylim=(-95, 100), xticks=weeks,
             xlabel="Study week", ylabel="Daily paying visitors vs. same week a year earlier")
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axes.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False, fontsize=9)
    figure.savefig(OUTPUT / "muttart_paid_attendance.png", dpi=200)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
