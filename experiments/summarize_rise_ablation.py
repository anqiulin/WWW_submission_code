"""Summarize the four RISE-MPC ablation strategies.

Reads the standard ``run_four_scenario_eval.py`` output and writes a compact
table with both normalized objective components and raw waiting loads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGIES = ("rise_mpc", "no_future", "no_poi", "no_road")
LABELS = {
    "rise_mpc": "RISE-MPC",
    "no_future": "Myopic-RISE (H=1)",
    "no_poi": "Road-only MPC",
    "no_road": "POI-only MPC",
}

FIELDS = (
    "scenario", "strategy", "label", "cumulative_global_total_load",
    "cumulative_global_road_load", "cumulative_global_poi_load",
    "cumulative_raw_road_vehicle_min", "cumulative_raw_poi_person_min",
    "delta_total_vs_rise", "delta_road_vs_rise", "delta_poi_vs_rise",
    "total_ratio_vs_rise", "road_ratio_vs_rise", "poi_ratio_vs_rise",
)


def _read_rows(root: Path) -> list[dict]:
    rows = []
    paths = list(root.glob("scenario_*/summary.json"))
    paths.extend(root.glob("scenario_*/*/summary.json"))
    for path in sorted(paths):
        scenario = (path.parent.parent.name
                    if path.parent.parent != root else path.parent.name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            if row.get("strategy") in STRATEGIES:
                rows.append({"scenario": scenario, **row})
    if not rows:
        raise SystemExit(f"no RISE ablation summaries found under {root}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = _read_rows(args.results)
    by_key = {(row["scenario"], row["strategy"]): row for row in rows}
    scenarios = sorted({row["scenario"] for row in rows})
    output = []
    for scenario in scenarios:
        reference = by_key.get((scenario, "rise_mpc"))
        if reference is None:
            raise SystemExit(f"{scenario} is missing rise_mpc")
        ref_total = float(reference["cumulative_global_total_load"])
        ref_road = float(reference["cumulative_global_road_load"])
        ref_poi = float(reference["cumulative_global_poi_load"])
        for strategy in STRATEGIES:
            row = by_key.get((scenario, strategy))
            if row is None:
                raise SystemExit(f"{scenario} is missing {strategy}")
            total = float(row["cumulative_global_total_load"])
            road = float(row["cumulative_global_road_load"])
            poi = float(row["cumulative_global_poi_load"])
            output.append({
                "scenario": scenario, "strategy": strategy, "label": LABELS[strategy],
                "cumulative_global_total_load": total,
                "cumulative_global_road_load": road,
                "cumulative_global_poi_load": poi,
                "cumulative_raw_road_vehicle_min": float(row["cumulative_raw_road_vehicle_min"]),
                "cumulative_raw_poi_person_min": float(row["cumulative_raw_poi_person_min"]),
                "delta_total_vs_rise": total - ref_total,
                "delta_road_vs_rise": road - ref_road,
                "delta_poi_vs_rise": poi - ref_poi,
                "total_ratio_vs_rise": total / ref_total if ref_total else "",
                "road_ratio_vs_rise": road / ref_road if ref_road else "",
                "poi_ratio_vs_rise": poi / ref_poi if ref_poi else "",
            })

    csv_path = args.results / "rise_ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)

    lines = [
        "# RISE-MPC Ablation Results", "",
        "All methods use the same 7000-request test scenes, candidates, initial states, and simulator.",
        "Normalized loads are the queue_wait objective; raw loads are vehicle-minutes/person-minutes.", "",
        "| Scenario | Method | Total load | Road component | POI component | Raw road | Raw POI | Total / RISE | Road / RISE | POI / RISE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            f"| {row['scenario']} | {row['label']} | {row['cumulative_global_total_load']:.6f} | "
            f"{row['cumulative_global_road_load']:.6f} | {row['cumulative_global_poi_load']:.6f} | "
            f"{row['cumulative_raw_road_vehicle_min']:.2f} | {row['cumulative_raw_poi_person_min']:.2f} | "
            f"{row['total_ratio_vs_rise']:.4f} | {row['road_ratio_vs_rise']:.4f} | {row['poi_ratio_vs_rise']:.4f} |"
        )
    lines.extend(["", "## Interpretation", "",
                  "- Myopic-RISE isolates the value of the multi-slot forecast.",
                  "- Road-only MPC removes POI externality from the decision objective.",
                  "- POI-only MPC removes road externality from the decision objective.",
                  "- Ratios above 1.0 indicate a higher load than full RISE-MPC.", ""])
    md_path = args.results / "rise_ablation_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    colors = {"rise_mpc": "#b23a38", "no_future": "#4477aa",
              "no_poi": "#22885a", "no_road": "#d18a26"}
    width = 0.2
    for index, strategy in enumerate(STRATEGIES):
        values = [float(by_key[(scenario, strategy)]["cumulative_global_total_load"])
                  for scenario in scenarios]
        positions = [number + (index - 1.5) * width for number in range(len(scenarios))]
        axis.bar(positions, values, width=width, label=LABELS[strategy],
                 color=colors[strategy])
    axis.set_xticks(range(len(scenarios)), [scenario.replace("scenario_", "Scenario ")
                                             for scenario in scenarios])
    axis.set_ylabel("Cumulative normalized total load")
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    figure.tight_layout()
    chart_path = args.results / "rise_ablation_grouped.pdf"
    figure.savefig(chart_path, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"rows": len(output), "csv": str(csv_path),
                      "markdown": str(md_path), "chart": str(chart_path)}, indent=2))


if __name__ == "__main__":
    main()
