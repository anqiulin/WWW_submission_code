"""Create paper-ready aggregate tables from four-scenario evaluation output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUMMARY_FIELDS = (
    "scenario", "strategy", "requests", "completed_requests", "unfinished_requests",
    "execution_rate", "runtime_sec",
    "requests_per_second", "mean_route_time_min", "p95_free_flow_route_min",
    "cumulative_global_total_load", "cumulative_global_road_load",
    "cumulative_global_poi_load", "cumulative_road_delay_min",
    "mean_road_congestion_ratio", "peak_road_congestion_ratio",
    "peak_vc_ratio", "mean_poi_congestion_ratio", "peak_poi_utilization",
    "max_road_queue_vehicles", "max_poi_waiting_people", "poi_assignment_gini",
)

DISPLAY_NAMES = {
    "ppo": "PPO",
    "rise_mpc": "RISE-MPC",
    "no_future": "Myopic-RISE (H=1)",
    "no_poi": "Road-only MPC",
    "no_road": "POI-only MPC",
    "scair": "SCAIR",
    "shortest": "shortest-road",
    "shortest_queue": "shortest-queue",
}


def _num(value):
    if value in (None, ""):
        return ""
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = []
    timeseries = []
    summary_paths = list(args.results.glob("scenario_*/summary.json"))
    summary_paths.extend(args.results.glob("scenario_*/*/summary.json"))
    for summary_path in sorted(summary_paths):
        scenario = (summary_path.parent.parent.name
                    if summary_path.parent.parent != args.results
                    else summary_path.parent.name)
        for item in json.loads(summary_path.read_text(encoding="utf-8")):
            row = {"scenario": scenario}
            row.update({field: item.get(field, "") for field in SUMMARY_FIELDS if field != "scenario"})
            rows.append(row)
        series_path = summary_path.parent / "timeseries.csv"
        if series_path.is_file():
            with series_path.open(encoding="utf-8", newline="") as stream:
                for item in csv.DictReader(stream):
                    timeseries.append({"scenario": scenario, **item})
    if not rows:
        raise SystemExit(f"no scenario summaries found under {args.results}")

    aggregate_path = args.results / "aggregate.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    timeseries_fields = sorted({key for row in timeseries for key in row})
    timeseries_path = args.results / "timeseries_all.csv"
    with timeseries_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=timeseries_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(timeseries)

    lines = [
        "# Four-Scenario Experiment Results", "",
        f"Source: `{args.results}`", "",
        f"Scenarios: {len({row['scenario'] for row in rows})}; strategies: "
        f"{len({row['strategy'] for row in rows})}.", "",
        "| Scenario | Strategy | Total load | Road load | POI load | Road delay | Mean route (min) | P95 route (min) | Mean road congestion | Peak V/C | Runtime (s) | Requests/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = DISPLAY_NAMES.get(row["strategy"], row["strategy"])
        lines.append(
            f"| {row['scenario']} | {label} | {float(row['cumulative_global_total_load']):.3f} | "
            f"{float(row['cumulative_global_road_load']):.3f} | "
            f"{float(row['cumulative_global_poi_load']):.3f} | "
            f"{float(row['cumulative_road_delay_min']):.2f} | {float(row['mean_route_time_min']):.3f} | "
            f"{float(row['p95_free_flow_route_min']):.3f} | {float(row['mean_road_congestion_ratio']):.4%} | "
            f"{float(row['peak_vc_ratio']):.3f} | {float(row['runtime_sec']):.3f} | "
            f"{float(row['requests_per_second']):.1f} |"
        )
    lines.extend(["", "## Automatic Analysis", ""])
    for scenario in sorted({row["scenario"] for row in rows}):
        group = [row for row in rows if row["scenario"] == scenario]
        lowest_load = min(group, key=lambda row: float(row["cumulative_global_total_load"]))
        shortest_route = min(group, key=lambda row: float(row["mean_route_time_min"]))
        fastest = max(group, key=lambda row: float(row["requests_per_second"]))
        lines.append(
            f"- **{scenario}**: lowest cumulative load is "
            f"{DISPLAY_NAMES.get(lowest_load['strategy'], lowest_load['strategy'])} "
            f"({float(lowest_load['cumulative_global_total_load']):.2f}); shortest mean route is "
            f"{DISPLAY_NAMES.get(shortest_route['strategy'], shortest_route['strategy'])} "
            f"({float(shortest_route['mean_route_time_min']):.3f} min); highest throughput is "
            f"{DISPLAY_NAMES.get(fastest['strategy'], fastest['strategy'])} "
            f"({float(fastest['requests_per_second']):.1f} requests/s)."
        )
    lines.extend([
        "", "## Files", "",
        "- `aggregate.csv`: one row per scenario and strategy.",
        "- `timeseries_all.csv`: one row per scenario, strategy, and slot.",
        "- `scenario_*/performance_metrics.json`: per-scenario performance details.",
        "- `scenario_*/summary.json`: complete metric records.",
    ])
    report_text = "\n".join(lines) + "\n"
    (args.results / "RESULTS.md").write_text(report_text, encoding="utf-8")
    (args.results / "results.txt").write_text(report_text, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "timeseries_rows": len(timeseries),
                      "aggregate": str(aggregate_path),
                      "timeseries": str(timeseries_path),
                      "report": str(args.results / "results.txt")}, indent=2))


if __name__ == "__main__":
    main()
