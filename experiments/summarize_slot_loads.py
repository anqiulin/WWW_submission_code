"""Extract per-slot normalized load values from four-scenario results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "scenario", "strategy", "slot", "global_total_load",
    "global_road_load", "global_poi_load", "incremental_global_total_load",
    "incremental_road_load", "incremental_poi_load",
)

RAW_FIELDS = (
    "scenario", "strategy", "slot", "raw_road_vehicle_min", "raw_poi_person_min",
)

DISPLAY_NAMES = {"rise_mpc": "MPC", "scair": "SCAIR"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    source = args.results / "timeseries_all.csv"
    if not source.is_file():
        raise SystemExit(f"missing {source}")
    rows = []
    with source.open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            rows.append({field: item.get(field, "") for field in FIELDS})
    rows.sort(key=lambda item: (item["scenario"], item["strategy"], int(item["slot"])))

    csv_path = args.results / "slot_load_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Per-Slot Load Summary", "",
        "Loads are normalized values from `timeseries_all.csv`; slots are simulation slots.", "",
        "| Scenario | Strategy | Slot | Total load | Road load | POI load |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['strategy']} | {row['slot']} | "
            f"{float(row['global_total_load']):.6f} | "
            f"{float(row['global_road_load']):.6f} | "
            f"{float(row['global_poi_load']):.6f} |"
        )
    md_path = args.results / "slot_load_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw_rows = []
    with source.open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            raw_rows.append({field: item.get(field, "") for field in RAW_FIELDS})
    raw_rows.sort(key=lambda item: (item["scenario"], item["strategy"], int(item["slot"])))
    raw_csv_path = args.results / "slot_load_raw_summary.csv"
    with raw_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(raw_rows)

    raw_lines = [
        "# Per-Slot Unnormalized Load Summary", "",
        "Road load is vehicle-minutes; POI load is person-minutes.", "",
        "| Scenario | Strategy | Slot | Road load (veh-min) | POI load (person-min) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in raw_rows:
        raw_lines.append(
            f"| {row['scenario']} | {row['strategy']} | {row['slot']} | "
            f"{float(row['raw_road_vehicle_min']):.0f} | "
            f"{float(row['raw_poi_person_min']):.0f} |"
        )
    raw_md_path = args.results / "slot_load_raw_summary.md"
    raw_md_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    grouped = defaultdict(dict)
    strategies = []
    for row in raw_rows:
        key = (row["scenario"], row["slot"])
        grouped[key][row["strategy"]] = row
        if row["strategy"] not in strategies:
            strategies.append(row["strategy"])
    wide_lines = ["# Per-Slot Unnormalized Load Summary", "",
                  "Each cell is `road vehicle-min / POI person-min`.", ""]
    for scenario in sorted({key[0] for key in grouped}):
        wide_lines.extend([f"## {scenario}", "",
                           "| Slot | " + " | ".join(DISPLAY_NAMES.get(s, s) for s in strategies) + " |",
                           "|---:|" + "---:|" * len(strategies)])
        scenario_slots = sorted(int(slot) for name, slot in grouped if name == scenario)
        for slot in scenario_slots:
            cells = []
            for strategy in strategies:
                row = grouped[(scenario, str(slot))].get(strategy)
                cells.append("" if row is None else
                             f"{float(row['raw_road_vehicle_min']):.0f}/"
                             f"{float(row['raw_poi_person_min']):.0f}")
            wide_lines.append(f"| {slot} | " + " | ".join(cells) + " |")
        wide_lines.append("")
    wide_md_path = args.results / "slot_load_raw_wide.md"
    wide_md_path.write_text("\n".join(wide_lines), encoding="utf-8")
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"markdown={md_path}")
    print(f"raw_csv={raw_csv_path}")
    print(f"raw_markdown={raw_md_path}")
    print(f"raw_wide_markdown={wide_md_path}")


if __name__ == "__main__":
    main()
