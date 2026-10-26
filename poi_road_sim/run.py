"""Run a small coupled scenario with trained background models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .model import CountModel
from .background_profile import DailyBackgroundProfile
from .simulator import CoupledSimulator, POIState, Request, RoadState


def run_scenario(scenario: dict, model_dir: Path) -> list[dict]:
    roads = {int(row["edge_id"]): RoadState(**row) for row in scenario["roads"]}
    pois = {int(row["poi_id"]): POIState(**row) for row in scenario["pois"]}
    simulator = CoupledSimulator(
        roads, pois,
        CountModel.load(model_dir / "road_background.json"),
        CountModel.load(model_dir / "poi_arrivals.json"),
        CountModel.load(model_dir / "poi_departures.json"),
        start_slot=int(scenario.get("start_slot", 0)),
        background_profile=(DailyBackgroundProfile.from_file(Path(scenario["background_profile_file"]))
                            if scenario.get("background_profile_file") else None),
    )
    requests_by_slot = defaultdict(list)
    for row in scenario.get("requests", []):
        requests_by_slot[int(row["departure_slot"])].append(Request(
            request_id=str(row["request_id"]),
            origin_node_id=int(row["origin_node_id"]),
            destination_poi_id=int(row["destination_poi_id"]),
            people=1.0, edge_ids=tuple(map(int, row["edge_ids"])),
            vehicles=1.0,
        ))
    key = "profile_observations" if scenario.get("background_profile_file") else "observations"
    observations = {int(item["slot"]): item for item in scenario.get(key, [])}
    results = []
    for _ in range(int(scenario["steps"])):
        slot = simulator.slot
        observed = observations.get(slot, {})
        results.append(simulator.step(
            requests=requests_by_slot[slot],
            observed_road_background={int(k): float(v) for k, v in observed.get("road_background", {}).items()},
            observed_poi_arrivals={int(k): float(v) for k, v in observed.get("poi_arrivals", {}).items()},
            observed_poi_departures={int(k): float(v) for k, v in observed.get("poi_departures", {}).items()},
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background-profile", type=Path)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    if args.background_profile is not None:
        scenario["background_profile_file"] = str(args.background_profile.resolve())
    result = run_scenario(scenario, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"provenance": "proxy simulation", "steps": result}, indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} steps to {args.output}")


if __name__ == "__main__":
    main()
