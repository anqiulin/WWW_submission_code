"""Build a small real-ID scenario from the v4-final static and baseline files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_edges, load_initial_rows, load_mta_signal, load_pois
from .parameters import ParameterConfig, poi_parameters, road_parameters
from .train import ROOT


def build_scenario(root: Path, poi_id: int, edge_ids: list[int], people: float, steps: int) -> dict:
    if not edge_ids or people <= 0 or steps <= 0:
        raise ValueError("Provide at least one edge, positive people, and positive steps")
    static_pois = load_pois(root / "static" / "POI.txt")
    poi = static_pois[poi_id]
    edges = load_edges(root / "static" / "edge.txt", set(edge_ids))
    if len(edges) != len(set(edge_ids)):
        raise ValueError("Some route edge IDs are missing")
    first_node = int(edges[edge_ids[0]]["from_node"])
    previous_node = first_node
    for edge_id in edge_ids:
        edge = edges[edge_id]
        if int(edge["from_node"]) != previous_node:
            raise ValueError(f"Route is disconnected at edge {edge_id}")
        previous_node = int(edge["to_node"])
    if previous_node != poi_id:
        raise ValueError("The route must end at the POI node ID")
    initial_road = load_initial_rows(root / "dynamic" / "road_state.txt", "road", set(edge_ids))
    initial_poi = load_initial_rows(root / "dynamic" / "POI_state_mta.txt", "poi", {poi_id})
    if len(initial_road) != len(set(edge_ids)) or poi_id not in initial_poi:
        raise ValueError("Missing initial road or POI state")
    mta = load_mta_signal(root / "calibration" / "mta_station_period.csv", root / "interfaces" / "poi_mta_links.csv")
    road_rows = []
    config = ParameterConfig()
    for edge_id in edge_ids:
        edge = edges[edge_id]
        vehicles_six_hours, _, speed, _ = initial_road[edge_id]
        vehicles_15min = vehicles_six_hours / 24
        parameters = road_parameters(edge_id, float(edge["speed_limit_kmh"]),
                                     [(vehicles_15min, 0.0, speed)], config)
        road_rows.append({
            "edge_id": edge_id, "from_node": edge["from_node"], "to_node": edge["to_node"],
            "length_m": edge["length_m"], "speed_limit_kmh": parameters["speed_limit_kmh"],
            "capacity_veh_per_slot": parameters["capacity_veh_per_slot"],
            "count_history": [vehicles_15min], "speed_kmh": speed,
        })
    six_hour_arrivals, six_hour_departures, occupancy, queue, _, _ = initial_poi[poi_id]
    parameters = poi_parameters(poi_id, poi["capacity"],
                                [(six_hour_arrivals / 24, 0.0, occupancy)], config)
    return {
        "provenance": {
            "initial_state": "four-period proxy at 00:00",
            "parameters": "deterministic simulated parameters from initial proxy state; seed=2025",
            "note": "full-store parameters use all 96 slots and can differ from this quick scenario",
        },
        "start_slot": 0, "steps": steps, "roads": road_rows,
        "pois": [{
            "poi_id": poi_id, "node_id": poi_id, "category": poi["category"],
            "capacity_people": parameters["capacity_people"],
            "service_people_per_slot": parameters["service_people_per_slot"],
            "occupancy": occupancy, "waiting": queue,
            "mta_period": mta.get(poi_id, (0.0,) * 4),
            "arrival_history": [six_hour_arrivals / 24],
            "departure_history": [six_hour_departures / 24],
        }],
        "requests": [{
            "departure_slot": 0, "request_id": "real-id-example-1",
            "origin_node_id": first_node, "destination_poi_id": poi_id,
            "people": people, "vehicles": 1, "edge_ids": edge_ids,
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--poi-id", type=int, required=True)
    parser.add_argument("--edge-ids", type=int, nargs="+", required=True)
    parser.add_argument("--people", type=float, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenario = build_scenario(args.root, args.poi_id, args.edge_ids, args.people, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(f"Wrote real-ID scenario to {args.output}")


if __name__ == "__main__":
    main()
