"""Materialize every v4-final object into sharded persistent state files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data import iter_histories, iter_road_histories_with_edges, load_pois
from .parameters import ParameterConfig, poi_parameters, road_parameters
from .state_store import StateStore
from .train import ROOT


def build(root: Path, output: Path, shards: int, batch_size: int,
          config: ParameterConfig | None = None) -> dict[str, int]:
    config = config or ParameterConfig()
    pois = load_pois(root / "static" / "POI.txt")
    with StateStore(output, shards=shards) as store:
        poi_batch = []
        for poi_id, history in iter_histories(root / "dynamic" / "POI_state_mta_15min.txt", "poi"):
            if poi_id not in pois:
                continue
            arrival, departure, occupancy, queue, wait, crowd = history[0]
            parameters = poi_parameters(poi_id, pois[poi_id]["capacity"], history, config)
            poi_batch.append((poi_id, {
                "poi_id": poi_id, "node_id": poi_id, "category": int(pois[poi_id]["category"]),
                **parameters,
                "occupancy": occupancy, "waiting": queue, "arrivals": arrival,
                "departures": departure, "wait_time_min": wait, "crowd_proxy": crowd,
                "arrival_history": [arrival], "departure_history": [departure],
            }, 0))
            if len(poi_batch) >= batch_size:
                store.upsert_pois(poi_batch)
                poi_batch.clear()
        if poi_batch:
            store.upsert_pois(poi_batch)

        road_batch = []
        for edge_id, edge, history in iter_road_histories_with_edges(
            root / "dynamic" / "road_state_15min.txt", root / "static" / "edge.txt"
        ):
            vehicles, flow, speed, travel = history[0]
            parameters = road_parameters(edge_id, float(edge["speed_limit_kmh"]), history, config)
            road_batch.append((edge_id, {
                "edge_id": edge_id, "from_node": edge["from_node"], "to_node": edge["to_node"],
                "length_m": edge["length_m"], **parameters,
                "vehicles": vehicles, "flow_vph": flow, "speed_kmh": speed,
                "travel_time_sec": travel, "count_history": [vehicles],
                "queue_vehicles": 0.0, "outflow_vehicles": 0.0,
                "bpr_alpha": 0.15, "bpr_beta": 4.0,
            }, 0))
            if len(road_batch) >= batch_size:
                store.upsert_roads(road_batch)
                road_batch.clear()
        if road_batch:
            store.upsert_roads(road_batch)
        result = store.counts()
    manifest = {
        "parameter_config": asdict(config),
        "road_unit": "directed edge",
        "time_step_minutes": 15,
        "source": "deterministic simulation parameters from 96-slot proxy histories",
        "counts": result,
    }
    (output / "parameter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "state_store")
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--road-capacity-scale", type=float, default=1.5)
    parser.add_argument("--poi-capacity-scale", type=float, default=1.25)
    parser.add_argument("--poi-service-scale", type=float, default=1.5)
    parser.add_argument("--parameter-spread", type=float, default=0.2)
    args = parser.parse_args()
    if args.shards < 1 or args.batch_size < 1:
        parser.error("--shards and --batch-size must be positive")
    config = ParameterConfig(args.seed, args.road_capacity_scale, args.poi_capacity_scale,
                             args.poi_service_scale, args.parameter_spread)
    print(build(args.root, args.output, args.shards, args.batch_size, config), flush=True)


if __name__ == "__main__":
    main()
