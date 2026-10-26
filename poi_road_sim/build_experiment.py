"""Build a bounded, route-verified NYC comparison scenario from observed OD demand.

OD endpoints and group sizes come from TLC-derived records. POI intent is not
observed; a reachable local POI category is assigned as a synthetic query.
Every policy receives the same directed H2H routes and stored initial states.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

from .data import load_pois, slot_of
from .grid_index import GridQuery, POIGridIndex
from .h2h import H2HRouter, generate_h2h_grid_candidates
from .train import ROOT


def _od_rows(path: Path, start_slot: int, steps: int, seed: int) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        source.readline()  # Declared row count.
        rows = [row for row in csv.DictReader(source, delimiter="\t")
                if start_slot <= slot_of(row["time"]) < start_slot + steps]
    random.Random(seed).shuffle(rows)
    return rows


def _node_locations(path: Path, wanted: set[int]) -> dict[int, tuple[float, float]]:
    result = {}
    with path.open(encoding="utf-8") as source:
        source.readline()
        for row in csv.reader(source, delimiter="\t"):
            node_id = int(row[0])
            if node_id in wanted:
                result[node_id] = float(row[1]), float(row[2])
                if len(result) == len(wanted):
                    break
    return result


def _poi_nodes(path: Path) -> dict[int, int]:
    with path.open(encoding="utf-8") as source:
        return {int(row["poi_id"]): int(row["nearest_node_id"])
                for row in csv.DictReader(source, delimiter="\t")}


def _snapshot(store: Path, kind: str, ids: set[int]) -> dict[int, dict]:
    paths = sorted(store.glob("state_*.sqlite3"))
    if not paths:
        raise ValueError(f"no state shards in {store}")
    groups: dict[int, list[int]] = defaultdict(list)
    for object_id in ids:
        groups[object_id % len(paths)].append(object_id)
    result = {}
    table = "road_state" if kind == "road" else "poi_state"
    for shard, objects in groups.items():
        connection = sqlite3.connect(f"file:{paths[shard]}?mode=ro", uri=True)
        try:
            for object_id in objects:
                row = connection.execute(
                    f"SELECT state_json, updated_slot FROM {table} WHERE object_id=?", (object_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"missing {kind} state {object_id}")
                if row[1] != 0:
                    raise ValueError(f"{kind} state {object_id} is at slot {row[1]}; expected initial slot 0")
                result[object_id] = json.loads(row[0])
        finally:
            connection.close()
    return result


def build_scenario(root: Path, routing_index: Path, state_store: Path,
                   request_count: int = 10, steps: int = 4, start_slot: int = 0,
                   max_scan: int = 1000, max_pois: int = 8, seed: int = 2025,
                   cell_size_deg: float = 0.005, max_layers: int = 4) -> dict:
    if min(request_count, steps, max_scan, max_pois) < 1 or start_slot < 0 or start_slot + steps > 96:
        raise ValueError("invalid request count, time range, scan limit, or POI count")
    od_file = root / "interfaces" / "group_requests_real.txt"
    if not od_file.exists():
        raise FileNotFoundError("real OD requests are required for this builder")
    rows = _od_rows(od_file, start_slot, steps, seed)[:max_scan]
    if not rows:
        raise ValueError("no OD requests in the selected time range")
    wanted_nodes = {int(row[field]) for row in rows
                    for field in ("origin_node_id", "destination_node_id")}
    locations = _node_locations(root / "static" / "node.txt", wanted_nodes)
    poi_records = load_pois(root / "static" / "POI.txt")
    poi_nodes = _poi_nodes(root / "interfaces" / "poi_node_map.txt")
    poi_records = {poi_id: {**row, "node_id": poi_nodes[poi_id]}
                   for poi_id, row in poi_records.items() if poi_id in poi_nodes}
    grid = POIGridIndex.from_poi_mapping(poi_records, cell_size_deg)
    selected = []
    stats = defaultdict(int)
    with H2HRouter(routing_index) as router:
        for row in rows:
            if len(selected) >= request_count:
                break
            stats["od_checked"] += 1
            if stats["od_checked"] % 250 == 0:
                print(f"OD checked={stats['od_checked']} feasible={len(selected)} "
                      f"unreachable={stats['directed_unreachable']}", flush=True)
            origin, target = int(row["origin_node_id"]), int(row["destination_node_id"])
            if origin not in locations or target not in locations or origin == target:
                stats["missing_or_same_node"] += 1
                continue
            if router.distance(origin, target) is None:
                stats["directed_unreachable"] += 1
                continue
            query = GridQuery(
                str(row["request_id"]), origin, *locations[origin], target, *locations[target],
                1.0, cell_size_deg=cell_size_deg,
                max_layers=max_layers, max_pois=max(32, max_pois * 8),
            )
            nearby = grid.search(query)
            chosen_category = None
            for poi_id, _, _ in nearby:
                poi_node = int(poi_records[poi_id]["node_id"])
                if poi_node in (origin, target):
                    continue  # The simulator requires both route legs to contain an edge.
                if router.distance(origin, poi_node) is not None and router.distance(poi_node, target) is not None:
                    chosen_category = int(poi_records[poi_id]["category"])
                    break
            if chosen_category is None:
                stats["no_reachable_poi"] += 1
                continue
            query = GridQuery(**{**query.__dict__, "category": chosen_category})
            candidates = generate_h2h_grid_candidates(
                grid, query, poi_records, router, max_results=max_pois
            )
            if not candidates:
                stats["no_connected_candidate"] += 1
                continue
            payload = []
            for item in candidates:
                request = item.request
                payload.append({
                    "destination_poi_id": request.destination_poi_id,
                    "final_destination_node_id": target,
                    "edge_ids": list(request.edge_ids),
                    "egress_edge_ids": list(request.egress_edge_ids),
                    "label": item.label,
                })
            selected.append({
                "departure_slot": slot_of(row["time"]), "request_id": str(row["request_id"]),
                "origin_node_id": origin, "final_destination_node_id": target,
                "category": chosen_category,
                "destination_poi_id": payload[0]["destination_poi_id"],
                "edge_ids": payload[0]["edge_ids"], "egress_edge_ids": payload[0]["egress_edge_ids"],
                "candidates": payload,
            })
    if not selected:
        raise ValueError(f"no feasible OD-to-POI requests among {stats['od_checked']} checked rows")
    road_ids = {edge for request in selected for candidate in request["candidates"]
                for edge in candidate["edge_ids"] + candidate["egress_edge_ids"]}
    poi_ids = {candidate["destination_poi_id"] for request in selected
               for candidate in request["candidates"]}
    roads = _snapshot(state_store, "road", road_ids)
    pois = _snapshot(state_store, "poi", poi_ids)
    road_fields = {"edge_id", "from_node", "to_node", "length_m", "speed_limit_kmh",
                   "capacity_veh_per_slot", "count_history", "speed_kmh", "background_vehicles",
                   "recommended_vehicles", "travel_time_sec", "queue_vehicles", "outflow_vehicles",
                   "bpr_alpha", "bpr_beta"}
    poi_fields = {"poi_id", "node_id", "category", "capacity_people", "service_people_per_slot",
                  "occupancy", "waiting", "mta_period", "arrival_history", "departure_history",
                  "arrivals", "departures", "admitted", "wait_time_min"}
    return {
        "start_slot": start_slot, "steps": steps,
        "roads": [{k: v for k, v in roads[edge].items() if k in road_fields} for edge in sorted(roads)],
        "pois": [{k: v for k, v in pois[poi].items() if k in poi_fields} for poi in sorted(pois)],
        "requests": selected,
        "provenance": {
            "od": "TLC-derived group_requests_real.txt; directed OD reachability verified",
            "poi_intent": "synthetic local reachable POI category; original OD has no POI intent",
            "candidates": "shared grid POIs with directed H2H shortest access and egress legs",
            "state": "slot-zero bounded snapshot from full state store",
            "seed": seed, "scanned_od": stats["od_checked"],
            "rejected": {k: v for k, v in stats.items() if k != "od_checked"},
            "selected_requests": len(selected), "road_edges": len(roads), "candidate_pois": len(pois),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--routing-index", type=Path, default=Path(__file__).resolve().parents[1] / "routing_index_calibrated")
    parser.add_argument("--state-store", type=Path, default=Path(__file__).resolve().parents[1] / "state_store")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--start-slot", type=int, default=0)
    parser.add_argument("--max-scan", type=int, default=1000)
    parser.add_argument("--max-pois", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cell-size-deg", type=float, default=0.005)
    parser.add_argument("--max-layers", type=int, default=4)
    args = parser.parse_args()
    scenario = build_scenario(args.root, args.routing_index, args.state_store,
                              args.requests, args.steps, args.start_slot, args.max_scan,
                              args.max_pois, args.seed, args.cell_size_deg, args.max_layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario), encoding="utf-8")
    print(json.dumps(scenario["provenance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
