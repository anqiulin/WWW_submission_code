"""Build regional high-demand NYC scenarios from the full POI catalog.

The catalog contains 2,000 real POIs per selected central region.  Candidate
routes are generated lazily with the native H2H index; only eight reachable
POIs per request and their route edges enter the simulator snapshot.  Demand
waves are synthetic interventions and are recorded as such in provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_road_sim.data import slot_of
from poi_road_sim.grid_index import GridQuery, POIGridIndex
from poi_road_sim.h2h import H2HRouter, generate_h2h_grid_candidates
from poi_road_sim.build_experiment import _node_locations, _od_rows, _snapshot


def _catalog(catalog: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((catalog / "manifest.json").read_text(encoding="utf-8"))
    records, centers = [], {}
    for region in range(int(manifest["regions"])):
        path = catalog / f"region_{region:02d}" / "pois.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                records.append({"poi_id": int(row["poi_id"]),
                                "node_id": int(row["nearest_node_id"]),
                                "latitude": float(row["lat"]),
                                "longitude": float(row["lon"]),
                                "category": int(row["category"]),
                                "region": region})
        summary = manifest["region_summary"][region]
        centers[region] = (float(summary["center_lat"]), float(summary["center_lon"]))
    return records, centers


def _regions_for_point(lat: float, lon: float, centers: dict[int, tuple[float, float]]) -> int:
    return min(centers, key=lambda r: (lat - centers[r][0]) ** 2 + (lon - centers[r][1]) ** 2)


def _state_rows(store: Path, kind: str, ids: set[int]) -> dict[int, dict]:
    """Read only selected object states and normalize fields for the simulator."""
    paths = sorted(store.glob("state_*.sqlite3"))
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
                    continue
                value = json.loads(row[0])
                result[object_id] = value
        finally:
            connection.close()
    return result


def _rate(condition: str, slot: int, region: int, rng: random.Random) -> int:
    peak = (24 <= slot < 40) if condition == "regional_delayed" else (36 <= slot < 52)
    if condition == "regional_shock":
        peak = (slot == 28 or 60 <= slot < 72)
    base = 3
    return max(1, (12 if peak else base) + rng.randint(-1, 1) + (region % 2))


def build_one(root: Path, routing_index: Path, state_store: Path, catalog: Path,
              output: Path, condition: str, seed: int, slots: int = 96,
              requests_scan: int = 10000, candidates: int = 8) -> dict:
    rng = random.Random(seed)
    records, centers = _catalog(catalog)
    by_region: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_region[record["region"]].append(record)
    grids = {region: POIGridIndex.from_records(items, 0.005)
             for region, items in by_region.items()}
    poi_map = {row["poi_id"]: row for row in records}
    rows = _od_rows(root / "interfaces" / "group_requests_real.txt", 0, slots, seed)
    rows = rows[:requests_scan]
    endpoints = {int(row[field]) for row in rows for field in
                 ("origin_node_id", "destination_node_id")}
    locations = _node_locations(root / "static" / "node.txt", endpoints)
    templates: dict[int, list[dict]] = defaultdict(list)
    stats = Counter()
    with H2HRouter(routing_index) as router:
        for row in rows:
            origin, target = int(row["origin_node_id"]), int(row["destination_node_id"])
            if origin not in locations or target not in locations or origin == target:
                continue
            if router.distance(origin, target) is None:
                stats["unreachable_od"] += 1
                continue
            region = _regions_for_point(*locations[origin], centers)
            # Query the corridor in the origin region. Categories are retained
            # from the actual catalog but the user request is synthetic.
            query = GridQuery(str(row["request_id"]), origin, *locations[origin],
                              target, *locations[target], category=None,
                              cell_size_deg=0.005, max_layers=8,
                              max_pois=max(8, candidates * 2))
            candidates_found = generate_h2h_grid_candidates(
                grids[region], query, {p["poi_id"]: p for p in by_region[region]},
                router, max_results=candidates)
            if len(candidates_found) < 2:
                stats["no_candidate"] += 1
                continue
            payload = []
            for candidate in candidates_found:
                request = candidate.request
                payload.append({"destination_poi_id": request.destination_poi_id,
                                "final_destination_node_id": target,
                                "edge_ids": list(request.edge_ids),
                                "egress_edge_ids": list(request.egress_edge_ids),
                                "label": candidate.label})
            templates[region].append({"origin_node_id": origin,
                "final_destination_node_id": target, "category": poi_map[payload[0]["destination_poi_id"]]["category"],
                "candidates": payload})
            stats["templates"] += 1
            if all(len(templates[r]) >= 2 for r in centers):
                break
        # The TLC OD interface is real, but its directed endpoints do not
        # cover every selected central region.  Fill missing regions with
        # synthetic OD pairs between real regional POI access nodes; H2H still
        # verifies every leg, so no unreachable request enters the dataset.
        for region in centers:
            if len(templates[region]) >= 2:
                continue
            pool = by_region[region]
            # Try local origins first, then all catalog nodes; some directed
            # components have no internal pair but still admit cross-region
            # requests into this POI region.
            origin_pool = pool + [p for p in records if p not in pool]
            for offset in range(min(500, len(origin_pool))):
                if len(templates[region]) >= 2:
                    break
                origin_rec = origin_pool[offset]
                target_rec = pool[(offset * 37 + 11) % len(pool)]
                origin, target = origin_rec["node_id"], target_rec["node_id"]
                if origin == target or router.distance(origin, target) is None:
                    continue
                fake_id = f"synthetic-{region}-{offset}"
                query = GridQuery(fake_id, origin, origin_rec["latitude"], origin_rec["longitude"],
                                  target, target_rec["latitude"], target_rec["longitude"],
                                  category=None, cell_size_deg=0.005, max_layers=8,
                                  max_pois=max(8, candidates * 2))
                found = generate_h2h_grid_candidates(
                    grids[region], query, {p["poi_id"]: p for p in by_region[region]},
                    router, max_results=candidates)
                if len(found) < 2:
                    continue
                payload = [{"destination_poi_id": c.request.destination_poi_id,
                            "final_destination_node_id": target,
                            "edge_ids": list(c.request.edge_ids),
                            "egress_edge_ids": list(c.request.egress_edge_ids),
                            "label": c.label} for c in found]
                templates[region].append({"origin_node_id": origin,
                    "final_destination_node_id": target,
                    "category": poi_map[payload[0]["destination_poi_id"]]["category"],
                    "candidates": payload})
                stats["synthetic_templates"] += 1
    missing = [r for r in centers if not templates[r]]
    fallback_regions = {}
    if missing:
        available_region = next((r for r in centers if templates[r]), None)
        if available_region is None:
            raise RuntimeError(f"no reachable regional templates: {stats}")
        for region in missing:
            # Keep the request executable when a selected region has no node
            # in the directed H2H component.  The catalog still records that
            # region's 2,000 POIs; this fallback is explicit in provenance.
            templates[region] = list(templates[available_region])
            fallback_regions[region] = available_region
            stats["template_region_fallback"] += 1

    selected_pois = {int(item["destination_poi_id"])
                     for values in templates.values() for row in values for item in row["candidates"]}
    selected_edges = {int(edge) for values in templates.values() for row in values
                      for item in row["candidates"] for edge in item["edge_ids"] + item["egress_edge_ids"]}
    road_states = _state_rows(state_store, "road", selected_edges)
    poi_states = _state_rows(state_store, "poi", selected_pois)
    road_fields = {"edge_id", "from_node", "to_node", "length_m", "speed_limit_kmh",
                   "capacity_veh_per_slot", "count_history", "speed_kmh", "background_vehicles",
                   "recommended_vehicles", "travel_time_sec", "queue_vehicles", "outflow_vehicles",
                   "bpr_alpha", "bpr_beta"}
    poi_fields = {"poi_id", "node_id", "category", "capacity_people", "service_people_per_slot",
                  "occupancy", "waiting", "mta_period", "arrival_history", "departure_history",
                  "arrivals", "departures", "admitted", "wait_time_min"}
    roads = [{k: (edge if k == "edge_id" else value) for k, value in {"edge_id": edge, **road_states[edge]}.items() if k in road_fields}
             for edge in sorted(selected_edges) if edge in road_states]
    pois = [{k: (poi_id if k == "poi_id" else value) for k, value in {"poi_id": poi_id, **poi_states[poi_id]}.items() if k in poi_fields}
            for poi_id in sorted(selected_pois) if poi_id in poi_states]
    if not roads or not pois:
        raise RuntimeError("selected routes have no matching state snapshots")
    # Keep the intervention locally congestible while retaining route topology.
    for road in roads:
        road["capacity_veh_per_slot"] = max(4.0, float(road.get("capacity_veh_per_slot", 20.0)) * 0.12)
        road["background_vehicles"] = 0.0
        road["recommended_vehicles"] = 0.0
        road["queue_vehicles"] = 0.0
        road["count_history"] = [0.1 * road["capacity_veh_per_slot"]] * 4
    for poi in pois:
        poi["capacity_people"] = max(10.0, float(poi.get("capacity_people", 80.0)) * 0.45)
        poi["service_people_per_slot"] = max(1.0, float(poi.get("service_people_per_slot", 10.0)) * 0.7)
        poi["occupancy"] = 0.1 * poi["capacity_people"]
        poi["waiting"] = 0.0
        poi["arrival_history"] = [0.0] * 4
        poi["departure_history"] = [0.0] * 4
    hot = {region: min((int(x["destination_poi_id"]) for row in templates[region] for x in row["candidates"]),
                       key=lambda p: poi_states.get(p, {}).get("service_people_per_slot", 1.0))
           for region in centers}
    requests = []
    for slot in range(slots):
        for region in centers:
            pool = templates[region]
            for index in range(_rate(condition, slot, region, rng)):
                template = pool[(slot * 17 + index + seed) % len(pool)]
                request = dict(template)
                request["request_id"] = f"regional-{condition}-{seed}-{slot}-{region}-{index}"
                request["departure_slot"] = slot
                request["destination_poi_id"] = request["candidates"][0]["destination_poi_id"]
                request["edge_ids"] = request["candidates"][0]["edge_ids"]
                request["egress_edge_ids"] = request["candidates"][0]["egress_edge_ids"]
                request["dwell_slots"] = 3
                requests.append(request)
    observations = []
    for slot in range(slots):
        arrivals = {str(p["poi_id"]): 0.0 for p in pois}
        if condition == "regional_delayed" and 20 <= slot < 44:
            for poi_id in hot.values():
                arrivals[str(poi_id)] = 3.0
        departures = {str(p["poi_id"]): 0.12 * float(p["service_people_per_slot"]) for p in pois}
        observations.append({"slot": slot,
            "road_background": {str(r["edge_id"]): 0.08 * float(r["capacity_veh_per_slot"]) for r in roads},
            "poi_arrivals": arrivals, "poi_departures": departures})
    scenario = {"start_slot": 0, "steps": slots, "roads": roads, "pois": pois,
        "requests": requests, "observations": observations, "allow_over_capacity": True,
        "condition": condition, "policy_use_forecast": True, "write_object_timeseries": False,
        "cost_normalization": {"road_reference_vehicle_min": 1000.0, "poi_reference_person_min": 1000.0,
                                "road_weight": 1.0, "poi_weight": 1.0},
        "provenance": {"source": "full NYC v4 real OD, road state and POI catalog",
            "type": "synthetic regional demand intervention; not observed request counts",
            "region_count": len(centers), "catalog_pois_per_region": 2000,
            "effective_template_regions": len(centers) - len(fallback_regions),
            "fallback_region_mapping": fallback_regions,
            "candidate_count_per_request": candidates, "selected_pois": len(pois),
            "selected_edges": len(roads), "templates": dict(stats), "hot_pois": hot,
            "condition": condition, "seed": seed}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario["provenance"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=here.parent / "experiment_data/nyc_v4_final")
    parser.add_argument("--routing-index", type=Path, default=here / "routing_index_calibrated")
    parser.add_argument("--state-store", type=Path, default=here / "state_store")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--slots", type=int, default=96)
    parser.add_argument("--conditions", nargs="+", default=["regional_delayed", "regional_coupling", "regional_shock"])
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()
    for condition in args.conditions:
        path = args.output / condition / f"seed_{args.seed}" / "scenario.json"
        print(json.dumps(build_one(args.root, args.routing_index, args.state_store, args.catalog,
                                   path, condition, args.seed, args.slots, candidates=args.candidates), indent=2), flush=True)


if __name__ == "__main__":
    main()
