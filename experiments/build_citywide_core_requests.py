"""Build route-verified synthetic requests concentrated in every city core."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.build_background_profile import _curve
from experiments.build_nyc_regional_scenarios import _catalog, _state_rows
from poi_road_sim.h2h import H2HRouter


ROAD_FIELDS = {"edge_id", "from_node", "to_node", "length_m", "speed_limit_kmh",
               "capacity_veh_per_slot", "count_history", "speed_kmh", "background_vehicles",
               "recommended_vehicles", "travel_time_sec", "queue_vehicles", "outflow_vehicles",
               "bpr_alpha", "bpr_beta"}
POI_FIELDS = {"poi_id", "node_id", "category", "capacity_people", "service_people_per_slot",
              "occupancy", "waiting", "mta_period", "arrival_history", "departure_history",
              "arrivals", "departures", "admitted", "wait_time_min"}


def _distance_km(record: dict, center: tuple[float, float]) -> float:
    return math.hypot((record["latitude"] - center[0]) * 111.0,
                      (record["longitude"] - center[1]) * 111.0 * math.cos(math.radians(center[0])))


def _route_variants(router: H2HRouter, origin: int, target: int,
                    primary: tuple[int, ...], limit: int) -> list[tuple[int, ...]]:
    """Request alternate directed paths by penalizing one primary edge at a time."""
    paths = [primary]
    seen = {primary}
    if limit <= 1:
        return paths
    order = sorted(range(len(primary)), key=lambda index: abs(index - len(primary) / 2))
    for index in order:
        try:
            router.update_travel_times({primary[index]: 1e8})
            alternate = router.route(origin, target)
        finally:
            router.reset_travel_times()
        if alternate and alternate[1] and alternate[1] not in seen and primary[index] not in alternate[1]:
            paths.append(alternate[1])
            seen.add(alternate[1])
        if len(paths) >= limit:
            break
    return paths


def _templates(router: H2HRouter, records: list[dict], center: tuple[float, float],
               radius_km: float, candidate_count: int, template_count: int,
               max_route_edges: int, rng: random.Random,
               routes_per_poi: int = 1) -> list[dict]:
    pool = [row for row in records if _distance_km(row, center) <= radius_km]
    if len(pool) < candidate_count + 2:
        raise ValueError(f"core has only {len(pool)} mapped POIs inside {radius_km} km")
    by_category: dict[int, list[dict]] = defaultdict(list)
    for row in pool:
        by_category[row["category"]].append(row)
    categories = sorted(by_category, key=lambda category: -len(by_category[category]))
    templates = []
    unique_pairs = set()
    for attempt in range(max(60, template_count * 50)):
        if len(templates) >= template_count:
            break
        origin = pool[rng.randrange(len(pool))]
        target = pool[rng.randrange(len(pool))]
        origin_node, target_node = origin["node_id"], target["node_id"]
        if origin_node == target_node or (origin_node, target_node) in unique_pairs:
            continue
        unique_pairs.add((origin_node, target_node))
        if router.distance(origin_node, target_node) is None:
            continue
        midpoint = ((origin["latitude"] + target["latitude"]) / 2,
                    (origin["longitude"] + target["longitude"]) / 2)
        for category in categories:
            ranked = sorted(by_category[category], key=lambda row:
                            ((row["latitude"] - midpoint[0]) ** 2 +
                             (row["longitude"] - midpoint[1]) ** 2, row["poi_id"]))
            candidates = []
            used_nodes = set()
            for poi in ranked[:max(40, candidate_count * 10)]:
                node = poi["node_id"]
                if node in (origin_node, target_node) or node in used_nodes:
                    continue
                if router.distance(origin_node, node) is None or router.distance(node, target_node) is None:
                    continue
                access, egress = router.route(origin_node, node), router.route(node, target_node)
                if (not access or not egress or not access[1] or not egress[1] or
                        len(access[1]) + len(egress[1]) > max_route_edges):
                    continue
                access_routes = _route_variants(router, origin_node, node, access[1], routes_per_poi)
                egress_routes = _route_variants(router, node, target_node, egress[1], routes_per_poi)
                alternatives = [(first, second) for first in access_routes for second in egress_routes
                                if len(first) + len(second) <= max_route_edges]
                alternatives.sort(key=lambda pair: len(pair[0]) + len(pair[1]))
                for route_index, (first, second) in enumerate(alternatives[:routes_per_poi]):
                    candidates.append({"destination_poi_id": poi["poi_id"],
                        "final_destination_node_id": target_node,
                        "edge_ids": list(first), "egress_edge_ids": list(second),
                        "label": f"poi-{poi['poi_id']}-route-{route_index}"})
                    if len(candidates) >= candidate_count:
                        break
                used_nodes.add(node)
                if len(candidates) >= candidate_count:
                    break
            if len(candidates) >= candidate_count:
                templates.append({"origin_node_id": origin_node,
                    "final_destination_node_id": target_node,
                    "category": category, "candidates": candidates[:candidate_count]})
                break
    if len(templates) != template_count:
        raise RuntimeError(f"only {len(templates)}/{template_count} feasible templates in this core; "
                           "adjust catalog centers/radius or reduce candidate_count")
    return templates


def build(catalog: Path, routing_index: Path, state_store: Path, background: Path,
          config: dict, output: Path, split: str, seed: int, slots: int = 96,
          start_slot: int = 0) -> dict:
    if output.exists():
        raise FileExistsError(f"scenario already exists: {output}")
    if not 1 <= slots <= 96:
        raise ValueError("slots must be in [1, 96]")
    if not 0 <= start_slot < 96 or start_slot + slots > 96:
        raise ValueError("start_slot + slots must stay inside one 96-slot day")
    if not background.exists():
        raise FileNotFoundError(background)
    records, centers = _catalog(catalog)
    region_count = len(centers)
    if region_count != int(config["region_count"]):
        raise ValueError("request region_count must match catalog regions")
    radius = float(config["core_radius_km"])
    candidate_count = int(config["candidate_count"])
    template_count = int(config["templates_per_region"])
    max_route_edges = int(config["max_route_edges"])
    routes_per_poi = int(config.get("routes_per_poi", 1))
    if (radius <= 0 or candidate_count < 2 or template_count < 1 or
            max_route_edges < 2 or routes_per_poi < 1):
        raise ValueError("invalid core radius, candidate count, or template count")
    curve = _curve(config["requests_per_region_anchors"])
    if any(value < 0 for value in curve):
        raise ValueError("request intensity must be non-negative")
    by_region: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_region[record["region"]].append(record)
    rng = random.Random(seed)
    templates = {}
    with H2HRouter(routing_index) as router:
        for region in range(region_count):
            print(f"[requests] {split} core {region + 1}/{region_count}: routing templates", flush=True)
            templates[region] = _templates(router, by_region[region], centers[region], radius,
                                           candidate_count, template_count, max_route_edges,
                                           rng, routes_per_poi)
    selected_pois = {candidate["destination_poi_id"] for group in templates.values()
                     for template in group for candidate in template["candidates"]}
    selected_edges = {edge for group in templates.values() for template in group
                      for candidate in template["candidates"]
                      for edge in candidate["edge_ids"] + candidate["egress_edge_ids"]}
    road_states = _state_rows(state_store, "road", selected_edges)
    poi_states = _state_rows(state_store, "poi", selected_pois)
    if road_states.keys() != selected_edges or poi_states.keys() != selected_pois:
        raise RuntimeError("state store is missing routed road or POI IDs")
    capacity = config["capacity_scales"]
    roads = []
    for edge_id in sorted(selected_edges):
        state = {key: value for key, value in road_states[edge_id].items() if key in ROAD_FIELDS}
        state["edge_id"] = edge_id
        state["capacity_veh_per_slot"] = max(1.0, float(state["capacity_veh_per_slot"]) * float(capacity["road"]))
        state["queue_vehicles"] = state["background_vehicles"] = state["recommended_vehicles"] = 0.0
        state["count_history"] = [0.1 * state["capacity_veh_per_slot"]] * 4
        roads.append(state)
    pois = []
    for poi_id in sorted(selected_pois):
        state = {key: value for key, value in poi_states[poi_id].items() if key in POI_FIELDS}
        state["poi_id"] = poi_id
        state["capacity_people"] = max(1.0, float(state["capacity_people"]) * float(capacity["poi"]))
        state["service_people_per_slot"] = max(1.0, float(state["service_people_per_slot"]) * float(capacity["poi_service"]))
        state["occupancy"] = 0.1 * state["capacity_people"]
        state["waiting"] = 0.0
        state["arrival_history"] = state["departure_history"] = [0.0] * 4
        pois.append(state)
    requests = []
    counts = Counter()
    region_factors = {int(key): float(value) for key, value in config.get("region_request_multipliers", {}).items()}
    if any(not 0 <= region < region_count or value <= 0 for region, value in region_factors.items()):
        raise ValueError("invalid region request multiplier")
    for offset in range(slots):
        slot = start_slot + offset
        for region in range(region_count):
            expected = curve[slot] * region_factors.get(region, 1.0)
            amount = int(expected) + (rng.random() < expected % 1)
            for index in range(amount):
                template = templates[region][(slot * 17 + index + seed) % template_count]
                first = template["candidates"][0]
                requests.append({"request_id": f"core-{split}-{seed}-{slot}-{region}-{index}",
                    "departure_slot": slot, "origin_node_id": template["origin_node_id"],
                    "final_destination_node_id": template["final_destination_node_id"],
                    "category": template["category"], "region_id": region,
                    "destination_poi_id": first["destination_poi_id"],
                    "forecast_poi_id": first["destination_poi_id"],
                    "edge_ids": first["edge_ids"], "egress_edge_ids": first["egress_edge_ids"],
                    "dwell_slots": int(config["dwell_slots"]), "candidates": template["candidates"]})
                counts[region] += 1
    if set(counts) != set(range(region_count)):
        raise ValueError("some core regions have no requests in the selected time window")
    scene = {"start_slot": start_slot, "steps": slots, "roads": roads, "pois": pois,
        "requests": requests, "background_profile_file": str(background.resolve()),
        "allow_over_capacity": True, "policy_use_forecast": True,
        "dynamics_mode": "event_driven",
        "future_request_limit": int(config.get("future_request_limit", 12)),
        "forecast_horizon": int(config.get("forecast_horizon", 4)),
        "write_object_timeseries": False, "condition": "citywide_ten_cores",
        "cost_normalization": {"mode": "queue_wait", "road_top_k": 0,
            "road_reference_vehicle_min": 1.0, "poi_reference_person_min": 1.0,
            "road_weight": 1.0, "poi_weight": 1.0},
        "provenance": {"source": "full NYC static graph and POI catalog; synthetic ten-core requests",
            "background_file": str(background.resolve()), "split": split, "seed": seed,
            "start_slot": start_slot, "simulation_slots": slots,
            "region_count": region_count, "active_request_regions": sorted(counts),
            "requests_by_region": dict(counts), "selected_edges": len(roads),
            "selected_pois": len(pois), "candidate_count": candidate_count,
            "routes_per_poi": routes_per_poi,
            "templates_per_region": template_count,
            "cost_reference_note": "raw road vehicle waiting minutes and POI person waiting minutes"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")
    return {"scenario": str(output), "split": split, "seed": seed,
            "requests": len(requests), "requests_by_region": dict(counts),
            "selected_edges": len(roads), "selected_pois": len(pois)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--routing-index", type=Path, default=here / "routing_index_calibrated")
    parser.add_argument("--state-store", type=Path, default=here / "state_store")
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--seed", type=int, default=9103)
    parser.add_argument("--slots", type=int, default=96)
    parser.add_argument("--start-slot", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build(args.catalog, args.routing_index, args.state_store,
                           args.background, config, args.output, args.split, args.seed,
                           args.slots, args.start_slot), indent=2), flush=True)


if __name__ == "__main__":
    main()
