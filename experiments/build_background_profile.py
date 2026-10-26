"""Build an ID-indexed 24-hour background profile for one fixed scenario."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_road_sim.background_profile import DailyBackgroundProfile


def _minute(label: str) -> int:
    hour, minute = map(int, label.split(":"))
    total = hour * 60 + minute
    if not 0 <= total <= 1440 or not 0 <= minute < 60 or total % 15:
        raise ValueError(f"invalid 15-minute anchor: {label}")
    return total


def _curve(anchors: dict[str, float]) -> list[float]:
    points = sorted((_minute(label), float(value)) for label, value in anchors.items())
    if len(points) < 2 or points[0][0] != 0 or points[-1][0] != 1440:
        raise ValueError("curves need both 00:00 and 24:00 anchors")
    if len({minute for minute, _ in points}) != len(points):
        raise ValueError("curve anchors must refer to distinct times")
    if not math.isclose(points[0][1], points[-1][1]):
        raise ValueError("00:00 and 24:00 values must match for a daily cycle")
    if any(not math.isfinite(value) or value < 0 for _, value in points):
        raise ValueError("curve values must be finite and non-negative")
    result = []
    segment = 0
    for minute in range(0, 1440, 15):
        while points[segment + 1][0] < minute:
            segment += 1
        start, left = points[segment]
        end, right = points[segment + 1]
        result.append(left + (right - left) * (minute - start) / (end - start))
    return result


def _distances(adjacency: dict[int, set[int]], start: int) -> dict[int, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _regions(scenario: dict, config: dict) -> tuple[dict[int, int], dict[int, int], list[int]]:
    roads, pois = scenario["roads"], scenario["pois"]
    count = int(config["region_count"])
    poi_nodes = sorted({int(poi["node_id"]) for poi in pois})
    if not 1 <= count <= len(poi_nodes):
        raise ValueError("region_count must be between 1 and the number of distinct POI access nodes")
    adjacency: dict[int, set[int]] = defaultdict(set)
    for road in roads:
        left, right = int(road["from_node"]), int(road["to_node"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    for node in poi_nodes:
        adjacency[node]
    specified = [int(poi_id) for poi_id in config.get("region_seed_poi_ids", [])]
    if specified:
        by_poi = {int(poi["poi_id"]): int(poi["node_id"]) for poi in pois}
        if len(specified) != count or any(poi_id not in by_poi for poi_id in specified):
            raise ValueError("region_seed_poi_ids must list one loaded POI per region")
        seeds = [by_poi[poi_id] for poi_id in specified]
        if len(set(seeds)) != count:
            raise ValueError("region seed POIs must have distinct access nodes")
    else:
        seeds = [poi_nodes[0]]
        nearest = _distances(adjacency, seeds[0])
        while len(seeds) < count:
            next_node = max((node for node in poi_nodes if node not in seeds),
                            key=lambda node: (nearest.get(node, float("inf")), -node))
            seeds.append(next_node)
            distances = _distances(adjacency, next_node)
            nearest = {node: min(nearest.get(node, float("inf")), distances.get(node, float("inf")))
                       for node in poi_nodes}
    owner = dict(zip(seeds, range(count)))
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in owner:
                owner[neighbor] = owner[node]
                queue.append(neighbor)
    # Components without a POI seed remain intact and are assigned deterministically.
    for node in sorted(adjacency):
        if node not in owner:
            owner[node] = node % count
            queue = deque([node])
            while queue:
                current = queue.popleft()
                for neighbor in sorted(adjacency[current]):
                    if neighbor not in owner:
                        owner[neighbor] = owner[current]
                        queue.append(neighbor)
    road_regions = {int(road["edge_id"]): owner[int(road["from_node"])] for road in roads}
    poi_regions = {int(poi["poi_id"]): owner[int(poi["node_id"])] for poi in pois}
    return road_regions, poi_regions, seeds


def _factor(identifier: int, seed: int, salt: int, spread: float) -> float:
    if not 0 <= spread < 1:
        raise ValueError("ID factor spread must lie in [0, 1)")
    return 1.0 + random.Random((seed << 32) ^ identifier ^ salt).uniform(-spread, spread)


def build_profile(scenario: dict, config: dict) -> dict:
    road_regions, poi_regions, seeds = _regions(scenario, config)
    count = len(seeds)
    hot = config.get("hot_region_ids")
    if hot is None:
        hot = list(range(int(config.get("hot_region_count", 0))))
    hot = [int(region) for region in hot]
    if len(hot) != len(set(hot)) or any(not 0 <= region < count for region in hot):
        raise ValueError("hot region IDs must be unique and inside region_count")
    road_config, poi_config = config["road"], config["poi"]
    seed = int(config.get("seed", 0))
    category_profiles = {int(k): str(v) for k, v in poi_config.get("category_profiles", {}).items()}
    poi_curves = {name: _curve(anchors) for name, anchors in poi_config["arrival_ratio_anchors"].items()}
    if "default" not in poi_curves:
        raise ValueError("POI arrival_ratio_anchors must include a default profile")
    road_override = {int(k): float(v) for k, v in road_config.get("edge_factor_overrides", {}).items()}
    poi_override = {int(k): float(v) for k, v in poi_config.get("poi_factor_overrides", {}).items()}
    profile_override = {int(k): str(v) for k, v in poi_config.get("poi_profile_overrides", {}).items()}
    if not set(road_override) <= road_regions.keys() or not set(poi_override) <= poi_regions.keys() or not set(profile_override) <= poi_regions.keys():
        raise ValueError("factor/profile override references an ID outside this scenario")
    road_factors = {edge_id: road_override.get(edge_id, _factor(edge_id, seed, 17,
                    float(road_config.get("edge_factor_spread", 0)))) for edge_id in road_regions}
    poi_factors = {poi_id: poi_override.get(poi_id, _factor(poi_id, seed, 29,
                   float(poi_config.get("poi_factor_spread", 0)))) for poi_id in poi_regions}
    poi_profiles = {int(poi["poi_id"]): profile_override.get(int(poi["poi_id"]),
                    category_profiles.get(int(poi["category"]), "default")) for poi in scenario["pois"]}
    if not set(poi_profiles.values()) <= poi_curves.keys():
        raise ValueError("POI category/profile assignment names an unknown arrival curve")
    departure_by_profile = {str(k): float(v) for k, v in poi_config["departure_fraction_by_profile"].items()}
    if not set(poi_profiles.values()) <= departure_by_profile.keys():
        raise ValueError("every used POI profile needs a departure fraction")
    payload = {"schema_version": 1, "slots_per_day": 96, "seed": seed,
        "road_ratios": _curve(road_config["capacity_ratio_anchors"]),
        "poi_ratios": poi_curves,
        "road_regions": road_regions, "poi_regions": poi_regions,
        "road_factors": road_factors, "poi_factors": poi_factors,
        "poi_profiles": poi_profiles,
        "poi_departure_fractions": {poi_id: departure_by_profile[name]
                                    for poi_id, name in poi_profiles.items()},
        "region_road_factors": {region: float(road_config["hot_region_multiplier"]) if region in hot else 1.0
                                for region in range(count)},
        "region_poi_factors": {region: float(poi_config["hot_region_multiplier"]) if region in hot else 1.0
                               for region in range(count)},
        "max_road_ratio": float(road_config["max_capacity_ratio"]),
        "max_poi_arrival_service_ratio": float(poi_config["max_arrival_service_ratio"]),
        "generation_config": config,
        "provenance": {"source": "synthetic 24-hour ID-indexed background demand",
                       "region_seed_nodes": seeds, "hot_region_ids": hot,
                       "road_counts_by_region": {region: sum(x == region for x in road_regions.values())
                                                 for region in range(count)},
                       "poi_counts_by_region": {region: sum(x == region for x in poi_regions.values())
                                                for region in range(count)}}}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", type=Path)
    source.add_argument("--dataset", type=Path,
                        help="source dataset with train/validation/test scenario.json files")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="profile JSON for --scenario, new dataset directory for --dataset")
    args = parser.parse_args()
    if args.dataset and args.dataset.resolve() == args.output.resolve():
        raise ValueError("output dataset must differ from the source dataset")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = ([args.scenario] if args.scenario else
             sorted(args.dataset.glob("*/**/scenario.json")))
    if not paths:
        raise FileNotFoundError("no scenario.json files found")
    summaries = []
    for source_path in paths:
        scenario = json.loads(source_path.read_text(encoding="utf-8"))
        payload = build_profile(scenario, config)
        if args.scenario:
            profile_path = args.output
        else:
            destination = args.output / source_path.relative_to(args.dataset)
            if destination.exists():
                raise FileExistsError(f"destination scenario exists; use a new output directory: {destination}")
            profile_path = destination.with_name("background_profile.json")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        profile = DailyBackgroundProfile.from_file(profile_path)
        profile.validate_ids((int(row["edge_id"]) for row in scenario["roads"]),
                             (int(row["poi_id"]) for row in scenario["pois"]))
        if args.dataset:
            scenario["background_profile_file"] = str(profile_path.resolve())
            scenario.setdefault("provenance", {})["background_profile"] = "synthetic_diurnal_v1"
            destination.write_text(json.dumps(scenario), encoding="utf-8")
        summaries.append({"scenario": str(source_path), "profile": str(profile_path),
            **payload["provenance"], "night_road_ratio": profile.road_ratios[8],
            "morning_road_ratio": profile.road_ratios[32],
            "evening_road_ratio": profile.road_ratios[72]})
    print(json.dumps(summaries if args.dataset else summaries[0], indent=2))


if __name__ == "__main__":
    main()
