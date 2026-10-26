"""Build the scaled four-condition NYC benchmark used by the paper table.

The request templates come from the existing real NYC core dataset.  This
builder expands the route-local snapshot to a fixed network size, fills each
condition's POI catalog to the requested size, and keeps the short-route
semantics used by the 32-hour benchmark.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .build_four_scenarios import (
        SCENARIOS, _apply_condition, _sample_request_window, _source_path,
    )
except ImportError:  # direct ``python code/experiments/...py`` invocation
    from build_four_scenarios import (
        SCENARIOS, _apply_condition, _sample_request_window, _source_path,
    )

CONDITION_LABELS = {
    "A": "Periodic",
    "B": "Bottleneck",
    "C": "Delayed",
    "D": "Shock",
}

ROAD_FIELDS = {"edge_id", "from_node", "to_node", "length_m", "speed_limit_kmh",
               "capacity_veh_per_slot", "count_history", "speed_kmh",
               "background_vehicles", "recommended_vehicles", "travel_time_sec",
               "queue_vehicles", "outflow_vehicles", "bpr_alpha", "bpr_beta"}
POI_FIELDS = {"poi_id", "node_id", "category", "capacity_people",
              "service_people_per_slot", "occupancy", "waiting", "mta_period",
              "arrival_history", "departure_history", "arrivals", "departures",
              "admitted", "wait_time_min"}


def _read_catalog(path: Path, regions: set[int]) -> list[dict]:
    records = []
    for region in sorted(regions):
        csv_path = path / f"region_{region:02d}" / "pois.csv"
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                records.append({
                    "poi_id": int(row["poi_id"]),
                    "node_id": int(row["nearest_node_id"]),
                    "category": int(row["category"]),
                    "region": region,
                })
    if len({row["poi_id"] for row in records}) != len(records):
        raise ValueError("POI catalog contains duplicate IDs")
    return records


def _state_rows(store: Path, kind: str, ids: set[int]) -> dict[int, dict]:
    paths = sorted(store.glob("state_*.sqlite3"))
    if not paths:
        raise FileNotFoundError(f"no state shards under {store}")
    groups: dict[int, list[int]] = defaultdict(list)
    for object_id in ids:
        groups[object_id % len(paths)].append(object_id)
    table = "road_state" if kind == "road" else "poi_state"
    result: dict[int, dict] = {}
    for shard, objects in groups.items():
        connection = sqlite3.connect(f"file:{paths[shard]}?mode=ro", uri=True)
        try:
            for object_id in objects:
                row = connection.execute(
                    f"SELECT state_json, updated_slot FROM {table} WHERE object_id=?",
                    (object_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"missing {kind} state {object_id}")
                if int(row[1]) != 0:
                    raise ValueError(f"{kind} state {object_id} is not at slot 0")
                result[object_id] = json.loads(row[0])
        finally:
            connection.close()
    return result


def _static_edge_path(path: Path) -> Path:
    return path / "static" / "edge.txt" if (path / "static").is_dir() else path / "edge.txt"


def _iter_static_edges(path: Path):
    with _static_edge_path(path).open(encoding="utf-8") as stream:
        stream.readline()
        for line in stream:
            row = line.split("\t", 3)
            yield int(row[0]), int(row[1]), int(row[2])


def _expand_roads(source_roads: list[dict], static_root: Path,
                  state_store: Path, target_edges: int, target_nodes: int,
                  capacity_scale: float) -> list[dict]:
    by_id = {int(row["edge_id"]): row for row in source_roads}
    selected = set(by_id)
    nodes = {int(row["from_node"]) for row in source_roads}
    nodes.update(int(row["to_node"]) for row in source_roads)
    if len(selected) > target_edges or len(nodes) > target_nodes:
        raise ValueError(
            f"route snapshot already has {len(selected)} edges/{len(nodes)} nodes; "
            f"cannot fit {target_edges}/{target_nodes}"
        )

    # First add boundary links until the exact node budget is reached.  The
    # later pass fills the remaining link budget with links internal to that
    # node set, so the resulting scene has deterministic target counts.
    for edge_id, left, right in _iter_static_edges(static_root):
        if edge_id in selected or len(selected) >= target_edges:
            continue
        if (left in nodes) ^ (right in nodes) and len(nodes) < target_nodes:
            selected.add(edge_id)
            nodes.update((left, right))
    if len(nodes) != target_nodes:
        raise RuntimeError(f"could not reach {target_nodes} nodes; got {len(nodes)}")
    for edge_id, left, right in _iter_static_edges(static_root):
        if edge_id in selected or len(selected) >= target_edges:
            continue
        if left in nodes and right in nodes:
            selected.add(edge_id)
    # A directed NYC subgraph can have fewer than 7,500 internal links at
    # exactly 3,000 nodes.  Keep the requested link count authoritative and
    # admit the smallest deterministic boundary expansion needed to fill it.
    if len(selected) < target_edges:
        for edge_id, left, right in _iter_static_edges(static_root):
            if edge_id in selected or len(selected) >= target_edges:
                continue
            if left in nodes or right in nodes:
                selected.add(edge_id)
                nodes.update((left, right))
    if len(selected) != target_edges:
        raise RuntimeError(
            f"could not construct network size {target_edges}/{target_nodes}; "
            f"got {len(selected)}/{len(nodes)}"
        )

    missing = selected - set(by_id)
    states = _state_rows(state_store, "road", missing)
    for edge_id in missing:
        row = {key: value for key, value in states[edge_id].items() if key in ROAD_FIELDS}
        row["edge_id"] = edge_id
        row["capacity_veh_per_slot"] = max(
            1.0, float(row["capacity_veh_per_slot"]) * capacity_scale
        )
        row["queue_vehicles"] = 0.0
        row["background_vehicles"] = 0.0
        row["recommended_vehicles"] = 0.0
        row["outflow_vehicles"] = 0.0
        row["count_history"] = [0.0] * 4
        by_id[edge_id] = row
    return [by_id[edge_id] for edge_id in sorted(selected)]


def _expand_pois(source_pois: list[dict], required_ids: set[int], catalog: list[dict],
                 state_store: Path, target: int, seed: int, capacity_scale: float,
                 service_scale: float) -> list[dict]:
    source_ids = set(required_ids)
    by_id = {int(row["poi_id"]): row for row in catalog}
    if len(by_id) < target:
        raise ValueError(f"catalog has {len(by_id)} POIs but {target} are required")
    if not source_ids <= set(by_id):
        raise ValueError("request candidates reference POIs missing from the catalog")
    rng = random.Random(seed)
    chosen = set(source_ids)
    remaining = [row for row in catalog if row["poi_id"] not in chosen]
    rng.shuffle(remaining)
    chosen.update(row["poi_id"] for row in remaining[:target - len(chosen)])
    if len(chosen) > target:
        # The source eight-slot templates may mention more IDs than condition
        # A's target.  Keep a deterministic target subset only when possible.
        chosen = set(sorted(source_ids)[:target])
    states = _state_rows(state_store, "poi", chosen)
    result = []
    for poi_id in sorted(chosen):
        state = {key: value for key, value in states[poi_id].items() if key in POI_FIELDS}
        catalog_row = by_id[poi_id]
        state.update({"poi_id": poi_id, "node_id": catalog_row["node_id"],
                      "category": catalog_row["category"]})
        state["capacity_people"] = max(1.0, float(state["capacity_people"]) * capacity_scale)
        state["service_people_per_slot"] = max(
            1.0, float(state["service_people_per_slot"]) * service_scale
        )
        state["occupancy"] = 0.10 * state["capacity_people"]
        state["waiting"] = 0.0
        state["arrivals"] = state["departures"] = state["admitted"] = 0.0
        state["wait_time_min"] = 0.0
        state["arrival_history"] = [0.0] * 4
        state["departure_history"] = [0.0] * 4
        result.append(state)
    return result


def _build_scene(source: dict, source_rows: list[dict], catalog: list[dict],
                 static_root: Path, state_store: Path,
                 condition: str, split: str, seed: int, request_count: int,
                 request_slots: int, simulation_slots: int, target_pois: int,
                 target_edges: int, target_nodes: int, background_profile: Path,
                 regions: set[int], source_start_slot: int,
                 road_template: list[dict] | None = None) -> dict:
    rows = _sample_request_window(
        source_rows, regions, 0, request_slots, request_count,
        seed, source_start_slot, short_routes=True,
    )
    for index, row in enumerate(rows):
        row["request_id"] = f"scaled-{condition}-{split}-{seed}-{index}"
    scene = copy.deepcopy(source)
    scene["requests"] = rows
    if road_template is None:
        route_edge_ids = {
            int(edge)
            for row in rows
            for candidate in (row.get("candidates") or [row])
            for edge in candidate.get("edge_ids", []) + candidate.get("egress_edge_ids", [])
        }
        route_roads = [row for row in source["roads"] if int(row["edge_id"]) in route_edge_ids]
        road_template = _expand_roads(
            route_roads, static_root, state_store, target_edges, target_nodes, 0.12
        )
    scene["roads"] = copy.deepcopy(road_template)
    used_pois = {int(candidate["destination_poi_id"])
                 for row in rows for candidate in (row.get("candidates") or [row])}
    # Keep every candidate target in the scene.  The four-region source has 46
    # request POIs, so all requested catalog sizes remain feasible.
    if len(used_pois) > target_pois:
        raise ValueError(f"condition {condition} uses {len(used_pois)} POIs, target is {target_pois}")
    scene["pois"] = _expand_pois(
        [row for row in source["pois"] if int(row["poi_id"]) in used_pois],
        used_pois, catalog, state_store, target_pois,
        seed + ord(condition), 0.45, 0.45,
    )
    scene["steps"] = simulation_slots
    scene["start_slot"] = 0
    scene["request_window_slots"] = request_slots
    scene["observation_tail_slots"] = simulation_slots - request_slots
    scene["condition"] = f"scenario_{condition}"
    scene["dynamics_mode"] = "event_driven"
    scene["allow_over_capacity"] = True
    scene["policy_use_forecast"] = True
    scene["background_profile_file"] = str(background_profile.resolve())
    scene["provenance"] = {
        "source": "NYC real core request templates with scaled route snapshot",
        "split": split, "seed": seed, "scenario": condition,
        "condition_label": CONDITION_LABELS[condition],
        "regions": sorted(regions), "requests": request_count,
        "request_window_slots": request_slots,
        "simulation_slots": simulation_slots,
        "selected_edges": len(scene["roads"]),
        "selected_nodes": len({int(row["from_node"]) for row in scene["roads"]} |
                              {int(row["to_node"]) for row in scene["roads"]}),
        "selected_pois": len(scene["pois"]),
        "target_edges": target_edges, "target_nodes": target_nodes,
        "target_pois": target_pois, "short_route_completion": "2-4 slots",
        "background": "lower-amplitude periodic profile",
    }
    # Apply bottleneck, delayed demand, and shock interventions after the
    # scaled network is present, so they cover the full 7,500-link snapshot.
    _apply_condition(scene, condition, list(range(simulation_slots)), seed, True)
    # The SQLite profile is authoritative at runtime.  The helper also emits
    # legacy inline observations; dropping them keeps each 7,500-link scene
    # compact and prevents duplicate background data in MPC signatures.
    scene.pop("profile_observations", None)
    scene["cost_normalization"] = {
        "mode": "queue_wait", "road_weight": 1.0, "poi_weight": 1.0,
        "road_reference_vehicle_min": max(
            1.0, sum(float(row["capacity_veh_per_slot"]) for row in scene["roads"])
            * simulation_slots * 15.0,
        ),
        "poi_reference_person_min": max(
            1.0, sum(float(row["service_people_per_slot"]) for row in scene["pois"])
            * simulation_slots * 15.0,
        ),
        "reference_basis": "scaled_network_capacity_minutes",
    }
    return scene


def build(dataset: Path, output: Path, catalog_path: Path, static_root: Path,
          state_store: Path, background_profile: Path, regions: set[int],
          request_count: int, request_slots: int, simulation_slots: int,
          poi_counts: dict[str, int], target_edges: int, target_nodes: int,
          seeds: dict[str, int]) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    if simulation_slots < request_slots:
        raise ValueError("simulation slots must cover the request window")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "regions": sorted(regions), "request_count": request_count,
        "request_slots": request_slots, "simulation_slots": simulation_slots,
        "poi_counts": poi_counts, "road_links": target_edges,
        "road_nodes": target_nodes, "background_profile": str(background_profile.resolve()),
        "condition_labels": CONDITION_LABELS,
        "scenarios": {},
    }
    for split, seed in seeds.items():
        split_road_template = None
        for condition in SCENARIOS:
            source_path = _source_path(dataset, split, seed)
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source_rows = source["requests"]
            scene = _build_scene(
                source, source_rows, _read_catalog(catalog_path, regions), static_root,
                state_store, condition, split, seed, request_count, request_slots,
                simulation_slots, poi_counts[condition], target_edges, target_nodes,
                background_profile, regions, int(source.get("start_slot", 28)),
                split_road_template,
            )
            if split_road_template is None:
                split_road_template = copy.deepcopy(scene["roads"])
            path = output / f"scenario_{condition}" / split / f"seed_{seed}" / "scenario.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")
            compatible = output / "dataset" / split / f"scenario_{condition}" / f"seed_{seed}"
            compatible.parent.mkdir(parents=True, exist_ok=True)
            if not compatible.exists():
                os.symlink(os.path.relpath(path.parent, compatible.parent), compatible,
                           target_is_directory=True)
            manifest["scenarios"].setdefault(condition, {})[split] = str(path.resolve())
            print(f"[scaled] {condition}/{split}: requests={len(scene['requests'])} "
                  f"roads={len(scene['roads'])} nodes={scene['provenance']['selected_nodes']} "
                  f"pois={len(scene['pois'])}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--static-root", type=Path,
                        default=here.parent / "experiment_data/nyc_v4_final")
    parser.add_argument("--state-store", type=Path, default=here / "state_store")
    parser.add_argument("--background-profile", type=Path, required=True)
    parser.add_argument("--regions", nargs=4, type=int, default=[0, 1, 2, 3])
    parser.add_argument("--requests", type=int, default=15000)
    parser.add_argument("--request-slots", type=int, default=112)
    parser.add_argument("--simulation-slots", type=int, default=128)
    parser.add_argument("--poi-counts", nargs=4, type=int, default=[100, 200, 300, 500])
    parser.add_argument("--road-links", type=int, default=7500)
    parser.add_argument("--road-nodes", type=int, default=3000,
                        help="target minimum; route closure may require additional nodes")
    parser.add_argument("--train-seed", type=int, default=9301)
    parser.add_argument("--validation-seed", type=int, default=9302)
    parser.add_argument("--test-seed", type=int, default=9303)
    args = parser.parse_args()
    if len(set(args.poi_counts)) != 4 or any(value < 1 for value in args.poi_counts):
        parser.error("poi-counts must contain four positive values")
    if args.request_slots < 1 or args.simulation_slots < args.request_slots:
        parser.error("simulation-slots must be >= request-slots")
    poi_counts = dict(zip(SCENARIOS, args.poi_counts))
    build(
        args.dataset, args.output, args.catalog, args.static_root, args.state_store,
        args.background_profile, set(args.regions), args.requests, args.request_slots,
        args.simulation_slots, poi_counts, args.road_links, args.road_nodes,
        {"train": args.train_seed, "validation": args.validation_seed,
         "test": args.test_seed},
    )


if __name__ == "__main__":
    main()
