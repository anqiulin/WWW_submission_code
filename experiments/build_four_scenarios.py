"""Build four scenarios from the ten-core eight-slot request dataset.

The builder reuses the already generated candidate routes and background SQLite
profile. It only samples requests and applies controlled interventions, so no
route search is repeated during training or evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path


SCENARIOS = ("A", "B", "C", "D")
# Keep the scenario ladder explicit: A is the concentrated baseline, while
# later scenarios expose progressively more of the POI catalog.
SCENARIO_POI_COUNTS = {"A": 20, "B": 40, "C": 60, "D": 100}
DEFAULT_SEEDS = {"train": 9301, "validation": 9302, "test": 9303}
SOURCE_SLOTS = 8


def log(message: str, started: float | None = None) -> None:
    suffix = f" elapsed={time.perf_counter() - started:.2f}s" if started else ""
    print(f"[four-scenarios] {message}{suffix}", flush=True)


def _source_path(dataset: Path, split: str, seed: int) -> Path:
    paths = sorted(dataset.glob(f"{split}/**/seed_{seed}/scenario.json"))
    if not paths:
        raise FileNotFoundError(f"no {split} scenario for seed {seed} under {dataset}")
    return paths[0]


def _route_score(row: dict) -> tuple[int, int]:
    """Return the shortest access and total route edge counts for a request."""
    candidates = row.get("candidates") or [row]
    scores = [(
        len(candidate.get("edge_ids", [])),
        len(candidate.get("edge_ids", [])) + len(candidate.get("egress_edge_ids", [])),
    ) for candidate in candidates]
    return min(scores) if scores else (10**9, 10**9)


def _shorten_request(row: dict, candidate_limit: int = 3) -> dict:
    """Keep short access alternatives and make completion occur at the POI.

    The event-driven simulator schedules access in 2-4 slots. Removing the
    optional return leg and dwell period makes that access arrival the request
    completion event, so the final observation tail can drain all requests.
    """
    candidates = list(row.get("candidates") or [row])
    candidates.sort(key=lambda item: (
        len(item.get("edge_ids", [])),
        len(item.get("edge_ids", [])) + len(item.get("egress_edge_ids", [])),
        str(item.get("label", "")),
    ))
    candidates = [copy.deepcopy(item) for item in candidates[:max(1, candidate_limit)]]
    for candidate in candidates:
        candidate["egress_edge_ids"] = []
        candidate["final_destination_node_id"] = None
        candidate["dwell_slots"] = 0
    row["candidates"] = candidates
    row["edge_ids"] = list(candidates[0].get("edge_ids", []))
    row["egress_edge_ids"] = []
    row["final_destination_node_id"] = None
    row["dwell_slots"] = 0
    return row


def _sample_requests(rows: list[dict], regions: set[int], slots: list[int], count: int,
                     seed: int, short_routes: bool = False) -> list[dict]:
    eligible = [row for row in rows if int(row.get("region_id", -1)) in regions
                and int(row["departure_slot"]) in slots]
    by_slot_region: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in eligible:
        by_slot_region[(int(row["departure_slot"]), int(row["region_id"]))].append(row)
    keys = sorted(by_slot_region)
    if len(eligible) < count or not keys:
        raise ValueError(f"only {len(eligible)} eligible requests; need {count}")
    rng = random.Random(seed)
    selected: list[dict] = []
    # Balanced allocation keeps every time slot and each selected region active.
    target_by_slot = {slot: count // len(slots) for slot in slots}
    for slot in slots[:count % len(slots)]:
        target_by_slot[slot] += 1
    for slot in slots:
        region_keys = [key for key in keys if key[0] == slot]
        base = target_by_slot[slot] // len(regions)
        remainder = target_by_slot[slot] % len(regions)
        for index, region in enumerate(sorted(regions)):
            amount = base + (index < remainder)
            pool = by_slot_region.get((slot, region), [])
            if len(pool) < amount:
                raise ValueError(f"slot {slot}, region {region} has only {len(pool)} requests")
            if short_routes:
                # Prefer the shortest access routes while retaining a seeded
                # tie-break so repeated builds remain reproducible.
                ranked = sorted(pool, key=lambda row: (
                    _route_score(row), rng.random(), str(row.get("request_id", ""))))
                selected.extend(ranked[:amount])
            else:
                selected.extend(rng.sample(pool, amount))
    if len(selected) != count:
        raise AssertionError(f"sampled {len(selected)} requests, expected {count}")
    return [copy.deepcopy(row) for row in selected]


def _sample_request_window(rows: list[dict], regions: set[int], start_slot: int,
                           slots: int, count: int, seed: int,
                           source_start_slot: int, short_routes: bool = False) -> list[dict]:
    if slots < 1 or count < slots:
        raise ValueError("request window must have at least one request per slot on average")
    selected = []
    # Preserve the source dataset's eight-slot rhythm. First distribute the
    # total across complete source cycles, then balance each cycle across its
    # slots. This keeps demand balanced without drifting between cycles.
    cycle_count, cycle_remainder = divmod(slots, SOURCE_SLOTS)
    if cycle_remainder:
        raise ValueError("request window slots must be a multiple of eight")
    cycle_base, cycle_extra = divmod(count, cycle_count)
    for offset in range(slots):
        cycle_index, cycle_offset = divmod(offset, SOURCE_SLOTS)
        this_cycle_count = cycle_base + (cycle_index < cycle_extra)
        slot_base, slot_extra = divmod(this_cycle_count, SOURCE_SLOTS)
        amount = slot_base + (cycle_offset < slot_extra)
        if not amount:
            continue
        source_slot = source_start_slot + (offset % SOURCE_SLOTS)
        batch = _sample_requests(rows, regions, [source_slot], amount,
                                 seed + offset * 10_000, short_routes=short_routes)
        for row in batch:
            row["departure_slot"] = start_slot + offset
            if short_routes:
                _shorten_request(row)
        selected.extend(batch)
    return selected


def _select_pois(source_pois: list[dict], used_pois: set[int], condition: str,
                 seed: int) -> list[dict]:
    """Retain request targets and deterministically fill the scenario catalog.

    Requests must never point at a POI omitted from the generated scene. The
    remaining slots are filled from the source catalog using a seeded order so
    repeated builds are stable while different scenario/split seeds can use
    different non-target POIs.
    """
    target_count = SCENARIO_POI_COUNTS[condition]
    by_id = {int(poi["poi_id"]): poi for poi in source_pois}
    if len(by_id) != len(source_pois):
        raise ValueError("source catalog contains duplicate POI IDs")
    missing = sorted(used_pois - set(by_id))
    if missing:
        raise ValueError(
            f"scenario {condition} requests reference missing POIs: {missing[:5]}"
        )
    if len(by_id) < target_count:
        raise ValueError(
            f"scenario {condition} requires {target_count} POIs, "
            f"but source catalog has only {len(by_id)}"
        )
    if len(used_pois) > target_count:
        raise ValueError(
            f"scenario {condition} requires {target_count} POIs, "
            f"but sampled requests use {len(used_pois)}"
        )

    selected_ids = set(used_pois)
    candidates = [poi for poi in source_pois if int(poi["poi_id"]) not in selected_ids]
    rng = random.Random(seed + ord(condition))
    rng.shuffle(candidates)
    selected_ids.update(int(poi["poi_id"]) for poi in candidates[:target_count - len(selected_ids)])
    # Return source records in a stable catalog order rather than the shuffled
    # selection order; this keeps serialized scenes and diffs reproducible.
    return [poi for poi in source_pois if int(poi["poi_id"]) in selected_ids]


def _category_ratio(category: int, offset: int) -> float:
    curves = (
        (0.70, 0.73, 0.76, 0.74, 0.78, 0.76, 0.72, 0.70),
        (0.70, 0.72, 0.74, 0.78, 0.76, 0.73, 0.71, 0.70),
        (0.70, 0.71, 0.73, 0.75, 0.80, 0.78, 0.74, 0.70),
    )
    return curves[category % len(curves)][offset % 8]


def _background_observations(scenario: dict, condition: str, slots: list[int],
                             bottleneck_edges: set[int], shock_category: int | None) -> list[dict]:
    roads = scenario["roads"]
    pois = scenario["pois"]
    observations = []
    normal_profile = (0.80, 0.82, 0.84, 0.83, 0.81, 0.85, 0.82, 0.84)
    cycle_end_pulses = len(slots) >= 128
    for offset, slot in enumerate(slots):
        # The 32-hour benchmark uses a 95% pulse at each 16-slot cycle end;
        # shorter legacy windows retain their original profile for compatibility.
        road_ratio = (0.95 if cycle_end_pulses and offset % 16 == 15
                      else normal_profile[offset % len(normal_profile)])
        # Scenario D retains its existing isolated road shock in addition to
        # the common four-hour cycle-end pulse.
        if condition == "D" and offset == 4:
            road_ratio = 0.95
        road_background = {}
        for road in roads:
            road_background[str(road["edge_id"])] = road_ratio * road["capacity_veh_per_slot"]
        poi_arrivals = {}
        for poi in pois:
            ratio = _category_ratio(int(poi["category"]), offset)
            if condition == "D" and offset == 4 and int(poi["category"]) == shock_category:
                ratio = 1.50
            poi_arrivals[str(poi["poi_id"])] = ratio * poi["service_people_per_slot"]
        observations.append({
            "slot": slot,
            "road_background": road_background,
            "poi_arrivals": poi_arrivals,
            "poi_departures": {
                str(poi["poi_id"]): 0.12 * poi["capacity_people"] for poi in pois
            },
        })
    return observations


def _apply_condition(scenario: dict, condition: str, slots: list[int], seed: int,
                     fixed_requests_per_slot: bool = False) -> dict:
    roads = scenario["roads"]
    pois = scenario["pois"]
    request_rows = scenario["requests"]
    edge_counts = Counter(edge for row in request_rows for candidate in (row.get("candidates") or [row])
                           for edge in candidate.get("edge_ids", []))
    bottlenecks = {edge for edge, _ in edge_counts.most_common(max(1, len(edge_counts) // 100))}
    if condition == "C":
        # A common access edge and one popular POI create delayed, concentrated arrivals.
        target_poi = Counter(int(row["destination_poi_id"]) for row in request_rows).most_common(1)[0][0]
        target_rows = [row for row in request_rows if int(row["destination_poi_id"]) == target_poi]
        if target_rows:
            delay_edges = Counter(edge for row in target_rows for edge in row.get("edge_ids", [])).most_common(2)
            bottlenecks = {edge for edge, _ in delay_edges} or bottlenecks
            for cycle_start in range(0, len(slots), SOURCE_SLOTS):
                cycle_targets = [row for row in target_rows
                                 if slots[cycle_start] <= row["departure_slot"]
                                 < slots[cycle_start] + SOURCE_SLOTS]
                if fixed_requests_per_slot:
                    peak_slots = set(slots[cycle_start + 2:cycle_start + 5])
                    donors = [row for row in request_rows
                              if row["departure_slot"] in peak_slots
                              and int(row["destination_poi_id"]) != target_poi]
                    random.Random(seed + cycle_start).shuffle(donors)
                    for target, donor in zip(
                            (row for row in cycle_targets
                             if row["departure_slot"] not in peak_slots), donors):
                        target["departure_slot"], donor["departure_slot"] = (
                            donor["departure_slot"], target["departure_slot"])
                else:
                    for index, row in enumerate(cycle_targets[:240]):
                        row["departure_slot"] = slots[cycle_start + 2 + (index % 3)]
        scenario["provenance"]["delayed_target_poi"] = target_poi
    for road in roads:
        if condition in ("B", "C") and int(road["edge_id"]) in bottlenecks:
            road["capacity_veh_per_slot"] = max(1.0, 0.55 * road["capacity_veh_per_slot"])
            road["speed_limit_kmh"] = max(5.0, 0.75 * road["speed_limit_kmh"])
    categories = Counter(int(poi["category"]) for poi in pois)
    shock_category = categories.most_common(1)[0][0] if categories else None
    scenario["profile_observations"] = _background_observations(
        scenario, condition, slots, bottlenecks, shock_category)
    # Keep the serialized initial state consistent with the first observed
    # slot.  Runtime updates still use profile_observations for each slot.
    first_background = scenario["profile_observations"][0]["road_background"]
    for road in roads:
        first_flow = float(first_background[str(road["edge_id"])])
        road["background_vehicles"] = first_flow
        road["count_history"] = [first_flow] * 4
        road["queue_vehicles"] = 0.0
        road["outflow_vehicles"] = 0.0
    for poi in pois:
        poi["occupancy"] = 0.75 * poi["capacity_people"]
        poi["waiting"] = 0.0
    scenario["provenance"].update({
        "bottleneck_edges": sorted(bottlenecks),
        "shock_category": shock_category if condition == "D" else None,
        "background_target": "road capacity ratio 0.80-0.85, with 0.95 at each 16-slot cycle end; POI category arrival ratio 0.70-0.80",
    })
    return scenario


def build(dataset: Path, output: Path, regions: set[int], request_count: int,
          slots: int, start_slot: int, seeds: dict[str, int],
          observation_tail: int = 0, fixed_requests_per_slot: bool = False,
          short_routes: bool = False) -> dict:
    started = time.perf_counter()
    if len(regions) != 4:
        raise ValueError("exactly four regions are required")
    if slots < 1 or request_count < 1:
        raise ValueError("this benchmark requires a positive request count and request window")
    if start_slot < 0 or observation_tail < 0:
        raise ValueError("start slot and observation tail must be nonnegative")
    slot_ids = list(range(start_slot, start_slot + slots + observation_tail))
    manifest = {"regions": sorted(regions), "request_count": request_count,
                "slots": slots, "observation_tail": observation_tail,
                "start_slot": start_slot, "fixed_requests_per_slot": fixed_requests_per_slot,
                "short_routes": short_routes, "poi_counts": SCENARIO_POI_COUNTS.copy(),
                "scenarios": {}}
    for condition in SCENARIOS:
        for split, seed in seeds.items():
            phase = time.perf_counter()
            source = _source_path(dataset, split, seed)
            log(f"loading {condition}/{split}: {source}")
            source_scene = json.loads(source.read_text(encoding="utf-8"))
            source_start_slot = int(source_scene.get("start_slot", 28))
            rows = _sample_request_window(source_scene["requests"], regions, start_slot,
                                          slots, request_count, seed + ord(condition),
                                          source_start_slot, short_routes=short_routes)
            for index, row in enumerate(rows):
                row["request_id"] = f"scenario-{condition}-{split}-{seed}-{index}"
            used_edges = {edge for row in rows for candidate in (row.get("candidates") or [row])
                          for edge in candidate.get("edge_ids", []) + candidate.get("egress_edge_ids", [])}
            used_pois = {int(candidate["destination_poi_id"])
                         for row in rows for candidate in (row.get("candidates") or [row])}
            scene = copy.deepcopy(source_scene)
            scene["requests"] = rows
            scene["roads"] = [road for road in source_scene["roads"] if int(road["edge_id"]) in used_edges]
            scene["pois"] = _select_pois(source_scene["pois"], used_pois, condition, seed)
            selected_pois = len(scene["pois"])
            scene["steps"] = slots + observation_tail
            scene["start_slot"] = start_slot
            scene["request_window_slots"] = slots
            scene["observation_tail_slots"] = observation_tail
            scene["condition"] = f"scenario_{condition}"
            scene["dynamics_mode"] = "event_driven"
            scene["provenance"] = {"source": str(source), "split": split, "seed": seed,
                                    "scenario": condition, "regions": sorted(regions),
                                    "requests": request_count, "request_window_slots": slots,
                                    "observation_tail_slots": observation_tail,
                                    "short_routes": short_routes,
                                    "selected_edges": len(used_edges),
                                    "selected_pois": selected_pois,
                                    "requested_pois": SCENARIO_POI_COUNTS[condition],
                                    "request_target_pois": len(used_pois)}
            scene = _apply_condition(scene, condition, slot_ids, seed,
                                     fixed_requests_per_slot)
            if fixed_requests_per_slot:
                by_slot = Counter(int(row["departure_slot"]) for row in scene["requests"])
                cycle_count, cycle_remainder = divmod(slots, SOURCE_SLOTS)
                if cycle_remainder:
                    raise ValueError("request window slots must be a multiple of eight")
                cycle_base, cycle_extra = divmod(request_count, cycle_count)
                expected_counts = {}
                for offset, slot in enumerate(slot_ids[:slots]):
                    cycle_index, cycle_offset = divmod(offset, SOURCE_SLOTS)
                    this_cycle_count = cycle_base + (cycle_index < cycle_extra)
                    slot_base, slot_extra = divmod(this_cycle_count, SOURCE_SLOTS)
                    expected_counts[slot] = slot_base + (cycle_offset < slot_extra)
                if by_slot != expected_counts:
                    raise AssertionError("request count changed after scenario intervention")
            # Use capacities after scenario-specific interventions.
            scene["cost_normalization"] = {
                "mode": "queue_wait", "road_weight": 1.0, "poi_weight": 1.0,
                "road_reference_vehicle_min": max(
                    1.0, sum(float(road["capacity_veh_per_slot"]) for road in scene["roads"])
                    * scene["steps"] * 15.0),
                "poi_reference_person_min": max(
                    1.0, sum(float(poi["service_people_per_slot"]) for poi in scene["pois"])
                    * scene["steps"] * 15.0),
                "reference_basis": "selected_network_capacity_minutes",
            }
            scene["background_profile_file"] = source_scene["background_profile_file"]
            destination = output / f"scenario_{condition}" / split / f"seed_{seed}" / "scenario.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")
            # Keep the layout expected by train_full_ppo.py without duplicating
            # the large scenario files.
            compatible = output / "dataset" / split / f"scenario_{condition}" / f"seed_{seed}"
            compatible.parent.mkdir(parents=True, exist_ok=True)
            if compatible.is_symlink() and compatible.resolve() == destination:
                compatible.unlink()
            if not compatible.exists():
                os.symlink(os.path.relpath(destination.parent, compatible.parent), compatible,
                           target_is_directory=True)
            manifest["scenarios"].setdefault(condition, {})[split] = str(destination)
            log(f"wrote {condition}/{split}: requests={len(rows)}, roads={len(used_edges)}, pois={selected_pois}", phase)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"completed all four scenarios", started)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", nargs=4, type=int, default=[0, 1, 2, 3])
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--observation-tail", type=int, default=0)
    parser.add_argument("--start-slot", type=int, default=28)
    parser.add_argument("--fixed-requests-per-slot", action="store_true",
                        help="balance counts across slots; when requests/slots is fractional, counts differ by at most one")
    parser.add_argument("--short-routes", action="store_true",
                        help="prefer shortest access candidates and complete requests at the POI in 2-4 slots")
    parser.add_argument("--train-seed", type=int, default=9301)
    parser.add_argument("--validation-seed", type=int, default=9302)
    parser.add_argument("--test-seed", type=int, default=9303)
    args = parser.parse_args()
    seeds = {"train": args.train_seed, "validation": args.validation_seed, "test": args.test_seed}
    build(args.dataset, args.output, set(args.regions), args.requests, args.slots,
          args.start_slot, seeds, args.observation_tail, args.fixed_requests_per_slot,
          args.short_routes)


if __name__ == "__main__":
    main()
