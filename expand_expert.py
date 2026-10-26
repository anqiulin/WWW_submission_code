"""Expand MPC expert states with documented simulator perturbations."""
from __future__ import annotations

import argparse
import copy
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from poi_road_sim.experiment import collect_mpc_expert_dataset


def variant(base: dict, index: int) -> dict:
    """Create a counterfactual state variant, preserving topology and requests."""
    scenario = copy.deepcopy(base)
    rng = random.Random(20261012 + index)
    road_scale = (0.75, 0.9, 1.0, 1.1, 1.25)[index % 5]
    poi_scale = (0.6, 0.85, 1.0, 1.15, 1.35)[(index // 5) % 5]
    for road in scenario["roads"]:
        capacity = float(road["capacity_veh_per_slot"])
        background = min(capacity * 1.15, max(0.0, capacity * 0.35 * road_scale))
        road["count_history"] = [background] * 4
        road["background_vehicles"] = background
        road["queue_vehicles"] = 0.0
        road["recommended_vehicles"] = 0.0
    for poi in scenario["pois"]:
        capacity = float(poi["capacity_people"])
        service = max(1.0, float(poi["service_people_per_slot"]))
        poi["occupancy"] = min(capacity * 0.95, max(0.0, capacity * 0.25 * poi_scale))
        poi["waiting"] = max(0.0, service * max(0.0, poi_scale - 0.5))
        poi["arrival_history"] = [service * poi_scale] * 4
        poi["departure_history"] = [service * 0.7] * 4
    requests = list(scenario.get("requests", []))
    rng.shuffle(requests)
    scenario["requests"] = requests
    scenario.setdefault("provenance", {})["expert_variant"] = {
        "index": index, "road_scale": road_scale, "poi_scale": poi_scale,
        "request_order_seed": 20261012 + index,
    }
    return scenario


def collect_one(args):
    path, scenario, model_dir, horizon = args
    examples = collect_mpc_expert_dataset(scenario, Path(model_dir), horizon)
    for item in examples:
        item["source_scenario"] = str(path)
        item["variant"] = scenario.get("provenance", {}).get("expert_variant", {})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--variants-per-scenario", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()
    paths = [Path(p) for p in sorted(glob.glob(str(args.input / "*" / "seed_*" / "scenario.json")))]
    if not paths:
        raise ValueError("no scenario.json files found")
    jobs = []
    for path in paths:
        base = json.loads(path.read_text(encoding="utf-8"))
        jobs.append((str(path), base, str(args.model_dir), args.horizon))
        for index in range(args.variants_per_scenario):
            jobs.append((str(path), variant(base, index), str(args.model_dir), args.horizon))
    examples = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, batch in enumerate(executor.map(collect_one, jobs), 1):
            examples.extend(batch)
            if index % 10 == 0:
                print(f"scenarios={index}/{len(jobs)} expert_states={len(examples)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(examples), encoding="utf-8")
    report = {"source_scenarios": len(paths), "expanded_scenarios": len(jobs),
              "variants_per_scenario": args.variants_per_scenario,
              "expert_states": len(examples), "horizon": args.horizon,
              "road_scales": [0.75, 0.9, 1.0, 1.1, 1.25],
              "poi_scales": [0.6, 0.85, 1.0, 1.15, 1.35],
              "provenance": "MPC labels regenerated for each perturbed state"}
    args.output.with_name("expert_expansion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
