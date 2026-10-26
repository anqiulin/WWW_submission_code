"""Prepare regional NYC scenes, train PPO, and compare recommendation methods."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_road_sim.experiment import (ExperimentConfig, collect_mpc_expert_dataset,
    run_experiment, train_ppo_scenarios, _policy_feature_count)
from poi_road_sim.torch_policy import PPOCandidatePolicy

CONDITIONS = ("regional_delayed", "regional_coupling", "regional_shock")
STRATEGIES = ("ppo", "rise_mpc", "scair", "shortest_distance", "shortest_queue", "random")


def _scenario_files(root: Path, split: str) -> list[Path]:
    return sorted((root / "dataset" / split).glob("*/seed_*/scenario.json"))


def _calibrate_training_references(output: Path, model_dir: Path, train: list[Path],
                                   all_paths: list[Path]) -> None:
    """Set fixed road/POI reference scales using training scenes only."""
    calibration_file = output / "cost_calibration.json"
    if calibration_file.exists():
        calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    else:
        pilot = []
        for path in train:
            scene = json.loads(path.read_text(encoding="utf-8"))
            report = run_experiment(scene, model_dir,
                output / "calibration" / scene["condition"],
                ExperimentConfig(("shortest_distance",), 1, 1.0, 0))
            item = report["summary"][0]
            pilot.append({"condition": scene["condition"],
                "raw_road_vehicle_min": item["cumulative_raw_road_vehicle_min"],
                "raw_poi_person_min": item["cumulative_raw_poi_person_min"]})
        calibration = {"reference_policy": "shortest_distance on training split only",
            "training_pilot": pilot,
            "road_reference_vehicle_min": max(1.0, sum(x["raw_road_vehicle_min"] for x in pilot) / len(pilot)),
            "poi_reference_person_min": max(1.0, sum(x["raw_poi_person_min"] for x in pilot) / len(pilot)),
            "road_weight": 1.0, "poi_weight": 1.0,
            "interpretation": "a unit cost is the training baseline's mean 24-hour component"}
        calibration_file.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    weights = {key: calibration[key] for key in
               ("road_reference_vehicle_min", "poi_reference_person_min", "road_weight", "poi_weight")}
    for path in all_paths:
        scene = json.loads(path.read_text(encoding="utf-8"))
        if scene.get("cost_normalization") != weights:
            scene["cost_normalization"] = weights
            path.write_text(json.dumps(scene), encoding="utf-8")


def _route_mean_top_k(scene: dict) -> int:
    total = count = 0
    for request in scene["requests"]:
        for candidate in request.get("candidates", [request]):
            total += len(candidate["edge_ids"]) + len(candidate.get("egress_edge_ids", []))
            count += 1
    return max(1, round(total / count)) if count else 1


def _positive_upper_quartile(values: list[float]) -> float:
    positive = sorted(value for value in values if value > 0)
    return max(1.0, positive[math.ceil(0.75 * len(positive)) - 1]) if positive else 1.0


def _calibrate_bounded_topk(output: Path, model_dir: Path, train: list[Path],
                            all_paths: list[Path]) -> dict:
    """Fix scene-local K from routes and shared scales from training rollouts."""
    for path in all_paths:
        scene = json.loads(path.read_text(encoding="utf-8"))
        k = _route_mean_top_k(scene)
        current = scene.get("cost_normalization", {})
        if current.get("mode") != "bounded_topk" or current.get("road_top_k") != k:
            scene["cost_normalization"] = {"mode": "bounded_topk", "road_top_k": k,
                "road_reference_vehicle_min": 1.0, "poi_reference_person_min": 1.0,
                "road_weight": 1.0, "poi_weight": 1.0}
            path.write_text(json.dumps(scene), encoding="utf-8")
    calibration_file = output / "cost_calibration.json"
    if calibration_file.exists():
        calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
        if calibration.get("mode") != "bounded_topk":
            raise ValueError(f"cost mode mismatch in {calibration_file}; use a new output directory")
    else:
        road_values, poi_values = [], []
        for path in train:
            scene = json.loads(path.read_text(encoding="utf-8"))
            destination = output / "calibration" / scene["condition"]
            print(f"[calibration] {scene['condition']} shortest-distance baseline", flush=True)
            run_experiment(scene, model_dir, destination,
                ExperimentConfig(("shortest_distance",), 1, 1.0, 0))
            with (destination / "timeseries.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    road_values.append(float(row["raw_topk_road_vehicle_min"]))
                    poi_values.append(float(row["raw_poi_person_min"]))
        calibration = {"mode": "bounded_topk", "road_weight": 1.0, "poi_weight": 1.0,
            "road_reference_vehicle_min": _positive_upper_quartile(road_values),
            "poi_reference_person_min": _positive_upper_quartile(poi_values),
            "reference_policy": "shortest_distance, training split only",
            "reference_statistic": "75th percentile of positive per-slot raw costs; fallback 1",
            "road_top_k_rule": "nearest integer mean candidate complete-route edge count, per scenario",
            "bounded_formula": "raw / (raw + reference), separately for road and POI",
            "training_slots": len(road_values)}
        calibration_file.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    for path in all_paths:
        scene = json.loads(path.read_text(encoding="utf-8"))
        normalization = scene["cost_normalization"]
        for key in ("road_reference_vehicle_min", "poi_reference_person_min"):
            normalization[key] = calibration[key]
        path.write_text(json.dumps(scene), encoding="utf-8")
    return calibration


def _write_comparison(output: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with (output / "per_scenario.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["strategy"]), []).append(row)
    episode_steps = sorted({int(row["steps"]) for row in rows})
    steps_label = str(episode_steps[0]) if len(episode_steps) == 1 else str(episode_steps)
    calibration_path = output / "cost_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    mode = calibration.get("mode", "vehicle_min")
    description = ("Per-slot road and POI components are separately bounded below 1; cumulative totals span all slots."
                   if mode == "bounded_topk" else
                   "Road and POI loads use the scenario reference scales; synthetic regional demand is not observed data.")
    lines = [f"Regional NYC benchmark ({mode})", "",
             f"Values aggregate {steps_label} 15-minute slots per episode; lower load is better.",
             description,
             "", "condition             strategy          total_load       road_load        poi_load      runtime_sec"]
    for (condition, strategy), values in sorted(grouped.items()):
        mean = lambda key: sum(float(v[key]) for v in values) / len(values)
        lines.append(f"{condition:<21} {strategy:<17} {mean('cumulative_global_total_load'):>14.6f} "
                     f"{mean('cumulative_global_road_load'):>14.6f} {mean('cumulative_global_poi_load'):>14.6f} "
                     f"{mean('runtime_sec'):>12.3f}")
    lines += ["", "Metrics per row include cumulative road/POI/total load, road delay, POI queue,",
              "first congestion onset slots, congestion duration, incremental load, and runtime.",
              "", "condition             strategy          road first  road min  poi first  poi min"]
    for (condition, strategy), values in sorted(grouped.items()):
        first_road = [int(v["first_road_congestion_slot"]) for v in values
                      if v.get("first_road_congestion_slot") not in (None, "")]
        first_poi = [int(v["first_poi_congestion_slot"]) for v in values
                     if v.get("first_poi_congestion_slot") not in (None, "")]
        road = f"{min(first_road) // 4:02d}:{(min(first_road) % 4) * 15:02d}" if first_road else "-"
        poi = f"{min(first_poi) // 4:02d}:{(min(first_poi) % 4) * 15:02d}" if first_poi else "-"
        mean = lambda key: sum(float(v.get(key, 0)) for v in values) / len(values)
        lines.append(f"{condition:<21} {strategy:<17} {str(road):>10} "
                     f"{mean('road_congestion_duration_min'):>9.1f} {str(poi):>10} "
                     f"{mean('poi_congestion_duration_min'):>8.1f}")
    (output / "experiment_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _augment_metrics(result_dir: Path, summaries: list[dict]) -> list[dict]:
    series = {}
    with (result_dir / "timeseries.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            series.setdefault(row["strategy"], []).append(row)
    output = []
    for summary in summaries:
        rows = series.get(summary["strategy"], [])
        road_first = summary.get("first_road_congestion_slot")
        poi_first = summary.get("first_poi_congestion_slot")
        if road_first is None:
            road_first = next((int(r["slot"]) for r in rows if float(r["active_road_edges"]) > 0), None)
        if poi_first is None:
            poi_first = next((int(r["slot"]) for r in rows if float(r.get("congested_pois", 0)) > 0), None)
        increment = sum(float(r["incremental_global_total_load"]) for r in rows)
        n = max(1, int(summary["requests"]))
        row = dict(summary)
        row.update({"first_road_congestion_slot": road_first,
                    "first_poi_congestion_slot": poi_first,
                    "recommendation_future_load_increment": increment,
                    "increment_per_recommendation": increment / n,
                    "mean_online_ms_per_recommendation": float(summary["runtime_sec"]) * 1000.0 / n})
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=here / "artifacts")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=here.parent / "experiment_data/nyc_v4_final")
    parser.add_argument("--routing-index", type=Path, default=here / "routing_index_calibrated")
    parser.add_argument("--state-store", type=Path, default=here / "state_store")
    parser.add_argument("--slots", type=int, default=96)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout-size", type=int, default=2048)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="12 slots, one condition, one PPO update")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--test-only", action="store_true",
                        help="reuse an existing dataset/checkpoint and only run test comparison")
    parser.add_argument("--checkpoint", type=Path,
                        help="PPO checkpoint for --test-only; defaults to output/global_policy_ppo_regional.pt")
    parser.add_argument("--profile", choices=("legacy", "predictive_v2", "predictive_v3"), default="legacy",
                        help="predictive_v2 adds delayed demand and avoidable road bottlenecks")
    parser.add_argument("--cost-mode", choices=("vehicle_min", "bounded_topk"), default="vehicle_min",
                        help="bounded_topk uses per-scene mean route length and training-only bounded scales")
    parser.add_argument("--source-dataset", type=Path,
                        help="reuse an existing verified routed dataset when creating predictive_v2 scenes")
    args = parser.parse_args()
    output = args.output.resolve(); dataset = output / "dataset"
    if args.source_dataset and args.source_dataset.resolve() == dataset:
        raise ValueError("source dataset and output dataset must differ to preserve the original scenes")
    slots = 12 if args.smoke else args.slots
    conditions = ("regional_delayed",) if args.smoke else CONDITIONS
    seeds = {"train": 9101, "validation": 9102, "test": 9103}
    for split, seed in seeds.items():
        for condition in conditions:
            path = dataset / split / condition / f"seed_{seed}" / "scenario.json"
            if not path.exists():
                print(f"[dataset] {split}/{condition}", flush=True)
                source = (args.source_dataset / split / condition / f"seed_{seed}" / "scenario.json"
                          if args.source_dataset else None)
                if source is not None:
                    if not source.exists():
                        raise FileNotFoundError(f"source routed scenario not found: {source}")
                    scenario = json.loads(source.read_text(encoding="utf-8"))
                    if int(scenario["steps"]) != slots:
                        raise ValueError(f"source scenario has {scenario['steps']} slots, requested {slots}")
                    source_profile = scenario.get("profile")
                    expected_profile = f"nyc_regional_{args.profile}" if args.profile != "legacy" else None
                    if source_profile not in (None, expected_profile):
                        raise ValueError(f"source profile mismatch at {source}: {source_profile} != {expected_profile}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(scenario), encoding="utf-8")
                else:
                    from build_nyc_regional_scenarios import build_one
                    build_one(args.root, args.routing_index, args.state_store, args.catalog,
                              path, condition, seed, slots,
                              requests_scan=500 if args.smoke else 2500)
                if args.profile in ("predictive_v2", "predictive_v3") and (
                        args.source_dataset is None or source_profile is None):
                    from regional_predictive_profile import apply_profile
                    scenario = apply_profile(json.loads(path.read_text(encoding="utf-8")), args.profile)
                    path.write_text(json.dumps(scenario), encoding="utf-8")
            scenario_profile = json.loads(path.read_text(encoding="utf-8")).get("profile")
            expected_profile = f"nyc_regional_{args.profile}" if args.profile != "legacy" else None
            if scenario_profile != expected_profile:
                raise ValueError(f"dataset profile mismatch at {path}; use a separate --output directory")
    train = _scenario_files(output, "train"); validation = _scenario_files(output, "validation")
    test = _scenario_files(output, "test")
    if args.cost_mode == "bounded_topk":
        _calibrate_bounded_topk(output, args.model_dir, train, train + validation + test)
    elif any(json.loads(path.read_text(encoding="utf-8")).get("cost_normalization", {}).get("mode") == "bounded_topk"
             for path in train + validation + test):
        raise ValueError("existing dataset uses bounded_topk; pass --cost-mode bounded_topk")
    elif args.profile == "predictive_v3":
        _calibrate_training_references(output, args.model_dir, train, train + validation + test)
    profile_rows = []
    for split, paths in (("train", train), ("validation", validation), ("test", test)):
        for path in paths:
            scene = json.loads(path.read_text(encoding="utf-8"))
            provenance = scene.get("provenance", {})
            profile_rows.append({"split": split, "condition": scene["condition"],
                "seed": path.parent.name, "requests": len(scene["requests"]),
                "route_edges": len(scene["roads"]), "candidate_pois": len(scene["pois"]),
                "catalog_pois_per_region": provenance.get("catalog_pois_per_region"),
                "effective_template_regions": provenance.get("effective_template_regions",
                    int(provenance.get("region_count", 0)) -
                    int(provenance.get("templates", {}).get("template_region_fallback", 0))),
                "fallback_regions": provenance.get("fallback_region_mapping",
                    provenance.get("templates", {}).get("template_region_fallback", 0)),
                "road_reference_vehicle_min": scene.get("cost_normalization", {}).get("road_reference_vehicle_min"),
                "poi_reference_person_min": scene.get("cost_normalization", {}).get("poi_reference_person_min"),
                "cost_mode": scene.get("cost_normalization", {}).get("mode", "vehicle_min"),
                "road_top_k": scene.get("cost_normalization", {}).get("road_top_k"),
                "bottleneck_edge_ids": provenance.get("bottleneck_edge_ids", []),
                "forecasted_requests": sum("forecast_poi_id" in row for row in scene["requests"])})
    (output / "dataset_profile_report.json").write_text(json.dumps(profile_rows, indent=2), encoding="utf-8")
    if args.skip_training:
        print(json.dumps({"train": len(train), "validation": len(validation), "test": len(test),
                          "dataset": str(dataset)}, indent=2)); return
    if args.test_only:
        objective_path = output / "training_objective.json"
        checkpoint = (args.checkpoint or (output / "global_policy_ppo_regional.pt")).resolve()
        if args.cost_mode == "bounded_topk":
            objective = (json.loads(objective_path.read_text(encoding="utf-8"))
                         if objective_path.exists() else {})
            permitted = {Path(value).resolve() for value in objective.get("checkpoints", [])}
            if objective.get("cost_mode") != "bounded_topk" or checkpoint not in permitted:
                raise ValueError("PPO checkpoint is not recorded as trained for this bounded_topk objective")
        if not checkpoint.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
        if not test:
            raise FileNotFoundError(f"test scenarios not found below {dataset}")
        rows = []
        for scenario_path in test:
            scenario = json.loads(scenario_path.read_text()); scenario["ppo_checkpoint"] = str(checkpoint)
            condition = scenario["condition"]; destination = output / "results" / condition
            report = run_experiment(scenario, args.model_dir, destination,
                                    ExperimentConfig(STRATEGIES, args.horizon, 0.95, 9103))
            for summary in _augment_metrics(destination, report["summary"]):
                rows.append({"condition": condition, "seed": scenario_path.parent.name, **summary})
        _write_comparison(output, rows)
        print(json.dumps({"output": str(output), "checkpoint": str(checkpoint),
                          "result_rows": len(rows), "mode": "test-only"}, indent=2))
        return
    prep = output / "preparation"; prep.mkdir(parents=True, exist_ok=True)
    expert = []
    for path in train:
        expert.extend(collect_mpc_expert_dataset(json.loads(path.read_text()), args.model_dir, args.horizon))
    (prep / "mpc_expert_dataset.json").write_text(json.dumps(expert), encoding="utf-8")
    policy = PPOCandidatePolicy(
        feature_count=_policy_feature_count(json.loads(train[0].read_text())))
    bc_report = policy.fit_behavior_cloning(expert, epochs=args.bc_epochs)
    bc_path = prep / "ppo_bc_init.pt"; policy.save(bc_path)
    (prep / "bc_report.json").write_text(json.dumps({**bc_report, "decision_states": len(expert)}, indent=2))
    checkpoint = output / "global_policy_ppo_regional.pt"
    reports = train_ppo_scenarios(train, args.model_dir, checkpoint, epochs=10,
        updates=1 if args.smoke else args.updates, seed=42, validation_scenarios=validation,
        rollout_size=min(args.rollout_size, 256) if args.smoke else args.rollout_size,
        batch_size=64, gamma=0.95, init_model=bc_path)
    (output / "training_report.json").write_text(json.dumps(reports, indent=2))
    (output / "training_objective.json").write_text(json.dumps({"cost_mode": args.cost_mode,
        "calibration": str(output / "cost_calibration.json"),
        "checkpoints": [str(checkpoint.resolve()),
                        str(checkpoint.with_name(checkpoint.stem + ".best" + checkpoint.suffix).resolve())]}, indent=2))
    rows = []
    strategies = STRATEGIES
    for scenario_path in test:
        scenario = json.loads(scenario_path.read_text()); scenario["ppo_checkpoint"] = str(checkpoint)
        condition = scenario["condition"]; destination = output / "results" / condition
        report = run_experiment(scenario, args.model_dir, destination,
                                ExperimentConfig(strategies, args.horizon, 0.95, 9103))
        for summary in _augment_metrics(destination, report["summary"]):
            rows.append({"condition": condition, "seed": scenario_path.parent.name, **summary})
    _write_comparison(output, rows)
    (output / "manifest.json").write_text(json.dumps({"dataset": str(dataset),
        "conditions": conditions, "strategies": strategies, "train_seed": seeds["train"],
        "validation_seed": seeds["validation"], "test_seed": seeds["test"],
        "region_catalog": str(args.catalog), "catalog_pois_per_region": 2000,
        "candidate_count_per_request": 8, "checkpoint": str(checkpoint),
        "cost_mode": args.cost_mode,
        "synthetic_demand_note": "regional high-demand interventions on real NYC topology"}, indent=2))
    print(json.dumps({"output": str(output), "expert_decision_states": len(expert),
                      "checkpoint": str(checkpoint), "result_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
