"""Improve BC on student-visited states, validating before any PPO update."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from poi_road_sim.experiment import (
    _policy_feature_count, apply_cost_mode, apply_feature_schema,
    collect_ppo_trajectory,
)
from poi_road_sim.torch_policy import PPOCandidatePolicy


def _load_scenario(path: Path, schema: str) -> dict:
    return apply_feature_schema(
        apply_cost_mode(json.loads(path.read_text(encoding="utf-8")), "queue_wait"),
        schema,
    )


def _reference_loads(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reference = {}
    for name, rows in data.items():
        reference[name] = {
            row["strategy"]: float(row["cumulative_global_total_load"])
            for row in rows
        }
        if "rise_mpc" not in reference[name] or "scair" not in reference[name]:
            raise ValueError(f"reference summary lacks MPC or SCAIR for {name}")
    return reference


def _validate(policy: PPOCandidatePolicy, paths: list[Path], model_dir: Path,
              schema: str, reference: dict[str, dict[str, float]]) -> dict:
    loads = {}
    for path in paths:
        name = path.parent.parent.parent.name
        trajectory = collect_ppo_trajectory(
            _load_scenario(path, schema), model_dir, policy, deterministic=True,
        )
        loads[name] = sum(
            float(slot["global_load"]["total_load"]) for slot in trajectory
        )
        print(f"BC validation {name}: load={loads[name]:.3f}", flush=True)
    if reference:
        missing = set(loads) - set(reference)
        if missing:
            raise ValueError(f"reference summary lacks scenarios: {sorted(missing)}")
        relative = {name: loads[name] / reference[name]["rise_mpc"] for name in loads}
        scair_relative = {name: loads[name] / reference[name]["scair"]
                          for name in loads}
        beats_scair = {
            name: loads[name] < reference[name]["scair"] for name in loads
        }
        return {
            "load_by_scenario": loads,
            "mean_mpc_ratio": sum(relative.values()) / len(relative),
            "mpc_ratio_by_scenario": relative,
            "scair_ratio_by_scenario": scair_relative,
            "worst_scair_ratio": max(scair_relative.values()),
            "beats_scair_by_scenario": beats_scair,
            "passes_scair_gate": all(beats_scair.values()),
            "passes_scair_5pct_margin": all(
                value <= 0.95 for value in scair_relative.values()),
        }
    return {"load_by_scenario": loads, "total_load": sum(loads.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--validation", nargs="+", type=Path, required=True)
    parser.add_argument("--expert-dataset", type=Path, required=True)
    parser.add_argument("--initial-dagger-dataset", type=Path,
                        help="previously collected student-state MPC labels")
    parser.add_argument("--init-model", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--feature-schema", choices=("legacy", "local_v2"),
                        default="local_v2")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--dagger-per-slot", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--future-limit", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.rounds < 1 or args.bc_epochs < 1 or args.dagger_per_slot < 1
            or args.horizon < 1 or args.future_limit < 0
            or args.learning_rate <= 0 or args.batch_size < 1):
        parser.error("rounds, epochs, labels, horizon, learning rate and batch size must be positive")
    inputs = [*args.train, *args.validation, args.expert_dataset, args.init_model]
    if args.initial_dagger_dataset is not None:
        inputs.append(args.initial_dagger_dataset)
    for path in inputs:
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.reference_summary and not args.reference_summary.is_file():
        parser.error(f"missing reference summary: {args.reference_summary}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output is not empty: {args.output}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    scenarios = [_load_scenario(path, args.feature_schema) for path in args.train]
    dimensions = {_policy_feature_count(scenario) for scenario in scenarios}
    if len(dimensions) != 1:
        parser.error("training scenarios use different feature dimensions")
    expert = json.loads(args.expert_dataset.read_text(encoding="utf-8"))
    if not expert or any(len(vector) != next(iter(dimensions))
                         for item in expert for vector in item["features"]):
        parser.error("expert dataset is empty or does not match the feature schema")
    online: list[dict] = (json.loads(args.initial_dagger_dataset.read_text(encoding="utf-8"))
                          if args.initial_dagger_dataset else [])
    if any(len(vector) != next(iter(dimensions))
           for item in online for vector in item["features"]):
        parser.error("initial DAgger dataset does not match the feature schema")
    reference = _reference_loads(args.reference_summary)
    policy = PPOCandidatePolicy(feature_count=dimensions.pop())
    policy.load(args.init_model)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "bc_dagger_report.json"
    report = {
        "ppo_updates": 0,
        "expert_examples": len(expert),
        "initial_dagger_examples": len(online),
        "feature_schema": args.feature_schema,
        "baseline": {
            "expert_fit": policy.evaluate_behavior_cloning(expert, args.batch_size),
            "validation": _validate(
                policy, args.validation, args.model_dir, args.feature_schema, reference),
        },
        "rounds": [],
    }
    best_score = report["baseline"]["validation"].get(
        "worst_scair_ratio", report["baseline"]["validation"].get("total_load"))
    policy.save(args.output / "bc.best.pt")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for round_number in range(1, args.rounds + 1):
        new_labels = []
        for path, scenario in zip(args.train, scenarios):
            trajectory = collect_ppo_trajectory(
                scenario, args.model_dir, policy, deterministic=True,
                dagger_per_slot=args.dagger_per_slot,
                dagger_horizon=args.horizon,
                dagger_future_limit=args.future_limit,
                dagger_sample_seed=args.seed + round_number,
            )
            labels = [item for slot in trajectory
                      for item in slot["expert_examples"]]
            new_labels.extend(labels)
            print(f"BC DAgger round {round_number}, {path.parent.parent.name}: "
                  f"labels={len(labels)}", flush=True)
        if not new_labels:
            raise ValueError("DAgger produced no MPC labels")
        before_fit = policy.evaluate_behavior_cloning(new_labels, args.batch_size)
        online.extend(new_labels)
        (args.output / "bc.online_examples.json").write_text(
            json.dumps(online), encoding="utf-8")
        # Equal replay mass preserves expert behavior while exposing the actor's states.
        replay = expert + rng.choices(online, k=len(expert))
        fit = policy.fit_behavior_cloning(
            replay, epochs=args.bc_epochs, learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        result = {
            "round": round_number,
            "new_labels": len(new_labels),
            "aggregated_labels": len(online),
            "on_policy_fit_before": before_fit,
            "replay_fit": fit,
            "expert_fit": policy.evaluate_behavior_cloning(expert, args.batch_size),
            "on_policy_fit": policy.evaluate_behavior_cloning(online, args.batch_size),
            "validation": _validate(
                policy, args.validation, args.model_dir, args.feature_schema, reference),
        }
        policy.save(args.output / f"bc.round_{round_number:02d}.pt")
        score = result["validation"].get(
            "worst_scair_ratio", result["validation"].get("total_load"))
        if score < best_score:
            best_score = score
            policy.save(args.output / "bc.best.pt")
            report["best_round"] = round_number
        report["rounds"].append(result)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    print(f"Best BC checkpoint: {args.output / 'bc.best.pt'}", flush=True)


if __name__ == "__main__":
    main()
