"""Prepare MPC expert data and a PPO behavior-cloning initialization.

This command intentionally stops before PPO reinforcement-learning updates.
It creates manifests, an environment report, expert candidate scores, and a
policy checkpoint suitable for the later PPO phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from poi_road_sim.experiment import (collect_mpc_expert_dataset, _policy_feature_count,
                                     apply_cost_mode, apply_feature_schema,
                                     find_mpc_expert_plan)
from poi_road_sim.torch_policy import PPOCandidatePolicy


def _expert_plan_path(output: Path, scenario_path: Path) -> Path:
    # Different scenarios often share scenario.json and the same seed folder.
    identity = hashlib.sha256(str(scenario_path.resolve()).encode()).hexdigest()[:16]
    return output / f"mpc_plan_{scenario_path.stem}_{scenario_path.parent.name}_{identity}.json"


def _collect_one(payload: tuple[str, str, str, int, int | None, int | None, str | None,
                                str | None, str]):
    """Worker-safe MPC collection for one scenario."""
    (path_text, model_dir_text, plan_text, horizon, per_slot, future_limit,
     cost_mode, cached_plan, feature_schema) = payload
    # One BLAS/Torch thread per worker prevents CPU oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    path = Path(path_text)
    scenario = apply_feature_schema(
        apply_cost_mode(json.loads(path.read_text(encoding="utf-8")), cost_mode),
        feature_schema)
    label = "/".join(path.parts[-4:-1])
    source = "cache replay" if cached_plan else "online MPC"
    print(f"MPC expert start: {label} ({source})", flush=True)
    examples = collect_mpc_expert_dataset(
        scenario, Path(model_dir_text), horizon,
        max_examples_per_slot=per_slot, future_request_limit=future_limit,
        mpc_plan_output=None if cached_plan else Path(plan_text),
        mpc_plan_input=Path(cached_plan) if cached_plan else None,
        progress=lambda done, total, count: print(
            f"MPC expert {label}: slots {done}/{total}, labels {count}", flush=True),
        # Per-label logging is prohibitively verbose for city-scale datasets.
        # The slot callback above is sufficient to monitor cache replay.
        label_progress=None)
    print(f"MPC expert complete: {label}, examples={len(examples)}", flush=True)
    return path_text, cached_plan or plan_text, examples


def inspect_scenarios(paths: list[Path]) -> dict:
    report = {"scenarios": len(paths), "total_requests": 0, "total_roads": 0,
              "total_pois": 0, "steps": 0, "empty_candidates": 0,
              "provenance": "proxy dynamic simulation; TLC-derived OD with synthetic POI intent"}
    for path in paths:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        report["total_requests"] += len(scenario.get("requests", []))
        report["total_roads"] += len(scenario.get("roads", []))
        report["total_pois"] += len(scenario.get("pois", []))
        report["steps"] += int(scenario.get("steps", 0))
        report["empty_candidates"] += sum(
            not (row.get("candidates") or [row]) for row in scenario.get("requests", []))
    for key in ("total_requests", "total_roads", "total_pois", "steps"):
        report[key] = int(report[key])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--validation", nargs="+", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--expert-per-slot", type=int,
                        help="maximum labels saved per slot; all MPC actions advance the expert trajectory")
    parser.add_argument("--expert-future-limit", type=int,
                        help="historical demand forecast samples per future slot; capped at 16, including when 0")
    parser.add_argument("--mpc-cache", type=Path,
                        help="reuse existing MPC plan directory/file; require a matching plan for every training scene")
    parser.add_argument("--cost-mode", choices=("queue_wait",),
                        help="use event-driven dynamics and physical waiting-minute cost")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel MPC expert workers; use 1 for deterministic logs")
    parser.add_argument("--expert-dataset", type=Path,
                        help="reuse an existing MPC-labeled JSON dataset and skip cache replay")
    parser.add_argument("--feature-schema", choices=("legacy", "local_v2"), default="legacy")
    args = parser.parse_args()
    # Resolve and validate every cache before writing output or starting BC.
    cached_plans = {}
    if args.expert_dataset is not None and not args.expert_dataset.is_file():
        parser.error(f"expert dataset not found: {args.expert_dataset}")
    if args.mpc_cache is not None and args.expert_dataset is None:
        for path in args.train:
            scenario = apply_feature_schema(
                apply_cost_mode(json.loads(path.read_text(encoding="utf-8")), args.cost_mode),
                args.feature_schema)
            limit = (int(scenario.get("future_request_limit", 12))
                     if args.expert_future_limit is None else args.expert_future_limit)
            try:
                cached_plans[str(path)] = str(find_mpc_expert_plan(
                    args.mpc_cache, scenario, args.horizon, limit))
            except (ValueError, FileNotFoundError) as exc:
                parser.error(f"{path}: {exc}")
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {"train": args.train, "validation": args.validation}
    manifest = {name: [str(path) for path in paths] for name, paths in splits.items()}
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = {name: inspect_scenarios(paths) for name, paths in splits.items()}
    (args.output / "environment_report.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    expert = (json.loads(args.expert_dataset.read_text(encoding="utf-8"))
              if args.expert_dataset is not None else [])
    if not isinstance(expert, list):
        parser.error("expert dataset must contain a JSON list")
    plan_files = []
    jobs = []
    if args.expert_dataset is None:
        for path in args.train:
            plan_path = _expert_plan_path(args.output, path)
            jobs.append((str(path), str(args.model_dir.resolve()), str(plan_path),
                         args.horizon, args.expert_per_slot, args.expert_future_limit,
                         args.cost_mode, cached_plans.get(str(path)), args.feature_schema))
        worker_count = max(1, min(args.workers, len(jobs)))
        if worker_count == 1:
            results = [_collect_one(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(_collect_one, job) for job in jobs]
                results = [future.result() for future in as_completed(futures)]
        for path_text, plan_text, examples in sorted(results, key=lambda item: item[0]):
            plan_files.append(plan_text)
            expert.extend(examples)
    (args.output / "mpc_expert_dataset.json").write_text(json.dumps(expert), encoding="utf-8")
    dimensions = {_policy_feature_count(apply_feature_schema(apply_cost_mode(
        json.loads(path.read_text(encoding="utf-8")), args.cost_mode), args.feature_schema))
                  for path in args.train}
    if len(dimensions) != 1:
        raise ValueError("training scenarios must use one policy feature schema")
    feature_count = dimensions.pop()
    if any(len(vector) != feature_count for item in expert
           for vector in item["features"]):
        parser.error("expert dataset feature schema does not match training scenarios")
    policy = PPOCandidatePolicy(feature_count=feature_count)
    report = policy.fit_behavior_cloning(expert, epochs=args.bc_epochs)
    report.update({"horizon": args.horizon, "stage": "pre_ppo_behavior_cloning",
                   "expert_examples": len(expert), "ppo_updates": 0,
                   "expert_per_slot": args.expert_per_slot,
                   "expert_future_limit": args.expert_future_limit})
    report["cost_mode"] = args.cost_mode or "scenario"
    report["feature_schema"] = args.feature_schema
    report["mpc_plan_files"] = plan_files
    report["expert_source"] = ("precomputed_expert_dataset" if args.expert_dataset is not None
                               else "mpc_cache_replay" if args.mpc_cache is not None
                               else "online_mpc")
    report["mpc_cache"] = str(args.mpc_cache.resolve()) if args.mpc_cache is not None else None
    policy.save(args.output / "ppo_bc_init.pt")
    (args.output / "bc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "environment": environment,
                      "bc_report": report}, indent=2))


if __name__ == "__main__":
    main()
