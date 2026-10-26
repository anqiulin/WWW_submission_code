"""Evaluate the four test scenarios in parallel.

Each worker owns one independent scenario simulator. This is safe because
scenario states, event queues, and strategy rollouts are never shared.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from poi_road_sim.experiment import (ExperimentConfig, apply_cost_mode,
                                     apply_feature_schema, run_experiment)


def _run_one(payload):
    (scenario_path, model_dir, output, strategies, horizon, discount, checkpoint,
     plan, compact, local_mpc, scair_travel_weight, feature_schema,
     slot_log_every) = payload
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass
    scenario = apply_feature_schema(apply_cost_mode(
        json.loads(Path(scenario_path).read_text(encoding="utf-8")), "queue_wait"),
        feature_schema)
    if checkpoint:
        scenario["ppo_checkpoint"] = checkpoint
    if plan:
        scenario["mpc_plan_file"] = plan
    result = run_experiment(
        scenario, Path(model_dir), Path(output),
        ExperimentConfig(strategies, horizon, discount, write_csv=not compact,
                          local_mpc=local_mpc,
                          scair_travel_weight=scair_travel_weight,
                          slot_log_every=slot_log_every),
    )
    return scenario_path, result["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="four-scenario dataset directory")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+",
                        default=["ppo", "rise_mpc", "scair", "shortest", "shortest_queue"])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel scenarios; use 2 when memory is limited")
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--mpc-cache", type=Path,
                        help="directory containing scenario_seed_*/mpc.json files")
    parser.add_argument("--compact-output", action="store_true",
                        help="skip detailed CSV files in each worker")
    parser.add_argument("--local-mpc", action="store_true",
                        help="use candidate-route scoped MPC aggregation")
    parser.add_argument("--scair-travel-weight", type=float, default=0.1,
                        help="SCAIR current travel-time coefficient")
    parser.add_argument("--split-strategies", action="store_true",
                        help="run each scenario/strategy pair in its own process")
    parser.add_argument("--feature-schema", choices=("legacy", "local_v2"), default="legacy")
    parser.add_argument("--slot-log-every", type=int, default=8,
                        help="print per-strategy progress every N slots; 0 disables it")
    args = parser.parse_args()
    if args.slot_log_every < 0:
        parser.error("slot-log-every must be nonnegative")
    if args.split_strategies and len(set(args.strategies)) != len(args.strategies):
        parser.error("split-strategies requires unique strategy names")
    paths = sorted(args.dataset.glob("test/scenario_*/seed_*/scenario.json"))
    if len(paths) != 4:
        raise ValueError(f"expected four test scenarios, found {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in paths:
        scenario_name = path.parent.parent.name
        destination = args.output / scenario_name
        plan = None
        if args.mpc_cache:
            candidate = args.mpc_cache / (
                f"{path.parent.parent.name}_{path.stem}_{path.parent.name}_mpc.json")
            if not candidate.is_file():
                raise FileNotFoundError(f"MPC cache missing for {scenario_name}: {candidate}")
            plan = str(candidate.resolve())
        strategy_jobs = args.strategies if args.split_strategies else [tuple(args.strategies)]
        for strategy_set in strategy_jobs:
            run_destination = destination / strategy_set if args.split_strategies else destination
            jobs.append((str(path), str(args.model_dir.resolve()), str(run_destination),
                         (strategy_set,) if args.split_strategies else strategy_set,
                         args.horizon, args.discount,
                         str(args.ppo_checkpoint.resolve()) if args.ppo_checkpoint else None,
                         plan, args.compact_output, args.local_mpc,
                         args.scair_travel_weight, args.feature_schema,
                         args.slot_log_every))
    results = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_run_one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            scenario_path, summary = future.result()
            name = Path(scenario_path).parent.parent.name
            key = f"{name}/{summary[0]['strategy']}" if args.split_strategies and summary else name
            results[key] = summary
            print(f"[parallel-eval] completed {key}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"scenarios": sorted(results), "workers": args.workers,
                      "strategies": args.strategies, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
