"""Precompute exact MPC plans for reuse in evaluation or expert preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poi_road_sim.experiment import (apply_cost_mode, collect_mpc_expert_dataset,
                                     load_mpc_expert_plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--future-request-limit", type=int, default=12)
    parser.add_argument("--cost-mode", choices=("queue_wait",))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for scenario_path in args.scenarios:
        scenario = apply_cost_mode(
            json.loads(scenario_path.read_text(encoding="utf-8")), args.cost_mode)
        # Include the scenario directory because all four files are named
        # ``scenario.json`` and share the same seed directory.
        scenario_name = (scenario_path.parent.parent.parent.name
                         if scenario_path.parent.parent.name in ("train", "validation", "test")
                         else scenario_path.parent.parent.name)
        output = args.output / f"{scenario_name}_{scenario_path.stem}_{scenario_path.parent.name}_mpc.json"
        print(f"MPC plan: {scenario_path} -> {output}", flush=True)
        if output.exists():
            load_mpc_expert_plan(scenario, output, args.horizon, args.future_request_limit)
            print(f"Reusing validated MPC plan: {output}", flush=True)
        else:
            collect_mpc_expert_dataset(
                scenario, args.model_dir, horizon=args.horizon,
                max_examples_per_slot=1,
                future_request_limit=args.future_request_limit,
                mpc_plan_output=output,
                progress=lambda done, total, count: print(
                    f"MPC plan slots {done}/{total}", flush=True))
        manifest[str(scenario_path)] = str(output)
    (args.output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "horizon": args.horizon,
                    "future_request_limit": args.future_request_limit,
                    "cost_mode": args.cost_mode or "scenario",
                    "plans": manifest}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
