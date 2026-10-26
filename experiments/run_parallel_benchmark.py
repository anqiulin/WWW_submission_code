"""Run independent strategies and scenarios in parallel processes."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from poi_road_sim.experiment import ExperimentConfig, STRATEGIES, run_experiment


def _run_one(job: tuple[str, str, str, str, int, float, int, str | None,
                         str | None]) -> dict[str, Any]:
    scenario_path, strategy, model_dir, output_dir, horizon, discount, seed, checkpoint, background = job
    scenario_file = Path(scenario_path)
    scenario = json.loads(scenario_file.read_text(encoding="utf-8"))
    if checkpoint:
        scenario["ppo_checkpoint"] = checkpoint
    if background:
        scenario["background_profile_file"] = str(Path(background).resolve())
    report = run_experiment(
        scenario, Path(model_dir), Path(output_dir),
        ExperimentConfig((strategy,), horizon, discount, seed),
    )
    summary = dict(report["summary"][0])
    summary.update({"scenario": scenario_path, "condition": scenario.get("condition", scenario_file.stem)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=["ppo", "rise_mpc", "scair", "shortest_distance", "shortest_queue", "random"])
    parser.add_argument("--workers", type=int, default=0,
                        help="processes; default is min(number of jobs, CPU count)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--background-profile", type=Path)
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scenario_path in args.scenarios:
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        condition = str(scenario.get("condition", scenario_path.stem))
        # Conditions commonly contain multiple independent seeds. Include the
        # scenario directory in the job path so those runs cannot overwrite
        # each other's assignments or time series.
        scenario_id = scenario_path.parent.name or scenario_path.stem
        if scenario_id in {".", ""}:
            scenario_id = scenario_path.stem
        for strategy in args.strategies:
            job_output = args.output / condition / scenario_id / strategy
            jobs.append((str(scenario_path), strategy, str(args.model_dir), str(job_output),
                         args.horizon, args.discount, args.seed,
                         str(args.ppo_checkpoint.resolve()) if args.ppo_checkpoint else None,
                         str(args.background_profile.resolve()) if args.background_profile else None))
    workers = args.workers or min(len(jobs), max(1, os.cpu_count() or 1))
    workers = min(workers, max(1, len(jobs)))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": len(results), "total": len(jobs),
                              "strategy": result["strategy"],
                              "condition": result["condition"],
                              "runtime_sec": result["runtime_sec"]}, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: (row["condition"], row["strategy"]))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({
        "scenarios": [str(path.resolve()) for path in args.scenarios],
        "strategies": args.strategies,
        "workers": workers,
        "horizon": args.horizon,
        "discount": args.discount,
        "seed": args.seed,
        "parallelism": "one independent process per scenario-strategy job",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "jobs": len(jobs),
                      "workers": workers, "summary": str(args.output / "summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
