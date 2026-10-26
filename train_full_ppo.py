"""Run sampled MPC expert generation, BC initialization, then PPO training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def discover_scenarios(dataset: Path, split: str) -> list[Path]:
    """Accept both scenario-first and split-first layouts, including symlinks."""
    paths = set()
    for root in (dataset, dataset / "dataset"):
        for pattern in (f"{split}/*/seed_*/scenario.json",
                        f"*/{split}/seed_*/scenario.json"):
            paths.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-scenarios", nargs="+", type=Path,
                        help="explicit training scenarios, for controlled single-dataset runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--expert-per-slot", type=int, default=16)
    parser.add_argument("--expert-future-limit", type=int, default=16)
    parser.add_argument("--mpc-cache", type=Path,
                        help="reuse existing training MPC plans; skip online MPC expert computation")
    parser.add_argument("--expert-dataset", type=Path,
                        help="reuse an existing MPC-labeled JSON dataset and skip cache replay")
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--ppo-bc-coef", type=float, default=0.1,
                        help="MPC behavior-cloning auxiliary loss weight during PPO updates")
    parser.add_argument("--dagger-per-slot", type=int, default=4,
                        help="sample this many requests per slot for online MPC labels")
    parser.add_argument("--dagger-bc-coef", type=float, default=0.5,
                        help="on-policy MPC behavior-cloning loss weight during PPO")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rollout-size", type=int, default=2048,
                        help="minimum decisions per update; collect complete passes over all training scenarios")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expert-workers", type=int, default=1,
                        help="parallel workers for MPC cache replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-mode", choices=("queue_wait",), default="queue_wait")
    parser.add_argument("--feature-schema", choices=("legacy", "local_v2"), default="legacy")
    args = parser.parse_args()

    scenarios = {"train": (args.train_scenarios if args.train_scenarios is not None
                            else discover_scenarios(args.dataset, "train")),
                 "validation": discover_scenarios(args.dataset, "validation")}
    for split, paths in scenarios.items():
        if not paths:
            parser.error(f"no {split} scenarios under {args.dataset}")
        for path in paths:
            if not path.is_file():
                parser.error(f"{split} scenario not found: {path}")
    if (args.expert_per_slot < 1 or args.expert_future_limit < 0 or
            args.dagger_per_slot < 0 or args.ppo_bc_coef < 0 or args.dagger_bc_coef < 0 or
            args.expert_workers < 1 or args.updates < 0):
        parser.error("expert-per-slot and workers must be positive; limits, updates and BC coefficients must be nonnegative")
    checkpoint = args.output / "ppo.pt"
    bc_checkpoint = args.output / "ppo_bc_init.pt"
    if checkpoint.exists() or bc_checkpoint.exists():
        parser.error(f"output contains a checkpoint: {args.output}; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    prepare = [sys.executable, "-u", str(here / "prepare_ppo.py"),
               "--train", *map(str, scenarios["train"]),
               "--validation", *map(str, scenarios["validation"]),
               "--model-dir", str(args.model_dir), "--output", str(args.output),
               "--horizon", str(args.horizon), "--expert-per-slot", str(args.expert_per_slot),
               "--expert-future-limit", str(args.expert_future_limit),
               "--bc-epochs", str(args.bc_epochs), "--workers", str(args.expert_workers),
               "--cost-mode", args.cost_mode, "--feature-schema", args.feature_schema]
    if args.mpc_cache is not None:
        prepare.extend(("--mpc-cache", str(args.mpc_cache)))
    if args.expert_dataset is not None:
        prepare.extend(("--expert-dataset", str(args.expert_dataset)))
    print("Stage 1/3: MPC cache replay" if args.mpc_cache else "Stage 1/3: MPC expert", flush=True)
    subprocess.run(prepare, check=True)
    train = [sys.executable, "-u", str(here / "train_ppo.py"),
             *map(str, scenarios["train"]), "--validation", *map(str, scenarios["validation"]),
             "--model-dir", str(args.model_dir), "--output", str(checkpoint),
             "--init-model", str(bc_checkpoint), "--updates", str(args.updates),
             "--expert-dataset", str(args.output / "mpc_expert_dataset.json"),
             "--bc-coef", str(args.ppo_bc_coef),
             "--epochs", str(args.epochs), "--rollout-size", str(args.rollout_size),
             "--batch-size", str(args.batch_size), "--seed", str(args.seed)]
    train.extend(("--cost-mode", args.cost_mode,
                  "--feature-schema", args.feature_schema))
    train.extend(("--dagger-per-slot", str(args.dagger_per_slot),
                  "--dagger-horizon", str(args.horizon),
                  "--dagger-bc-coef", str(args.dagger_bc_coef)))
    if args.expert_future_limit is not None:
        train.extend(("--dagger-future-limit", str(args.expert_future_limit)))
    print("Stage 2/3: BC checkpoint saved; stage 3/3: " +
          ("BC validation" if args.updates == 0 else "PPO"), flush=True)
    subprocess.run(train, check=True)
    print(f"Training complete: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
