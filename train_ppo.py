"""Train the MPC-compatible global recommendation policy with PPO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from poi_road_sim.experiment import train_ppo_scenarios


def main() -> None:
    # Avoid BLAS/OpenMP oversubscription when training alongside evaluators.
    thread_count = os.environ.get("TORCH_NUM_THREADS")
    if thread_count:
        import torch
        torch.set_num_threads(max(1, int(thread_count)))
        torch.set_num_interop_threads(max(1, int(os.environ.get("TORCH_NUM_INTEROP_THREADS", thread_count))))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="+", type=Path)
    parser.add_argument("--validation", nargs="*", type=Path, default=[])
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-size", type=int, default=2048,
                        help="minimum request decisions per update; collect complete passes over all training scenarios")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="environment time slices per optimization mini-batch")
    parser.add_argument("--slot-log-every", type=int, default=1,
                        help="print slot progress every N slots; 0 disables slot logs")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--expert-dataset", type=Path,
                        help="MPC-labeled examples used as a PPO auxiliary behavior-cloning loss")
    parser.add_argument("--bc-coef", type=float, default=0.1,
                        help="weight of the auxiliary MPC behavior-cloning loss during PPO")
    parser.add_argument("--dagger-per-slot", type=int, default=0,
                        help="sample this many requests per slot for online MPC labels; 0 disables DAgger")
    parser.add_argument("--dagger-horizon", type=int, default=4)
    parser.add_argument("--dagger-future-limit", type=int,
                        help="cap recently observed requests used by online MPC labels")
    parser.add_argument("--dagger-bc-coef", type=float, default=0.5,
                        help="weight of the on-policy DAgger loss during PPO")
    parser.add_argument("--cost-mode", choices=("queue_wait",),
                        help="use event-driven dynamics and physical waiting-minute cost")
    parser.add_argument("--feature-schema", choices=("legacy", "local_v2"), default="legacy")
    args = parser.parse_args()
    if args.slot_log_every < 0:
        parser.error("slot-log-every must be nonnegative")
    if (args.learning_rate <= 0 or args.clip_ratio <= 0 or
            args.value_coef < 0 or args.entropy_coef < 0):
        parser.error("learning-rate and clip-ratio must be positive; coefficients must be nonnegative")
    expert_examples = (json.loads(args.expert_dataset.read_text(encoding="utf-8"))
                       if args.expert_dataset else [])

    def log_slot(stage: str, update: int, done: int, total: int,
                 decisions: int) -> None:
        if not args.slot_log_every:
            return
        if done != total and done % args.slot_log_every:
            return
        print(f"PPO {stage} update {update}/{args.updates}, "
              f"slot {done}/{total}: decisions={decisions}", flush=True)

    reports = train_ppo_scenarios(
        args.scenarios, args.model_dir, args.output,
        epochs=args.epochs, updates=args.updates, seed=args.seed,
        validation_scenarios=args.validation, rollout_size=args.rollout_size,
        batch_size=args.batch_size, gamma=args.gamma, init_model=args.init_model,
        expert_examples=expert_examples, bc_coef=args.bc_coef,
        dagger_per_slot=args.dagger_per_slot,
        dagger_horizon=args.dagger_horizon,
        dagger_future_limit=args.dagger_future_limit,
        dagger_bc_coef=args.dagger_bc_coef,
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        cost_mode=args.cost_mode, feature_schema=args.feature_schema,
        progress=lambda report: print(
            f"PPO update {report['update'] + 1}/{args.updates}: "
            f"decisions={report['transitions']}, "
            f"bc_auxiliary_loss={report.get('bc_auxiliary_loss', 'n/a')}, "
            f"dagger_examples={report.get('dagger_examples', 0)}, "
            f"dagger_auxiliary_loss={report.get('dagger_auxiliary_loss', 'n/a')}, "
            f"validation_reward={report.get('validation_reward', 'n/a')}, "
            f"validation_mean_absolute_load={report.get('validation_mean_absolute_load', 'n/a')}",
            flush=True),
        slot_progress=log_slot)
    print(json.dumps({"output": str(args.output), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
