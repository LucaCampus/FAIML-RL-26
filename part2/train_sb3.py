import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import SAC, PPO
from rand_wrapper import RandomizationWrapper
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor


DEFAULT_WANDB_PROJECT = "FAIML_RL_Part2"


def parse_args() -> argparse.Namespace:
    # All relevant experiment choices are CLI arguments so the run name and W&B config stay reproducible.
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--task",
        type=str,
        default="task4",
        choices=["task4", "task5"],
        help="Project phase; controls the default W&B group",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="target",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="RL algorithm to use for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment and SB3 model",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project name",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Shared identifier used by W&B, TensorBoard logs, and saved model folders.
    timesteps_k = args.timesteps // 1000
    run_name = f"{args.algorithm}_{args.sampling_strategy}_{args.env_type}_{timesteps_k}k_seed{args.seed}"
    wandb_group = f"{args.task}_training"

    # The local panda-gym version maps env_type source/target to different object masses.
    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    # Monitor records episode return/length/success so SB3 and W&B can log rollout metrics.
    env = Monitor(env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    

    # sync_tensorboard=True forwards SB3 TensorBoard scalars from runs/<run_name> to W&B.
    run = wandb.init(
        project=args.wandb_project,
        group=wandb_group,
        name=run_name,
        config={
            "task": args.task,
            "wandb_group": wandb_group,
            "run_name": run_name,
            "algorithm": args.algorithm,
            "env": args.env_type,
            "timesteps": args.timesteps,
            "timesteps_k": timesteps_k,
            "sampling_strategy": args.sampling_strategy,
            "seed": args.seed,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    if args.sampling_strategy != "none":
        # UDR/ADR randomize the block mass on top of the chosen source/target domain.
        env = RandomizationWrapper(env, mass_range=(0.5, 25.0), mode=args.sampling_strategy)

    tensorboard_log = f"runs/{run_name}"

    # MultiInputPolicy is required because PandaPush observations are Dict observations.
    if args.algorithm == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log, seed=args.seed)
    elif args.algorithm == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log, seed=args.seed)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    save_path = f"models/{run_name}"
    try:
        # WandbCallback saves the final model to models/<run_name>/model.zip at training end.
        model.learn(total_timesteps=args.timesteps,
                        callback=WandbCallback(
                            model_save_path=save_path,
                            verbose=2
                        )
                    )
    finally:
        env.close()
        run.finish()

if __name__ == "__main__":
    main()
