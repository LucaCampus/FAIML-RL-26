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
    timesteps_k = args.timesteps // 1000
    run_name = f"{args.algorithm}_{args.sampling_strategy}_{args.env_type}_{timesteps_k}k_seed{args.seed}"
    wandb_group = f"{args.task}_training"

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    env = Monitor(env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

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
        env = RandomizationWrapper(env, args.sampling_strategy)

    tensorboard_log = f"runs/{run_name}"
    if args.algorithm == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log, seed=args.seed)
    elif args.algorithm == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log, seed=args.seed)

    save_path = f"models/{run_name}"
    try:
        model.learn(total_timesteps=args.timesteps,
                        callback=WandbCallback(
                            model_save_path=save_path,
                            verbose=2
                        )
                    )
    finally:
        env.close()
        run.finish()
    # model.save(save_path) useless because wandb already saves the model 

if __name__ == "__main__":
    main()
