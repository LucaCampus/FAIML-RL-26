import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import SAC, PPO
from rand_wrapper import RandomizationWrapper
import wandb
from wandb.integration.sb3 import WandbCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
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
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["sac", "ppo"],
        help="RL algorithm to use for training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    wandb.init(
    project="faiml-rl",
    name = "SAC-PPO-PandaPush",
    config={
        "algorithm": args.algorithm,
        "env": args.env_type,
        "timesteps": args.timesteps,
        "sampling_strategy": args.sampling_strategy
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,

)

    #TODO: add randomization wrapper here
    # if args.sampling_strategy != "none":
    #     env = RandomizationWrapper(env, args.sampling_strategy)
    #TODO: create model and train it
    if args.algorithm == "sac":
        model = SAC("MultiInputPolicy", env)
    elif args.algorithm == "ppo":
        model = PPO("MultiInputPolicy", env)
    model.learn(total_timesteps=args.timesteps,
                callback=WandbCallback(
                    model_save_path=f"models/{args.algorithm}_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k",
                    verbose=2
                )
                )
    # save_name = f"{args.algorithm}_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    # TODO: model.save(save_name)
    model.save(f"models/{args.algorithm}_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k")

if __name__ == "__main__":
    main()