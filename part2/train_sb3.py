import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import DDPG, PPO, SAC
from rand_wrapper import RandomizationWrapper

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPush-v3")
    parser.add_argument("--sampling-strategy",type=str,default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument("--env-type",type=str,default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument("--timesteps",type=int,default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument("--alpha",type=float,default=3e-4,
        help="Learning Rate",
    )
    parser.add_argument("--n-steps",type=int,default=2048,
        help="Number of rollout steps recorded before updating the Neural Network",
    )
    parser.add_argument("--n-eval-episodes",type=int,default=15,
        help="Number of episodes used for source/target evaluation",
    )
    return parser.parse_args()

def evaluate_model(model, eval_env, n_eval_episodes: int = 15):
    episode_rewards = []
    successes = []

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            episode_reward += reward
            done = terminated or truncated
            last_info = info

        episode_rewards.append(episode_reward)

        if "is_success" in last_info:
            successes.append(float(last_info["is_success"]))

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

    if len(successes) > 0:
        metrics["success_rate"] = float(np.mean(successes))

    return metrics


def make_env(env_type: str, render_mode: str = "rgb_array"):
    env = gym.make(
        "PandaPush-v3",
        render_mode=render_mode,
        type=env_type,
        reward_type="dense",
    )
    return Monitor(env)


def print_reward_log(
    iteration: int,
    timesteps: int,
    source_metrics: dict[str, float],
    target_metrics: dict[str, float],
) -> None:
    source_success = source_metrics.get("success_rate")
    target_success = target_metrics.get("success_rate")

    message = (
        f"[Iteration {iteration:04d} | {timesteps} steps] "
        f"source_reward={source_metrics['mean_reward']:.3f} "
        f"+/- {source_metrics['std_reward']:.3f} | "
        f"target_reward={target_metrics['mean_reward']:.3f} "
        f"+/- {target_metrics['std_reward']:.3f}"
    )

    if source_success is not None:
        message += f" | source_success={source_success:.2%}"

    if target_success is not None:
        message += f" | target_success={target_success:.2%}"

    print(message, flush=True)



def main() -> None:
    args = parse_args()


    train_env = make_env(env_type="source", render_mode="rgb_array")
    source_eval_env = make_env(env_type="source", render_mode="rgb_array")
    target_eval_env = make_env(env_type="target", render_mode="rgb_array")

    save_name = f"ppo_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"

    run = wandb.init(
        project="faiml-panda-push-test1",
        name = save_name,
        config={
            "algorithm": "PPO",
            "sampling_strategy": args.sampling_strategy,
            "train_env_type": args.env_type,
            "timesteps": args.timesteps,
            "learning_rate": args.alpha,
            "n_steps": args.n_steps,
            "n_eval_episodes": args.n_eval_episodes,
        },
        sync_tensorboard=True,
    )

    # TODO: Apply mass randomization only when a sampling strategy is selected.
    # if args.sampling_strategy != "none":
    #     env = RandomizationWrapper(env, mode=args.sampling_strategy)


    #Create model
    
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.alpha,
        verbose=1,
        n_steps=args.n_steps,
        tensorboard_log=f"runs/{run.id}",
    )
    

    #Train model
    num_iterations = args.timesteps//args.n_steps
    best_target_reward = -float("inf")

    for iteration in range(num_iterations):
        model.learn(total_timesteps=args.n_steps, reset_num_timesteps=False)
        current_timesteps = (iteration + 1) * args.n_steps

        source_metrics = evaluate_model(
            model,
            source_eval_env,
            n_eval_episodes=args.n_eval_episodes,
        )

        target_metrics = evaluate_model(
            model,
            target_eval_env,
            n_eval_episodes=args.n_eval_episodes,
        )

        log_data = {
            "timesteps": current_timesteps,

            "eval/source_mean_reward": source_metrics["mean_reward"],
            "eval/source_std_reward": source_metrics["std_reward"],

            "eval/target_mean_reward": target_metrics["mean_reward"],
            "eval/target_std_reward": target_metrics["std_reward"],
        }

        if "success_rate" in source_metrics:
            log_data["eval/source_success_rate"] = source_metrics["success_rate"]

        if "success_rate" in target_metrics:
            log_data["eval/target_success_rate"] = target_metrics["success_rate"]

        wandb.log(log_data, step=current_timesteps)
        print_reward_log(
            iteration=iteration + 1,
            timesteps=current_timesteps,
            source_metrics=source_metrics,
            target_metrics=target_metrics,
        )

        if target_metrics["mean_reward"] > best_target_reward:
            best_target_reward = target_metrics["mean_reward"]
            model.save("best_model_target")



    # Save the model
    model.save(save_name)
    

    #Close every env and wandb
    train_env.close()
    source_eval_env.close()
    target_eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()



# https://chatgpt.com/share/6a0ad069-1474-8396-917c-0bc31261e151
# look for "Step 1 — Collect transitions (n_steps=2048)"
