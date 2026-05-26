import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Type

import gymnasium as gym
import numpy as np
import numpy.core.numeric as numpy_numeric
import panda_gym  # noqa: F401 - required so Panda envs are registered
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

sys.modules["numpy._core.numeric"] = numpy_numeric


@dataclass
class EpisodeMetric:
    model_name: str
    episode: int
    episode_return: float
    episode_length: int
    success: float


@dataclass
class EvaluationResult:
    name: str
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    success_rate: float
    mean_length: float
    episode_metrics: list[EpisodeMetric]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PPO and SAC models on PandaPush-v3")
    parser.add_argument("--ppo-path", type=str, required=True, help="Path to the saved PPO model zip")
    parser.add_argument("--sac-path", type=str, required=True, help="Path to the saved SAC model zip")
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="Environment type used for evaluation",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible evaluation episodes")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes")
    parser.add_argument("--render-delay", type=float, default=0.03, help="Delay between rendered steps")
    parser.add_argument("--wandb-project", type=str, default="FAIML_RL_Part2", help="W&B project name")
    parser.add_argument("--wandb-group", type=str, default="task4_comparison", help="W&B run group")

    return parser.parse_args()


def make_env(env_type: str, render: bool) -> gym.Env:
    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
    return Monitor(env)


def load_model(
    algorithm: Type[BaseAlgorithm],
    model_path: str,
    env: gym.Env,
) -> BaseAlgorithm:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return algorithm.load(
        model_path,
        env=env,
        custom_objects={
            "action_space": env.action_space,
            "observation_space": env.observation_space,
        },
    )


def evaluate_model(
    name: str,
    model: BaseAlgorithm,
    env: gym.Env,
    episodes: int,
    seed: int,
    deterministic: bool,
    render_delay: float,
) -> EvaluationResult:
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    successes: list[float] = []
    episode_metrics: list[EpisodeMetric] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)     #seed changed to evaluate different random situations
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        last_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, last_info = env.step(action)
            episode_return += float(reward)
            episode_length += 1

            if render_delay > 0:
                time.sleep(render_delay)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if isinstance(last_info, dict) and "is_success" in last_info:
            successes.append(float(last_info["is_success"]))

        success = float(last_info["is_success"]) if isinstance(last_info, dict) and "is_success" in last_info else np.nan
        episode_metrics.append(
            EpisodeMetric(
                model_name=name,
                episode=episode + 1,
                episode_return=episode_return,
                episode_length=episode_length,
                success=success,
            )
        )

    returns = np.array(episode_returns, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.float32)
    success_rate = float(np.mean(successes)) if successes else float("nan")

    return EvaluationResult(
        name=name,
        mean_return=float(returns.mean()),
        std_return=float(returns.std()),
        min_return=float(returns.min()),
        max_return=float(returns.max()),
        success_rate=success_rate,
        mean_length=float(lengths.mean()),
        episode_metrics=episode_metrics,
    )


def print_result_table(results: list[EvaluationResult]) -> None:
    print("\n=== Model comparison ===")
    print(
        f"{'Model':<8}"
        f"{'Mean return':>14}"
        f"{'Std':>10}"
        f"{'Min':>10}"
        f"{'Max':>10}"
        f"{'Success':>12}"
        f"{'Mean len':>12}"
    )
    print("-" * 76)

    for result in results:
        success = "n/a" if np.isnan(result.success_rate) else f"{result.success_rate:.2%}"
        print(
            f"{result.name:<8}"
            f"{result.mean_return:>14.3f}"
            f"{result.std_return:>10.3f}"
            f"{result.min_return:>10.3f}"
            f"{result.max_return:>10.3f}"
            f"{success:>12}"
            f"{result.mean_length:>12.1f}"
        )


def print_winner(results: list[EvaluationResult]) -> None:
    ranked = sorted(
        results,
        key=lambda result: (
            result.mean_return,
            -result.std_return,
            -result.mean_length,
        ),
        reverse=True,
    )
    best = ranked[0]
    print(f"\nBest model by mean return: {best.name}")


def init_wandb(args: argparse.Namespace):

    import wandb

    run_name = f"ppo_vs_sac_{args.env_type}_{args.episodes}eps_seed{args.seed}"
    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config={
            "comparison": "ppo_vs_sac",
            "env_type": args.env_type,
            "episodes": args.episodes,
            "seed": args.seed,
            "deterministic": not args.stochastic,
            "ppo_path": args.ppo_path,
            "sac_path": args.sac_path,
        },
    )


def log_to_wandb(results: list[EvaluationResult]) -> None:
    import wandb

    episode_rows = []
    for result in results:
        for metric in result.episode_metrics:
            episode_rows.append(
                [
                    metric.model_name,
                    metric.episode,
                    metric.episode_return,
                    metric.episode_length,
                    metric.success,
                ]
            )

    episode_table = wandb.Table(
        columns=["model", "episode", "return", "length", "success"],
        data=episode_rows,
    )
    summary_table = wandb.Table(
        columns=["model", "mean_return", "std_return", "min_return", "max_return", "success_rate", "mean_length"],
        data=[
            [
                result.name,
                result.mean_return,
                result.std_return,
                result.min_return,
                result.max_return,
                result.success_rate,
                result.mean_length,
            ]
            for result in results
        ],
    )

    episodes = [metric.episode for metric in results[0].episode_metrics]
    return_series = [[metric.episode_return for metric in result.episode_metrics] for result in results]
    success_series = [[metric.success for metric in result.episode_metrics] for result in results]
    length_series = [[metric.episode_length for metric in result.episode_metrics] for result in results]
    keys = [result.name for result in results]

    log_payload = {
        "comparison/episode_metrics": episode_table,
        "comparison/summary": summary_table,
        "comparison/return_by_episode": wandb.plot.line_series(
            xs=episodes,
            ys=return_series,
            keys=keys,
            title="PPO vs SAC return by episode",
            xname="Episode",
        ),
        "comparison/episode_length_by_episode": wandb.plot.line_series(
            xs=episodes,
            ys=length_series,
            keys=keys,
            title="PPO vs SAC episode length by episode",
            xname="Episode",
        ),
    }

    if not any(np.isnan(value) for values in success_series for value in values):
        log_payload["comparison/success_by_episode"] = wandb.plot.line_series(
            xs=episodes,
            ys=success_series,
            keys=keys,
            title="PPO vs SAC success by episode",
            xname="Episode",
        )

    for result in results:
        prefix = f"comparison/{result.name.lower()}"
        log_payload[f"{prefix}/mean_return"] = result.mean_return
        log_payload[f"{prefix}/std_return"] = result.std_return
        log_payload[f"{prefix}/success_rate"] = result.success_rate
        log_payload[f"{prefix}/mean_length"] = result.mean_length

    wandb.log(log_payload)


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic
    render_delay = args.render_delay if args.render else 0.0

    if args.episodes <= 0:
        raise ValueError("Episode count must be a positive integer")

    run = init_wandb(args)

    ppo_env = make_env(args.env_type, args.render)
    sac_env = make_env(args.env_type, args.render)

    try:
        ppo_model = load_model(PPO, args.ppo_path, ppo_env)
        sac_model = load_model(SAC, args.sac_path, sac_env)

        results = [
            evaluate_model("PPO", ppo_model, ppo_env, args.episodes, args.seed, deterministic, render_delay),
            evaluate_model("SAC", sac_model, sac_env, args.episodes, args.seed, deterministic, render_delay),
        ]

        print_result_table(results)
        print_winner(results)
        if run is not None:
            log_to_wandb(results)
    finally:
        ppo_env.close()
        sac_env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
