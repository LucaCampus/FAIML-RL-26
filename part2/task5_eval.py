import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Type

import gymnasium as gym
import numpy as np
import numpy.core.numeric as numpy_numeric
import panda_gym  # noqa: F401 - required so Panda envs are registered
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

sys.modules["numpy._core.numeric"] = numpy_numeric


ALGORITHMS: dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "sac": SAC,
}

TASK5_LABELS = ["source->source", "source->target", "target->target"]


@dataclass
class EpisodeMetric:
    label: str
    train_env: str
    test_env: str
    episodes_count: int
    episode: int
    episode_return: float
    episode_length: int
    success: float


@dataclass
class EvaluationResult:
    label: str
    train_env: str
    test_env: str
    episodes: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    success_rate: float
    mean_length: float
    episode_metrics: list[EpisodeMetric]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task 5 source/target baselines")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=sorted(ALGORITHMS),
        help="Algorithm used to train both models",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy used by the evaluated models",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Training timesteps used by the evaluated models",
    )
    parser.add_argument(
        "--source-model-path",
        type=str,
        required=True,
        help="Path to the model trained on the source environment",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to the model trained on the target environment",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[10, 25, 50],
        help="Episode counts to evaluate, e.g. --episodes 10 25 50",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible evaluations")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes")
    parser.add_argument("--render-delay", type=float, default=0.03, help="Delay between rendered steps")
    parser.add_argument("--wandb-project", type=str, default="FAIML_RL_Part2", help="W&B project name")
    parser.add_argument("--wandb-group", type=str, default="task5_eval", help="W&B run group")

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
    label: str,
    train_env: str,
    test_env: str,
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
        obs, _ = env.reset(seed=seed + episode)
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
                label=label,
                train_env=train_env,
                test_env=test_env,
                episodes_count=episodes,
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
        label=label,
        train_env=train_env,
        test_env=test_env,
        episodes=episodes,
        mean_return=float(returns.mean()),
        std_return=float(returns.std()),
        min_return=float(returns.min()),
        max_return=float(returns.max()),
        success_rate=success_rate,
        mean_length=float(lengths.mean()),
        episode_metrics=episode_metrics,
    )

def print_results(results: list[EvaluationResult]) -> None:
    print("\n=== Task 5 evaluations ===")
    print(
        f"{'Config':<18}"
        f"{'Episodes':>10}"
        f"{'Mean return':>14}"
        f"{'Std':>10}"
        f"{'Min':>10}"
        f"{'Max':>10}"
        f"{'Success':>12}"
        f"{'Mean len':>12}"
    )
    print("-" * 96)

    for result in results:
        success = "n/a" if np.isnan(result.success_rate) else f"{result.success_rate:.2%}"
        print(
            f"{result.label:<18}"
            f"{result.episodes:>10}"
            f"{result.mean_return:>14.3f}"
            f"{result.std_return:>10.3f}"
            f"{result.min_return:>10.3f}"
            f"{result.max_return:>10.3f}"
            f"{success:>12}"
            f"{result.mean_length:>12.1f}"
        )

def print_aggregate(results: list[EvaluationResult]) -> None:
    print("\n=== Average across requested episode counts ===")
    print(f"{'Config':<18}{'Avg mean return':>18}{'Avg success':>14}{'Interpretation':>20}")
    print("-" * 70)

    for label in TASK5_LABELS:
        selected = [result for result in results if result.label == label]
        avg_return = float(np.mean([result.mean_return for result in selected]))
        success_values = [result.success_rate for result in selected if not np.isnan(result.success_rate)]
        avg_success = float(np.mean(success_values)) if success_values else float("nan")
        success = "n/a" if np.isnan(avg_success) else f"{avg_success:.2%}"

        interpretation = {
            "source->source": "source reference",
            "source->target": "lower bound",
            "target->target": "upper bound",
        }[label]

        print(f"{label:<18}{avg_return:>18.3f}{success:>14}{interpretation:>20}")

    task_episodes = max(result.episodes for result in results)
    final_results = [result for result in results if result.episodes == task_episodes]
    print(f"\nTask 5 report values should normally use the {task_episodes}-episode rows.")

    for label in TASK5_LABELS:
        result = next(result for result in final_results if result.label == label)
        print(f"{label}: mean return {result.mean_return:.3f}, success {result.success_rate:.2%}")

def init_wandb(args: argparse.Namespace):

    import wandb

    episode_suffix = "-".join(str(episodes) for episodes in args.episodes)
    timesteps_k = args.timesteps // 1000
    run_name = f"{args.algorithm}_{args.sampling_strategy}_{timesteps_k}k_{episode_suffix}eps_seed{args.seed}"
    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config={
            "task": "task5_source_target_baselines",
            "algorithm": args.algorithm,
            "sampling_strategy": args.sampling_strategy,
            "timesteps": args.timesteps,
            "timesteps_k": timesteps_k,
            "episodes": args.episodes,
            "seed": args.seed,
            "deterministic": not args.stochastic,
            "source_model_path": args.source_model_path,
            "target_model_path": args.target_model_path,
        },
    )

def aggregate_by_label(results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
    aggregate = {}
    for label in TASK5_LABELS:
        selected = [result for result in results if result.label == label]
        success_values = [result.success_rate for result in selected if not np.isnan(result.success_rate)]
        aggregate[label] = {
            "avg_mean_return": float(np.mean([result.mean_return for result in selected])),
            "avg_success_rate": float(np.mean(success_values)) if success_values else float("nan"),
            "avg_mean_length": float(np.mean([result.mean_length for result in selected])),
        }
    return aggregate

def log_to_wandb(results: list[EvaluationResult]) -> None:
    import wandb

    episode_rows = []
    for result in results:
        for metric in result.episode_metrics:
            episode_rows.append(
                [
                    metric.label,
                    metric.train_env,
                    metric.test_env,
                    metric.episodes_count,
                    metric.episode,
                    metric.episode_return,
                    metric.episode_length,
                    metric.success,
                ]
            )

    summary_rows = [
        [
            result.label,
            result.train_env,
            result.test_env,
            result.episodes,
            result.mean_return,
            result.std_return,
            result.min_return,
            result.max_return,
            result.success_rate,
            result.mean_length,
        ]
        for result in results
    ]

    aggregate = aggregate_by_label(results)
    aggregate_rows = [
        [
            label,
            aggregate[label]["avg_mean_return"],
            aggregate[label]["avg_success_rate"],
            aggregate[label]["avg_mean_length"],
        ]
        for label in TASK5_LABELS
    ]

    log_payload: dict[str, Any] = {
        "task5/episode_metrics": wandb.Table(
            columns=["config", "train_env", "test_env", "episodes_count", "episode", "return", "length", "success"],
            data=episode_rows,
        ),
        "task5/summary": wandb.Table(
            columns=[
                "config",
                "train_env",
                "test_env",
                "episodes",
                "mean_return",
                "std_return",
                "min_return",
                "max_return",
                "success_rate",
                "mean_length",
            ],
            data=summary_rows,
        ),
        "task5/aggregate": wandb.Table(
            columns=["config", "avg_mean_return", "avg_success_rate", "avg_mean_length"],
            data=aggregate_rows,
        ),
    }

    episode_counts = sorted({result.episodes for result in results})
    for episodes in episode_counts:
        selected_by_label = {result.label: result for result in results if result.episodes == episodes}
        xs = list(range(1, episodes + 1))
        return_series = [
            [metric.episode_return for metric in selected_by_label[label].episode_metrics]
            for label in TASK5_LABELS
        ]
        length_series = [
            [metric.episode_length for metric in selected_by_label[label].episode_metrics]
            for label in TASK5_LABELS
        ]
        success_series = [
            [metric.success for metric in selected_by_label[label].episode_metrics]
            for label in TASK5_LABELS
        ]

        log_payload[f"task5/{episodes}_episodes/return_by_episode"] = wandb.plot.line_series(
            xs=xs,
            ys=return_series,
            keys=TASK5_LABELS,
            title=f"Task 5 return by episode ({episodes} episodes)",
            xname="Episode",
        )
        log_payload[f"task5/{episodes}_episodes/episode_length_by_episode"] = wandb.plot.line_series(
            xs=xs,
            ys=length_series,
            keys=TASK5_LABELS,
            title=f"Task 5 episode length by episode ({episodes} episodes)",
            xname="Episode",
        )

        if not any(np.isnan(value) for values in success_series for value in values):
            log_payload[f"task5/{episodes}_episodes/success_by_episode"] = wandb.plot.line_series(
                xs=xs,
                ys=success_series,
                keys=TASK5_LABELS,
                title=f"Task 5 success by episode ({episodes} episodes)",
                xname="Episode",
            )

    mean_return_by_count = []
    success_by_count = []
    length_by_count = []
    for label in TASK5_LABELS:
        by_count = {result.episodes: result for result in results if result.label == label}
        mean_return_by_count.append([by_count[episodes].mean_return for episodes in episode_counts])
        success_by_count.append([by_count[episodes].success_rate for episodes in episode_counts])
        length_by_count.append([by_count[episodes].mean_length for episodes in episode_counts])

    log_payload["task5/mean_return_by_episode_count"] = wandb.plot.line_series(
        xs=episode_counts,
        ys=mean_return_by_count,
        keys=TASK5_LABELS,
        title="Task 5 mean return by evaluation size",
        xname="Evaluation episodes",
    )
    log_payload["task5/mean_length_by_episode_count"] = wandb.plot.line_series(
        xs=episode_counts,
        ys=length_by_count,
        keys=TASK5_LABELS,
        title="Task 5 mean episode length by evaluation size",
        xname="Evaluation episodes",
    )

    if not any(np.isnan(value) for values in success_by_count for value in values):
        log_payload["task5/success_rate_by_episode_count"] = wandb.plot.line_series(
            xs=episode_counts,
            ys=success_by_count,
            keys=TASK5_LABELS,
            title="Task 5 success rate by evaluation size",
            xname="Evaluation episodes",
        )

    final_episodes = max(episode_counts)
    final_results = [result for result in results if result.episodes == final_episodes]
    for result in final_results:
        metric_prefix = result.label.replace("->", "_to_")
        log_payload[f"task5/final/{metric_prefix}/mean_return"] = result.mean_return
        log_payload[f"task5/final/{metric_prefix}/std_return"] = result.std_return
        log_payload[f"task5/final/{metric_prefix}/success_rate"] = result.success_rate
        log_payload[f"task5/final/{metric_prefix}/mean_length"] = result.mean_length

    for label, values in aggregate.items():
        metric_prefix = label.replace("->", "_to_")
        log_payload[f"task5/aggregate/{metric_prefix}/avg_mean_return"] = values["avg_mean_return"]
        log_payload[f"task5/aggregate/{metric_prefix}/avg_success_rate"] = values["avg_success_rate"]
        log_payload[f"task5/aggregate/{metric_prefix}/avg_mean_length"] = values["avg_mean_length"]

    wandb.log(log_payload)


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic
    algorithm = ALGORITHMS[args.algorithm]
    render_delay = args.render_delay if args.render else 0.0

    if any(episodes <= 0 for episodes in args.episodes):
        raise ValueError("All episode counts must be positive integers")

    run = init_wandb(args)

    source_eval_env = make_env("source", args.render)
    target_eval_env = make_env("target", args.render)

    try:
        source_model = load_model(algorithm, args.source_model_path, source_eval_env)
        target_model = load_model(algorithm, args.target_model_path, target_eval_env)

        results: list[EvaluationResult] = []
        for episodes in args.episodes:
            results.extend(
                [
                    evaluate_model(
                        "source->source",
                        "source",
                        "source",
                        source_model,
                        source_eval_env,
                        episodes,
                        args.seed,
                        deterministic,
                        render_delay,
                    ),
                    evaluate_model(
                        "source->target",
                        "source",
                        "target",
                        source_model,
                        target_eval_env,
                        episodes,
                        args.seed,
                        deterministic,
                        render_delay,
                    ),
                    evaluate_model(
                        "target->target",
                        "target",
                        "target",
                        target_model,
                        target_eval_env,
                        episodes,
                        args.seed,
                        deterministic,
                        render_delay,
                    ),
                ]
            )

        print_results(results)
        print_aggregate(results)
        if run is not None:
            log_to_wandb(results)

    finally:
        source_eval_env.close()
        target_eval_env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
