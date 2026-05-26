import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import numpy.core.numeric as numpy_numeric
from stable_baselines3 import SAC
import panda_gym  # noqa: F401 - required so Panda envs are registered

sys.modules["numpy._core.numeric"] = numpy_numeric


def evaluate(
    model_path: str,
    n_episodes: int,
    deterministic: bool,
    render: bool,
    env_type: str,
    render_delay: float,
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")

    sim = env.unwrapped.task.sim
    object_body_id = sim._bodies_idx["object"]
    mass = sim.physics_client.getDynamicsInfo(object_body_id, -1)[0]
    print(f"Evaluation env_type={env_type}, object mass={mass}")

    model = SAC.load(
        model_path,
        env=env,
        custom_objects={
            "action_space": env.action_space,
            "observation_space": env.observation_space,
        },
    )

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action,_ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if render and render_delay > 0:
                time.sleep(render_delay)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=r"C:\Users\lucac\Documents\UNI\\2_MAGISTRALE\ANNO_1\Semestre_2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\FAIML-RL-26\part2\Model_1M_Source.zip",
        help="Path to a PPO model zip file (e.g., ppo_panda_push.zip)",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=50, 
        help="Number of eval episodes"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.03,
        help="Seconds to wait after each rendered step",
    )
    parser.add_argument(
        "--env-type",
        type=str, default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        render_delay=args.render_delay,
    )
