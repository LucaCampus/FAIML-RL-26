import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import numpy.core.numeric as numpy_numeric
from stable_baselines3 import SAC
import panda_gym  # noqa: F401 - required so Panda envs are registered

# #TODO: useful?
# # Compatibility alias for models/checkpoints saved with a NumPy build that used this module path.
# sys.modules["numpy._core.numeric"] = numpy_numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO model zip file (e.g., ppo_panda_push.zip)",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500, 
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


    # This quick evaluator supports SAC checkpoints (which is the best performing model)
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
            # Deterministic evaluation uses the policy mean; --stochastic samples from the policy.
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

    # Report raw returns plus success rate when the environment provides is_success.
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
