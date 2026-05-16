import argparse

import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Hopper policy.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="hopper_policy.pt",
        help="Path to the saved PyTorch policy weights.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of inference episodes to run.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Hopper environment in a window.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save mp4 videos of the evaluation episodes.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of using the deterministic mean action.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save_video:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"
    else:
        render_mode = None

    # render_mode = "human"

    env = gym.make("Hopper-v4", render_mode=render_mode,  healthy_angle_range=(-0.5, 0.5))

    if args.save_video:
        # This will save files into a new folder named 'videos'
        # episode_trigger records every episode we run
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder="videos", 
            episode_trigger=lambda episode_id: True,
            name_prefix="hopper_eval"
        )
        print("Video recording enabled. Videos will be saved to the './videos' folder.")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    agent = Agent(policy, device=device)

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(state, evaluation=True)

            action = action.detach().cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)

        print(f"Episode {episode:03d} | reward = {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
