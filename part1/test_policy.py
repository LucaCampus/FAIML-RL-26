"""Test a trained policy on the Hopper environment"""
import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent import Policy, Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pt')
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--save_video', action='store_true', default=True)
    parser.add_argument('--video_dir', type=str, default='videos')
    args = parser.parse_args()

    render_mode = 'rgb_array' if args.save_video else ('human' if args.render else None)
    env = gym.make('Hopper-v4', render_mode=render_mode)

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda _: True)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space)
    policy.load_state_dict(torch.load(args.model, map_location='cpu'))
    policy.eval()

    agent = Agent(policy)

    episode_rewards = []
    for ep in range(args.n_episodes):
        done = False
        state, _ = env.reset()
        episode_reward = 0

        while not done:
            action, _ = agent.get_action(state, infer=True)
            state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f'Episode {ep+1}: reward = {episode_reward:.2f}')

    print(f'\nMean reward over {args.n_episodes} episodes: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}')
    env.close()


if __name__ == '__main__':
    main()
