"""Test a trained policy on the Hopper environment"""
import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import wandb
from gymnasium.wrappers import RecordVideo
from agent import Policy, Agent

DEFAULT_WANDB_PROJECT = "REINFORCE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pt')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--save_video', action='store_true', default=True)
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb-project', type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument('--wandb-entity', type=str, default='terr1veneto')
    args = parser.parse_args()

    render_mode = 'rgb_array' if args.save_video else ('human' if args.render else None)
    env = gym.make('Hopper-v4', render_mode=render_mode)

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda _: True)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space)
    policy.load_state_dict(torch.load(args.model, map_location='cpu'), strict=False)
    policy.eval()

    agent = Agent(policy)
    model_tag = os.path.splitext(os.path.basename(args.model))[0]
    run_name = f"eval_{model_tag}_hopper_{args.n_episodes}ep_seed{args.seed}"
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config={
            "model": args.model,
            "env": "hopper",
            "n_episodes": args.n_episodes,
            "seed": args.seed,
        },
    )

    episode_rewards = []
    try:
        for ep in range(args.n_episodes):
            done = False
            state, _ = env.reset(seed=args.seed + ep)  # Reset environment to initial state 
            episode_reward = 0

            while not done:
                action, _ = agent.get_action(state, evaluation=True)
                state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)
            average_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            wandb.log({
                'episode_reward': episode_reward,
                'average_reward': average_reward,
                'std_reward': std_reward,
            }, step=ep)
            print(f'Episode {ep+1}: reward = {episode_reward:.2f}')

        run.summary['mean_reward'] = np.mean(episode_rewards)
        run.summary['std_reward'] = np.std(episode_rewards)
        run.summary['min_reward'] = np.min(episode_rewards)
        run.summary['max_reward'] = np.max(episode_rewards)
        print(f'\nMean reward over {args.n_episodes} episodes: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}\nMin: {np.min(episode_rewards):.2f}, Max: {np.max(episode_rewards):.2f}')
    finally:
        env.close()
        run.finish()


if __name__ == '__main__':
    main()
