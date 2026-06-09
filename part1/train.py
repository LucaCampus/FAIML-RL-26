"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import argparse
import numpy as np
import torch
import gymnasium as gym
import wandb
from agent import Policy, Agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--baseline', type=float, default=None)
    parser.add_argument('--algorithm', type=str, default='reinforce', choices=['reinforce', 'actor-critic'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    # setup policy and agent

    state_space = env.observation_space.shape[0]  # 11
    action_space = env.action_space.shape[0]      # 3

    policy = Policy(state_space, action_space)
    agent = Agent(policy, gamma=args.gamma, lr=args.lr, baseline=args.baseline, algorithm=args.algorithm)

    run_name = f"{args.algorithm}_baseline{args.baseline}_ep_{args.n_episodes}_gamma{args.gamma}_lr{args.lr}_seed{args.seed}"
    baseline_str = f"_baseline{args.baseline}" if args.baseline is not None else ""
    model_name = f"best_model_{args.algorithm}{baseline_str}.pt"
    wandb.init(entity="terr1veneto", project='FinalRunsPart1', name=run_name, config=vars(args))

    best_avg_reward = -float('inf')
    rewards_history=[]

    for ep in range(args.n_episodes):
        done = False
        state, info = env.reset(seed=args.seed if ep == 0 else None)  
        episode_reward = 0

        while not done:  
            action, log_prob = agent.get_action(state)  # Sample action from policy
            next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())  # Step the simulator to the next timestep
            done = terminated or truncated

            agent.store_outcome(state, next_state, log_prob, reward, done)
            state = next_state
            episode_reward += reward

        loss = agent.update_policy()
        rewards_history.append(episode_reward)
        average_reward = np.mean(rewards_history[-100:])
        std_reward = np.std(rewards_history[-100:])
        
        wandb.log({'episode_reward': episode_reward, 'loss': loss, 'average_reward': average_reward, 'std_reward': std_reward}, step=ep)
        print(f'Episode {ep+1} reward: {episode_reward:.2f}, average reward: {average_reward:.2f}, loss: {loss:.4f}')

        if average_reward > best_avg_reward:
            best_avg_reward = average_reward
            torch.save(policy.state_dict(), model_name)
            wandb.save(model_name)

    wandb.finish()   

if __name__ == '__main__':
    main()
