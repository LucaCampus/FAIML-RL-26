"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
from agent import Policy, Agent

def main():
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim)
    agent = Agent(policy)

    num_episodes = 500

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            done = terminated or truncated

            agent.store_outcome(state, next_state, log_prob, reward, done)

            state = next_state
            episode_reward += reward

        agent.update_policy()

        print(f"Episode {ep}, Reward: {episode_reward}")

    env.close()

if __name__ == '__main__':
    main()