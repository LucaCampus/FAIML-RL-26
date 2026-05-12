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

    # Get dimensions of state and action spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize policy and agent
    policy = Policy(state_dim, action_dim)
    agent = Agent(policy)

    num_episodes = 1000
    for ep in range(num_episodes):
        # Reset environment at the start of each episode
        state, _ = env.reset()
        # Initialize variables to track episode reward and done flag
        done = False
        episode_reward = 0

        while not done:
            # Get action from the agent's policy
            action, log_prob = agent.get_action(state)

            # Take a step in the environment using the action
            next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            # The episode is done if either terminated or truncated is True
            done = terminated or truncated

            # Store the outcome in the agent's memory
            agent.store_outcome(state, next_state, log_prob, reward, done)

            # Update the current state and accumulate the episode reward
            state = next_state
            episode_reward += reward

        # After the episode is done, update the policy using the collected experience
        agent.update_policy()

        print(f"Episode {ep}, Reward: {episode_reward}")

    # Close the environment after training is complete
    env.close()

if __name__ == '__main__':
    main()