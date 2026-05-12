"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
from agent import Policy, Agent

def main():
    render = False

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    # Get dimensions of state and action spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize policy and agent
    policy = Policy(state_dim, action_dim)
    use_baseline = False # change to False for vanilla REINFORCE
    agent = Agent(policy, use_baseline=use_baseline, baseline_value=20)

    print("Used baseline:", agent.baseline_value if agent.use_baseline else None)

    num_episodes = 500
    episodes_reward = 0
    for ep in range(num_episodes):
        # Reset environment at the start of each episode
        state, _ = env.reset()
        # Initialize variables to track episode reward and done flag
        done = False

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


    # Close the environment after training is complete
    print(f"Mean Reward: {episodes_reward/num_episodes}")
    env.close()


if __name__ == '__main__':
    main()

   
