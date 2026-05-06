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
    use_baseline = True # change to False for vanilla REINFORCE
    mean_rewards = []
    for baseline_value in range (20):
        agent = Agent(policy, use_baseline=use_baseline, baseline_value=baseline_value)

        print("Used baseline:", agent.baseline_value if agent.use_baseline else None)

        num_episodes = 500
        tot = 0
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

            # print(f"Episode {ep}, Reward: {episode_reward}")
            tot += episode_reward

        # Close the environment after training is complete
        print(f"Mean Reward: {tot/num_episodes}")
        mean_rewards[baseline_value] = tot/num_episodes
        env.close()
    best_index = mean_rewards.index(max(mean_rewards))
    print(f"Best baseline index: {best_index}")

    print(f"Best mean reward: {mean_rewards[best_index]}")


if __name__ == '__main__':
    main()

   
# The REINFORCE algorithm is able to learn a control policy for the Hopper environment, 
# as shown by the increasing trend in episode rewards. However, the training process is highly unstable, 
# with significant variance in performace across episodes. 
# This behavior is expected due to the high variance nature of the Monte Carlo policy gradient estimator.  

#Analyze the performance of the trained policies in terms of reward and time consumption:
#Compared to the no-baseline case, using a constant baseline of 20 leads to higher variability in performance, 
#especially during early training. While the no-baseline approach results in more stable but lower rewards, 
#the constant baseline enables the agent to achieve higher peak performance. 
#However, because the baseline is not adapted to the scale of returns, 
#it introduces noisy and sometimes misleading updates, resulting in instability and occasional performance drops.

#How would you choose a good value for the baseline?
#A good baseline should approximate the expected return so that the advantage reflects whether 
#an outcome is better or worse than average. In practice, this can be achieved by using the mean return 
#of an episode or a running average over multiple episodes. 
#Fixed constant baselines are generally suboptimal because they do not adapt to changes
#in the reward scale during training.

#How does the baseline affect the training, and why?
#The baseline reduces variance by turning rewards into relative performance, 
#making learning more stable without changing the final solution.
#However, if the baseline is not well-chosen (e.g., too high or too low), it can introduce bias and lead to
#suboptimal policies. A poorly chosen baseline can cause the agent to underestimate or overestimate the advantage, 
#leading to noisy updates and slower convergence.
#without baseline -> gradient ∝ log π(a|s) * return
#with baseline -> gradient ∝ log π(a|s) * (return - baseline)
