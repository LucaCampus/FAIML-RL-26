"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
import wandb
import numpy as np
from agent import Policy, Agent

def main():
    render = False

    if render:
        env = gym.make('Hopper-v4', render_mode='human',  healthy_angle_range=(-0.5, 0.5))
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array',  healthy_angle_range=(-0.5, 0.5))

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    # Get dimensions of state and action spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize policy and agent
    policy = Policy(state_dim, action_dim)
    use_baseline = True  # change to False for vanilla REINFORCE
    agent = Agent(policy, device=torch.device('cuda'), algorithm = 'reinforce', use_baseline=use_baseline)
    print(agent.train_device)

    if use_baseline:
        print("Using baseline:", use_baseline)

    num_episodes = 20000

        #
    # WANDB
    #
    wandb.init(
        project="hopper-rl",
        name="hopper-reinforce-baseline",
        config={
            "environment": "Hopper-v4",
            "algorithm": agent.algorithm,
            "gamma": agent.gamma,
            "episodes": num_episodes,
            "device": str(agent.train_device)
        }
    )

    #
    # WATCH MODEL
    #
    wandb.watch(policy, log="all")

    #
    # TRAINING STATS
    #
    rewards_history = []

    best_avg_reward = -1e9

    for ep in range(num_episodes):
        # Reset environment at the start of each episode
        state, _ = env.reset()
        # Initialize variables to track episode reward and done flag
        done = False
        episode_reward = 0

        while not done:
            # Get action from the agent's policy
            action, log_prob, state_value = agent.get_action(state)

            # Take a step in the environment using a valid clipped action
            action_np = action.detach().cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            # The episode is done if either terminated or truncated is True
            done = terminated or truncated
            if done:
                print(
                    "terminated:", terminated,
                    "truncated:", truncated,
                    "height:", env.unwrapped.data.qpos[1],
                    "angle:", env.unwrapped.data.qpos[2],
                )

            # Store the outcome in the agent's memory
            agent.store_outcome(state, next_state, log_prob, state_value, reward, done)

            # Update the current state and accumulate the episode reward
            state = next_state
            episode_reward += reward

        # After the episode is done, update the policy using the collected experience
        agent.update_policy()

        #
        # SAVE REWARD
        #
        rewards_history.append(episode_reward)

        #
        # MOVING AVERAGE
        #
        avg_reward = np.mean(rewards_history[-100:])

        #
        # SAVE BEST MODEL
        #
        if avg_reward > best_avg_reward:

            best_avg_reward = avg_reward

            torch.save(policy.state_dict(),"best_model.pt")

            wandb.save("best_model.pt")

            print(f"\n[BEST MODEL SAVED] "f"Avg100 Reward: {best_avg_reward:.2f}\n")

            #
            # PRINT LOG
            #
            print(
                f"Episode: {ep:5d} | "
                f"Reward: {episode_reward:10.2f} | "
                f"Avg100: {avg_reward:10.2f}"
            )

            #
            # WANDB LOGGING
            #
            log_dict = {"episode": ep, "reward": episode_reward, "avg_reward_100": avg_reward}

            #
            # OPTIONAL LOSSES
            #
            if hasattr(agent, "last_actor_loss"):
                log_dict["actor_loss"] = agent.last_actor_loss

            if hasattr(agent, "last_critic_loss"):
                log_dict["critic_loss"] = agent.last_critic_loss

            wandb.log(log_dict)

        print(f"Episode {ep}, Reward: {episode_reward}")

    torch.save(policy.state_dict(), "hopper_policy.pt")

    # Close the environment after training is complete
    env.close()

    wandb.finish()

    print("Training completed.")

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
#However, if the baseline is not well-chosen (e.g., too high or too low), it may fail to reduce variance effectively and can 
#lead to unstable or slower learning. A poorly chosen baseline can cause the agent to underestimate or overestimate the advantage, 
#leading to noisy updates and slower convergence.
#without baseline -> gradient ∝ log π(a|s) * return
#with baseline -> gradient ∝ log π(a|s) * (return - baseline)
#In terms of computational cost, using a constant baseline introduces almost no additional
#overhead compared to standard REINFORCE, since it only requires subtracting a scalar value from the returns.

#Analyze the performance of the trained policies in terms of reward and time consumption.
#The Actor-Critic approach achieved strong performance in its best run, reaching high rewards and effective policy learning.
#However, training exhibited high variance across executions and required significant computational time due 
#to the joint optimization of actor and critic networks.

#Compare the results with the REINFORCE algorithm you have previously obtained, highlighting 
#any notable differences in terms of learning stability and convergence speed.
#“Compared to REINFORCE, the Actor-Critic algorithm achieved faster convergence and higher rewards 
#thanks to the critic-guided updates. While both methods exhibited some instability, 
#Actor-Critic generally provided more efficient and stable learning behavior.”
