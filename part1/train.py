"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch

from agent import Agent, Policy

def main():
    n_episodes = 5000
    log_interval = 10
    seed = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make('Hopper-v4', render_mode='human')
    env.action_space.seed(seed)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    agent = Agent(policy, device=device, algorithm='actor_critic')

    best_return = float('-inf')
    running_return = 0.0

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0

        while not done:
            action, action_log_prob = agent.get_action(state)
            action_np = action.detach().cpu().numpy()
            action_np = action_np.clip(env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            env.render()

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state
            episode_return += reward

        agent.update_policy()

        running_return = episode_return if episode == 1 else 0.95 * running_return + 0.05 * episode_return
        best_return = max(best_return, episode_return)

        if episode % log_interval == 0:
            print(
                f'Episode {episode:5d} | '
                f'return {episode_return:8.2f} | '
                f'running {running_return:8.2f} | '
                f'best {best_return:8.2f}'
            )

    env.close()

if __name__ == '__main__':
    main()
