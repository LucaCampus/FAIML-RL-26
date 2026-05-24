"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gymnasium as gym
import time
import panda_gym # type: ignore[import-not-found]
def main():
    render = True

    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        # type="target",
        reward_type="dense",
    )
    
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 500

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            time.sleep(0.04)

            action = env.action_space.sample()  # Sample random action
            action = 0.05 * env.action_space.sample()

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            time.sleep(0.04)

            done = terminated or truncated


            if render:
                env.render()


if __name__ == '__main__':
    main()