"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gymnasium as gym
import panda_gym # type: ignore[import-not-found]
def main():
    render = True

    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        reward_type="dense",
    )
    
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 10

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated

            if render:
                env.render()


if __name__ == '__main__':
    main()



#What is an action here?
#Action space: Box(-1.0, 1.0, (3,), float32)
#We have a 3-dimensional vector with values from -1 to 1 representing the action (continuous).
#These are Cartesian velocity/displacement commands for the robot end-effector.
#The end goal is to push the box to the target location, so the actions control the movement of the 
#robot's end-effector in 3D space.