"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    Resetting th environment puts the agent back to the initial state,
    adds a small noise to the join angles (so each episode starts differently) and
    returns the initial observation. 
    Without reset the environment has no valid physical state from, so it raises an error.
    
    When exactly is the episode over?
    The episode is over when the agent falls down, 
    which happens when the height of the torso is less than 0.7 or 
    when the angle of the torso is greater than 0.2 radians.
    
    What is an action here?
    An action is a 3-dimensional vector of torques applied 
    to the 3 joints of the hopper.
    The action space is continuous, so the agent can apply any torque.
"""
import gymnasium as gym

def main():
    render = True

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    model = env.unwrapped.model

    for i in range(model.nbody):
        name = model.body(i).name
        mass = model.body(i).mass
        print(f'{name}:{mass}')

    n_episodes = 50

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated

            if render:
                env.render()


    env.close()


if __name__ == '__main__':
    main()