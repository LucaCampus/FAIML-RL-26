"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gymnasium as gym

def main():
    render = True

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    state, info = env.reset()

    model = env.unwrapped.model

    for i in range(model.nbody):
        name = model.body(i).name
        mass = model.body_mass[i]
        print(f"{name}: {mass}")

    n_episodes = 50

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()

        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if render:
                env.render()

    env.close()

if __name__ == '__main__':
    main()

#What is the state space in the Hopper environment? Is it discrete or continuous?
#The state space in the Hopper environment is continuous. It consists of 11 dimensions, which include the position 
#and velocity of the hopper's body and its joints.   
#Box(low=-inf, high=inf, shape=(11,), dtype=float32)

#What is the action space in the Hopper environment? Is it discrete or continuous?
#The action space in the Hopper environment is continuous. It consists of 3 dimensions, 
#which represent the torques applied to the hopper's joints. 
#Box(low=-1.0, high=1.0, shape=(3,), dtype=float32)

#What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
#The masses of the Hopper’s bodies (obtained from the MuJoCo model) are: world: 0.0 kg, torso: 3.665 kg, thigh: 4.058 kg, 
#leg: 2.781 kg, foot: 5.316 kg. They are the same in both the source and target variants of the environment.

#what happens if you don't reset the environment even after the episode is over?
#If you don't reset the environment after the episode is over, the environment will continue to run in its current state.
#This means that the agent will keep taking actions and receiving rewards based on the last state of the episode.
#However, since the episode is already over, the rewards received will not be meaningful, and the agent will 
#not learn anything useful from these actions. Additionally, if the environment has a maximum episode length, 
#it may eventually terminate on its own, but this is not guaranteed. 
#Therefore, it is important to reset the environment after each episode to ensure that the agent can learn effectively.

#When exactly is the episode over?
#In the Hopper environment, an episode is considered over when either of the following conditions is met:
#1. (Truncated) The hopper falls over: This occurs when the height of the hopper's torso falls below a certain threshold 
# (usually around 0.7 meters). This indicates that the hopper has lost balance and is no longer able to continue standing.
#2. (Terminated) The episode reaches a maximum time step limit: This is a predefined limit on the number of time steps
# that an episode can last. If the hopper does not fall over within this time limit, the episode will end automatically.
#In both cases, the environment will return a "done" signal to indicate that the episode has ended, 
# and the agent will need to reset the environment before starting a new episode.

#What is an action in the Hopper environment?
#An action in the Hopper environment is a vector of three continuous values that represent the torques applied to the 
#hopper's joints. These torques control the movement of the hopper's legs and allow it to hop forward.
#The three values in the action vector correspond to the torques applied to the hip, knee, and ankle joints of the hopper.
#The values can range from -1.0 to 1.0, where negative values indicate a torque in one direction and positive values 
#indicate a torque in the opposite direction.
