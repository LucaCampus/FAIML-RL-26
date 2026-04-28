import gymnasium as gym

def main():
    render = True

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 50000
    a=1
    
    for ep in range(n_episodes):  
        done = False
        if(a==1):
            state, info = env.reset()  # Reset environment to initial state
            a = 0

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action
            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated

            if render:
                env.render()


if __name__ == '__main__':
    main()