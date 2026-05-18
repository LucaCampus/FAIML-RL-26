import argparse
from collections import deque
from xml.parsers.expat import model

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback
from rand_wrapper import RandomizationWrapper

# This allows you to run the same Python file with 
# different settings directly from the terminal instead of hardcoding values in the script.
def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Train SAC and PPO on PandaPush-v3")
    
    #We can run --algo ppo or --algo sac to choose the algorithm we want to train with.
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["ppo", "sac"],
        help="RL algorithm",
)
    #This controls the domain randomization method.
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )

    #Controls if we train on source or target enviroment.
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )

    #Controls total environment interactions.
    #A timestep is ONE interaction with the environment.
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Number of training timesteps",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run = wandb.init(
    project="panda-push-rl",
    config={
        "algorithm": args.algo,
        "timesteps": args.timesteps,
        "sampling_strategy": args.sampling_strategy,
        "env_type": args.env_type,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        reward_type="dense",
    )

    if args.sampling_strategy == "none":

        if args.env_type == "source":
            env = RandomizationWrapper(
            env,
            mass_range=(1.0, 1.0),
            mode="none",
            )

        elif args.env_type == "target":
            env = RandomizationWrapper(
            env,
            mass_range=(5.0, 5.0),
            mode="none",
        )

    elif args.sampling_strategy == "udr":
        env = RandomizationWrapper(
        env,
        mass_range=(0.5, 6.0),
        mode="udr",
    )
    
    elif args.sampling_strategy == "adr":

        env = RandomizationWrapper(
        env,
        mass_range=(0.5, 5.5),
        mode="adr",
    )

    
    #TODO: create model and train it
    if args.algo == "ppo":

        #This creates a PPO agent from Sable-Baseline3.
        #We do not implement the gradients manually anymore SB3 takes care of that for us.
        model = PPO(
            #Since our observation space is a dictionary, we need to use the MultiInputPolicy which can handle that.
            #This combines the different parts of the observation space into a single input for the neural network.
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            #Stores training metrics for visualization in TensorBoard.
            tensorboard_log="./tensorboard_logs/"
        )

    elif args.algo == "sac":

        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )

    # Train model
    model.learn(
        total_timesteps=args.timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    # Save model
    save_name = (
        f"{args.algo}_push_"
        f"{args.sampling_strategy}_"
        f"{args.env_type}_"
        f"{args.timesteps // 1000}k"
    )
    
    # TODO: model.save(save_name)
    model.save(save_name)

    print(f"Model saved as {save_name}")

    #TODO: add randomization wrapper here
    
    env.close()
    wandb.finish()
    
if __name__ == "__main__":
    main()


#How does PPO work?
#PPO is a actor-critic method that works in cycles of collecting data and updating the policy.
#1. Data collection: The agent interacts with the environment using its current policy to collect trajectories of states, 
#actions, rewards, and next states. This creates a rollout of experience that the agent can learn from.
#2. Compute advantages: The agent computes the advantage to determine wether the action was betetr than expected.
#3. Update policy: The agent updates its policy using the collected data and the computed advantages.
#The main idea of PPO is that it prevents HUGE policy changes. It compares new policy to old policy
#and clips updates if they become too large.
#PPO is an on-policy algorithm, it ONLY learns from fresh data collected from the current policy. 
#It does not reuse old data like off-policy algorithms (e.g., SAC).
#This wastes experience but improves stability.

#How does SAC work?
#Unlike PPO, SAC is an off-policy algorithm that can learn from past experience stored in a replay buffer.
#Sac store experience in memory and reuses it many times.
#This makes it Much more sample efficient, which is really good with robots since
#they learn more from the same data.
#SAC also learns the Q-function, which estimates the expected return of taking an action in a state, so that
#the actor can learn to select actions that maximize this expected return.
#But SAC doesn't just maximize rewards, it maximizes rewards + exploration bonus (entropy). 
#This encourages the agent to explore more and not get stuck in local optima.
#Entropy in facts means randomness; high entropy means that our policy allows diverse actions and
#exploration, while low entropy means that our policy is more deterministic and focused on exploitation.
#Without exploration, the robot may get stuck and repeat mediocre behaviour. 
#SAC encourages continued exploration.

#SAC performs better than PPO on the PandaPush task because it is an off-policy algorithm that can reuse past experiences 
#through a replay buffer, making it much more sample efficient. 
#This allows SAC to learn more effectively from the same interactions with the environment, 
#which is especially important in robotic tasks where data collection is expensive. 
#In addition, SAC maximizes both reward and entropy, encouraging continuous exploration and preventing 
#the agent from getting stuck in poor behaviors too early. In the experiments, SAC achieved higher rewards, 
#a significantly higher success rate, and shorter episode lengths than PPO, showing that it learned more efficient
#and reliable pushing strategies.